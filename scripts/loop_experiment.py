"""Durable, bounded experiments; never equate offline gates with live training."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import threading
import uuid
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener

from env_factory.evidence.execution_provenance import (
    collect_execution_provenance,
    verify_execution_provenance,
)
from env_factory.tasks.task_similarity import task_partition_isolation
from env_factory.tasks.task_portability import valid_task_lineage
from env_factory.evidence.material_artifacts import digest_json

MODEL = "gpt-6-luna"
ACTIVE_PROCESSES = set()
ACTIVE_CONTAINER_CIDFILES = set()
PROCESS_LOCK = threading.Lock()
CONTAINER_PORT_LOCK = threading.Lock()
PROCESS_DEADLINE = None
BUDGET_STATE_PATH = None
ACTIVE_RUN_STARTED = None


def finish_active_budget(status):
    """Persist active execution time; paused wall time consumes no budget."""
    global ACTIVE_RUN_STARTED
    if BUDGET_STATE_PATH is None or ACTIVE_RUN_STARTED is None:
        return
    try:
        state = json.loads(BUDGET_STATE_PATH.read_text()) if BUDGET_STATE_PATH.exists() else {}
    except (OSError, json.JSONDecodeError):
        state = {}
    elapsed = max(0.0, time.time() - ACTIVE_RUN_STARTED)
    state.update(
        active_seconds=float(state.get("active_seconds", 0)) + elapsed,
        status=status,
        updated_at=time.time(),
    )
    state.pop("run_started_at", None)
    write_json(BUDGET_STATE_PATH, state)
    ACTIVE_RUN_STARTED = None


def interrupt(signum, frame):
    with PROCESS_LOCK:
        for pid in tuple(ACTIVE_PROCESSES):
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        cidfiles = tuple(ACTIVE_CONTAINER_CIDFILES)
    for cidfile in cidfiles:
        try:
            container_id = cidfile.read_text(encoding="utf-8").strip()
            if container_id:
                subprocess.run(
                    ["docker", "rm", "-f", container_id],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=10, check=False,
                )
        except (OSError, subprocess.SubprocessError):
            pass
        finally:
            cidfile.unlink(missing_ok=True)
    finish_active_budget("paused")
    raise SystemExit(128 + signum)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_process(command, cwd, log, timeout):
    """Own a process group so timeout/cancellation does not orphan builders."""
    started = time.monotonic()
    if PROCESS_DEADLINE is not None:
        timeout = min(timeout, PROCESS_DEADLINE - time.time())
        if timeout <= 0:
            return {"exit_code": 124, "timed_out": True, "seconds": 0, "budget_exhausted": True}
    process = None
    with log.open("w") as stream:
        try:
            process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            with PROCESS_LOCK:
                ACTIVE_PROCESSES.add(process.pid)
            code = process.wait(timeout=timeout)
            return {"exit_code": code, "timed_out": False, "seconds": time.monotonic() - started}
        except subprocess.TimeoutExpired:
            return {"exit_code": 124, "timed_out": True, "seconds": time.monotonic() - started}
        finally:
            if process is not None:
                # Kill remaining descendants even if their immediate parent exited.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=2)
                    finally:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError):
                            pass
                except (ProcessLookupError, PermissionError):
                    pass
                process.wait()
                with PROCESS_LOCK:
                    ACTIVE_PROCESSES.discard(process.pid)


def source_digest(project):
    digest = hashlib.sha256()
    for folder in ("src", "scripts", "examples", "docs", "tests"):
        for path in sorted((project / folder).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".sh", ".md"}:
                digest.update(str(path.relative_to(project)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def input_digest(path):
    digest = hashlib.sha256()
    for item in sorted(path.parent.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(path.parent)).encode())
            digest.update(item.read_bytes())
    return digest.hexdigest()


def report_task_documents(report):
    documents = []
    for job in report.get("jobs", []):
        result = job.get("result", {}) if isinstance(job, dict) else {}
        path = Path(str(result.get("task_path", "")))
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            documents.append(value)
    return documents


def sandbox_image_tag(output: Path) -> str:
    """Give every concurrent attempt a collision-resistant local image tag."""
    identity = hashlib.sha256(str(output.resolve()).encode("utf-8")).hexdigest()[:20]
    return f"envfactory-sandbox-{identity}"


def stop_rollout_container(project, output, container_id, config):
    result = run_process(
        ["docker", "rm", "-f", container_id], project,
        output / "live_container_stop.log",
        min(config.get("rollout_timeout", 600), 60),
    )
    (output / ".live_rollout.cid").unlink(missing_ok=True)
    with PROCESS_LOCK:
        ACTIVE_CONTAINER_CIDFILES.discard(output / ".live_rollout.cid")
    return result


def start_rollout_container(project, output, image_tag, config):
    """Start the validated image on a random loopback port for live rollout."""
    os.environ.setdefault("SANDBOX_TRAINER_API_KEY", "local-rollout-trainer")
    for target, source in {
        "SANDBOX_LLM_API_KEY": "LLM_API_KEY",
        "SANDBOX_LLM_BASE_URL": "LLM_BASE_URL",
        "SANDBOX_LLM_MODEL": "LLM_MODEL",
        "SANDBOX_LLM_TIMEOUT_SECONDS": "LLM_TIMEOUT",
    }.items():
        if not os.environ.get(target) and os.environ.get(source):
            os.environ[target] = os.environ[source]
    os.environ["SANDBOX_EVALUATOR_MOCK"] = "0"
    os.environ["SANDBOX_MUTATION_MODE"] = "disabled"
    cid_path = output / ".live_rollout.cid"
    cid_path.unlink(missing_ok=True)
    with PROCESS_LOCK:
        ACTIVE_CONTAINER_CIDFILES.add(cid_path)
    command_prefix = [
        "docker", "run", "-d", "--rm", "--cidfile", str(cid_path),
        "--read-only", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges:true",
        "--pids-limit", "128", "--memory", "512m", "--cpus", "1.0",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--tmpfs", "/app/.runtime:rw,nosuid,size=64m,uid=10001,gid=10001,mode=0700",
    ]
    for name in (
        "SANDBOX_TRAINER_API_KEY", "SANDBOX_LLM_API_KEY",
        "SANDBOX_LLM_BASE_URL", "SANDBOX_LLM_MODEL",
        "SANDBOX_LLM_TIMEOUT_SECONDS", "SANDBOX_LLM_MAX_RETRIES",
        "SANDBOX_EVALUATOR_MOCK", "SANDBOX_MUTATION_MODE",
    ):
        if os.environ.get(name):
            command_prefix.extend(["--env", name])
    start_attempts = []
    host_port = 0
    with CONTAINER_PORT_LOCK:
        for attempt in range(1, 4):
            cid_path.unlink(missing_ok=True)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                probe.bind(("127.0.0.1", 0))
                host_port = probe.getsockname()[1]
            started = run_process(
                [
                    *command_prefix,
                    "--publish", f"127.0.0.1:{host_port}:8000", image_tag,
                ],
                project,
                output / (
                    "live_container_start.log" if attempt == 1
                    else f"live_container_start_{attempt}.log"
                ),
                min(config.get("rollout_timeout", 600), 120),
            )
            start_attempts.append({"attempt": attempt, **started})
            if started["exit_code"] == 0 and cid_path.is_file():
                break
            if cid_path.is_file():
                failed_container = cid_path.read_text(encoding="utf-8").strip()
                if failed_container:
                    stop_rollout_container(
                        project, output, failed_container, config
                    )
                with PROCESS_LOCK:
                    ACTIVE_CONTAINER_CIDFILES.add(cid_path)
    if started["exit_code"] != 0 or not cid_path.is_file():
        with PROCESS_LOCK:
            ACTIVE_CONTAINER_CIDFILES.discard(cid_path)
        return {
            "started": False, "process": started,
            "start_attempts": start_attempts,
        }
    container_id = cid_path.read_text(encoding="utf-8").strip()

    def docker_value(arguments, log_name):
        path = output / log_name
        result = run_process(
            ["docker", *arguments], project, path,
            min(config.get("rollout_timeout", 600), 60),
        )
        value = path.read_text(encoding="utf-8").strip() if path.is_file() else ""
        return result, value

    inspected, image_id = docker_value(
        ["inspect", "--format", "{{.Image}}", container_id],
        "live_container_inspect.log",
    )
    import re
    if (
        inspected["exit_code"] != 0
        or re.fullmatch(r"sha256:[0-9a-f]{64}", image_id) is None
    ):
        stop_rollout_container(project, output, container_id, config)
        return {"started": False, "process": started, "inspect": inspected}
    base_url = f"http://127.0.0.1:{host_port}"
    loopback = build_opener(ProxyHandler({}))
    deadline = time.monotonic() + min(30, config.get("rollout_timeout", 600))
    healthy = False
    while time.monotonic() < deadline:
        try:
            with loopback.open(base_url + "/health", timeout=2) as response:
                healthy = response.status == 200
            if healthy:
                break
        except (URLError, TimeoutError, OSError):
            time.sleep(0.2)
    if not healthy:
        stop_rollout_container(project, output, container_id, config)
        return {"started": False, "process": started, "health": False}
    return {
        "started": True,
        "container_id": container_id,
        "image_id": image_id,
        "base_url": base_url,
        "start_attempts": start_attempts,
        "security": {
            "read_only_root": True, "cap_drop": "ALL",
            "no_new_privileges": True, "non_root_user": True,
        },
    }


def failure(stage, detail, **extra):
    targets = {"generation": "task_pipeline", "task_quality": "task_contract",
               "build": "sandbox_builder", "offline_validation": "runtime_or_contract",
               "data_governance": "inspect_outbound_payload",
               "trajectory_privacy": "inspect_policy_visible_trajectory",
               "infrastructure": "runner_or_provider", "live_rollout": "inspect_live_trajectory",
               "live_reward_calibration": "inspect_reward_evaluator"}
    codes = {"generation": "GEN_SEMANTIC", "task_quality": "TASK_BUILDABILITY",
             "build": "BUILD_BUSINESS", "offline_validation": "REWARD_OR_RUNTIME",
             "data_governance": "DATA_GOVERNANCE",
             "trajectory_privacy": "TRAJECTORY_PRIVACY",
             "infrastructure": "INFRA", "live_rollout": "ROLLOUT_ENVIRONMENT",
             "live_reward_calibration": "LIVE_REWARD_CALIBRATION"}
    return {"passed": False, "failure_class": stage, "repair_target": targets.get(stage, "inspect"),
            "failure_code": codes.get(stage, "UNKNOWN"), "detail": detail,
            "live_rollout_verified": False, **extra}


BUILD_PHASE_FAILURES = {
    "node_development": ("BUILD_MODULE", "sandbox_builder"),
    "semantic_review": ("BUILD_SEMANTIC_REVIEW", "sandbox_builder"),
    "defect_repair": ("BUILD_DEFECT_REPAIR", "sandbox_builder"),
    "defect_validation": ("BUILD_DEFECT_VALIDATION", "sandbox_builder"),
    "contract_validation": ("BUILD_CONTRACT", "sandbox_builder"),
    "trace_validation": ("BUILD_TRACE", "sandbox_builder"),
    "runtime_validation": ("BUILD_RUNTIME_INTEGRITY", "platform_runtime_boundary"),
    "mutation_testing": ("BUILD_MUTATION", "sandbox_builder"),
    "training_readiness": ("BUILD_TRAINING_READINESS", "sandbox_builder"),
    "agentic_training_value": ("BUILD_AGENTIC_VALUE", "sandbox_builder"),
    "docker_build": ("BUILD_DOCKER", "container_builder"),
}


def classify_build_failure(output):
    """Use the builder's final gate instead of labeling every exit as business logic."""
    try:
        status = json.loads((output / "status.json").read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    if status.get("status") != "failed":
        return {}
    phase = status.get("failed_phase")
    if not isinstance(phase, str) or not phase:
        return {}
    if status.get("failure_code") == "INFRA" and status.get("failure_category") == "infrastructure":
        return {"build_failed_phase": phase, "failure_class": "infrastructure",
                "failure_code": "INFRA", "repair_target": "runner_or_provider"}
    code, target = BUILD_PHASE_FAILURES.get(
        phase, ("BUILD_BUSINESS", "sandbox_builder")
    )
    return {"build_failed_phase": phase, "failure_code": code, "repair_target": target}


def summarize(results, threshold, *, targets=None):
    total = len(results)
    qualified_items = [
        item for item in results
        if item.get("passed") is True and item.get("score", 0) >= threshold
    ]
    qualified = len(qualified_items)
    generated = sum(bool(item.get("task_path")) for item in results)
    task_qualified = sum(
        bool(item.get("task_score", {}).get("eligible"))
        and item.get("task_score", {}).get("score", 0) >= threshold
        for item in results
    )
    offline_qualified = sum(
        item.get("sandbox_score", {}).get("passed") is True
        and item.get("sandbox_score", {}).get("score", 0) >= threshold
        for item in results
    )
    end_to_end_rate = qualified / total if total else 0
    task_good_yield = task_qualified / total if total else 0
    sandbox_build_yield = offline_qualified / task_qualified if task_qualified else 0
    mean_qualified_score = (
        sum(item.get("score", 0) for item in qualified_items) / qualified if qualified else 0
    )
    summary = {
        "requested": total, "generated": sum(bool(item.get("task_path")) for item in results),
        "generation_completion_rate": generated / total if total else 0,
        "task_qualified": task_qualified,
        "task_good_yield": task_good_yield,
        "qualified": qualified,
        "sandbox_build_yield": sandbox_build_yield,
        "offline_qualified": offline_qualified,
        "end_to_end_rate": end_to_end_rate,
        "mean_score_all_requests": sum(item.get("score", 0) for item in results) / total if total else 0,
        "mean_qualified_score": mean_qualified_score,
        "all_passed": total > 0 and qualified == total,
        "failures": dict(Counter(item.get("failure_class", "unknown") for item in results if not item.get("passed"))),
        "failure_codes": dict(Counter(item.get("failure_code", "UNKNOWN") for item in results if not item.get("passed"))),
        "live_rollout_verified": total > 0 and all(item.get("live_rollout_verified") for item in results),
        "by_category": {category: {"requested": len(items), "qualified": sum(
                            item.get("passed") is True and item.get("score", 0) >= threshold
                            for item in items
                        )}
                        for category in sorted({item.get("category", "unknown") for item in results})
                        for items in [[item for item in results if item.get("category", "unknown") == category]]},
    }
    infrastructure_failures = sum(
        item.get("failure_class") == "infrastructure"
        or item.get("failure_code") == "INFRA"
        for item in results
    )
    # Provider, network, database and runner outages are not measurements of
    # EnvFactory quality and must not consume the user's quality-round budget.
    summary["valid_quality_round"] = not (total > 0 and infrastructure_failures == total)
    if targets:
        category_floor = targets.get("category_rate", 0)
        category_ok = all(
            values["qualified"] / values["requested"] >= category_floor
            for values in summary["by_category"].values() if values["requested"]
        )
        summary["target_met"] = (
            total > 0
            and task_good_yield >= targets.get("task_yield", 0)
            and sandbox_build_yield >= targets.get("build_yield", 0)
            and end_to_end_rate >= targets.get("end_to_end_rate", 0)
            and mean_qualified_score >= targets.get("qualified_mean", threshold)
            and category_ok
        )
    else:
        summary["target_met"] = summary["all_passed"]
    return summary


def summarize_holdout(
    results, threshold, *, expected_count, end_to_end_target,
    rollout_success_target, previous_seeds=(), previous_task_digests=(),
    minimum_materialized=None,
):
    """Apply the stricter, distribution-shifted release gate."""
    summary = summarize(results, threshold)
    live_results = [
        item["live_rollout"] for item in results
        if isinstance(item.get("live_rollout"), dict)
    ]
    qualified_results = [
        item for item in results
        if item.get("passed") is True and item.get("score", 0) >= threshold
    ]
    source_counts = {platform: sum(item.get("dataset_platform") == platform for item in results)
                     for platform in ("kaggle", "data_gov_hk")}
    source_balance_verified = (len(results) == expected_count and expected_count % 2 == 0
                               and all(count == expected_count // 2
                                       for count in source_counts.values()))
    seeds = [item.get("sample_seed") for item in results]
    task_digests = []
    for item in results:
        path = Path(item.get("task_path", ""))
        if path.is_file():
            task_digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
    minimum_materialized = (
        expected_count if minimum_materialized is None else minimum_materialized
    )
    fresh_tasks_verified = (
        len(results) == expected_count
        and all(isinstance(seed, int) for seed in seeds)
        and len(set(seeds)) == expected_count
        and set(seeds).isdisjoint(previous_seeds)
        and len(task_digests) >= minimum_materialized
        and len(set(task_digests)) == len(task_digests)
        and set(task_digests).isdisjoint(previous_task_digests)
    )
    qualified_rollout_floor_met = bool(qualified_results) and all(
        isinstance(item.get("live_rollout"), dict)
        and item["live_rollout"].get("agent_success_rate", 0) >= rollout_success_target
        for item in qualified_results
    )
    all_episodes_environment_clean = bool(live_results) and all(
        report.get("all_episodes_environment_clean") is True
        for report in live_results
    )
    all_episodes_fallback_free = bool(live_results) and all(
        report.get("all_episodes_fallback_free") is True
        for report in live_results
    )
    summary.update({
        "holdout_expected": expected_count,
        "holdout_minimum_materialized": minimum_materialized,
        "fresh_tasks_verified": fresh_tasks_verified,
        "source_counts": source_counts,
        "source_balance_verified": source_balance_verified,
        "rollout_success_target": rollout_success_target,
        "qualified_rollout_floor_met": qualified_rollout_floor_met,
        "all_episodes_environment_clean": all_episodes_environment_clean,
        "all_episodes_fallback_free": all_episodes_fallback_free,
    })
    summary["target_met"] = (
        fresh_tasks_verified
        and source_balance_verified
        and summary["end_to_end_rate"] >= end_to_end_target
        and qualified_rollout_floor_met
        and all_episodes_environment_clean
        and all_episodes_fallback_free
    )
    return summary


def round_numbers(max_rounds):
    """Yield finite round numbers, or an unbounded sequence when configured as zero."""
    return itertools.count(1) if max_rounds == 0 else range(1, max_rounds + 1)


def run_holdout(project, root, config, previous_reports, *, batch_number=1):
    """Run one fresh, stricter release set after development targets converge."""
    holdout_root = root / "holdout"
    holdout_config = dict(config)
    holdout_config.update(
        generate_count=config["holdout_count"],
        task_paths=[],
        build_mode="clean",
        experiment_seed=config["experiment_seed"] + config["holdout_seed_offset"],
        rollout_episodes=config["holdout_rollout_episodes"],
        rollout_min_success_rate=config["holdout_rollout_success_rate"],
    )
    report_path = holdout_root / f"round-{batch_number:02d}" / "round_report.json"
    report = (
        json.loads(report_path.read_text())
        if report_path.exists()
        else {"round": batch_number, "state": "running"}
    )
    if report.get("state") != "complete":
        report = run_round(project, holdout_root, holdout_config, report)
    previous_seeds = {
        job.get("result", {}).get("sample_seed")
        for previous in previous_reports
        for job in previous.get("jobs", [])
        if isinstance(job.get("result", {}).get("sample_seed"), int)
    }
    previous_task_digests = set()
    for previous in previous_reports:
        for job in previous.get("jobs", []):
            task_path = Path(job.get("result", {}).get("task_path", ""))
            if task_path.is_file():
                previous_task_digests.add(hashlib.sha256(task_path.read_bytes()).hexdigest())
    report["summary"] = summarize_holdout(
        [job["result"] for job in report.get("jobs", [])],
        config["threshold"],
        expected_count=config["holdout_count"],
        end_to_end_target=config["holdout_end_to_end_rate"],
        rollout_success_target=config["holdout_rollout_success_rate"],
        previous_seeds=previous_seeds,
        previous_task_digests=previous_task_digests,
        minimum_materialized=(
            300 if config.get("certification_profile") == "production"
            else config["holdout_count"]
        ),
    )
    isolation = task_partition_isolation({
        "previous": [
            task for previous in previous_reports
            for task in report_task_documents(previous)
        ],
        "candidate": report_task_documents(report),
    })
    report["summary"]["partition_isolation"] = isolation
    report["summary"]["fresh_tasks_verified"] = (
        report["summary"]["fresh_tasks_verified"]
        and isolation["isolated"]
    )
    report["summary"]["target_met"] = (
        report["summary"]["target_met"]
        and isolation["isolated"]
    )
    report["phase"] = "holdout"
    report["holdout_batch"] = batch_number
    write_json(report_path, report)
    return report


def _build_one(project, task_path, output, config, seed=None):
    from env_factory.tasks.task_quality import score_file
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    task_score = score_file(task_path, min_score=config["threshold"]).to_dict()
    common = {"task_path": str(task_path), "output": str(output), "task_score": task_score,
              "category": task_score.get("training_category"), "build_mode": config["build_mode"]}
    if not task_score.get("eligible") or task_score["score"] < config["threshold"]:
        return failure("task_quality", task_score.get("findings", []), **common)
    # Only use a complete seed with identical inputs; do not copy runtime DBs or evidence.
    if seed:
        import shutil
        for name in ("task_impl.py", "acceptance.sh", "tests"):
            source = seed / name
            if source.is_dir():
                shutil.copytree(source, output / name)
            elif source.is_file():
                shutil.copy2(source, output / name)
    image_tag = sandbox_image_tag(output)
    common["container_image_tag"] = image_tag
    command = ["bash", str(project / "scripts/develop_sandbox_with_agent.sh"),
               "--input", str(task_path), "--output", str(output), "--agent", "codex",
               "--review-agent", "codex", "--model", MODEL, "--review-model", MODEL,
               "--runtime", config.get("sandbox_runtime", "none"),
               "--tag", image_tag,
               "--max-attempts", str(config["max_attempts"]),
               "--foreground", "--skip-auto-score"]
    if seed:
        command.append("--resume")
    build_attempts = []
    built = None
    for infrastructure_attempt in range(config.get("infrastructure_retries", 1) + 1):
        attempt_command = list(command)
        if infrastructure_attempt and "--resume" not in attempt_command:
            attempt_command.append("--resume")
        log_name = "build.log" if infrastructure_attempt == 0 else f"build-infra-retry-{infrastructure_attempt}.log"
        built = run_process(
            attempt_command, project, output / log_name, config["build_timeout"]
        )
        build_attempts.append({"attempt": infrastructure_attempt + 1, **built})
        if not built.get("timed_out") or built.get("budget_exhausted"):
            break
    assert built is not None
    common["build"] = built
    common["build_attempts"] = build_attempts
    if built["exit_code"] != 0:
        buildability_path = output / "buildability.json"
        if buildability_path.is_file():
            try:
                buildability = json.loads(buildability_path.read_text())
            except (OSError, json.JSONDecodeError):
                buildability = {}
            if buildability.get("buildable") is False:
                issue_codes = [
                    item.get("code") for item in buildability.get("issues", [])
                    if isinstance(item, dict) and item.get("code")
                ]
                return failure(
                    "task_quality", buildability.get("issues", []),
                    failure_code=issue_codes[0] if issue_codes else "TASK_BUILDABILITY",
                    buildability=buildability, **common,
                )
        if built["timed_out"]:
            return failure("infrastructure", "see build.log", **common)
        diagnosis = classify_build_failure(output)
        stage = diagnosis.pop("failure_class", "build")
        return failure(stage, "see build.log", **common, **diagnosis)
    try:
        task_lineage = json.loads((output / "task_lineage.json").read_text())
    except (OSError, json.JSONDecodeError):
        task_lineage = {}
    common["task_lineage"] = task_lineage
    if not valid_task_lineage(
        task_lineage, task_path, output / "task.json"
    ):
        return failure(
            "task_lineage",
            "generated task identity changed while preparing the sandbox",
            failure_code="TASK_LINEAGE",
            repair_target="task_generation_or_artifact_layout",
            **common,
        )
    report_path = output / "score_summary.json"
    scored = run_process([sys.executable, str(project / "scripts/sandbox/score_sandbox_offline.py"), str(output),
                          "--project", str(project), "--threshold", str(config["threshold"]),
                          "--output", str(report_path)], project, output / "score.log", config["score_timeout"])
    common["scoring"] = scored
    if scored["timed_out"]:
        return failure("infrastructure", "scoring timeout", **common)
    report = json.loads(report_path.read_text())["sandboxes"][0]
    passed = (scored["exit_code"] == 0 and report.get("passed") is True
              and report.get("score", 0) >= config["threshold"]
              and report.get("model") == MODEL and report.get("review_model") == MODEL)
    result = {**common, "passed": passed, "score": report.get("score", 0),
            "offline_score": report.get("score", 0), "sandbox_score": report,
            "failure_class": None if passed else "offline_validation",
            "repair_target": None if passed else "runtime_or_contract",
            "detail": report.get("failed_critical_gates", []),
            "elapsed_seconds": time.monotonic() - started, "live_rollout_verified": False}
    if passed and config["validation"] == "live":
        governance_path = output / "data_governance.json"
        governance_run = run_process([
            sys.executable,
            str(project / "scripts/rollout/audit_data_governance.py"),
            "--root", str(output),
            "--output", str(governance_path),
        ], project, output / "data_governance.log", config["score_timeout"])
        try:
            governance = json.loads(governance_path.read_text())
        except (OSError, json.JSONDecodeError):
            governance = {}
        governance_passed = (
            governance_run["exit_code"] == 0
            and governance.get("eligible_for_external_model_processing") is True
        )
        result.update(
            data_governance=governance,
            data_governance_process=governance_run,
            data_governance_verified=governance_passed,
        )
        if not governance_passed:
            result.update(failure(
                "infrastructure" if governance_run["timed_out"] else "data_governance",
                (
                    "data governance audit timed out"
                    if governance_run["timed_out"]
                    else governance or "data governance report unavailable"
                ),
                **common,
            ))
            result["elapsed_seconds"] = time.monotonic() - started
            return result
        live_path = output / "live_rollout.json"
        live_command = [
            sys.executable, str(project / "scripts/rollout/run_live_rollout.py"),
            str(output), "--output", str(live_path),
            "--episodes", str(config["rollout_episodes"]),
            "--max-steps", str(config["rollout_steps"]),
            "--min-success-rate", str(config.get("rollout_min_success_rate", 0.0)),
        ]
        container_runtime = None
        if config.get("sandbox_runtime", "none") == "docker":
            container_runtime = start_rollout_container(
                project, output, image_tag, config
            )
            result["container_rollout"] = {
                key: value for key, value in container_runtime.items()
                if key not in {"base_url", "container_id"}
            }
            if container_runtime.get("started") is not True:
                result.update(failure(
                    "infrastructure", "validated rollout container failed to start",
                    **common,
                ))
                result["elapsed_seconds"] = time.monotonic() - started
                return result
            live_command.extend([
                "--base-url", container_runtime["base_url"],
                "--container-image-id", container_runtime["image_id"],
            ])
        try:
            live_run = run_process(
                live_command, project, output / "rollout.log",
                config["rollout_timeout"],
            )
        finally:
            if container_runtime and container_runtime.get("container_id"):
                stopped = stop_rollout_container(
                    project, output, container_runtime["container_id"], config
                )
                result.setdefault("container_rollout", {})["stop"] = stopped
        live = json.loads(live_path.read_text()) if live_path.exists() else {}
        result.update(live_rollout=live, live_process=live_run,
                      live_rollout_verified=live.get("live_rollout_verified", False))
        # Offline executable evidence owns 9/10 points; real trajectories own
        # the final point. Hard live failures still make the sample ineligible.
        result["score"] = round(
            min(10.0, result["offline_score"] * 0.9 + float(live.get("quality_score", 0))), 2
        )
        result["passed"] = (
            live_run["exit_code"] == 0
            and live.get("passed") is True
            and (
                not container_runtime
                or result.get("container_rollout", {}).get("stop", {}).get("exit_code") == 0
            )
        )
        if not result["passed"]:
            owner = live.get("failure_owner")
            infrastructure = (
                live_run["timed_out"]
                or owner == "infrastructure"
                or container_runtime is not None
                and result.get("container_rollout", {}).get("stop", {}).get(
                    "exit_code"
                ) != 0
            )
            result.update(
                failure_class="infrastructure" if infrastructure else "live_rollout",
                failure_code=("INFRA" if infrastructure else
                              "ROLLOUT_AGENT" if owner == "agent" else "ROLLOUT_ENVIRONMENT"),
                repair_target=("rollout_policy" if owner == "agent" else "inspect_live_trajectory"),
                detail=live.get("conclusion", "rollout failed"),
            )
        else:
            privacy_path = output / "trajectory_privacy.json"
            privacy_run = run_process([
                sys.executable,
                str(project / "scripts/rollout/audit_trajectory_privacy.py"),
                "--rollout", str(live_path),
                "--output", str(privacy_path),
            ], project, output / "trajectory_privacy.log", config["score_timeout"])
            try:
                privacy = json.loads(privacy_path.read_text())
            except (OSError, json.JSONDecodeError):
                privacy = {}
            privacy_passed = (
                privacy_run["exit_code"] == 0
                and privacy.get("eligible_for_policy_training_export") is True
            )
            result.update(
                trajectory_privacy=privacy,
                trajectory_privacy_process=privacy_run,
                trajectory_privacy_verified=privacy_passed,
            )
            if not privacy_passed:
                result.update(
                    passed=False,
                    live_rollout_verified=False,
                    failure_class=(
                        "infrastructure" if privacy_run["timed_out"]
                        else "trajectory_privacy"
                    ),
                    failure_code=(
                        "INFRA" if privacy_run["timed_out"]
                        else "TRAJECTORY_PRIVACY"
                    ),
                    repair_target="inspect_policy_visible_trajectory",
                    detail=privacy or "trajectory privacy report unavailable",
                )
        if result["passed"]:
            calibration_path = output / "agentic_training_value_live.json"
            calibration_command = [
                sys.executable,
                str(project / "scripts/sandbox/validate_agentic_training_value.py"),
                "--root", str(output), "--output", str(calibration_path),
                "--evaluator-mode", "live",
            ]
            calibration_runtime = None
            calibration_runtime_dir = output / ".reward_calibration_runtime"
            if config.get("sandbox_runtime", "none") == "docker":
                calibration_runtime_dir.mkdir(exist_ok=True)
                calibration_runtime = start_rollout_container(
                    project, calibration_runtime_dir, image_tag, config
                )
                result["container_reward_calibration"] = {
                    key: value for key, value in calibration_runtime.items()
                    if key not in {"base_url", "container_id"}
                }
                if calibration_runtime.get("started") is not True:
                    result.update(
                        passed=False,
                        live_rollout_verified=False,
                        failure_class="infrastructure",
                        failure_code="INFRA",
                        repair_target="runner_or_provider",
                        detail="validated reward calibration container failed to start",
                    )
                    result["elapsed_seconds"] = time.monotonic() - started
                    return result
                calibration_command.extend([
                    "--base-url", calibration_runtime["base_url"],
                    "--container-image-id", calibration_runtime["image_id"],
                ])
            try:
                calibration_run = run_process(
                    calibration_command,
                    project,
                    output / "live_reward_calibration.log",
                    config["rollout_timeout"],
                )
            finally:
                if calibration_runtime and calibration_runtime.get("container_id"):
                    stopped = stop_rollout_container(
                        project,
                        calibration_runtime_dir,
                        calibration_runtime["container_id"],
                        config,
                    )
                    result.setdefault("container_reward_calibration", {})[
                        "stop"
                    ] = stopped
            try:
                calibration = json.loads(calibration_path.read_text())
            except (OSError, json.JSONDecodeError):
                calibration = {}
            calibration_execution = calibration.get("runtime_execution", {})
            calibration_passed = (
                calibration_run["exit_code"] == 0
                and calibration.get("curriculum_training_ready") is True
                and calibration.get("validation_mode") == "live_evaluator"
                and (
                    calibration_runtime is None
                    or (
                        calibration_execution.get("mode") == "docker_http"
                        and calibration_execution.get("container_image_id")
                        == calibration_runtime.get("image_id")
                        and result.get("container_reward_calibration", {})
                        .get("stop", {}).get("exit_code") == 0
                    )
                )
            )
            result.update(
                live_reward_calibration=calibration,
                live_reward_calibration_process=calibration_run,
                live_reward_calibration_verified=calibration_passed,
            )
            if not calibration_passed:
                result.update(
                    passed=False,
                    live_rollout_verified=False,
                    failure_class=(
                        "infrastructure" if calibration_run["timed_out"]
                        else "live_reward_calibration"
                    ),
                    failure_code=(
                        "INFRA" if calibration_run["timed_out"]
                        else "LIVE_REWARD_CALIBRATION"
                    ),
                    repair_target="inspect_reward_evaluator",
                    detail=calibration.get("failed_gates", ["live calibration unavailable"]),
                )
    result["elapsed_seconds"] = time.monotonic() - started
    return result


def build_one(project, task_path, output, config, seed=None):
    """Build and validate one sandbox, always reclaiming its local image."""
    result = None
    try:
        result = _build_one(project, task_path, output, config, seed)
        return result
    finally:
        if config.get("sandbox_runtime", "none") == "docker":
            cleanup = run_process(
                ["docker", "image", "rm", sandbox_image_tag(output)],
                project,
                output / "container_cleanup.log",
                min(config.get("score_timeout", 1800), 120),
            )
            if result is not None:
                result["container_cleanup"] = cleanup
                built = result.get("build", {})
                if built.get("exit_code") == 0 and cleanup["exit_code"] != 0:
                    result.update(
                        passed=False,
                        live_rollout_verified=False,
                        failure_class="infrastructure",
                        failure_code="INFRA",
                        repair_target="runner_or_provider",
                        detail="validated container image cleanup failed",
                    )


def run_round(project, root, config, report):
    round_root = root / f"round-{report['round']:02d}"
    round_root.mkdir(parents=True, exist_ok=True)
    state_path = round_root / "round_report.json"
    if "jobs" not in report:
        if config["generate_count"]:
            generation_root = round_root / "generation"
            # Do not repeat an ambiguous interrupted generation and accidentally change the sample.
            interrupted = report.get("generation_started", False)
            if not interrupted:
                report["generation_started"] = True
                write_json(state_path, report)
                report["generation"] = run_process([
                    sys.executable, str(project / "examples/generate_task.py"),
                    "--count", str(config["generate_count"]), "--max-workers", str(config["max_concurrency"]),
                    "--dataset-platform", "balanced",
                    "--seed", str(config.get("experiment_seed", 0) + report["round"] - 1),
                    "--hops", str(config.get("generation_hops", 3)),
                    "--route-attempts", str(config.get("route_attempts", 3)),
                    "--training-mix", str(config.get(
                        "training_mix", "direct_response=0.20,simple_agentic=0.30,multi_step_agentic=0.50"
                    )),
                    "--output", str(generation_root), "--log-file", str(round_root / "generation.log"),
                ], project, round_root / "generation_process.log", config["generation_timeout"])
            count = config["generate_count"]
        else:
            paths = [Path(value) for value in config["task_paths"]]
            count = len(paths)
        report["jobs"] = []
        if config["generate_count"]:
            manifests = {}
            for manifest_path in sorted((generation_root / "task_artifacts").glob("task-*/sample_manifest.json")):
                try:
                    sample = json.loads(manifest_path.read_text())
                    manifests[int(sample["batch_index"])] = (manifest_path, sample)
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    continue
            for index in range(1, count + 1):
                manifest_entry = manifests.get(index)
                if manifest_entry:
                    manifest_path, sample = manifest_entry
                    task_path = manifest_path.parent / "task.json"
                    if task_path.is_file():
                        report["jobs"].append({
                            "id": index, "task_path": str(task_path), "state": "pending",
                            "sample_manifest": str(manifest_path),
                            "category": sample.get("training_category", "unknown"),
                            "sample_seed": sample.get("sample_seed"),
                            "dataset_platform": sample.get("dataset_platform"),
                        })
                        continue
                    failure_path = manifest_path.parent / "failure.json"
                    detail = json.loads(failure_path.read_text()) if failure_path.is_file() else {
                        "failure_class": "GEN_SEMANTIC", "message": "no completed task artifact; see generation log"
                    }
                    result = failure(
                        "generation", detail.get("message", detail),
                        failure_code=detail.get("failure_class", "GEN_SEMANTIC"),
                        category=sample.get("training_category", "unknown"),
                        sample_manifest=str(manifest_path), sample_seed=sample.get("sample_seed"),
                        dataset_platform=sample.get("dataset_platform"),
                    )
                else:
                    result = failure("generation", "sample manifest missing; see generation log",
                                     failure_code="INFRA", category="unknown")
                report["jobs"].append({"id": index, "state": "complete", "result": result})
        else:
            for index, task_path in enumerate(paths, start=1):
                report["jobs"].append({"id": index, "task_path": str(task_path), "state": "pending"})
        write_json(state_path, report)

    def execute(job):
        task_path = Path(job["task_path"])
        output = round_root / f"sample-{job['id']:03d}" / f"attempt-{job['attempt']}"
        seed = None
        if config["build_mode"] == "repair" and not config["generate_count"]:
            for previous in sorted(root.glob("round-*/round_report.json"), reverse=True):
                old = json.loads(previous.read_text())
                if old["round"] >= report["round"]:
                    continue
                matches = [entry for entry in old.get("jobs", []) if entry["id"] == job["id"]]
                if matches and matches[0].get("result", {}).get("output"):
                    candidate = Path(matches[0]["result"]["output"])
                    if (candidate / "task_impl.py").is_file():
                        seed = candidate
                        break
        try:
            result = build_one(project, task_path, output, config, seed)
            if job.get("sample_manifest"):
                result.setdefault("sample_manifest", job["sample_manifest"])
            if job.get("sample_seed") is not None:
                result.setdefault("sample_seed", job["sample_seed"])
            if job.get("dataset_platform") is not None:
                result.setdefault("dataset_platform", job["dataset_platform"])
            return result
        except Exception as exc:
            return failure("infrastructure", f"{type(exc).__name__}: {exc}",
                           task_path=str(task_path), output=str(output),
                           category=job.get("category", "unknown"),
                           sample_manifest=job.get("sample_manifest"),
                           sample_seed=job.get("sample_seed"),
                           dataset_platform=job.get("dataset_platform"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=config["max_concurrency"]) as pool:
        futures = {}
        for job in report["jobs"]:
            if job["state"] == "complete":
                continue
            job.update(state="running", attempt=job.get("attempt", 0) + 1)
            write_json(state_path, report)
            futures[pool.submit(execute, job)] = job
        for future in concurrent.futures.as_completed(futures):
            job = futures[future]
            job.update(state="complete", result=future.result())
            write_json(state_path, report)
            print(json.dumps({"round": report["round"], "sample": job["id"], "passed": job["result"]["passed"]}), flush=True)
    targets = {
        "task_yield": config.get("target_task_yield", 0),
        "build_yield": config.get("target_build_yield", 0),
        "end_to_end_rate": config.get("target_end_to_end_rate", 0),
        "qualified_mean": config.get("target_qualified_mean", config["threshold"]),
        "category_rate": config.get("target_category_rate", 0),
    }
    report["summary"] = summarize(
        [job["result"] for job in report["jobs"]], config["threshold"], targets=targets
    )
    report["state"] = "complete"
    write_json(state_path, report)
    return report


def main():
    global PROCESS_DEADLINE, BUDGET_STATE_PATH, ACTIVE_RUN_STARTED
    parser = argparse.ArgumentParser(description="可恢复的端到端/固定集实验：离线验收后执行真实模型 rollout")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("output/loop_experiment"))
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--task-ids", default=None)
    source.add_argument("--generate-count", type=int, default=0)
    parser.add_argument("--generation-hops", type=int, default=3)
    parser.add_argument("--route-attempts", type=int, default=3)
    parser.add_argument(
        "--training-mix",
        default="direct_response=0.20,simple_agentic=0.30,multi_step_agentic=0.50",
    )
    parser.add_argument("--task-root", type=Path, default=Path("output/task_artifacts"))
    parser.add_argument(
        "--max-rounds", type=int, default=0,
        help="第一阶段最大质量轮数；0 表示不设置轮数上限",
    )
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="每个沙箱开发节点及每个独立缺陷的最大修复次数",
    )
    parser.add_argument(
        "--infrastructure-retries", type=int, default=1,
        help="构建进程超时后的自动 resume 次数；不用于业务缺陷重试",
    )
    parser.add_argument("--threshold", type=float, default=8)
    parser.add_argument("--consecutive-rounds", type=int, default=2)
    parser.add_argument("--build-mode", choices=("clean", "repair"), default="clean")
    parser.add_argument(
        "--sandbox-runtime", choices=("none", "docker"), default="docker",
        help="production 必须实际构建并安全冒烟验证 Docker 镜像；pilot 可显式使用 none",
    )
    parser.add_argument("--build-timeout", type=int, default=3600)
    parser.add_argument("--score-timeout", type=int, default=1800)
    parser.add_argument("--generation-timeout", type=int, default=3600)
    parser.add_argument(
        "--max-total-seconds",
        type=int,
        default=259200,
        help="实验的累计活跃运行时间预算；暂停期间不计时",
    )
    parser.add_argument("--validation", choices=("offline", "live"), default="live")
    parser.add_argument("--rollout-episodes", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=20)
    parser.add_argument("--rollout-timeout", type=int, default=1800)
    parser.add_argument(
        "--rollout-min-success-rate", type=float, default=0.0,
        help="第一阶段每个沙箱的最低 rollout 成功率；0 保留至少一次成功语义",
    )
    parser.add_argument(
        "--certification-profile", choices=("pilot", "production"), default="production",
        help="pilot 仅执行候选门禁；production 追加生产级训练素材准备认证",
    )
    parser.add_argument("--bundle-signing-private-key", type=Path)
    parser.add_argument("--bundle-trusted-public-key", type=Path)
    parser.add_argument("--holdout-count", type=int, default=400)
    parser.add_argument("--holdout-batches", type=int, default=3)
    parser.add_argument("--holdout-end-to-end-rate", type=float, default=0.85)
    parser.add_argument("--holdout-rollout-episodes", type=int, default=10)
    parser.add_argument("--holdout-rollout-success-rate", type=float, default=2 / 3)
    parser.add_argument(
        "--holdout-seed-offset", type=int, default=1_000_000,
        help="留出集相对开发轮 seed 的固定偏移，保证独立抽样",
    )
    parser.add_argument("--hypothesis", default="baseline", help="本实验要验证的改进假设")
    parser.add_argument("--experiment-seed", type=int, default=20260925, help="跨版本配对实验种子")
    parser.add_argument("--target-task-yield", type=float, default=0.85)
    parser.add_argument("--target-build-yield", type=float, default=0.80)
    parser.add_argument("--target-end-to-end-rate", type=float, default=0.70)
    parser.add_argument("--target-qualified-mean", type=float, default=8.5)
    parser.add_argument("--target-category-rate", type=float, default=0.60)
    args = parser.parse_args()
    if (args.max_rounds < 0 or not 0 <= args.threshold < 10
            or args.generate_count < 0 or not 0 <= args.generation_hops <= 20):
        parser.error("invalid rounds, threshold or generation count")
    if args.generate_count % 2 or args.holdout_count % 2:
        parser.error("balanced Kaggle/DATA.GOV.HK batches require even generation and holdout counts")
    if not 0 <= args.infrastructure_retries <= 3:
        parser.error("infrastructure retries must be between 0 and 3")
    for key in ("max_concurrency", "max_attempts", "route_attempts", "consecutive_rounds",
                "build_timeout", "score_timeout", "generation_timeout", "rollout_episodes", "rollout_steps", "rollout_timeout", "max_total_seconds", "holdout_count", "holdout_batches", "holdout_rollout_episodes", "holdout_seed_offset"):
        if getattr(args, key) <= 0:
            parser.error(f"{key} must be positive")
    for key in ("target_task_yield", "target_build_yield", "target_end_to_end_rate", "target_category_rate",
                "rollout_min_success_rate", "holdout_end_to_end_rate", "holdout_rollout_success_rate"):
        if not 0 <= getattr(args, key) <= 1:
            parser.error(f"{key} must be between 0 and 1")
    if not args.threshold <= args.target_qualified_mean <= 10:
        parser.error("target_qualified_mean must be between threshold and 10")
    if args.certification_profile == "production" and (
        args.holdout_count < 300 or args.holdout_batches < 3
        or args.holdout_rollout_episodes < 10
    ):
        parser.error("production certification requires 3 batches, 300 requests per batch and 10 episodes per sandbox")
    if args.certification_profile == "production" and args.threshold < 8:
        parser.error("production certification requires --threshold >= 8")
    if args.certification_profile == "production" and args.sandbox_runtime != "docker":
        parser.error("production certification requires --sandbox-runtime docker")
    project = args.project.resolve()
    from dotenv import load_dotenv
    load_dotenv(project / ".env")
    from env_factory.evidence.model_roles import resolve_model_roles
    model_roles = resolve_model_roles(os.environ)
    bundle_private_key = args.bundle_signing_private_key or (
        Path(os.environ["ENVFACTORY_BUNDLE_SIGNING_PRIVATE_KEY"])
        if os.getenv("ENVFACTORY_BUNDLE_SIGNING_PRIVATE_KEY") else None
    )
    bundle_public_key = args.bundle_trusted_public_key or (
        Path(os.environ["ENVFACTORY_BUNDLE_TRUSTED_PUBLIC_KEY"])
        if os.getenv("ENVFACTORY_BUNDLE_TRUSTED_PUBLIC_KEY") else None
    )
    bundle_key_identity = None
    if args.certification_profile == "production":
        if bundle_private_key is None or bundle_public_key is None:
            parser.error(
                "production certification requires bundle signing private and "
                "trusted public keys"
            )
        try:
            from env_factory.evidence.material_attestation import (
                private_key_public_identity,
                public_key_identity,
            )
            private_identity = private_key_public_identity(bundle_private_key)
            bundle_key_identity = public_key_identity(bundle_public_key)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if private_identity != bundle_key_identity:
            parser.error("bundle signing private key does not match trusted public key")
    if args.generate_count and not all(
        model_roles["generation"][key] for key in ("model", "api_key")
    ):
        parser.error("task generation requires LLM_MODEL and LLM_API_KEY")
    if args.validation == "live" and not all(
        model_roles["agent"][key] for key in ("model", "api_key")
    ):
        parser.error(
            "live rollout requires ROLLOUT_LLM_MODEL/API_KEY or LLM fallback"
        )
    root = args.output if args.output.is_absolute() else project / args.output
    root = root.resolve()
    task_root = args.task_root if args.task_root.is_absolute() else project / args.task_root
    try:
        ids = list(dict.fromkeys(int(value) for value in (args.task_ids or "45,78,92,175").split(",")))
        if not ids or any(value <= 0 for value in ids):
            raise ValueError()
    except ValueError:
        parser.error("task IDs must be positive integers")
    paths = [] if args.generate_count else [(task_root / f"task-{value}/task.json").resolve() for value in ids]
    if any(not path.is_file() for path in paths):
        parser.error("task input is missing")
    production_preflight = None
    if args.certification_profile == "production":
        try:
            minimum_free_gib = float(
                os.getenv("ENVFACTORY_MIN_FREE_GIB", "10")
            )
        except ValueError:
            parser.error("ENVFACTORY_MIN_FREE_GIB must be a positive number")
        if minimum_free_gib <= 0:
            parser.error("ENVFACTORY_MIN_FREE_GIB must be a positive number")
        from env_factory.evidence.production_preflight import run_production_preflight
        production_preflight = run_production_preflight(
            project,
            root.parent,
            signing_private_key=bundle_private_key,
            trusted_public_key=bundle_public_key,
            environment=os.environ,
            minimum_free_bytes=int(minimum_free_gib * 1024**3),
        )
    config = {key: value for key, value in vars(args).items() if key not in {
        "project", "output", "task_root", "task_ids",
        "bundle_signing_private_key", "bundle_trusted_public_key",
    }}
    from env_factory.evidence.data_governance import provider_identity
    generation_provider = provider_identity(
        model_roles["generation"]["base_url"],
        model_roles["generation"]["model"],
    )
    rollout_provider = provider_identity(
        model_roles["agent"]["base_url"],
        model_roles["agent"]["model"],
    )
    runtime_provider = provider_identity(
        model_roles["runtime"]["base_url"],
        model_roles["runtime"]["model"],
    )
    config.update(project=str(project), task_paths=list(map(str, paths)), model=MODEL,
                  source_digest=source_digest(project), input_digests=[input_digest(path) for path in paths],
                  execution_provenance=collect_execution_provenance(project),
                  bundle_attestation_key_identity_sha256=bundle_key_identity,
                  generation_model=model_roles["generation"]["model"],
                  rollout_model=model_roles["agent"]["model"],
                  runtime_model=model_roles["runtime"]["model"],
                  generation_allowed_response_models=model_roles[
                      "generation"
                  ]["allowed_response_models"],
                  rollout_allowed_response_models=model_roles[
                      "agent"
                  ]["allowed_response_models"],
                  runtime_allowed_response_models=model_roles[
                      "runtime"
                  ]["allowed_response_models"],
                  generation_provider=generation_provider,
                  rollout_provider=rollout_provider,
                  runtime_provider=runtime_provider,
                  provider_digest=digest_json({
                      "generation": generation_provider,
                      "agent": rollout_provider,
                      "runtime": runtime_provider,
                  }))
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("experiment already running")
        if production_preflight is not None:
            write_json(root / "production_preflight.json", production_preflight)
            if not production_preflight["ready"]:
                parser.error(
                    "production preflight failed: "
                    + ", ".join(production_preflight["failed_checks"])
                    + f"; see {root / 'production_preflight.json'}"
                )
        manifest = root / "experiment.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != config:
                parser.error("experiment configuration/code/inputs changed; use a new --output")
        else:
            if list(root.glob("round-*")):
                parser.error("legacy or unrecognized experiment; use a new --output")
            write_json(manifest, config)
        BUDGET_STATE_PATH = root / "runtime_state.json"
        try:
            budget_state = json.loads(BUDGET_STATE_PATH.read_text()) if BUDGET_STATE_PATH.exists() else {}
        except (OSError, json.JSONDecodeError):
            budget_state = {}
        active_seconds = float(budget_state.get("active_seconds", 0))
        remaining_seconds = args.max_total_seconds - active_seconds
        if remaining_seconds <= 0:
            write_json(root / "history.json", {"config": config, "stop_reason": "active_time_budget",
                       "rounds": [], "live_rollout_verified": False})
            return 1
        ACTIVE_RUN_STARTED = time.time()
        PROCESS_DEADLINE = ACTIVE_RUN_STARTED + remaining_seconds
        write_json(BUDGET_STATE_PATH, {
            "active_seconds": active_seconds, "status": "running",
            "run_started_at": ACTIVE_RUN_STARTED, "updated_at": ACTIVE_RUN_STARTED,
        })
        reports = []
        streak = 0
        for number in round_numbers(args.max_rounds):
            if (
                source_digest(project) != config["source_digest"]
                or [input_digest(path) for path in paths] != config["input_digests"]
                or not verify_execution_provenance(
                    project, config["execution_provenance"]
                )["verified"]
            ):
                parser.error(
                    "code/inputs/execution environment changed during experiment; "
                    "start a new --output"
                )
            path = root / f"round-{number:02d}/round_report.json"
            report = json.loads(path.read_text()) if path.exists() else {"round": number, "state": "running"}
            if report["state"] != "complete":
                if time.time() >= PROCESS_DEADLINE:
                    write_json(root / "history.json", {"config": config, "stop_reason": "active_time_budget",
                               "rounds": reports, "live_rollout_verified": False})
                    finish_active_budget("active_time_budget")
                    return 1
                report = run_round(project, root, config, report)
            reports.append(report)
            summary = report["summary"]
            if not summary.get("valid_quality_round", True):
                reason = "infrastructure_abort"
                write_json(root / "history.json", {
                    "config": config, "stop_reason": reason,
                    "consecutive_passes": streak,
                    "live_rollout_verified": False, "rounds": reports,
                })
                print(json.dumps({"stop_reason": reason, "round": number, **summary}, ensure_ascii=False))
                finish_active_budget(reason)
                return 2
            streak = streak + 1 if summary["target_met"] else 0
            target = "development_target_met"
            reason = (target if streak >= args.consecutive_rounds else
                      "round_budget" if args.max_rounds and number == args.max_rounds else "running")
            write_json(root / "history.json", {"config": config, "stop_reason": reason,
                       "consecutive_passes": streak, "live_rollout_verified": summary["live_rollout_verified"], "rounds": reports})
            if reason == target:
                if args.validation != "live":
                    offline_reason = "offline_target_met"
                    write_json(root / "history.json", {
                        "config": config, "stop_reason": offline_reason,
                        "consecutive_passes": streak,
                        "live_rollout_verified": False, "rounds": reports,
                    })
                    print(json.dumps({"stop_reason": offline_reason, **summary}, ensure_ascii=False))
                    finish_active_budget(offline_reason)
                    return 0
                if time.time() >= PROCESS_DEADLINE:
                    finish_active_budget("active_time_budget")
                    return 1
                holdouts = []
                previous_evidence = list(reports)
                batch_count = (
                    args.holdout_batches if args.certification_profile == "production" else 1
                )
                for batch_number in range(1, batch_count + 1):
                    holdout = run_holdout(
                        project, root, config, previous_evidence,
                        batch_number=batch_number,
                    )
                    holdouts.append(holdout)
                    previous_evidence.append(holdout)
                history = {
                    "config": config, "stop_reason": "certification_pending",
                    "consecutive_passes": streak,
                    "live_rollout_verified": (
                        summary["live_rollout_verified"]
                        and all(
                            item["summary"]["all_episodes_environment_clean"]
                            and item["summary"]["all_episodes_fallback_free"]
                            for item in holdouts
                        )
                    ),
                    "rounds": reports, "holdout": holdouts[0], "holdouts": holdouts,
                }
                if args.certification_profile == "production":
                    from env_factory.evidence.production_preflight import (
                        run_production_preflight,
                    )
                    from certify_training_materials import (
                        attach_artifact_verification, certify, default_policy,
                        run_sandbox_revalidation,
                    )
                    policy = default_policy()
                    policy["score_threshold"] = args.threshold
                    certification_preflight = run_production_preflight(
                        project,
                        root.parent,
                        signing_private_key=bundle_private_key,
                        trusted_public_key=bundle_public_key,
                        environment=os.environ,
                        minimum_free_bytes=int(minimum_free_gib * 1024**3),
                        experiment_config=config,
                    )
                    write_json(
                        root / "production_certification_preflight.json",
                        certification_preflight,
                    )
                    sandbox_revalidation = (
                        run_sandbox_revalidation(
                            history,
                            project=project,
                            policy=policy,
                            max_workers=args.max_concurrency,
                            timeout=args.score_timeout,
                        )
                        if certification_preflight["ready"] else {}
                    )
                    certification = certify(
                        history,
                        policy,
                        project=project,
                        sandbox_revalidation=sandbox_revalidation,
                        production_preflight=certification_preflight,
                    )
                    from verify_training_materials import verify
                    certification = attach_artifact_verification(
                        certification,
                        verify(certification["materials_manifest"], project),
                    )
                    if certification["certified"]:
                        from certify_training_materials import attach_bundle_verification
                        from export_training_materials import export_bundle, verify_bundle
                        bundle_root = root / "training_materials_bundle"
                        try:
                            if bundle_root.is_dir() and any(bundle_root.iterdir()):
                                bundle_verification = verify_bundle(
                                    bundle_root,
                                    trusted_public_key=bundle_public_key,
                                )
                            else:
                                bundle_verification = export_bundle(
                                    certification, bundle_root, project,
                                    signing_private_key=bundle_private_key,
                                    trusted_public_key=bundle_public_key,
                                )
                        except Exception as exc:
                            bundle_verification = {
                                "verified": False,
                                "failed_gates": ["bundle_export"],
                                "error_type": type(exc).__name__,
                            }
                        certification = attach_bundle_verification(
                            certification, bundle_verification
                        )
                    write_json(
                        root / "training_materials_manifest.json",
                        certification["materials_manifest"],
                    )
                    # The report is the final commit marker. Its referenced
                    # manifest has already been atomically published.
                    write_json(root / "production_readiness.json", certification)
                    passed = certification["certified"]
                    reason = (
                        "production_prepared_for_agentic_rl"
                        if passed else "production_readiness_failed"
                    )
                    history["production_readiness"] = certification
                else:
                    passed = holdouts[0]["summary"]["target_met"]
                    reason = "holdout_target_met" if passed else "holdout_failed"
                history["stop_reason"] = reason
                write_json(root / "history.json", history)
                print(json.dumps({
                    "stop_reason": reason,
                    "holdout_batches": [item["summary"] for item in holdouts],
                }, ensure_ascii=False))
                finish_active_budget(reason)
                return 0 if passed else 1
            if reason == "round_budget":
                print(json.dumps({"stop_reason": reason, "round": number, **summary}, ensure_ascii=False))
                finish_active_budget(reason)
                return 1
        finish_active_budget("round_budget")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
