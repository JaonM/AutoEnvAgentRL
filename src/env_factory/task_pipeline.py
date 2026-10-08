"""Code Agent task generation, sandbox construction and training gates.

Run with python -m env_factory.task_pipeline or scripts/run_pipeline.sh.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import threading
from concurrent.futures import CancelledError, ThreadPoolExecutor
import uuid

from env_factory.tasks.task_portability import valid_task_lineage
from env_factory.sandbox_scoring import valid_score_report
from env_factory.evidence.training_status import write_training_status

MODEL = "gpt-6.1-sol"

def sandbox_model(config):
    """Resolve construction/review separately from task authoring."""
    return config.get("sandbox_model") or ("gpt-6.1-sol" if config.get("code_agent", "codex") == "codex"
                                           else config.get("code_agent_model", MODEL))

ACTIVE_PROCESSES = set()

PROCESS_LOCK = threading.Lock()
PROCESS_CONTEXT = threading.local()


def interrupt(signum, frame):
    with PROCESS_LOCK:
        for pid in tuple(ACTIVE_PROCESSES):
            try:
                os.killpg(pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
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
    timeout = None if timeout == 0 else timeout
    process = None
    with log.open("w") as stream:
        try:
            with PROCESS_LOCK:
                cancelled = getattr(PROCESS_CONTEXT, 'cancelled', None)
                if cancelled is not None and cancelled.is_set():
                    raise CancelledError('pipeline cancelled before subprocess launch')
                process = subprocess.Popen(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
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


def sandbox_image_tag(output: Path) -> str:
    """Give every concurrent attempt a collision-resistant local image tag."""
    identity = hashlib.sha256(str(output.resolve()).encode("utf-8")).hexdigest()[:20]
    return f"envfactory-sandbox-{identity}"


def failure(stage, detail, **extra):
    targets = {"generation": "task_pipeline", "task_quality": "task_contract",
               "build": "sandbox_builder", "offline_validation": "runtime_or_contract",
               "data_governance": "inspect_outbound_payload",
               "infrastructure": "runner_or_provider"}
    codes = {"generation": "GEN_SEMANTIC", "task_quality": "TASK_BUILDABILITY",
             "build": "BUILD_BUSINESS", "offline_validation": "REWARD_OR_RUNTIME",
             "data_governance": "DATA_GOVERNANCE",
             "infrastructure": "INFRA"}
    return {"passed": False, "failure_class": stage, "repair_target": targets.get(stage, "inspect"),
            "failure_code": codes.get(stage, "UNKNOWN"), "detail": detail,
            "live_rollout_verified": False, **extra}

BUILD_PHASE_FAILURES = {
    "node_development": ("BUILD_MODULE", "sandbox_builder"),
    "semantic_review": ("BUILD_SEMANTIC_REVIEW", "sandbox_builder"),
    "defect_repair": ("BUILD_DEFECT_REPAIR", "sandbox_builder"),
    "repair_no_progress": ("BUILD_REPAIR_NO_PROGRESS", "sandbox_builder"),
    "repair_budget_exhausted": ("BUILD_REPAIR_BUDGET", "sandbox_builder"),
    "defect_validation": ("BUILD_DEFECT_VALIDATION", "sandbox_builder"),
    "contract_validation": ("BUILD_CONTRACT", "sandbox_builder"),
    "trace_validation": ("BUILD_TRACE", "sandbox_builder"),
    "runtime_validation": ("BUILD_RUNTIME_INTEGRITY", "platform_runtime_boundary"),
    "mutation_testing": ("BUILD_MUTATION", "sandbox_builder"),
    "training_readiness": ("BUILD_TRAINING_READINESS", "sandbox_builder"),
    "agentic_training_value": ("BUILD_AGENTIC_VALUE", "sandbox_builder"),
    "docker_build": ("BUILD_DOCKER", "container_builder"),
    "task_contract_rejected": ("TASK_CONTRACT", "task_pipeline"),
    "platform_contract_rejected": ("PLATFORM_RUNTIME", "platform_runtime_boundary"),
    "runtime_timeout_rejected": ("BUILD_RUNTIME_TIMEOUT", "platform_or_task_contract"),
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
    if status.get("failure_code") == "REVIEW_UNRESOLVED":
        return {"build_failed_phase": phase, "failure_class": "review",
                "failure_code": "REVIEW_UNRESOLVED", "repair_target": "review_evidence"}
    explicit = status.get("failure_code")
    if explicit in {"TASK_CONTRACT", "PLATFORM_RUNTIME", "BUILD_RUNTIME_TIMEOUT"}:
        target = {
            "TASK_CONTRACT": "task_pipeline",
            "PLATFORM_RUNTIME": "platform_runtime_boundary",
            "BUILD_RUNTIME_TIMEOUT": "platform_or_task_contract",
        }[explicit]
        return {"build_failed_phase": phase, "failure_code": explicit, "repair_target": target}
    code, target = BUILD_PHASE_FAILURES.get(
        phase, ("BUILD_BUSINESS", "sandbox_builder")
    )
    return {"build_failed_phase": phase, "failure_code": code, "repair_target": target}

def _build_one_unfinalized(project, task_path, output, config):
    from env_factory.tasks.task_quality import error_report, score_file
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        task_score = score_file(task_path, min_score=config["score_threshold"]).to_dict()
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        task_score = error_report(
            task_path, exc, min_score=config["score_threshold"],
        ).to_dict()
    common = {"task_path": str(task_path), "output": str(output), "task_score": task_score,
              "category": task_score.get("training_category"), "build_mode": "resume" if config.get("resume_existing") and (output / "task_impl.py").is_file() else "clean"}
    if not task_score.get("eligible") or task_score["score"] < config["score_threshold"]:
        return failure("task_quality", task_score.get("findings", []), **common)
    generation_seconds = 0.0
    construction_deadline = None
    try:
        task_document = json.loads(task_path.read_text())
        pipeline = task_document.get("artifacts", {}).get("generation_pipeline", {})
        if pipeline.get("backend") == "code_agent":
            generation_seconds = float(pipeline.get("seconds", 0))
            manifest_path = task_path.parent / "sample_manifest.json"
            if manifest_path.is_file():
                generation_seconds = float(json.loads(manifest_path.read_text()).get("generation_seconds", generation_seconds))
            budget = config.get("sample_build_budget", 0)
            if budget > 0:
                construction_deadline = started + budget - max(0.0, generation_seconds)
            common["generation_build_budget_seconds"] = budget or None
            common["generation_build_target_seconds"] = 300
    except (OSError, ValueError, TypeError, AttributeError):
        pass
    def remaining_timeout(limit):
        if construction_deadline is None:
            return limit
        return max(0.001, min(limit, construction_deadline - time.monotonic()))
    image_tag = sandbox_image_tag(output)
    common["container_image_tag"] = image_tag
    command = ["bash", str(project / "scripts/develop_sandbox_with_agent.sh"),
               "--input", str(task_path), "--output", str(output), "--agent", config.get("code_agent", "codex"),
               "--review-agent", config.get("code_agent", "codex"), "--model", sandbox_model(config), "--review-model", sandbox_model(config),
               "--runtime", config.get("sandbox_runtime", "none"),
               "--tag", image_tag,
               "--max-attempts", str(config["max_attempts"]),
               "--max-total-repairs", str(config.get("max_total_repairs", 4)),
               "--foreground", "--skip-auto-score"]
    if config.get("resume_existing") and (output / "task_impl.py").is_file():
        command.append("--resume")
    build_attempts = []
    built = None
    for infrastructure_attempt in range(config.get("infrastructure_retries", 1) + 1):
        attempt_command = list(command)
        if infrastructure_attempt and "--resume" not in attempt_command:
            attempt_command.append("--resume")
        log_name = "build.log" if infrastructure_attempt == 0 else f"build-infra-retry-{infrastructure_attempt}.log"
        built = run_process(
            attempt_command, project, output / log_name, remaining_timeout(config["build_timeout"])
        )
        build_attempts.append({"attempt": infrastructure_attempt + 1, **built})
        if construction_deadline is not None and time.monotonic() >= construction_deadline:
            common["generation_and_build_seconds"] = generation_seconds + time.monotonic() - started
            return failure("performance_budget", "generation and sandbox construction exceeded the sample budget",
                           failure_code="SAMPLE_BUILD_BUDGET", **common)
        infrastructure_status = False
        status_path = output / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
                infrastructure_status = (
                    status.get("failure_code") == "INFRA"
                    and status.get("failure_category") == "infrastructure"
                )
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        if not built["timed_out"] and not infrastructure_status:
            break
    assert built is not None
    common["build"] = built
    common["build_attempts"] = build_attempts
    common["build_completed_at"] = datetime.now(timezone.utc).isoformat()
    common["generation_and_build_seconds"] = generation_seconds + time.monotonic() - started
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
    report_path.unlink(missing_ok=True)
    scored = run_process([sys.executable, str(project / "scripts/sandbox/score_sandbox_offline.py"), str(output),
                          "--project", str(project), "--threshold", str(config["score_threshold"]),
                          "--output", str(report_path)], project, output / "score.log", remaining_timeout(config["score_timeout"]))
    common["scoring"] = scored
    common["generation_and_build_seconds"] = generation_seconds + time.monotonic() - started
    if construction_deadline is not None and time.monotonic() >= construction_deadline:
        return failure("performance_budget", "sandbox qualification exceeded the sample budget",
                       failure_code="SAMPLE_BUILD_BUDGET", **common)
    if scored["timed_out"]:
        return failure("infrastructure", "scoring timeout", **common)
    try:
        reports = json.loads(report_path.read_text(encoding="utf-8"))["sandboxes"]
        if not isinstance(reports, list) or len(reports) != 1 or not isinstance(reports[0], dict):
            raise ValueError("scorer must return exactly one sandbox report")
        report = reports[0]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, IndexError) as exc:
        return failure("offline_validation", f"scorer report unavailable or invalid: {type(exc).__name__}", **common)
    score_structure_valid = valid_score_report(
        report, root=output, project=project, threshold=float(config["score_threshold"]),
    )
    passed = (scored["exit_code"] == 0 and report.get("passed") is True
              and report.get("score", 0) >= config["score_threshold"]
              and report.get("model") == sandbox_model(config) and report.get("review_model") == sandbox_model(config)
              and score_structure_valid)
    result = {**common, "passed": passed, "score": report.get("score", 0),
            "offline_score": report.get("score", 0), "sandbox_score": report,
            "failure_class": None if passed else "offline_validation",
            "repair_target": None if passed else "runtime_or_contract",
            "detail": report.get("failed_critical_gates", []),
            "elapsed_seconds": time.monotonic() - started}
    result.update(
        qualification_basis="executable_v1",
        executable_validation_verified=passed and config.get("sandbox_runtime") == "docker",
        live_rollout_verified=False,
        live_reward_calibration_verified=False,
        trajectory_privacy_verified=False,
        validation_checks={
            "live_rollout": "not_run",
            "live_reward_calibration": "not_run",
            "trajectory_privacy": "not_run",
        },
    )
    if passed:
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
    result["elapsed_seconds"] = time.monotonic() - started
    return result

def finalize_delivery_score(result):
    """Keep a failed gate's numeric evidence separate from its delivery score."""
    result["delivery_score_kind"] = "gated_final_10_point"
    if result.get("passed") is not True:
        diagnostic_score = result.get("score")
        if (
            isinstance(diagnostic_score, (int, float))
            and not isinstance(diagnostic_score, bool)
            and "diagnostic_score" not in result
        ):
            result["diagnostic_score"] = diagnostic_score
        result["score"] = 0.0
    return result

def _build_one(project, task_path, output, config):
    """Score direct replays with the same final gate as the batch entry point."""
    write_training_status(output)
    result = finalize_delivery_score(
        _build_one_unfinalized(project, task_path, output, config)
    )
    result["training_ready"] = False
    return result

def build_one(project, task_path, output, config):
    """Build and validate one sandbox, always reclaiming its local image."""
    result = None
    try:
        result = _build_one(project, task_path, output, config)
        return result
    finally:
        write_training_status(output)
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
        if result is not None:
            finalize_delivery_score(result)
            write_training_status(output, result)

def resolve_generated_job(index, manifest_entry, *, generation_done):
    """Admit only atomically completed generation artifacts."""
    if manifest_entry is None:
        if not generation_done:
            return None
        result = failure("generation", "sample manifest missing; see generation log",
                         failure_code="INFRA", category="unknown")
        return {"id": index, "state": "complete", "result": result}
    manifest_path, sample = manifest_entry
    task_path = manifest_path.parent / "task.json"
    declared_hash = sample.get("task_sha256")
    complete = (
        sample.get("status") == "completed"
        and isinstance(declared_hash, str)
        and len(declared_hash) == 64
        and task_path.is_file()
    )
    if complete and hashlib.sha256(task_path.read_bytes()).hexdigest() == declared_hash:
        return {
            "id": index, "task_path": str(task_path), "state": "pending",
            "sample_manifest": str(manifest_path),
            "category": sample.get("training_category", "unknown"),
            "sample_seed": sample.get("sample_seed"),
        }
    if sample.get("status") != "failed" and not generation_done:
        return None
    failure_path = manifest_path.parent / "failure.json"
    if sample.get("status") == "failed" and not failure_path.is_file() and not generation_done:
        # The generator commits the failed manifest before its failure report.
        # Wait for the report rather than misclassifying a semantic rejection.
        return None
    if sample.get("status") == "failed" and failure_path.is_file():
        try:
            detail = json.loads(failure_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            detail = {"failure_class": "INFRA", "message": "invalid generation failure report"}
    else:
        detail = {
            "failure_class": "INFRA",
            "message": "generation artifact incomplete or checksum mismatch; refusing sandbox build",
        }
    if not isinstance(detail, dict):
        detail = {"failure_class": "INFRA", "message": "invalid generation failure report"}
    result = failure(
        "generation", detail.get("message", detail),
        failure_code=detail.get("failure_class", "GEN_SEMANTIC"),
        category=sample.get("training_category", "unknown"),
        failure_history=detail.get("failure_history", []),
        sample_manifest=str(manifest_path), sample_seed=sample.get("sample_seed"),
    )
    return {"id": index, "state": "complete", "result": result}

def run_pipeline(project, root, config):
    """Build committed tasks while the generation process produces subsequent tasks."""
    generation_workers = config.get('generation_workers', 1)
    build_workers = config.get('build_workers', 1)
    if any(type(value) is not int or value <= 0 for value in (generation_workers, build_workers)):
        raise ValueError('generation_workers and build_workers must be positive integers')
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("pipeline already running for this output")
        started = time.monotonic()
        run_id = f"{time.time_ns()}-{os.getpid()}"
        def event(stage, **fields):
            record = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "run_id": run_id, "stage": stage,
                      "elapsed_seconds": round(time.monotonic() - started, 3), **fields}
            with (root / "pipeline_events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
        event("pipeline_started", count=config["generate_count"] or len(config["task_paths"]),
              agent=config.get("code_agent", "codex"), model=config.get("code_agent_model", MODEL),
              sandbox_model=sandbox_model(config), generation_workers=generation_workers,
              build_workers=build_workers)
        paths = [Path(path) for path in config["task_paths"]]
        attempted_dirs = [path.parent for path in paths]
        exit_code = 0
        built_count = 0
        cancelled = threading.Event()
        active_builds = {}

        def run_build(task_path, output):
            PROCESS_CONTEXT.cancelled = cancelled
            try:
                if cancelled.is_set():
                    raise CancelledError('pipeline cancelled before build')
                return build_one(project, task_path, output, {**config, "resume_existing": True})
            finally:
                del PROCESS_CONTEXT.cancelled

        def start_build(builders, task_path):
            if task_path in (item[0] for item in active_builds.values()):
                raise ValueError('task is already being built')
            if task_path.parent.name != "task" or not task_path.parent.parent.name.startswith("artifact-"):
                raise ValueError("pipeline requires artifacts/artifact-N/task/task.json inputs")
            output = task_path.parent.parent / "sandbox"
            event("sandbox_started", task=output.parent.name, log=str(output / "build.log"))
            active_builds[builders.submit(run_build, task_path, output)] = (task_path, time.monotonic())

        def finish_builds():
            nonlocal exit_code, built_count
            for future, (task_path, task_started) in list(active_builds.items()):
                if not future.done():
                    continue
                try:
                    result = future.result()
                except Exception as exc:
                    result = failure('infrastructure', f'{type(exc).__name__}: {exc}',
                                     task_path=str(task_path), training_ready=False)
                del active_builds[future]
                output = task_path.parent.parent / 'sandbox'
                built_count += 1
                write_json(output / "pipeline_result.json", result)
                event("sandbox_finished", task=output.parent.name,
                      duration_seconds=round(time.monotonic() - task_started, 3),
                      passed=result.get("passed", False), training_ready=result.get("training_ready", False),
                      failure_class=result.get("failure_class"), failure_code=result.get("failure_code"),
                      detail=result.get("detail"), result_path=str(output / "pipeline_result.json"))
                status = 0 if result.get("passed") is True else 2 if result.get("failure_code") == "INFRA" else 1
                exit_code = max(exit_code, status)

        def cancel_builds():
            # Publish cancellation under the same lock as Popen, preventing a
            # worker still in preflight from launching after the group sweep.
            with PROCESS_LOCK:
                cancelled.set()
                for future in active_builds:
                    future.cancel()
                for pid in list(ACTIVE_PROCESSES):
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

        if config["generate_count"]:
            task_root = root / "artifacts"
            def artifact_directories():
                return {path for path in task_root.glob("artifact-*")
                        if path.is_dir() and path.name.removeprefix("artifact-").isdigit()}
            previous = artifact_directories()
            command = [sys.executable, "-m", "env_factory.generate_task",
                       "--count", str(config["generate_count"]), "--max-workers", str(generation_workers), "--output", str(root),
                       "--log-file", str(root / "generation.log"),
                       "--hops", str(config["generation_hops"]),
                       "--user-script-count", str(config.get("user_script_count", 3)),
                       "--code-agent", config.get("code_agent", "codex"),
                       "--code-agent-model", config.get("code_agent_model", MODEL),
                       "--language", config.get("language", "zh-CN"),
                       "--code-agent-timeout", str(config["code_agent_timeout"]),
                       "--route-attempts", str(config["route_attempts"]),
                       "--training-mix", config["training_mix"],
                       "--seed", str(config["seed"])]
            for key, flag in (("task_type", "--task-type"),
                              ("training_category", "--training-category"),
                              ("generation_intent", "--task-intent"),
                              ("generation_environment_mode", "--environment-mode")):
                if config.get(key):
                    command.extend([flag, config[key]])
            event("generation_started", count=config["generate_count"], log=str(root / "generation.log"))
            # Generation owns its worker pool. Only checksum-committed tasks
            # enter the bounded build pool; events and accounting stay here.
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="task-producer") as executor, \
                    ThreadPoolExecutor(max_workers=build_workers, thread_name_prefix="sandbox-builder") as builders:
                producer = executor.submit(run_process, command, project,
                                           root / "generation_process.log", config["generation_timeout"])
                consumed = set()
                try:
                    while True:
                        finish_builds()
                        generation_done = producer.done()
                        created = sorted(artifact_directories() - previous,
                                         key=lambda path: int(path.name.split("-")[-1]))
                        for artifact in created:
                            if artifact in consumed:
                                continue
                            directory = artifact / "task"
                            manifest_path = directory / "sample_manifest.json"
                            try:
                                manifest = json.loads(manifest_path.read_text())
                            except (OSError, ValueError):
                                manifest = {}
                            job = resolve_generated_job(len(consumed) + 1, (manifest_path, manifest),
                                                        generation_done=generation_done)
                            if job is None:
                                continue
                            if job["state"] == "pending":
                                if len(active_builds) >= build_workers:
                                    continue
                                event("task_generated", task=artifact.name)
                                start_build(builders, Path(job["task_path"]))
                            else:
                                result = job["result"]
                                result["training_ready"] = False
                                event("task_generation_failed", task=artifact.name, result=result)
                                write_json(directory / "pipeline_result.json", result)
                                exit_code = max(exit_code, 2 if result.get("failure_code") == "INFRA" else 1)
                            consumed.add(artifact)
                        if generation_done and len(consumed) == len(created) and not active_builds:
                            break
                        time.sleep(0.1)
                    generated = producer.result()
                except BaseException:
                    # Stop subprocess groups before joining the producer thread.
                    cancel_builds()
                    raise
            event("generation_finished", exit_code=generated["exit_code"])
            attempted_dirs = [artifact / "task" for artifact in created]
            if len(created) != config["generate_count"]:
                event("generation_count_mismatch", expected=config["generate_count"], actual=len(created))
                exit_code = 2
            if generated["exit_code"] != 0 and exit_code == 0:
                exit_code = 2
        else:
            if not paths:
                raise ValueError("pipeline requires artifacts/artifact-N/task/task.json inputs")
            if len(set(path.resolve() for path in paths)) != len(paths):
                raise ValueError('duplicate task paths cannot be built concurrently')
            with ThreadPoolExecutor(max_workers=build_workers, thread_name_prefix="sandbox-builder") as builders:
                pending = iter(paths)
                exhausted = False
                try:
                    while not exhausted or active_builds:
                        finish_builds()
                        while not exhausted and len(active_builds) < build_workers:
                            task_path = next(pending, None)
                            if task_path is None:
                                exhausted = True
                            else:
                                start_build(builders, task_path)
                        if not exhausted or active_builds:
                            time.sleep(0.1)
                except BaseException:
                    cancel_builds()
                    raise
        from env_factory.generation.realism_reporting import summarize_realism
        summary = summarize_realism(attempted_dirs, root / 'sandbox',
                                    requested=config['generate_count'] or len(paths))
        write_json(root / 'realism_summary.json', summary)
        event("pipeline_finished", exit_code=exit_code, built=built_count, realism_summary=str(root / 'realism_summary.json'))
        return exit_code


def main(argv=None):
    """Run the single supported generation/construction chain."""
    parser = argparse.ArgumentParser(description="Code Agent 任务生成与沙箱构建流水线")
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--output', type=Path, default=Path('output'))
    parser.add_argument('--count', type=int, help='生成并构建数量，默认 1')
    parser.add_argument('--generation-workers', type=int, default=1,
                        help='并发任务生成数，正整数，默认 1')
    parser.add_argument('--build-workers', type=int, default=1,
                        help='并发沙箱构建数，正整数，默认 1')
    parser.add_argument('--code-agent', choices=('codex', 'claude', 'opencode'), default='codex')
    parser.add_argument('--code-agent-model', help='任务生成模型；Codex 默认 gpt-6.1-sol')
    parser.add_argument('--sandbox-model', help='沙箱构建与审查模型；Codex 默认 gpt-6.1-sol')
    parser.add_argument('--language', default='zh-CN')
    parser.add_argument('--code-agent-timeout', type=float, default=600)
    parser.add_argument('--generation-hops', type=int, default=2,
                        help='Scene 图谱随机跳数上限 N，每次从 0..N 均匀抽取，N 范围 0..20，默认 2')
    parser.add_argument('--user-script-count', type=int, default=3,
                        help='每个新任务生成的用户交互剧本（FSM）数量，正整数，默认 3')
    parser.add_argument('--task-type',
                        help='任务类型：QA、Event、Chat、Research；支持逗号分隔多选，默认从全部类型随机选择')
    parser.add_argument('--environment-mode', dest='generation_environment_mode',
                        choices=('stateless', 'reference_data', 'stateful', 'external_capability'),
                        help='限定新任务环境模式：stateless 无状态公开输入；reference_data 查询参考数据；'
                             'stateful 修改并持久化业务状态；external_capability 使用已配置外部服务或 fixture'
                             '（预先准备的测试数据或模拟响应，例如固定物流查询结果）；'
                             '默认由 Agent 在可用且兼容的模式中选择')
    parser.add_argument('--seed', type=int, help='可选随机种子，默认随机')
    from env_factory.tasks.task_routing import TRAINING_CATEGORIES
    parser.add_argument('--training-category', choices=TRAINING_CATEGORIES,
                        help='固定全部新任务的训练类别，便于同类别良品率评测')
    args = parser.parse_args(argv)
    if args.task_type is not None:
        from env_factory.graph.knowledge_graph import TaskType
        values = [value.strip() for value in args.task_type.split(",")]
        if any(value not in {item.value for item in TaskType} for value in values):
            parser.error("--task-type must be comma-separated values from QA, Event, Chat, Research")
        args.task_type = ",".join(dict.fromkeys(values))
    if not 0 <= args.generation_hops <= 20:
        parser.error('--generation-hops must be between 0 and 20')
    if args.user_script_count <= 0:
        parser.error('--user-script-count must be positive')
    if args.count is not None and args.count <= 0:
        parser.error('--count must be positive')
    if args.generation_workers <= 0 or args.build_workers <= 0:
        parser.error('--generation-workers and --build-workers must be positive')
    if not 0 < args.code_agent_timeout < float('inf'):
        parser.error('--code-agent-timeout must be positive and finite')
    if any(value is not None and not value.strip() for value in
           (args.language, args.code_agent_model, args.sandbox_model)):
        parser.error('language and model names cannot be empty')
    if not args.code_agent_model:
        if args.code_agent != 'codex':
            parser.error('--code-agent-model is required for claude/opencode')
        args.code_agent_model = MODEL
    project = args.project.resolve()
    root = (project / args.output).resolve()
    from dotenv import load_dotenv
    load_dotenv(project / '.env')
    config = dict(vars(args), generate_count=args.count or 1,
                  task_paths=[],
                  generation_timeout=0, sample_build_budget=0, route_attempts=3,
                  max_attempts=3, max_total_repairs=3, infrastructure_retries=1,
                  score_threshold=8, build_timeout=3600, score_timeout=1800,
                  sandbox_runtime='docker',
                  training_mix='direct_response=0.20,simple_agentic=0.30,multi_step_agentic=0.50')
    config['seed'] = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), 'big')
    config['sandbox_model'] = sandbox_model(config)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        return run_pipeline(project, root, config)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
