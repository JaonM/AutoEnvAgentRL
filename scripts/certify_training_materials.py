#!/usr/bin/env python3
"""Certify an EnvFactory holdout as production-prepared Agentic-RL material.

This is a pre-training certification.  It verifies task/sandbox yield, runtime
integrity, reward counterfactuals and trajectory collectability; it deliberately
does not claim that an RL algorithm will improve a policy.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid
from typing import Any, Callable, Iterable, Mapping

from env_factory.evidence.material_artifacts import (
    MATERIAL_MANIFEST_VERSION,
    digest_json,
    evidence_artifact_digests,
    portable_artifact_digest,
    portable_artifact_digests,
)
from env_factory.evidence.material_privacy import audit_rollout_privacy
from env_factory.evidence.trajectory_schema import complete_episode
from env_factory.evidence.data_governance import (
    FORBIDDEN_OUTBOUND,
    OUTBOUND_SURFACES,
    REQUIRED_OUTBOUND_SURFACES,
    valid_governance_report,
    valid_provider_binding as valid_governed_provider_binding,
)
from env_factory.evidence.container_provenance import inspect_local_image, verify_container_provenance
from env_factory.evidence.certification_policy import (
    canonical_certification_policy,
    policy_for_experiment,
)
from env_factory.evidence.execution_provenance import verify_execution_provenance
from env_factory.evidence.experiment_contract import (
    build_experiment_contract,
    valid_experiment_contract,
)
from env_factory.tasks.task_similarity import near_duplicate_rate, task_partition_isolation
from env_factory.evidence.generation_provenance import generation_provenance_snapshot
from env_factory.evidence.runtime_provenance import valid_container_rollout_execution
from env_factory.sandbox_scoring import valid_score_report
from env_factory.tasks.task_quality import score_file
from env_factory.contracts.reward_contract import numeric_answer_counterfactual, terminal_outcome_weight
from env_factory.evidence.material_consumer import BUNDLE_VERSION
from env_factory.evidence.model_response_provenance import response_provenance
from env_factory.evidence.production_preflight import (
    REQUIRED_CHECKS,
    run_production_preflight,
    valid_production_preflight,
)


Z_95 = 1.959963984540054
USER_OUTCOMES = {
    "goal_satisfied", "information_required", "user_correction",
    "user_rejection", "user_acceptance", "agent_off_topic",
    "agent_premature_completion", "unrecognized",
}
NEGATIVE_COUNTERFACTUALS = (
    "goal_failure", "no_tools", "noise_selection", "reordered_tools",
    "wrong_final_answer", "wrong_numeric_answer",
)
MUST_COMPLETE_NEGATIVE_COUNTERFACTUALS = frozenset({
    "goal_failure", "no_tools", "noise_selection",
    "wrong_final_answer", "wrong_numeric_answer",
})
def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    """Publish one JSON document without exposing partial file contents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sandbox_revalidation_key(root: Path, task_path: Path) -> str:
    try:
        task_sha256 = hashlib.sha256(task_path.read_bytes()).hexdigest()
    except OSError:
        task_sha256 = "missing"
    return digest_json({
        "sandbox_root": str(root.resolve()),
        "task_sha256": task_sha256,
    })


def score_envelope(report: Mapping[str, Any] | Any) -> dict[str, Any] | None:
    """Return deterministic score semantics; diagnostic text may vary."""
    if not isinstance(report, Mapping):
        return None
    checks = report.get("checks")
    if not isinstance(checks, list):
        return None
    return {
        "score": report.get("score"),
        "eligible": report.get("eligible"),
        "passed": report.get("passed"),
        "threshold": report.get("threshold"),
        "failed_critical_gates": report.get("failed_critical_gates"),
        "checks": [
            {
                key: check.get(key)
                for key in ("name", "weight", "passed", "critical")
            }
            if isinstance(check, Mapping) else None
            for check in checks
        ],
        "mode": report.get("mode"),
        "network_used": report.get("network_used"),
        "model_used": report.get("model_used"),
        "model": report.get("model"),
        "review_model": report.get("review_model"),
        "evidence_fingerprint": report.get("evidence_fingerprint"),
    }


def run_sandbox_revalidation(
    history: Mapping[str, Any],
    *,
    project: Path,
    policy: Mapping[str, Any],
    max_workers: int = 4,
    timeout: int = 1800,
) -> dict[str, dict[str, Any]]:
    """Re-execute every claimed passing sandbox with offline hard gates."""
    raw_holdouts = history.get("holdouts")
    holdouts = (
        [item for item in raw_holdouts if isinstance(item, Mapping)]
        if isinstance(raw_holdouts, list) and raw_holdouts
        else [history.get("holdout")]
    )
    targets: dict[str, tuple[Path, Path]] = {}
    for holdout in holdouts:
        if not isinstance(holdout, Mapping):
            continue
        for job in holdout.get("jobs", []):
            result = job.get("result") if isinstance(job, Mapping) else None
            if not isinstance(result, Mapping):
                continue
            score = result.get("sandbox_score")
            if not (
                isinstance(score, Mapping)
                and score.get("passed") is True
                and score.get("score", 0) >= policy["score_threshold"]
            ):
                continue
            root = Path(str(result.get("output", "")))
            task_path = Path(str(result.get("task_path", "")))
            targets[sandbox_revalidation_key(root, task_path)] = (root, task_path)

    def execute(key: str, root: Path, task_path: Path) -> tuple[str, dict[str, Any]]:
        if not root.is_dir() or not task_path.is_file():
            return key, {"verified": False, "error": "missing_artifact"}
        environment = os.environ.copy()
        environment["SANDBOX_EVALUATOR_MOCK"] = "1"
        for name in (
            "SANDBOX_LLM_API_KEY", "SANDBOX_LLM_BASE_URL",
            "SANDBOX_EXTERNAL_CAPABILITY_URL", "LLM_API_KEY", "LLM_BASE_URL",
        ):
            environment.pop(name, None)
        with tempfile.TemporaryDirectory(prefix="envfactory-certify-") as directory:
            output = Path(directory) / "score_summary.json"
            command = [
                sys.executable,
                str(project / "scripts/sandbox/score_sandbox_offline.py"),
                str(root),
                "--project", str(project),
                "--threshold", str(policy["score_threshold"]), "--fresh",
                "--output", str(output),
                "--no-individual",
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=project,
                    env=environment,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout,
                )
                summary = load(output)
                reports = summary.get("sandboxes")
                report = (
                    reports[0]
                    if isinstance(reports, list) and len(reports) == 1
                    else None
                )
                return key, {
                    "verified": completed.returncode == 0,
                    "exit_code": completed.returncode,
                    "report": report,
                    "error": (
                        None if completed.returncode == 0 else "scorer_failed"
                    ),
                }
            except subprocess.TimeoutExpired:
                return key, {"verified": False, "error": "timeout"}
            except (OSError, TypeError, json.JSONDecodeError):
                return key, {"verified": False, "error": "invalid_output"}

    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = {
            executor.submit(execute, key, root, task_path): key
            for key, (root, task_path) in targets.items()
        }
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result
    return results


def artifact_digests(root: Path) -> dict[str, str]:
    """Backward-compatible public helper used by existing report consumers."""
    return portable_artifact_digests(root)


def wilson_lower(successes: int, total: int, *, z: float = Z_95) -> float:
    """Return the lower bound of a two-sided Wilson score interval."""
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    )
    return max(0.0, (centre - margin) / denominator)


def wilson_upper(errors: int, total: int, *, z: float = Z_95) -> float:
    """Upper bound for an error rate; no observations give no assurance."""
    if total <= 0:
        return 1.0
    return min(1.0, 1.0 - wilson_lower(total - errors, total, z=z))


def _artifact(result: Mapping[str, Any], name: str) -> dict[str, Any]:
    root = Path(str(result.get("output", "")))
    path = root / name
    try:
        value = load(path)
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def valid_data_governance(
    report: Mapping[str, Any], sandbox_root: Path, task_path: Path
) -> bool:
    """Independently recompute claims over the frozen task and fixture payloads."""
    try:
        task = load(task_path)
    except (OSError, json.JSONDecodeError, TypeError):
        return False
    return valid_governance_report(report, sandbox_root, task)


def _counterfactual_counts(reports: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    positive_total = positive_errors = negative_total = negative_errors = 0
    for report in reports:
        cases = report.get("evidence", {}).get("counterfactuals", {})
        if not isinstance(cases, Mapping):
            continue
        success = cases.get("goal_success")
        if isinstance(success, Mapping):
            positive_total += 1
            reward = success.get("reward")
            if (success.get("status") != "completed"
                    or not _valid_reward_number(reward) or reward < 0.6):
                positive_errors += 1
        for name, case in cases.items():
            if name == "goal_success" or not isinstance(case, Mapping):
                continue
            if not (
                name in NEGATIVE_COUNTERFACTUALS
                or name.startswith(("corrupted_arguments_", "skipped_tool_"))
            ):
                continue
            negative_total += 1
            reward = case.get("reward")
            answer_case = name in {"wrong_final_answer", "wrong_numeric_answer"}
            limit = 0.6 if answer_case else (0.0 if name == "noise_selection" else 0.2)
            must_complete_failed = (
                name in MUST_COMPLETE_NEGATIVE_COUNTERFACTUALS
                and case.get("status") != "completed"
            )
            invalid_reward = case.get("status") != "rejected" and (
                not _valid_reward_number(reward)
                or (reward >= limit if answer_case else reward > limit)
            )
            if must_complete_failed or invalid_reward:
                negative_errors += 1
    return {
        "positive_total": positive_total,
        "positive_errors": positive_errors,
        "negative_total": negative_total,
        "negative_errors": negative_errors,
    }


def _counterfactual_sandbox_counts(reports: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Treat correlated counterfactuals from one sandbox as one audit unit."""
    total = positive_errors = negative_errors = 0
    for report in reports:
        total += 1
        counts = _counterfactual_counts((report,))
        positive_errors += counts["positive_total"] == 0 or counts["positive_errors"] > 0
        negative_errors += counts["negative_total"] == 0 or counts["negative_errors"] > 0
    return {
        "total": total,
        "positive_errors": positive_errors,
        "negative_errors": negative_errors,
    }


def _valid_reward_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and -1 <= value <= 1 and math.isfinite(value)
    )


def valid_reward_calibration(
    report: Mapping[str, Any], task: Mapping[str, Any]
) -> bool:
    """Require the task-derived counterfactual matrix, not a cherry-picked subset."""
    evidence = report.get("evidence")
    cases = evidence.get("counterfactuals") if isinstance(evidence, Mapping) else None
    if not isinstance(cases, Mapping):
        return False
    training = task.get("training_contract")
    if not isinstance(training, Mapping):
        training = {}
    category = training.get(
        "category", task.get("training_category", "multi_step_agentic")
    )
    tool_required = category != "direct_response"
    environment = task.get("environment_plan")
    mode = environment.get("mode") if isinstance(environment, Mapping) else None
    if tool_required and mode in {"reference_data", "stateful"}:
        access = evidence.get("business_data_access")
        read_tables = access.get("tables_read") if isinstance(access, Mapping) else None
        tool_reads = access.get("tool_reads") if isinstance(access, Mapping) else None
        artifacts = task.get("artifacts")
        manifest = artifacts.get("data_manifest") if isinstance(artifacts, Mapping) else None
        declared = {
            item.get("table_name") for item in manifest.get("tables", [])
            if isinstance(item, Mapping) and isinstance(item.get("table_name"), str)
        } if isinstance(manifest, Mapping) else set()
        if (
            not isinstance(read_tables, list)
            or not read_tables
            or not declared
            or any(not isinstance(name, str) or name not in declared for name in read_tables)
        ):
            return False
        acceptance = task.get("acceptance_contract")
        scenarios = acceptance.get("executable_scenarios") if isinstance(acceptance, Mapping) else None
        success = next((item for item in scenarios if isinstance(item, Mapping)
                        and item.get("kind") == "goal_success"), None) if isinstance(scenarios, list) else None
        expected_tools = [
            step.get("tool_name") for step in success.get("steps", [])
            if isinstance(step, Mapping) and step.get("operation") == "tool_call"
        ] if isinstance(success, Mapping) else []
        if (
            not expected_tools
            or not isinstance(tool_reads, list)
            or len(tool_reads) < len(expected_tools)
            or any(
                not isinstance(item, Mapping)
                or item.get("tool_name") != expected_tools[index]
                or not isinstance(item.get("tables_read"), list)
                or not item["tables_read"]
                or any(name not in declared for name in item["tables_read"])
                for index, item in enumerate(tool_reads[:len(expected_tools)])
            )
        ):
            return False
        if mode == "reference_data":
            causality = evidence.get("business_data_causality")
            immutability = evidence.get("reference_data_immutability")
            if (
                not isinstance(causality, Mapping)
                or causality.get("proved") is not True
                or causality.get("table") not in read_tables
                or not isinstance(causality.get("tool"), str)
                or not causality["tool"]
                or not isinstance(immutability, Mapping)
                or immutability.get("unchanged") is not True
            ):
                return False
            terminal_weight = terminal_outcome_weight(dict(task))
            if terminal_weight > 0:
                sensitivity = evidence.get("business_data_reward_sensitivity")
                baseline = sensitivity.get("baseline_reward") if isinstance(sensitivity, Mapping) else None
                baseline_repeats = sensitivity.get("baseline_rewards") if isinstance(sensitivity, Mapping) else None
                changed = sensitivity.get("changed_reward") if isinstance(sensitivity, Mapping) else None
                repeats = sensitivity.get("changed_rewards") if isinstance(sensitivity, Mapping) else None
                if (
                    not isinstance(sensitivity, Mapping)
                    or sensitivity.get("proved") is not True
                    or sensitivity.get("tool_results_changed") is not True
                    or sensitivity.get("baseline_tool_results_stable") is not True
                    or not _valid_reward_number(baseline)
                    or not isinstance(baseline_repeats, list) or len(baseline_repeats) != 2
                    or any(not _valid_reward_number(value) or value < 0.6 or value != baseline
                           for value in baseline_repeats)
                    or not _valid_reward_number(changed)
                    or not isinstance(repeats, list) or len(repeats) != 2
                    or any(not _valid_reward_number(value)
                           or value >= 0.6 or baseline - value < max(0.05, 0.5 * terminal_weight) - 1e-9
                           for value in repeats)
                    or changed != max(repeats)
                    or baseline < 0.6 or changed >= 0.6
                    or baseline - changed < max(0.05, 0.5 * terminal_weight) - 1e-9
                ):
                    return False
    dependency = training.get("dependency")
    dependency_required = (
        dependency.get("required", category == "multi_step_agentic")
        if isinstance(dependency, Mapping)
        else category == "multi_step_agentic"
    )
    acceptance = task.get("acceptance_contract")
    scenarios = acceptance.get("executable_scenarios") if isinstance(
        acceptance, Mapping
    ) else None
    if not isinstance(scenarios, list):
        return False
    by_kind = {
        item.get("kind"): item for item in scenarios
        if isinstance(item, Mapping) and isinstance(item.get("kind"), str)
    }
    success = by_kind.get("goal_success")
    failure = by_kind.get("goal_failure")
    if not isinstance(success, Mapping) or not isinstance(failure, Mapping):
        return False
    steps = success.get("steps")
    if not isinstance(steps, list):
        return False
    tool_steps = [
        step for step in steps
        if isinstance(step, Mapping) and step.get("operation") == "tool_call"
    ]
    required = {"goal_success", "goal_failure"}
    terminal_weight = terminal_outcome_weight(dict(task))
    if terminal_weight > 0:
        answer_step = next((step for step in steps
                            if isinstance(step, Mapping) and step.get("operation") == "agent_response"), None)
        if answer_step is None:
            return False
        required.add("wrong_final_answer")
        if numeric_answer_counterfactual(dict(task), answer_step.get("content")) is not None:
            required.add("wrong_numeric_answer")
    if tool_required and any(
        isinstance(step, Mapping) and step.get("operation") == "agent_response"
        for step in steps
    ):
        required.add("no_tools")
    tool_schemas = {
        item["function"].get("name"): item["function"].get("parameters", {})
        for item in task.get("tools", [])
        if isinstance(item, Mapping) and isinstance(item.get("function"), Mapping)
    }
    for index, step in enumerate(tool_steps, start=1):
        arguments = step.get("arguments")
        if not isinstance(arguments, Mapping) or not arguments:
            continue
        schema = tool_schemas.get(step.get("tool_name"), {})
        schema_required = schema.get("required", []) if isinstance(schema, Mapping) else []
        keys = ([key for key in schema_required if isinstance(key, str) and key in arguments]
                if isinstance(schema_required, list) else [])
        for argument_index in range(1, len(keys or [next(iter(arguments))]) + 1):
            suffix = f"__{argument_index}" if argument_index > 1 else ""
            required.add(f"corrupted_arguments_{index}{suffix}")
    if dependency_required and len(tool_steps) >= 2:
        required.update(
            f"skipped_tool_{index}" for index in range(1, len(tool_steps) + 1)
        )
        required.add("reordered_tools")
    if task.get("noise_tools") and "noise_selection" in by_kind:
        required.add("noise_selection")
    if not required <= set(cases):
        return False

    positive = cases.get("goal_success")
    if not (
        isinstance(positive, Mapping)
        and positive.get("status") == "completed"
        and _valid_reward_number(positive.get("reward"))
        and positive["reward"] >= 0.6
    ):
        return False
    for name in required - {"goal_success"}:
        case = cases.get(name)
        if (name in MUST_COMPLETE_NEGATIVE_COUNTERFACTUALS
                and (not isinstance(case, Mapping) or case.get("status") != "completed")):
            return False
        if name in {"wrong_final_answer", "wrong_numeric_answer"}:
            reward = case.get("reward") if isinstance(case, Mapping) else None
            required_drop = max(0.05, 0.5 * terminal_weight)
            if (
                not isinstance(case, Mapping)
                or case.get("status") != "completed"
                or not _valid_reward_number(reward)
                or reward >= 0.6
                or positive["reward"] - reward < required_drop - 1e-9
            ):
                return False
            continue
        if not isinstance(case, Mapping) or case.get("status") not in {
            "completed", "rejected",
        }:
            return False
        reward = case.get("reward")
        limit = 0.0 if name == "noise_selection" else 0.2
        if case.get("status") == "completed":
            if (
                not _valid_reward_number(reward)
                or reward > limit
            ):
                return False
        elif (
            reward is not None
            or not isinstance(case.get("http_status"), int)
            or not 400 <= case["http_status"] < 500
            or not isinstance(case.get("error_type"), str)
            or not case["error_type"]
            or "message" in case
        ):
            return False
    return (
        report.get("curriculum_training_ready") is True
        and report.get("agentic_training_ready") is True
        and report.get("hard_gates_passed") is True
        and report.get("validation_mode") == "live_evaluator"
        and report.get("failed_gates") == []
        and report.get("failures") == []
    )


def consistent_causality_witness(
    offline: Mapping[str, Any], live: Mapping[str, Any], task: Mapping[str, Any],
) -> bool:
    """Bind the live sensitivity decision to the deterministic offline mutation."""
    if task.get("environment_plan", {}).get("mode") != "reference_data":
        return True
    offline_evidence = offline.get("evidence")
    live_evidence = live.get("evidence")
    if not isinstance(offline_evidence, Mapping) or not isinstance(live_evidence, Mapping):
        return False
    before = offline_evidence.get("business_data_causality")
    after = live_evidence.get("business_data_causality")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return False
    fields = ("proved", "table", "tool", "field", "answer_value_mentioned")
    return (
        isinstance(before.get("answer_value_mentioned"), bool)
        and all(before.get(field) == after.get(field) for field in fields)
    )


def valid_training_readiness(report: Mapping[str, Any]) -> bool:
    """Validate the complete deterministic runtime evidence envelope."""
    evidence = report.get("evidence")
    if not isinstance(evidence, Mapping):
        return False
    state_causality = evidence.get("state_causality")
    rewards = evidence.get("rewards")
    observation = evidence.get("observation_scan")
    runtime = evidence.get("runtime_state")
    if not all(isinstance(value, Mapping) for value in (
        state_causality, rewards, observation, runtime,
    )):
        return False
    success = rewards.get("success")
    failure = rewards.get("failure")
    reward_separated = (
        isinstance(success, (int, float))
        and not isinstance(success, bool)
        and isinstance(failure, (int, float))
        and not isinstance(failure, bool)
        and success > failure
        and success - failure >= 0.1
    )
    return (
        report.get("training_ready") is True
        and report.get("validation_mode") == "offline_mock"
        and report.get("live_rollout_verified") is False
        and report.get("hard_gates_passed") is True
        and report.get("failed_gates") == []
        and report.get("failures") == []
        and evidence.get("determinism") is True
        and isinstance(state_causality.get("archetype"), str)
        and bool(state_causality["archetype"])
        and isinstance(state_causality.get("expected_delta"), list)
        and state_causality.get("invalid") == []
        and reward_separated
        and observation.get("status") == 200
        and observation.get("forbidden_paths") == []
        and runtime.get("reset_reproducible") is True
        and runtime.get("episode_isolation") is True
        and runtime.get("replay_consistent") is True
        and isinstance(runtime.get("episode_a_event_count"), int)
        and not isinstance(runtime.get("episode_a_event_count"), bool)
        and runtime["episode_a_event_count"] >= 1
        and runtime.get("episode_b_event_count") == 0
    )


def valid_user_turn(
    step: Mapping[str, Any], task: Mapping[str, Any]
) -> bool:
    result = step.get("result")
    if not (
        step.get("status") == 200
        and isinstance(result, Mapping)
        and isinstance(result.get("user_query"), str)
        and bool(result["user_query"].strip())
        and isinstance(result.get("should_end"), bool)
        and result.get("match_status") in {"matched", "unmatched", "ambiguous"}
        and result.get("outcome_category") in USER_OUTCOMES
        and isinstance(result.get("reason_code"), str)
        and bool(result["reason_code"])
        and isinstance(result.get("fsm_script_id"), str)
        and bool(result["fsm_script_id"])
        and isinstance(result.get("fsm_state_before"), str)
        and bool(result["fsm_state_before"])
        and isinstance(result.get("fsm_state_after"), str)
        and bool(result["fsm_state_after"])
        and isinstance(result.get("fsm_transition_applied"), bool)
        and isinstance(result.get("fsm_recovery_count"), int)
        and not isinstance(result.get("fsm_recovery_count"), bool)
        and result["fsm_recovery_count"] >= 0
        and (
            not result.get("should_end")
            or isinstance(result.get("termination_reason"), str)
        )
    ):
        return False
    scripts = {
        str(script.get("script_id")): script
        for script in task.get("user_scripts", [])
        if isinstance(script, Mapping) and script.get("script_id")
    }
    script = scripts.get(result["fsm_script_id"])
    if not isinstance(script, Mapping):
        return False
    outcome = result["outcome_category"]
    if result["match_status"] == "matched":
        transition_id = result.get("fsm_transition_id")
        transition = next((
            item for item in script.get("transitions", [])
            if isinstance(item, Mapping)
            and item.get("transition_id") == transition_id
        ), None)
        if not isinstance(transition, Mapping):
            return False
        terminal_states = {
            item.get("state_id") for item in script.get("states", [])
            if isinstance(item, Mapping) and item.get("terminal") is True
        }
        expected_end = bool(
            transition.get("should_end")
            or transition.get("to_state") in terminal_states
        )
        return (
            outcome in {
                "goal_satisfied", "information_required", "user_correction",
                "user_rejection", "user_acceptance",
            }
            and result.get("fsm_transition_applied") is True
            and transition.get("outcome_category") == outcome
            and transition.get("from_state") == result["fsm_state_before"]
            and transition.get("to_state") == result["fsm_state_after"]
            and result["should_end"] is expected_end
            and (
                not expected_end
                or result.get("termination_reason") == "completed"
            )
        )
    return (
        outcome in {
            "agent_off_topic", "agent_premature_completion", "unrecognized",
        }
        and result.get("fsm_transition_id") is None
        and result.get("fsm_transition_applied") is False
        and result["fsm_state_before"] == result["fsm_state_after"]
        and (
            not result["should_end"]
            or result.get("termination_reason") == "unresolved_dialogue"
        )
    )


def category_mix(results: Iterable[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    values = list(results)
    total = len(values)
    counts = Counter(str(item.get("category", "unknown")) for item in values)
    shares = {name: count / total if total else 0.0 for name, count in counts.items()}
    minimums = policy.get("min_category_shares", {})
    passed = total > 0 and all(
        shares.get(str(name), 0.0) >= float(minimum)
        for name, minimum in minimums.items()
    )
    maximums = policy.get("max_category_shares", {})
    passed = passed and all(
        shares.get(str(name), 0.0) <= float(maximum)
        for name, maximum in maximums.items()
    )
    return {"counts": dict(counts), "shares": shares, "passed": passed}


def valid_rollout_provenance(item: Mapping[str, Any]) -> bool:
    live = item.get("live_rollout")
    task_path = Path(str(item.get("task_path", "")))
    root = Path(str(item.get("output", "")))
    if not isinstance(live, Mapping) or not task_path.is_file() or not root.is_dir():
        return False
    try:
        task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
        artifact_sha = portable_artifact_digest(root)
    except OSError:
        return False
    return (
        live.get("schema_version") == "2.0"
        and live.get("task_sha256") == task_sha
        and live.get("sandbox_artifacts_digest") == artifact_sha
        and isinstance(live.get("agent_model"), str) and bool(live["agent_model"])
        and isinstance(live.get("runtime_model"), str) and bool(live["runtime_model"])
        and all(
            isinstance(live.get(name), str)
            and re.fullmatch(r"[0-9a-f]{64}", live[name]) is not None
            for name in ("agent_provider_sha256", "runtime_provider_sha256")
        )
        and valid_container_rollout_execution(live, root)
        and response_provenance(live)["verified"] is True
    )


def valid_provider_binding(
    item: Mapping[str, Any],
    *,
    expected_agent_provider: Mapping[str, Any] | None = None,
    expected_runtime_provider: Mapping[str, Any] | None = None,
) -> bool:
    """Bind rollout model identities to the provider governance authorization."""
    governance = _artifact(item, "data_governance.json")
    providers = governance.get("providers") if isinstance(governance, Mapping) else None
    return (
        valid_governed_provider_binding(
            item.get("live_rollout"),
            _artifact(item, "agentic_training_value_live.json"),
            governance,
        )
        and isinstance(providers, Mapping)
        and (
            expected_agent_provider is None
            or providers.get("agent") == dict(expected_agent_provider)
        )
        and (
            expected_runtime_provider is None
            or providers.get("user_simulator_and_reward")
                == dict(expected_runtime_provider)
        )
    )


def valid_rollout_outcome(
    report: Mapping[str, Any], policy: Mapping[str, Any],
    *, task: Mapping[str, Any] | None = None,
) -> bool:
    """Independently recompute the per-sandbox production rollout verdict."""
    episodes = report.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        return False
    spec = task.get("task_spec") if isinstance(task, Mapping) else None
    goal = spec.get("goal_contract", {}) if isinstance(spec, Mapping) else {}
    predicates = goal.get("row_predicates") if isinstance(goal, Mapping) else None
    if predicates and not isinstance(predicates, list):
        return False
    for episode in episodes:
        if not isinstance(episode, Mapping):
            return False
        initial_reward = episode.get("initial_reward")
        final_reward = episode.get("final_reward")
        issues = episode.get("issues")
        if (
            not _valid_reward_number(initial_reward)
            or not _valid_reward_number(final_reward)
            or not isinstance(issues, list)
        ):
            return False
        state_success = episode.get("state_goal_satisfied")
        if predicates:
            from env_factory.sandbox_runtime import BusinessGoalEvaluator
            baseline = episode.get("initial_state")
            state = episode.get("final_state")
            if not isinstance(baseline, Mapping) or not isinstance(state, Mapping):
                return False
            try:
                state_success = (
                    BusinessGoalEvaluator.evaluate(predicates, state, baseline)
                    and BusinessGoalEvaluator.preserves_unrelated(goal, baseline, state)
                    and (not goal.get("requires_state_change") or baseline != state)
                )
            except (AttributeError, KeyError, TypeError, ValueError):
                return False
            if episode.get("state_goal_satisfied") is not state_success:
                return False
        expected_success = (
            episode.get("termination") == "completed"
            and final_reward >= 0.8
            and initial_reward < 0.8
            and state_success is not False
            and not issues
        )
        if episode.get("agent_success") is not expected_success:
            return False
    successes = sum(
        episode.get("agent_success") is True
        for episode in episodes if isinstance(episode, Mapping)
    )
    if len(episodes) != sum(isinstance(item, Mapping) for item in episodes):
        return False
    success_rate = successes / len(episodes)
    minimum = float(policy["min_agent_success_rate_per_sandbox"])
    clean = all(not episode.get("issues") for episode in episodes)
    fallback_free = all(
        "runtime_llm_fallback" not in episode.get("issues", [])
        for episode in episodes
    )
    return (
        len(episodes) >= policy["min_episodes_per_qualified_sandbox"]
        and success_rate >= minimum
        and clean
        and fallback_free
        and report.get("passed") is True
        and report.get("live_rollout_verified") is True
        and report.get("environment_checks_passed") is True
        and report.get("all_episodes_environment_clean") is True
        and report.get("all_episodes_fallback_free") is True
        and report.get("success_rate_gate_passed") is True
        and isinstance(report.get("agent_success_rate"), (int, float))
        and math.isclose(float(report["agent_success_rate"]), success_rate)
        and isinstance(report.get("minimum_success_rate"), (int, float))
        and float(report["minimum_success_rate"]) >= minimum
        and report.get("quality_score") == 1.0
        and report.get("failure_owner") is None
        and report.get("conclusion") == "live_success_witness"
    )


def _generation_attempt_seeds(item: Mapping[str, Any]) -> list[int] | None:
    try:
        manifest = load(Path(str(item.get("sample_manifest", ""))))
        attempts = manifest.get("attempts") if isinstance(manifest, Mapping) else None
        if not isinstance(attempts, list) or not attempts:
            return None
        seeds = [attempt.get("seed") for attempt in attempts]
        if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds):
            return None
        return seeds
    except (OSError, TypeError, ValueError, json.JSONDecodeError, AttributeError):
        return None


def fresh_holdout_evidence(
    holdouts: Iterable[Mapping[str, Any]], policy: Mapping[str, Any],
    *, development_reports: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    batches = list(holdouts)
    batch_ids = [item.get("holdout_batch") for item in batches]
    seen_seeds: set[int] = set()
    seen_attempt_seeds: set[int] = set()
    development_seeds: set[int] = set()
    development_attempt_seeds: set[int] = set()
    development_attempts_valid = True
    for report in development_reports:
        if not isinstance(report, Mapping):
            continue
        for job in report.get("jobs", []):
            item = job.get("result", {}) if isinstance(job, Mapping) else {}
            if not isinstance(item, Mapping):
                continue
            seed = item.get("sample_seed")
            if isinstance(seed, int) and not isinstance(seed, bool):
                development_seeds.add(seed)
                development_attempt_seeds.add(seed)
            if item.get("sample_manifest"):
                actual = _generation_attempt_seeds(item)
                if actual is None:
                    development_attempts_valid = False
                else:
                    development_attempt_seeds.update(actual)
    seen_tasks: set[str] = set()
    details = []
    all_fresh = (
        len(batches) >= policy["min_holdout_batches"]
        and all(isinstance(item, int) for item in batch_ids)
        and len(set(batch_ids)) == len(batch_ids)
        and development_attempts_valid
    )
    for batch in batches:
        results = [
            job.get("result", {}) for job in batch.get("jobs", [])
            if isinstance(job, Mapping)
        ]
        seeds = [item.get("sample_seed") for item in results]
        attempt_seeds: list[int] = []
        complete_attempts = True
        for item in results:
            actual = _generation_attempt_seeds(item)
            if actual is not None:
                attempt_seeds.extend(actual)
            else:
                complete_attempts = False
        digests = []
        for item in results:
            path = Path(str(item.get("task_path", "")))
            try:
                digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
            except OSError:
                continue
        batch_fresh = (
            len(results) >= policy["min_tasks"]
            and len(digests) >= policy["min_tasks"]
            and all(isinstance(seed, int) for seed in seeds)
            and len(set(seeds)) == len(seeds)
            and seen_seeds.isdisjoint(seeds)
            and development_seeds.isdisjoint(seeds)
            and complete_attempts
            and len(set(attempt_seeds)) == len(attempt_seeds)
            and seen_attempt_seeds.isdisjoint(attempt_seeds)
            and development_attempt_seeds.isdisjoint(attempt_seeds)
            and len(set(digests)) == len(digests)
            and seen_tasks.isdisjoint(digests)
            and batch.get("summary", {}).get("fresh_tasks_verified") is True
        )
        details.append({
            "batch": batch.get("holdout_batch"),
            "requests": len(results),
            "materialized": len(digests),
            "generation_attempts": len(attempt_seeds),
            "fresh": batch_fresh,
        })
        all_fresh = all_fresh and batch_fresh
        seen_seeds.update(seed for seed in seeds if isinstance(seed, int))
        seen_attempt_seeds.update(attempt_seeds)
        seen_tasks.update(digests)
    return {
        "verified": all_fresh,
        "unique_seeds": len(seen_seeds),
        "unique_attempt_seeds": len(seen_attempt_seeds),
        "development_seeds": len(development_seeds),
        "development_attempt_seeds": len(development_attempt_seeds),
        "unique_tasks": len(seen_tasks),
        "batches": details,
    }


def user_outcome_coverage(records, minimum: int) -> bool:
    """Coverage follows the declared dialogue policy, not accidental failures."""
    if not records:
        return False
    interactive = []
    fixed = []
    for step, task in records:
        if not valid_user_turn(step, task):
            continue
        outcome = step.get("result", {}).get("outcome_category")
        policy = task.get("user_simulation_policy", {})
        if policy == {"mode": "fixed_goal", "required_outcomes": ["goal_satisfied"]}:
            fixed.append(outcome)
        else:
            interactive.append(outcome)
    if fixed and not set(fixed) & {"goal_satisfied", "user_acceptance"}:
        return False
    if interactive:
        outcomes = set(interactive)
        return (len(outcomes) >= minimum and bool(outcomes & {"goal_satisfied", "user_acceptance"})
                and bool(outcomes & {"information_required", "user_correction", "user_rejection", "agent_off_topic", "agent_premature_completion", "unrecognized"}))
    return bool(fixed)


def _report_task_documents(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    documents = []
    for job in report.get("jobs", []):
        if not isinstance(job, Mapping):
            continue
        result = job.get("result", {})
        path = Path(str(result.get("task_path", ""))) if isinstance(
            result, Mapping
        ) else Path("")
        try:
            value = load(path)
        except (OSError, json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            documents.append(value)
    return documents


def partition_isolation_evidence(
    history: Mapping[str, Any], holdouts: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Check development/holdout and cross-holdout near-duplicate leakage."""
    partitions: dict[str, list[dict[str, Any]]] = {}
    development = [
        task
        for report in history.get("rounds", [])
        if isinstance(report, Mapping)
        for task in _report_task_documents(report)
    ]
    if development:
        partitions["development"] = development
    for index, holdout in enumerate(holdouts, start=1):
        batch = holdout.get("holdout_batch", index)
        partitions[f"holdout:{index}:batch-{batch}"] = _report_task_documents(holdout)
    return task_partition_isolation(partitions)


def certify(
    history: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    project: Path | None = None,
    sandbox_revalidation: Mapping[str, Mapping[str, Any]] | None = None,
    production_preflight: Mapping[str, Any] | None = None,
    image_inspector: Callable[[str], Mapping[str, Any] | None] = inspect_local_image,
) -> dict[str, Any]:
    project = project or Path(__file__).resolve().parents[1]
    config = history.get("config", {})
    expected_policy, policy_config_valid = policy_for_experiment(config)
    certification_policy_verified = (
        policy_config_valid
        and isinstance(policy, Mapping)
        and dict(policy) == expected_policy
    )
    policy = expected_policy
    certification_policy_sha256 = digest_json(policy)
    production_experiment = (
        isinstance(config, Mapping)
        and config.get("certification_profile") == "production"
    )
    expected_generation_provider = (
        config.get("generation_provider") if isinstance(config, Mapping) else None
    )
    expected_agent_provider = (
        config.get("rollout_provider") if isinstance(config, Mapping) else None
    )
    expected_runtime_provider = (
        config.get("runtime_provider") if isinstance(config, Mapping) else None
    )
    expected_signing_key_identity = (
        config.get("bundle_attestation_key_identity_sha256")
        if isinstance(config, Mapping) else None
    )
    expected_generation_response_models = (
        config.get("generation_allowed_response_models")
        if isinstance(config, Mapping) else None
    )
    expected_agent_response_models = (
        config.get("rollout_allowed_response_models")
        if isinstance(config, Mapping) else None
    )
    expected_runtime_response_models = (
        config.get("runtime_allowed_response_models")
        if isinstance(config, Mapping) else None
    )
    experiment_config_sha256 = (
        digest_json(config) if isinstance(config, Mapping) else None
    )
    experiment_contract = build_experiment_contract(config)
    experiment_contract_verified = valid_experiment_contract(
        experiment_contract,
        expected_source_config_sha256=experiment_config_sha256,
    )
    experiment_contract_sha256 = digest_json(experiment_contract)
    preflight_verified = (
        isinstance(expected_generation_provider, Mapping)
        and isinstance(expected_agent_provider, Mapping)
        and isinstance(expected_runtime_provider, Mapping)
        and isinstance(expected_signing_key_identity, str)
        and all(
            isinstance(models, list) and bool(models)
            and all(isinstance(model, str) and model for model in models)
            for models in (
                expected_generation_response_models,
                expected_agent_response_models,
                expected_runtime_response_models,
            )
        )
        and valid_production_preflight(
            production_preflight,
            expected_generation_provider=expected_generation_provider,
            expected_agent_provider=expected_agent_provider,
            expected_runtime_provider=expected_runtime_provider,
            expected_signing_key_identity=expected_signing_key_identity,
            expected_experiment_config_sha256=experiment_config_sha256,
            expected_generation_response_models=expected_generation_response_models,
            expected_agent_response_models=expected_agent_response_models,
            expected_runtime_response_models=expected_runtime_response_models,
        )
    )
    recorded_execution = (
        config.get("execution_provenance") if isinstance(config, Mapping) else None
    )
    execution_verification = verify_execution_provenance(project, recorded_execution)
    raw_holdouts = history.get("holdouts")
    if isinstance(raw_holdouts, list) and raw_holdouts:
        holdouts = [item for item in raw_holdouts if isinstance(item, Mapping)]
    else:
        holdout = history.get("holdout")
        holdouts = [holdout] if isinstance(holdout, Mapping) else []
    jobs = [job for holdout in holdouts for job in holdout.get("jobs", [])]
    results = [job.get("result", {}) for job in jobs if isinstance(job, Mapping)]
    total = len(results)
    holdout_freshness = fresh_holdout_evidence(
        holdouts, policy, development_reports=history.get("rounds") or [],
    )
    partition_isolation = partition_isolation_evidence(history, holdouts)
    generated = [item for item in results if Path(str(item.get("task_path", ""))).is_file()]
    verified_task_scores: set[int] = set()
    task_score_failures = []
    for item in results:
        claimed = item.get("task_score")
        if not isinstance(claimed, Mapping):
            continue
        task_path = Path(str(item.get("task_path", "")))
        try:
            expected = score_file(
                task_path, min_score=float(policy["score_threshold"])
            ).to_dict()
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            expected = None
        # The source path is operational metadata and may change when a frozen
        # experiment is relocated. Every semantic score field must still match.
        claimed_semantics = {key: value for key, value in claimed.items() if key != "path"}
        expected_semantics = (
            {key: value for key, value in expected.items() if key != "path"}
            if isinstance(expected, Mapping) else None
        )
        category_matches = (
            isinstance(expected, Mapping)
            and item.get("category") == expected.get("training_category")
        )
        if (
            expected_semantics is not None
            and digest_json(claimed_semantics) == digest_json(expected_semantics)
            and category_matches
        ):
            verified_task_scores.add(id(item))
        elif (
            claimed.get("eligible") is True
            and claimed.get("score", 0) >= policy["score_threshold"]
        ):
            task_score_failures.append(str(task_path))
    structurally_verified_sandbox_scores: set[int] = set()
    verified_sandbox_scores: set[int] = set()
    executable_revalidated_scores: set[int] = set()
    executable_revalidation_digests: dict[int, str] = {}
    sandbox_score_failures = []
    executable_revalidation_failures = []
    sandbox_revalidation = sandbox_revalidation or {}
    for item in results:
        claimed = item.get("sandbox_score")
        if not isinstance(claimed, Mapping):
            continue
        root = Path(str(item.get("output", "")))
        try:
            summary = load(root / "score_summary.json")
            reports = summary.get("sandboxes")
            stored = next(
                (
                    report for report in reports
                    if isinstance(report, Mapping) and report == claimed
                ),
                None,
            ) if isinstance(reports, list) else None
        except (OSError, json.JSONDecodeError, TypeError, IndexError):
            stored = None
        task_path = Path(str(item.get("task_path", "")))
        revalidation = sandbox_revalidation.get(
            sandbox_revalidation_key(root, task_path), {}
        )
        rerun = (
            revalidation.get("report")
            if isinstance(revalidation, Mapping) else None
        )
        executable_verified = (
            isinstance(revalidation, Mapping)
            and revalidation.get("verified") is True
            and score_envelope(rerun) == score_envelope(claimed)
            and valid_score_report(
                rerun,
                root=root,
                project=project,
                threshold=float(policy["score_threshold"]),
                task_path=task_path,
            )
        )
        if executable_verified:
            executable_revalidated_scores.add(id(item))
            executable_revalidation_digests[id(item)] = digest_json(
                score_envelope(rerun)
            )
        structurally_verified = (
            stored == claimed
            and valid_score_report(
                stored,
                root=root,
                project=project,
                threshold=float(policy["score_threshold"]),
                task_path=task_path,
            )
        )
        if structurally_verified:
            structurally_verified_sandbox_scores.add(id(item))
        if structurally_verified and executable_verified:
            verified_sandbox_scores.add(id(item))
        elif (
            claimed.get("passed") is True
            and claimed.get("score", 0) >= policy["score_threshold"]
        ):
            if not structurally_verified:
                sandbox_score_failures.append(str(item.get("task_path", "")))
            if not executable_verified:
                executable_revalidation_failures.append(str(task_path))
    def task_passes(item: Mapping[str, Any]) -> bool:
        return (
            id(item) in verified_task_scores
            and item.get("task_score", {}).get("eligible") is True
            and item.get("task_score", {}).get("score", 0)
            >= policy["score_threshold"]
        )

    def sandbox_passes(item: Mapping[str, Any]) -> bool:
        return (
            task_passes(item)
            and id(item) in verified_sandbox_scores
            and item.get("sandbox_score", {}).get("passed") is True
            and item.get("sandbox_score", {}).get("score", 0)
            >= policy["score_threshold"]
        )

    verified_final_results: set[int] = set()
    expected_final_scores: dict[int, float] = {}
    final_result_failures = []
    def frozen_task(item: Mapping[str, Any]) -> Mapping[str, Any]:
        try:
            document = load(Path(str(item.get("task_path", ""))))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}
        return document if isinstance(document, Mapping) else {}

    for item in results:
        live = item.get("live_rollout")
        expected_score = None
        if (
            sandbox_passes(item)
            and isinstance(live, Mapping)
            and valid_rollout_outcome(live, policy, task=frozen_task(item))
        ):
            expected_score = round(min(
                10.0,
                float(item["sandbox_score"]["score"]) * 0.9
                + float(live["quality_score"]),
            ), 2)
            expected_final_scores[id(item)] = expected_score
        claimed_score = item.get("score")
        if (
            expected_score is not None
            and item.get("passed") is True
            and isinstance(claimed_score, (int, float))
            and not isinstance(claimed_score, bool)
            and math.isclose(float(claimed_score), expected_score)
        ):
            verified_final_results.add(id(item))
        elif (
            item.get("passed") is True
            and isinstance(claimed_score, (int, float))
            and not isinstance(claimed_score, bool)
            and claimed_score >= policy["score_threshold"]
        ):
            final_result_failures.append(str(item.get("task_path", "")))

    def result_passes(item: Mapping[str, Any]) -> bool:
        return (
            id(item) in verified_final_results
            and expected_final_scores[id(item)] >= policy["score_threshold"]
        )

    task_good = [item for item in results if task_passes(item)]
    built = [item for item in results if sandbox_passes(item)]
    qualified = [item for item in results if result_passes(item)]
    qualified_ids = {id(item) for item in qualified}
    post_score_qualified = sum(id(item) in qualified_ids for item in built)
    claimed_qualified = [
        item for item in results
        if item.get("passed") is True
        and item.get("score", 0) >= policy["score_threshold"]
    ]
    lineage_violations = sum(
        item.get("passed") is True
        and item.get("score", 0) >= policy["score_threshold"]
        and not sandbox_passes(item)
        for item in results
    )
    qualification_lineage = lineage_violations == 0
    task_score_provenance = not task_score_failures
    sandbox_score_provenance = not sandbox_score_failures
    sandbox_executable_revalidation = (
        bool(executable_revalidated_scores)
        and not executable_revalidation_failures
    )
    final_result_provenance = not final_result_failures
    rates = {
        "generation_completion": len(generated) / total if total else 0.0,
        "task_good_yield": len(task_good) / total if total else 0.0,
        "task_good_yield_ci95_lower": wilson_lower(len(task_good), total),
        "sandbox_build_yield": len(built) / len(task_good) if task_good else 0.0,
        "sandbox_build_yield_ci95_lower": wilson_lower(len(built), len(task_good)),
        "post_score_qualified": post_score_qualified,
        "post_score_survival_rate": post_score_qualified / len(built) if built else None,
        "post_score_survival_ci95_lower": (
            wilson_lower(post_score_qualified, len(built)) if built else None
        ),
        "end_to_end_rate": len(qualified) / total if total else 0.0,
        "end_to_end_ci95_lower": wilson_lower(len(qualified), total),
    }

    category_counts: dict[str, dict[str, Any]] = {}
    categories = sorted({str(item.get("category", "unknown")) for item in results})
    for category in categories:
        members = [item for item in results if str(item.get("category", "unknown")) == category]
        passed = sum(result_passes(item) for item in members)
        category_counts[category] = {
            "requested": len(members), "qualified": passed,
            "rate": passed / len(members) if members else 0.0,
        }

    task_documents = []
    task_digests = []
    for item in generated:
        path = Path(str(item["task_path"]))
        try:
            raw = path.read_bytes()
            document = json.loads(raw)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(document, dict):
            task_documents.append(document)
            task_digests.append(hashlib.sha256(raw).hexdigest())
    exact_duplicate_rate = (
        (len(task_digests) - len(set(task_digests))) / len(task_digests)
        if task_digests else 1.0
    )
    expected_generation_provider = history.get("config", {}).get(
        "generation_provider"
    )
    if not isinstance(expected_generation_provider, Mapping):
        expected_generation_provider = {}
    generation_provenance: dict[str, dict[str, Any]] = {}
    generation_failures = []
    for item in generated:
        task_path = Path(str(item.get("task_path", "")))
        manifest_path = Path(str(item.get("sample_manifest", "")))
        try:
            task_document = load(task_path)
            sample_manifest = load(manifest_path)
            snapshot = generation_provenance_snapshot(
                sample_manifest,
                task_document,
                expected_provider=(None if task_document.get("artifacts", {}).get("generation_pipeline", {}).get("backend") == "spec"
                                   else expected_generation_provider),
                expected_task_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(),
            )
            if snapshot["sample_seed"] != item.get("sample_seed"):
                raise ValueError("result sample seed does not match generation manifest")
            generation_provenance[str(task_path.resolve())] = snapshot
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            generation_failures.append({
                "task_path": str(task_path), "error_type": type(exc).__name__,
            })
    actual_generation_models: Counter[str] = Counter()
    for snapshot in generation_provenance.values():
        for attempt in snapshot.get("attempts", []):
            actual_generation_models.update(
                attempt.get("llm_trace", {}).get("models", {})
            )
    generation_response_models_authorized = (
        isinstance(expected_generation_response_models, list)
        and bool(actual_generation_models)
        and set(actual_generation_models) <= set(expected_generation_response_models)
    )

    claimed_live_reports = [
        item.get("live_rollout") for item in claimed_qualified
        if isinstance(item.get("live_rollout"), Mapping)
    ]
    rollout_outcomes_verified = sum(
        valid_rollout_outcome(
            item["live_rollout"], policy, task=frozen_task(item),
        )
        for item in claimed_qualified
        if isinstance(item.get("live_rollout"), Mapping)
    )
    rollout_outcome_integrity = (
        bool(claimed_qualified)
        and len(claimed_live_reports) == len(claimed_qualified)
        and rollout_outcomes_verified == len(claimed_qualified)
    )
    live_reports = [
        item.get("live_rollout") for item in qualified
        if isinstance(item.get("live_rollout"), Mapping)
    ]
    audited_episodes = [
        episode for report in claimed_live_reports
        for episode in report.get("episodes", []) if isinstance(episode, Mapping)
    ]
    episodes = [
        episode for report in live_reports
        for episode in report.get("episodes", []) if isinstance(episode, Mapping)
    ]
    environment_errors = sum(bool(item.get("issues")) for item in audited_episodes)
    fallbacks = sum(
        "runtime_llm_fallback" in item.get("issues", [])
        for item in audited_episodes
    )
    successes = sum(item.get("agent_success") is True for item in episodes)
    complete_episodes = sum(complete_episode(item) for item in audited_episodes)
    user_turn_records = []
    for item in claimed_qualified:
        try:
            task = load(Path(str(item.get("task_path", ""))))
        except (OSError, json.JSONDecodeError, TypeError):
            task = {}
        live = item.get("live_rollout")
        if not isinstance(live, Mapping) or not isinstance(task, Mapping):
            continue
        user_turn_records.extend(
            (step, task)
            for episode in live.get("episodes", [])
            if isinstance(episode, Mapping)
            for step in episode.get("trajectory", [])
            if isinstance(step, Mapping)
            and step.get("path") == "/v1/user_simulator"
        )
    user_turns = [step for step, _ in user_turn_records]
    valid_user_turns = sum(
        valid_user_turn(step, task) for step, task in user_turn_records
    )
    user_outcomes = Counter(
        str(step.get("result", {}).get("outcome_category"))
        for step, task in user_turn_records if valid_user_turn(step, task)
    )
    interactive_outcomes = {
        "information_required", "user_correction", "user_rejection",
        "agent_off_topic", "agent_premature_completion", "unrecognized",
    }
    rollout_coverage = all(
        len(item.get("live_rollout", {}).get("episodes", []))
        >= policy["min_episodes_per_qualified_sandbox"]
        for item in qualified
    ) and bool(qualified)
    rollout_provenance = bool(qualified) and all(
        valid_rollout_provenance(item) for item in qualified
    )
    response_provenance_reports = [
        response_provenance(item.get("live_rollout")) for item in qualified
    ]
    response_provenance_verified = (
        bool(qualified)
        and all(report["verified"] is True for report in response_provenance_reports)
    )
    actual_agent_models = Counter(
        {
            model: sum(
                report["agent_response_models"].get(model, 0)
                for report in response_provenance_reports
            )
            for model in {
                model
                for report in response_provenance_reports
                for model in report["agent_response_models"]
            }
        }
    )
    actual_runtime_models = Counter(
        {
            model: sum(
                report["runtime_response_models"].get(model, 0)
                for report in response_provenance_reports
            )
            for model in {
                model
                for report in response_provenance_reports
                for model in report["runtime_response_models"]
            }
        }
    )
    rollout_response_models_authorized = (
        isinstance(expected_agent_response_models, list)
        and isinstance(expected_runtime_response_models, list)
        and bool(actual_agent_models)
        and bool(actual_runtime_models)
        and set(actual_agent_models) <= set(expected_agent_response_models)
        and set(actual_runtime_models) <= set(expected_runtime_response_models)
    )
    response_models_authorized = (
        generation_response_models_authorized
        and rollout_response_models_authorized
    )
    provider_bindings_verified = sum(
        valid_provider_binding(
            item,
            expected_agent_provider=expected_agent_provider,
            expected_runtime_provider=expected_runtime_provider,
        )
        for item in qualified
    )
    provider_identity_consistency = (
        bool(qualified) and provider_bindings_verified == len(qualified)
    )
    configured_same_provider = (
        isinstance(expected_agent_provider, Mapping)
        and isinstance(expected_runtime_provider, Mapping)
        and expected_agent_provider.get("host")
            == expected_runtime_provider.get("host")
    )
    same_provider_evaluator_items = sum(
        configured_same_provider
        or item.get("live_rollout", {}).get("agent_provider_sha256")
            == item.get("live_rollout", {}).get("runtime_provider_sha256")
        for item in qualified
    )
    same_provider_evaluator_rate = (
        same_provider_evaluator_items / len(qualified) if qualified else 1.0
    )
    container_rollout_execution = bool(qualified) and all(
        valid_container_rollout_execution(
            item.get("live_rollout"), Path(str(item.get("output", "")))
        )
        for item in qualified
    )

    readiness_reports = [_artifact(item, "training_readiness.json") for item in qualified]
    agentic_reports = [
        _artifact(item, "agentic_training_value_live.json") for item in qualified
    ]
    offline_agentic_reports = [
        _artifact(item, "agentic_training_value.json") for item in qualified
    ]
    # Re-scan every sample that claims final success, even if another
    # independent gate already removed it from the exportable set. This keeps
    # governance diagnostics fail-closed under post-run task tampering.
    governance_reports = [
        _artifact(item, "data_governance.json") for item in claimed_qualified
    ]
    governed_materials = sum(
        valid_data_governance(
            report,
            Path(str(item.get("output", ""))),
            Path(str(item.get("task_path", ""))),
        )
        for item, report in zip(claimed_qualified, governance_reports)
    )
    observed_images: dict[str, Mapping[str, Any] | None] = {}
    for item in qualified:
        tag = str(item.get("container_image_tag", ""))
        if tag not in observed_images:
            observed = image_inspector(tag)
            observed_images[tag] = dict(observed) if isinstance(observed, Mapping) else None
    container_reports = [
        verify_container_provenance(
            Path(str(item.get("output", ""))),
            expected_tag=str(item.get("container_image_tag", "")),
            observed_image=observed_images[str(item.get("container_image_tag", ""))],
            require_local_image=True,
        )
        for item in qualified
    ]
    reproducible_containers = sum(
        report.get("verified") is True for report in container_reports
    )
    privacy_reports = [_artifact(item, "trajectory_privacy.json") for item in qualified]
    privacy_verified = sum(
        report == audit_rollout_privacy(item.get("live_rollout", {}))
        and report.get("eligible_for_policy_training_export") is True
        for item, report in zip(qualified, privacy_reports)
    )
    verified_readiness_reports = sum(
        valid_training_readiness(report) for report in readiness_reports
    )
    runtime_integrity = (
        bool(qualified)
        and len(readiness_reports) == len(qualified)
        and verified_readiness_reports == len(qualified)
    )
    qualified_tasks = []
    for item in qualified:
        try:
            value = load(Path(str(item.get("task_path", ""))))
        except (OSError, json.JSONDecodeError, TypeError):
            value = None
        qualified_tasks.append(value if isinstance(value, Mapping) else {})
    verified_reward_calibrations = sum(
        valid_reward_calibration(report, task)
        and consistent_causality_witness(offline, report, task)
        for offline, report, task in zip(
            offline_agentic_reports, agentic_reports, qualified_tasks
        )
    )
    tool_and_reward_integrity = (
        bool(qualified)
        and len(agentic_reports) == len(qualified)
        and len(offline_agentic_reports) == len(qualified)
        and len(qualified_tasks) == len(qualified)
        and verified_reward_calibrations == len(qualified)
    )
    container_reward_calibration = bool(qualified) and all(
        valid_container_rollout_execution(report, Path(str(item.get("output", ""))))
        for item, report in zip(qualified, agentic_reports)
    )
    counterfactuals = _counterfactual_counts(agentic_reports)
    counterfactual_sandboxes = _counterfactual_sandbox_counts(agentic_reports)
    false_positive_rate = (
        counterfactuals["negative_errors"] / counterfactuals["negative_total"]
        if counterfactuals["negative_total"] else 1.0
    )
    false_negative_rate = (
        counterfactuals["positive_errors"] / counterfactuals["positive_total"]
        if counterfactuals["positive_total"] else 1.0
    )
    false_positive_upper = wilson_upper(
        counterfactual_sandboxes["negative_errors"], counterfactual_sandboxes["total"]
    )
    false_negative_upper = wilson_upper(
        counterfactual_sandboxes["positive_errors"], counterfactual_sandboxes["total"]
    )
    batch_measurements = []
    for batch in holdouts:
        batch_results = [
            job.get("result", {}) for job in batch.get("jobs", [])
            if isinstance(job, Mapping)
        ]
        batch_total = len(batch_results)
        batch_generated = sum(
            Path(str(item.get("task_path", ""))).is_file() for item in batch_results
        )
        batch_task_good = [item for item in batch_results if task_passes(item)]
        batch_built = [item for item in batch_results if sandbox_passes(item)]
        batch_qualified = [item for item in batch_results if result_passes(item)]
        batch_lineage_integrity = not any(
            item.get("passed") is True
            and item.get("score", 0) >= policy["score_threshold"]
            and not sandbox_passes(item)
            for item in batch_results
        )
        batch_categories = {}
        for category in sorted({str(item.get("category", "unknown")) for item in batch_results}):
            members = [
                item for item in batch_results
                if str(item.get("category", "unknown")) == category
            ]
            count = sum(result_passes(item) for item in members)
            batch_categories[category] = count / len(members) if members else 0.0
        task_rate = len(batch_task_good) / batch_total if batch_total else 0.0
        build_rate = len(batch_built) / len(batch_task_good) if batch_task_good else 0.0
        e2e_rate = len(batch_qualified) / batch_total if batch_total else 0.0
        batch_mix = category_mix(batch_results, policy)
        batch_passed = (
            batch_generated >= policy["min_tasks"]
            and batch.get("summary", {}).get("fresh_tasks_verified") is True
            and task_rate >= policy["min_task_yield"]
            and wilson_lower(len(batch_task_good), batch_total)
                >= policy["min_task_yield_ci95_lower"]
            and build_rate >= policy["min_build_yield"]
            and wilson_lower(len(batch_built), len(batch_task_good))
                >= policy["min_build_yield_ci95_lower"]
            and e2e_rate >= policy["min_end_to_end_rate"]
            and wilson_lower(len(batch_qualified), batch_total)
                >= policy["min_end_to_end_ci95_lower"]
            and bool(batch_categories)
            and all(rate >= policy["min_category_rate"] for rate in batch_categories.values())
            and batch_mix["passed"]
            and batch_lineage_integrity
        )
        batch_measurements.append({
            "batch": batch.get("holdout_batch"),
            "requested": batch_total,
            "generated": batch_generated,
            "task_good_yield": task_rate,
            "task_good_yield_ci95_lower": wilson_lower(len(batch_task_good), batch_total),
            "sandbox_build_yield": build_rate,
            "sandbox_build_yield_ci95_lower": wilson_lower(len(batch_built), len(batch_task_good)),
            "end_to_end_rate": e2e_rate,
            "end_to_end_ci95_lower": wilson_lower(len(batch_qualified), batch_total),
            "by_category": batch_categories,
            "category_mix": batch_mix,
            "qualification_lineage": batch_lineage_integrity,
            "passed": batch_passed,
        })
    material_items = []
    for item in qualified:
        task_path = Path(str(item.get("task_path", "")))
        sandbox_score = item.get("sandbox_score", {})
        fingerprint = sandbox_score.get("evidence_fingerprint")
        live = item.get("live_rollout")
        if not task_path.is_file() or not isinstance(fingerprint, str) or not fingerprint:
            continue
        if not isinstance(live, Mapping):
            continue
        item_episodes = live.get("episodes", [])
        material_items.append({
            "task_path": str(task_path.resolve()),
            "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
            "sandbox_root": str(Path(str(item.get("output", ""))).resolve()),
            "sandbox_evidence_fingerprint": fingerprint,
            "sandbox_artifacts_sha256": artifact_digests(
                Path(str(item.get("output", "")))
            ),
            "sandbox_evidence_sha256": evidence_artifact_digests(
                Path(str(item.get("output", "")))
            ),
            "category": str(item.get("category", "unknown")),
            "score": expected_final_scores[id(item)],
            "rollout_sha256": digest_json(live),
            "episode_count": len(item_episodes) if isinstance(item_episodes, list) else 0,
            "successful_episodes": sum(
                episode.get("agent_success") is True
                for episode in item_episodes if isinstance(episode, Mapping)
            ) if isinstance(item_episodes, list) else 0,
            "generation_provenance": generation_provenance.get(
                str(task_path.resolve())
            ),
        })
    material_fingerprints = [
        item["sandbox_evidence_fingerprint"] for item in material_items
    ]
    material_manifest = {
        "version": MATERIAL_MANIFEST_VERSION,
        "kind": "agentic_rl_pretraining_materials",
        "evaluator_source_digest": history.get("config", {}).get("source_digest"),
        "execution_provenance": recorded_execution,
        "experiment_config_sha256": experiment_config_sha256,
        "experiment_contract": experiment_contract,
        "experiment_contract_sha256": experiment_contract_sha256,
        "certification_policy": dict(policy),
        "certification_policy_sha256": certification_policy_sha256,
        "items": material_items,
    }
    material_manifest["dataset_sha256"] = digest_json(material_manifest)

    measurements = {
        "certification_policy": {
            "version": policy["version"],
            "sha256": certification_policy_sha256,
            "verified": certification_policy_verified,
        },
        "experiment_contract": {
            "version": experiment_contract["version"],
            "sha256": experiment_contract_sha256,
            "verified": experiment_contract_verified,
        },
        "requested_tasks": total,
        "generated_tasks": len(generated),
        "parseable_tasks": len(task_documents),
        "task_qualified": len(task_good),
        "sandbox_built": len(built),
        "training_ready": len(qualified),
        "task_score_provenance": {
            "verified": len(verified_task_scores),
            "failures": len(task_score_failures),
        },
        "qualification_lineage": {
            "verified": qualification_lineage,
            "violations": lineage_violations,
        },
        "sandbox_score_provenance": {
            "verified": len(structurally_verified_sandbox_scores),
            "failures": len(sandbox_score_failures),
        },
        "sandbox_executable_revalidation": {
            "verified": len(executable_revalidated_scores),
            "expected": (
                len(executable_revalidated_scores)
                + len(executable_revalidation_failures)
            ),
            "failures": len(executable_revalidation_failures),
            "evidence_set_sha256": (
                digest_json(sorted(executable_revalidation_digests.values()))
                if executable_revalidation_digests else None
            ),
        },
        "final_result_provenance": {
            "verified": len(verified_final_results),
            "failures": len(final_result_failures),
        },
        **rates,
        "by_category": category_counts,
        "category_mix": category_mix(results, policy),
        "exact_duplicate_rate": exact_duplicate_rate,
        "near_duplicate_rate": near_duplicate_rate(task_documents),
        "live_sandboxes": len(live_reports),
        "episodes": len(episodes),
        "audited_episodes": len(audited_episodes),
        "successful_episodes": successes,
        "failed_episodes": len(episodes) - successes,
        "complete_episodes": complete_episodes,
        "agent_success_rate": successes / len(episodes) if episodes else 0.0,
        "environment_errors": environment_errors,
        "environment_error_rate": (
            environment_errors / len(audited_episodes)
            if audited_episodes else 1.0
        ),
        "llm_fallbacks": fallbacks,
        "rollout_coverage": rollout_coverage,
        "rollout_outcome_integrity": {
            "verified": rollout_outcomes_verified,
            "expected": len(claimed_qualified),
            "all_verified": rollout_outcome_integrity,
        },
        "rollout_provenance": rollout_provenance,
        "model_response_provenance": {
            "verified": sum(
                report["verified"] is True
                for report in response_provenance_reports
            ),
            "expected": len(qualified),
            "all_verified": response_provenance_verified,
            "agent_response_models": dict(sorted(actual_agent_models.items())),
            "runtime_response_models": dict(sorted(actual_runtime_models.items())),
        },
        "model_response_authorization": {
            "all_authorized": response_models_authorized,
            "allowed": {
                "generation": list(expected_generation_response_models or []),
                "agent": list(expected_agent_response_models or []),
                "runtime": list(expected_runtime_response_models or []),
            },
            "actual": {
                "generation": dict(sorted(actual_generation_models.items())),
                "agent": dict(sorted(actual_agent_models.items())),
                "runtime": dict(sorted(actual_runtime_models.items())),
            },
        },
        "provider_identity_consistency": {
            "verified": provider_bindings_verified,
            "expected": len(qualified),
            "all_verified": provider_identity_consistency,
        },
        "evaluator_independence": {
            "same_provider_items": same_provider_evaluator_items,
            "total_items": len(qualified),
            "same_provider_rate": same_provider_evaluator_rate,
        },
        "container_rollout_execution": container_rollout_execution,
        "trajectory_schema_complete": (
            bool(audited_episodes)
            and complete_episodes == len(audited_episodes)
        ),
        "user_simulator_calls": len(user_turns),
        "user_simulator_valid_calls": valid_user_turns,
        "user_simulator_protocol_rate": (
            valid_user_turns / len(user_turns) if user_turns else 0.0
        ),
        "user_simulator_outcomes": dict(user_outcomes),
        "runtime_integrity": runtime_integrity,
        "runtime_readiness_coverage": {
            "verified": verified_readiness_reports,
            "expected": len(qualified),
        },
        "tool_and_reward_integrity": tool_and_reward_integrity,
        "reward_calibration_coverage": {
            "verified": verified_reward_calibrations,
            "expected": len(qualified),
        },
        "container_reward_calibration": container_reward_calibration,
        "data_governance": {
            "reports": len(governance_reports),
            "verified": governed_materials,
            "all_verified": (
                bool(claimed_qualified)
                and len(governance_reports) == len(claimed_qualified)
                and governed_materials == len(claimed_qualified)
            ),
        },
        "container_reproducibility": {
            "reports": len(container_reports),
            "verified": reproducible_containers,
            "observed_images_sha256": digest_json(observed_images),
            "all_verified": (
                bool(qualified)
                and len(container_reports) == len(qualified)
                and reproducible_containers == len(qualified)
            ),
            "failed_gates": dict(Counter(
                gate
                for report in container_reports
                for gate in report.get("failed_gates", [])
            )),
        },
        "trajectory_privacy": {
            "reports": len(privacy_reports),
            "verified": privacy_verified,
            "all_verified": (
                bool(qualified)
                and len(privacy_reports) == len(qualified)
                and privacy_verified == len(qualified)
            ),
        },
        "reward_counterfactuals": counterfactuals,
        "reward_counterfactual_sandboxes": counterfactual_sandboxes,
        "reward_false_positive_rate": false_positive_rate,
        "reward_false_negative_rate": false_negative_rate,
        "reward_false_positive_ci95_upper": false_positive_upper,
        "reward_false_negative_ci95_upper": false_negative_upper,
        "fresh_holdout_verified": holdout_freshness["verified"],
        "fresh_holdout_evidence": holdout_freshness,
        "partition_isolation": partition_isolation,
        "holdout_batches": batch_measurements,
        "material_manifest_items": len(material_items),
        "generation_provenance": {
            "verified": len(generation_provenance),
            "expected": len(generated),
            "failures": generation_failures,
        },
        "execution_environment": execution_verification,
        "production_experiment_profile": production_experiment,
        "production_preflight": dict(production_preflight or {}),
    }

    gates = {
        "certification_policy": certification_policy_verified,
        "portable_experiment_contract": experiment_contract_verified,
        "production_experiment_profile": production_experiment,
        "production_preflight": preflight_verified,
        "materialized_sample_size": (
            len(task_documents) >= policy["min_tasks"] * policy["min_holdout_batches"]
            and len(task_documents) == len(generated)
        ),
        "independent_holdout_batches": (
            len(batch_measurements) >= policy["min_holdout_batches"]
            and all(item["passed"] for item in batch_measurements)
        ),
        "fresh_holdout": measurements["fresh_holdout_verified"],
        "holdout_partition_isolation": partition_isolation["isolated"],
        "task_score_provenance": task_score_provenance,
        "qualification_lineage": qualification_lineage,
        "sandbox_score_provenance": sandbox_score_provenance,
        "sandbox_executable_revalidation": sandbox_executable_revalidation,
        "final_result_provenance": final_result_provenance,
        "task_good_yield": (
            rates["task_good_yield"] >= policy["min_task_yield"]
            and rates["task_good_yield_ci95_lower"] >= policy["min_task_yield_ci95_lower"]
        ),
        "sandbox_build_yield": (
            rates["sandbox_build_yield"] >= policy["min_build_yield"]
            and rates["sandbox_build_yield_ci95_lower"] >= policy["min_build_yield_ci95_lower"]
        ),
        "end_to_end_yield": (
            rates["end_to_end_rate"] >= policy["min_end_to_end_rate"]
            and rates["end_to_end_ci95_lower"] >= policy["min_end_to_end_ci95_lower"]
        ),
        "category_floor": bool(category_counts) and all(
            item["rate"] >= policy["min_category_rate"] for item in category_counts.values()
        ),
        "category_mix": measurements["category_mix"]["passed"],
        "exact_deduplication": exact_duplicate_rate == 0,
        "near_deduplication": measurements["near_duplicate_rate"] <= policy["max_near_duplicate_rate"],
        "rollout_coverage": rollout_coverage and len(episodes) >= policy["min_total_episodes"],
        "rollout_outcome_integrity": rollout_outcome_integrity,
        "rollout_provenance": rollout_provenance,
        "model_response_provenance": response_provenance_verified,
        "model_response_authorization": response_models_authorized,
        "provider_identity_consistency": provider_identity_consistency,
        "evaluator_independence": (
            bool(qualified)
            and same_provider_evaluator_rate
                <= policy["max_same_provider_evaluator_rate"]
        ),
        "container_rollout_execution": container_rollout_execution,
        "trajectory_schema": measurements["trajectory_schema_complete"],
        "user_simulator_protocol": (
            bool(user_turns)
            and measurements["user_simulator_protocol_rate"]
            >= policy["min_user_simulator_protocol_rate"]
        ),
        "user_simulator_outcome_coverage": user_outcome_coverage(user_turn_records, policy["min_user_outcome_categories"]),
        "trajectory_diversity": successes > 0 and counterfactuals.get("negative_total", 0) > 0,
        "environment_integrity": (
            measurements["environment_error_rate"] <= policy["max_environment_error_rate"]
            and fallbacks == 0
        ),
        "runtime_state_integrity": runtime_integrity,
        "tool_and_reward_integrity": tool_and_reward_integrity,
        "container_reward_calibration": container_reward_calibration,
        "data_governance": measurements["data_governance"]["all_verified"],
        "container_reproducibility": measurements["container_reproducibility"][
            "all_verified"
        ],
        "trajectory_privacy": measurements["trajectory_privacy"]["all_verified"],
        "execution_environment": execution_verification["verified"],
        "generation_provenance": (
            bool(generated)
            and len(generation_provenance) == len(generated)
            and not generation_failures
            and all(item.get("generation_provenance") for item in material_items)
        ),
        "material_identity": (
            len(material_items) == len(qualified)
            and len(material_fingerprints) == len(set(material_fingerprints))
            and isinstance(material_manifest.get("evaluator_source_digest"), str)
            and bool(material_manifest["evaluator_source_digest"])
            and all(item.get("sandbox_artifacts_sha256") for item in material_items)
            and all(item.get("sandbox_evidence_sha256") for item in material_items)
        ),
        "reward_false_positive_rate": (
            false_positive_rate <= policy["max_reward_false_positive_rate"]
            and false_positive_upper <= policy["max_reward_false_positive_rate"]
        ),
        "reward_false_negative_rate": (
            false_negative_rate <= policy["max_reward_false_negative_rate"]
            and false_negative_upper <= policy["max_reward_false_negative_rate"]
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    return {
        "certification": "production_prepared_for_agentic_rl",
        "scope": "pre_training_material_readiness",
        "certified": not failed,
        "does_not_certify": [
            "rl_training_execution",
            "downstream_training_system_compatibility",
            "rl_training_convergence",
            "post_training_policy_improvement",
            "cross_model_generalization",
            "offline_rl_algorithm_compatibility",
            "exported_rollouts_as_direct_policy_optimization_targets",
            "data_license_or_distribution_rights",
            "absence_of_same_model_evaluation_bias",
        ],
        "policy": dict(policy),
        "measurements": measurements,
        "gates": gates,
        "failed_gates": failed,
        "materials_manifest": material_manifest,
    }


def default_policy() -> dict[str, Any]:
    return canonical_certification_policy()


def attach_artifact_verification(
    report: dict[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    """Make post-score artifact immutability part of the certification claim."""
    report["material_verification"] = dict(verification)
    report["gates"]["material_artifacts_immutable"] = verification.get("verified") is True
    report["failed_gates"] = [
        name for name, passed in report["gates"].items() if not passed
    ]
    report["certified"] = not report["failed_gates"]
    return report


def attach_bundle_verification(
    report: dict[str, Any], verification: Mapping[str, Any]
) -> dict[str, Any]:
    report["materials_bundle_verification"] = dict(verification)
    report["gates"]["portable_materials_bundle"] = (
        verification.get("verified") is True
        and verification.get("bundle_version") == BUNDLE_VERSION
        and verification.get("production_contract_ready") is True
        and verification.get("trusted_attestation") is True
        and verification.get("dataset_split_ready") is True
        and verification.get("task_family_split_ready") is True
        and verification.get("generation_provenance_ready") is True
        and verification.get("container_rollout_ready") is True
        and verification.get("portable_build_context_ready") is True
        and verification.get("certification_policy_ready") is True
        and verification.get("container_reward_calibration_ready") is True
        and verification.get("provider_identity_ready") is True
        and verification.get("data_governance_ready") is True
        and verification.get("task_lineage_ready") is True
        and verification.get("production_preflight_ready") is True
        and verification.get("experiment_config_ready") is True
        and verification.get("experiment_contract_ready") is True
        and verification.get("trajectory_purpose_ready") is True
        and verification.get("model_response_provenance_ready") is True
        and verification.get("model_response_authorization_ready") is True
        and verification.get("consumer_records_ready") is True
        and verification.get("metadata_privacy_ready") is True
        and verification.get("evaluator_independence_ready") is True
        and verification.get("source_dataset_sha256")
        == report.get("materials_manifest", {}).get("dataset_sha256")
    )
    report["failed_gates"] = [
        name for name, passed in report["gates"].items() if not passed
    ]
    report["certified"] = not report["failed_gates"]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", type=Path, help="loop experiment history.json")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle-output", type=Path)
    parser.add_argument("--bundle-signing-private-key", type=Path)
    parser.add_argument("--bundle-trusted-public-key", type=Path)
    parser.add_argument("--revalidation-workers", type=int, default=4)
    parser.add_argument("--revalidation-timeout", type=int, default=1800)
    args = parser.parse_args()
    history = load(args.history.resolve())
    policy, _ = policy_for_experiment(history.get("config", {}))
    project = args.project.resolve()
    production_preflight = run_production_preflight(
        project,
        args.history.resolve().parent,
        signing_private_key=args.bundle_signing_private_key,
        trusted_public_key=args.bundle_trusted_public_key,
        environment=os.environ,
        experiment_config=history.get("config", {}),
    )
    print(
        "production certification: re-executing sandbox evidence",
        file=sys.stderr,
    )
    sandbox_revalidation = (
        run_sandbox_revalidation(
            history,
            project=project,
            policy=policy,
            max_workers=args.revalidation_workers,
            timeout=args.revalidation_timeout,
        )
        if production_preflight["ready"] else {}
    )
    print(
        "production certification: sandbox evidence revalidation complete "
        f"({sum(item.get('verified') is True for item in sandbox_revalidation.values())}"
        f"/{len(sandbox_revalidation)})",
        file=sys.stderr,
    )
    report = certify(
        history,
        policy,
        project=project,
        sandbox_revalidation=sandbox_revalidation,
        production_preflight=production_preflight,
    )
    from verify_training_materials import verify
    report = attach_artifact_verification(
        report, verify(report["materials_manifest"], project)
    )
    if report["certified"]:
        from export_training_materials import export_bundle, verify_bundle
        bundle_root = (
            args.bundle_output.resolve() if args.bundle_output
            else args.history.resolve().parent / "training_materials_bundle"
        )
        try:
            if bundle_root.is_dir() and any(bundle_root.iterdir()):
                bundle_verification = verify_bundle(
                    bundle_root, trusted_public_key=args.bundle_trusted_public_key
                )
            else:
                bundle_verification = export_bundle(
                    report, bundle_root, project,
                    signing_private_key=args.bundle_signing_private_key,
                    trusted_public_key=args.bundle_trusted_public_key,
                )
        except Exception as exc:
            bundle_verification = {
                "verified": False,
                "failed_gates": ["bundle_export"],
                "error_type": type(exc).__name__,
            }
        report = attach_bundle_verification(report, bundle_verification)
    output = args.output or args.history.with_name("production_readiness.json")
    manifest_path = output.with_name("training_materials_manifest.json")
    # Publish dependencies first. The readiness report is the final commit
    # marker and must never precede its referenced material manifest.
    write_json_atomic(manifest_path, report["materials_manifest"])
    write_json_atomic(output, report)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    return 0 if report["certified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
