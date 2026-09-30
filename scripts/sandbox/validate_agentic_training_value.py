#!/usr/bin/env python3
"""Deterministic gate for whether a sandbox teaches its declared tool policy."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

from env_factory.sandbox_http import HTTPSandboxClient, evaluator_http_timeout
from env_factory.sandbox_runtime import AcceptanceScenarioRunner
from env_factory.evidence.data_governance import provider_identity
from env_factory.runtime_llm import RuntimeLLMConfig
from env_factory.contracts.reward_contract import numeric_answer_counterfactual, terminal_outcome_weight


PLACEHOLDERS = {"fixture-value", "example", "placeholder", "todo", "unknown", "test"}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def import_app(root: Path):
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("agentic_value_app", root / "app.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import generated app.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_app()


def reward_of(run: Mapping[str, Any]) -> float | None:
    for event in reversed(run.get("history", [])):
        body = event.get("body") if isinstance(event, Mapping) else None
        if event.get("operation") == "reward" and isinstance(body, Mapping):
            reward = body.get("reward")
            if (isinstance(reward, (int, float)) and not isinstance(reward, bool)
                    and -1 <= reward <= 1 and math.isfinite(reward)):
                return float(reward)
            return None
    return None


def has_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in PLACEHOLDERS or "fixture-value" in value.casefold()
    if isinstance(value, list):
        return any(has_placeholder(item) for item in value)
    if isinstance(value, Mapping):
        return any(has_placeholder(item) for item in value.values())
    return False


def empty_critical_collection(value: Any) -> bool:
    if isinstance(value, list):
        return not value or any(empty_critical_collection(item) for item in value)
    if isinstance(value, Mapping):
        return any(empty_critical_collection(item) for item in value.values())
    return False


def meaningful_result(body: Any) -> bool:
    if not isinstance(body, Mapping):
        return body not in (None, "", [], {})
    if isinstance(body.get("count"), int):
        return body["count"] > 0
    for key in ("records", "items", "results", "rows", "data"):
        if key in body:
            return isinstance(body[key], (list, Mapping)) and bool(body[key])
    ignored = {"status", "accepted", "request_id", "episode_id"}
    return any(key not in ignored and value not in (None, "", [], {}) for key, value in body.items())


def business_data_reads_from_replay(replay: Any, table_names: set[str]) -> set[str]:
    """Count private table reads made inside successful business tool calls."""
    if not isinstance(replay, Mapping):
        return set()
    reads: set[str] = set()
    for event in replay.get("events", []):
        if not isinstance(event, Mapping) or event.get("event") != "tool_call":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or payload.get("noise") is True:
            continue
        declared = payload.get("business_data_reads")
        if isinstance(declared, list):
            reads.update(name for name in declared if isinstance(name, str) and name in table_names)
    return reads


def business_tool_reads_from_replay(replay: Any, table_names: set[str]) -> list[dict[str, Any]]:
    if not isinstance(replay, Mapping):
        return []
    result: list[dict[str, Any]] = []
    for event in replay.get("events", []):
        if not isinstance(event, Mapping) or event.get("event") != "tool_call":
            continue
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or payload.get("noise") is True:
            continue
        declared = payload.get("business_data_reads")
        result.append({
            "tool_name": payload.get("tool_name"),
            "tables_read": sorted({
                name for name in declared
                if isinstance(name, str) and name in table_names
            }) if isinstance(declared, list) else [],
        })
    return result


def _changed_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, str) and value:
        return value + "__envfactory_probe__"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value + 1
    return None


def _answer_mentions_value(answer: str, value: Any) -> bool:
    """Conservatively identify a private scalar stated in the final answer."""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return False
    rendered = str(value).strip()
    if len(rendered) < 3:
        return False
    if isinstance(value, (int, float)):
        return re.search(rf"(?<![\d.]){re.escape(rendered)}(?![\d.])", answer) is not None
    return rendered in answer


def semantic_terminal_requires_live(task: Mapping[str, Any]) -> bool:
    """Fixture matching cannot test whether a semantic judge follows changed facts."""
    return any(
        isinstance(metric, Mapping)
        and metric.get("category") == "outcome"
        and isinstance(metric.get("evaluator"), Mapping)
        and metric["evaluator"].get("kind") in {"external_llm_judge", "hybrid_outcome"}
        and isinstance(metric.get("evaluation_inputs"), list)
        and "final_agent_response" in metric["evaluation_inputs"]
        for metric in task.get("metrics", [])
    )


def probe_reference_data_causality(
    app: Any, trainer_headers: Mapping[str, str],
    success_scenario: Mapping[str, Any], manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove that a private table change affects a tool's returned evidence."""
    snapshot = getattr(app, "business_snapshot", None)
    mutate = getattr(app, "mutate_business_state", None)
    if not callable(snapshot) or not callable(mutate):
        return {"proved": False, "reason": "private mutation interface unavailable"}
    runner = AcceptanceScenarioRunner(
        app.handle, trainer_headers=trainer_headers,
        business_snapshot=snapshot, mutate_business_state=mutate,
    )
    steps = success_scenario.get("steps", [])
    if not isinstance(steps, list):
        return {"proved": False, "reason": "success steps unavailable"}
    primary_keys = {
        item["table_name"]: set(item.get("primary_key", []))
        for item in manifest.get("tables", [])
        if isinstance(item, Mapping) and isinstance(item.get("table_name"), str)
    }
    answer = " ".join(
        str(step.get("content", "")) for step in steps
        if isinstance(step, Mapping) and step.get("operation") == "agent_response"
    )
    candidate_count = 0
    tool_count = sum(isinstance(step, Mapping) and step.get("operation") == "tool_call" for step in steps)
    per_tool_limit = max(1, min(24, 96 // max(1, tool_count)))
    fallback: dict[str, Any] | None = None
    fallback_mutations: list[dict[str, Any]] = []
    for end, step in enumerate(steps):
        if not isinstance(step, Mapping) or step.get("operation") != "tool_call":
            continue
        tool_candidates = 0
        prefix = [dict(item) for item in steps[:end + 1] if isinstance(item, Mapping)]
        try:
            baseline_run = runner.run({"steps": prefix, "assertions": []})
            baseline_output = baseline_run["history"][-1]["body"]
            status, replay, _ = app.handle("GET", "/v1/replay", None, trainer_headers)
            tables = snapshot()
        except Exception:
            continue
        if status != 200 or not isinstance(tables, Mapping):
            continue
        read_events = [
            event for event in replay.get("events", [])
            if isinstance(event, Mapping) and event.get("event") == "tool_call"
        ] if isinstance(replay, Mapping) else []
        tool_events = [
            event for event in read_events
            if isinstance(event.get("payload"), Mapping)
            and event["payload"].get("tool_name") == step.get("tool_name")
        ]
        if not tool_events:
            continue
        reads = tool_events[-1]["payload"].get("business_data_reads", [])
        for table_name in reads if isinstance(reads, list) else []:
            if table_name not in primary_keys or not isinstance(tables.get(table_name), list):
                continue
            returned_rows = baseline_output.get("records", []) if isinstance(baseline_output, Mapping) else []
            candidate_rows = sorted(tables[table_name], key=lambda row: not any(
                isinstance(observed, Mapping) and observed
                and all(row.get(key) == value for key, value in observed.items())
                for observed in returned_rows
            ))
            for row in candidate_rows:
                if not isinstance(row, Mapping):
                    continue
                selector = {key: row[key] for key in primary_keys[table_name] if key in row}
                if len(selector) != len(primary_keys[table_name]) or not selector:
                    continue
                for field, value in sorted(
                    row.items(),
                    key=lambda item: (
                        not _answer_mentions_value(answer, item[1]),
                        not isinstance(item[1], (int, float)) or isinstance(item[1], bool),
                    ),
                ):
                    if field in primary_keys[table_name]:
                        continue
                    changed = _changed_scalar(value)
                    if changed is None:
                        continue
                    candidate_count += 1
                    tool_candidates += 1
                    altered_steps = [copy.deepcopy(prefix[0]), {
                        "operation": "mutate_business_state",
                        "mutation": {"table": table_name, "selector": selector,
                                     "changes": {field: changed}},
                    }, *copy.deepcopy(prefix[1:])]
                    try:
                        altered = runner.run({"steps": altered_steps, "assertions": []})
                    except Exception:
                        if tool_candidates >= per_tool_limit or candidate_count >= 96:
                            break
                        continue
                    if altered["history"][-1]["body"] != baseline_output:
                        result = {
                            "proved": True, "table": table_name,
                            "tool": step.get("tool_name"), "field": field,
                            "candidates_tried": candidate_count,
                            "answer_value_mentioned": _answer_mentions_value(answer, value),
                            "_probe_mutation": {
                                "table": table_name, "selector": selector,
                                "changes": {field: changed},
                            },
                        }
                        if fallback is None or (
                            result["answer_value_mentioned"] and not fallback["answer_value_mentioned"]
                        ):
                            fallback = result
                        fallback_mutations.append(result["_probe_mutation"])
                    if tool_candidates >= per_tool_limit or candidate_count >= 96:
                        break
                if tool_candidates >= per_tool_limit or candidate_count >= 96:
                    break
            if tool_candidates >= per_tool_limit or candidate_count >= 96:
                break
        if candidate_count >= 96:
            break
    if fallback is not None:
        fallback["_fallback_mutations"] = fallback_mutations
    return fallback or {"proved": False, "candidates_tried": candidate_count}


def probe_reference_data_reward_sensitivity(
    app: Any, trainer_headers: Mapping[str, str],
    success_scenario: Mapping[str, Any], mutation: Mapping[str, Any] | list[Mapping[str, Any]],
    terminal_weight: float,
) -> dict[str, Any]:
    """Keep the final answer fixed after changing an answer-bearing private input."""
    snapshot = getattr(app, "business_snapshot", None)
    mutate = getattr(app, "mutate_business_state", None)
    if not callable(snapshot) or not callable(mutate):
        return {"proved": False, "reason": "private mutation interface unavailable"}
    runner = AcceptanceScenarioRunner(
        app.handle, trainer_headers=trainer_headers,
        business_snapshot=snapshot, mutate_business_state=mutate,
    )
    scenario = scenario_without_assertions(success_scenario)
    steps = scenario.get("steps", [])
    reset_index = next((index for index, step in enumerate(steps)
                        if isinstance(step, Mapping) and step.get("operation") == "reset"), None)
    if reset_index is None or not any(
        isinstance(step, Mapping) and step.get("operation") == "agent_response"
        for step in steps
    ):
        return {"proved": False, "reason": "success trajectory lacks reset or answer"}
    try:
        baseline = runner.run(scenario)
        repeated_baseline = runner.run(scenario)
        baseline_state = snapshot()
    except Exception as exc:
        return {"proved": False, "reason": type(exc).__name__}
    baseline_rewards = [reward_of(baseline), reward_of(repeated_baseline)]
    baseline_reward = baseline_rewards[0]
    required_drop = 0.5 * terminal_weight
    baseline_tools = [item.get("body") for item in baseline["history"]
                      if item.get("operation") == "tool_call"]
    repeated_tools = [item.get("body") for item in repeated_baseline["history"]
                      if item.get("operation") == "tool_call"]
    baseline_tool_results_stable = baseline_tools == repeated_tools
    if (not baseline_tool_results_stable or baseline_reward is None
            or baseline_reward < 1.0 - 1e-9 or baseline_rewards[1] != baseline_reward):
        return {
            "proved": False, "reason": "baseline reward or tool results are unstable",
            "baseline_reward": baseline_reward,
            "baseline_rewards": baseline_rewards,
            "baseline_tool_results_stable": baseline_tool_results_stable,
        }
    variants = [copy.deepcopy(mutation)]
    single = mutation if isinstance(mutation, Mapping) else {}
    table = single.get("table")
    selector = single.get("selector")
    changes = single.get("changes")
    if (isinstance(baseline_state, Mapping) and isinstance(table, str)
            and isinstance(selector, Mapping) and isinstance(changes, Mapping)
            and len(changes) == 1):
        field = next(iter(changes))
        original = next((row.get(field) for row in baseline_state.get(table, [])
                         if isinstance(row, Mapping)
                         and all(row.get(key) == value for key, value in selector.items())), None)
        if isinstance(original, (int, float)) and not isinstance(original, bool):
            candidates = [0]
            if original > 1:
                candidates.append(1)
            for value in candidates:
                decisive = copy.deepcopy(dict(mutation))
                decisive["changes"] = {field: value}
                if decisive not in variants:
                    variants.append(decisive)
    best: dict[str, Any] | None = None
    variants_tried = 0
    for variant in variants:
        altered = copy.deepcopy(scenario)
        mutations = variant if isinstance(variant, list) else [variant]
        altered["steps"][reset_index + 1:reset_index + 1] = [
            {"operation": "mutate_business_state", "mutation": item} for item in mutations
        ]
        changed_rewards: list[float | None] = []
        changed_tool_flags: list[bool] = []
        for _ in range(2):
            try:
                changed = runner.run(altered)
            except Exception:
                break
            changed_rewards.append(reward_of(changed))
            changed_tools = [item.get("body") for item in changed["history"]
                             if item.get("operation") == "tool_call"]
            changed_tool_flags.append(baseline_tools != changed_tools)
        if len(changed_rewards) != 2:
            continue
        variants_tried += 1
        changed_reward = max(value for value in changed_rewards if value is not None) if all(
            value is not None for value in changed_rewards
        ) else None
        tools_changed = all(changed_tool_flags)
        proved = (
            tools_changed and baseline_reward is not None
            and baseline_reward >= 1.0 - 1e-9 and changed_reward is not None
            and changed_reward < 1.0 - 1e-9
            and baseline_reward - changed_reward >= required_drop - 1e-9
        )
        result = {
            "proved": proved, "baseline_reward": baseline_reward,
            "mutation": variant,
            "baseline_rewards": baseline_rewards,
            "baseline_tool_results_stable": baseline_tool_results_stable,
            "changed_reward": changed_reward,
            "changed_rewards": changed_rewards,
            "required_drop": required_drop,
            "tool_results_changed": tools_changed,
            "variants_tried": variants_tried,
        }
        if proved:
            return result
        if best is None or (changed_reward is not None and
                            (best["changed_reward"] is None or changed_reward < best["changed_reward"])):
            best = result
    return best or {"proved": False, "reason": "no executable private mutation",
                    "baseline_reward": baseline_reward,
                    "baseline_rewards": baseline_rewards,
                    "baseline_tool_results_stable": baseline_tool_results_stable,
                    "variants_tried": variants_tried}


def probe_reference_data_reward_candidates(
    app: Any, trainer_headers: Mapping[str, str], success_scenario: Mapping[str, Any],
    mutations: list[Mapping[str, Any]], terminal_weight: float,
) -> dict[str, Any]:
    """Try executable counterfactuals, then combine independently stale facts.

    Several correct outcome components can legitimately retain partial credit
    when just one fact changes. Combine only mutations already observed to
    change tool evidence and reduce stale-answer reward; replay the combination
    twice against the original strict threshold.
    """
    attempts = []
    partial = []
    best = None
    seen = set()
    for mutation in mutations:
        key = json.dumps(mutation, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        result = probe_reference_data_reward_sensitivity(
            app, trainer_headers, success_scenario, mutation, terminal_weight,
        )
        attempts.append({"mutation": mutation, "result": result})
        if result.get("proved"):
            return {**result, "candidate_attempts": attempts}
        changed, baseline = result.get("changed_reward"), result.get("baseline_reward")
        if best is None or (changed is not None and (
                best.get("changed_reward") is None or changed < best["changed_reward"])):
            best = result
        if (result.get("tool_results_changed") and changed is not None
                and baseline is not None and changed < baseline - 1e-9):
            partial.append(result["mutation"])
    if len(partial) > 1:
        result = probe_reference_data_reward_sensitivity(
            app, trainer_headers, success_scenario, partial, terminal_weight,
        )
        attempts.append({"mutation": partial, "result": result})
        if result.get("proved"):
            return {**result, "candidate_attempts": attempts, "combined_counterfactual": True}
    return {**(best or {"proved": False, "reason": "no causal mutation found"}),
            "candidate_attempts": attempts}


def corrupt(value: Any) -> Any:
    if isinstance(value, str):
        return "__envfactory_unknown__"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 987654
    if isinstance(value, list):
        return ["__envfactory_unknown__"]
    if isinstance(value, Mapping):
        changed = dict(value)
        if changed:
            key = next(iter(changed))
            changed[key] = corrupt(changed[key])
        return changed
    return "__envfactory_unknown__"


def argument_probe_keys(step: Mapping[str, Any], tools: list[Any]) -> list[str]:
    """Probe every required tool input, or one supplied input if none is required."""
    arguments = step.get("arguments")
    if not isinstance(arguments, Mapping) or not arguments:
        return []
    schema = next((item.get("function", {}).get("parameters", {}) for item in tools
                   if isinstance(item, Mapping)
                   and isinstance(item.get("function"), Mapping)
                   and item["function"].get("name") == step.get("tool_name")), {})
    required = schema.get("required", []) if isinstance(schema, Mapping) else []
    keys = [key for key in required if isinstance(key, str) and key in arguments] if isinstance(required, list) else []
    return keys or [next(iter(arguments))]


def scenario_without_assertions(scenario: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(scenario))
    result["assertions"] = []
    return result


def dependency_swap_positions(
    steps: list[dict[str, Any]], capability_dag: Mapping[str, Any]
) -> tuple[int, int] | None:
    """Return one actual producer/consumer pair to reverse.

    Multi-step routes may have independent prerequisite tools. Swapping two
    such producers is valid and must not be reported as an order-sensitivity
    failure. Only reverse a pair connected by a declared DAG edge.
    """
    positions: dict[str, int] = {}
    for index, step in enumerate(steps):
        if step.get("operation") == "tool_call" and isinstance(step.get("tool_name"), str):
            positions.setdefault(str(step["tool_name"]), index)
    for edge in capability_dag.get("edges", []):
        if not isinstance(edge, Mapping):
            continue
        source, target = edge.get("from_tool"), edge.get("to_tool")
        if source in positions and target in positions and positions[source] < positions[target]:
            return positions[source], positions[target]
    return None


def validate(
    root: Path,
    *,
    evaluator_mode: str = "mock",
    base_url: str | None = None,
    container_image_id: str | None = None,
) -> dict[str, Any]:
    if evaluator_mode not in {"mock", "live"}:
        raise ValueError("evaluator_mode must be mock or live")
    evaluator_provider = None
    if evaluator_mode == "live":
        runtime = RuntimeLLMConfig.from_env()
        evaluator_provider = provider_identity(runtime.base_url, runtime.model)
    task = load(root / "task.json")
    contract = task.get("training_contract") if isinstance(task.get("training_contract"), Mapping) else {}
    category = contract.get("category", task.get("training_category", "multi_step_agentic"))
    tool_required = category != "direct_response"
    dependency_required = bool(contract.get("dependency", {}).get("required", category == "multi_step_agentic"))
    scenarios = task.get("acceptance_contract", {}).get("executable_scenarios", [])
    by_kind = {
        item.get("kind"): item for item in scenarios
        if isinstance(item, Mapping) and isinstance(item.get("kind"), str)
    }
    failures: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {"counterfactuals": {}}
    required_scenarios = contract.get("required_scenarios", ["goal_success", "goal_failure"])
    for kind in required_scenarios:
        if kind not in by_kind:
            failures.append({"gate": "scenario_coverage", "message": f"missing {kind}"})
    if failures:
        return {
            "curriculum_training_ready": False,
            "agentic_training_ready": False,
            "training_category": category,
            "agentic_eligible": tool_required,
            "validation_mode": (
                "offline_mock" if evaluator_mode == "mock" else "live_evaluator"
            ),
            "evaluator_provider": evaluator_provider,
            "failed_gates": ["scenario_coverage"],
            "evidence": evidence,
            "failures": failures,
        }

    success_scenario = scenario_without_assertions(by_kind["goal_success"])
    tool_steps = [step for step in success_scenario.get("steps", []) if isinstance(step, Mapping) and step.get("operation") == "tool_call"]
    if tool_required and not tool_steps:
        failures.append({"gate": "agentic_path", "message": "goal_success has no tool calls"})
    if not tool_required and tool_steps:
        failures.append({"gate": "direct_response_path", "message": "direct_response goal_success must not call tools"})
    for index, step in enumerate(tool_steps):
        arguments = step.get("arguments")
        schema = next((item.get("function", {}).get("parameters", {}) for item in task.get("tools", [])
                       if item.get("function", {}).get("name") == step.get("tool_name")), {})
        if not isinstance(arguments, Mapping) or not set(schema.get("required", [])) <= set(arguments):
            failures.append({"gate": "semantic_fixture", "message": f"tool step {index} has no arguments"})
        elif has_placeholder(arguments):
            failures.append({"gate": "semantic_fixture", "message": f"tool step {index} uses placeholder arguments"})
        elif empty_critical_collection(arguments):
            failures.append({"gate": "semantic_fixture", "message": f"tool step {index} uses an empty collection"})

    if bool(base_url) != bool(container_image_id):
        raise ValueError("base_url and container_image_id must be provided together")
    if base_url and evaluator_mode != "live":
        raise ValueError("container execution requires evaluator_mode=live")
    if base_url and re.fullmatch(r"sha256:[0-9a-f]{64}", str(container_image_id)) is None:
        raise ValueError("container_image_id must be a Docker sha256 image ID")
    os.environ.setdefault("SANDBOX_TRAINER_API_KEY", "envfactory-agentic-value-key")
    os.environ["SANDBOX_EVALUATOR_MOCK"] = "1" if evaluator_mode == "mock" else "0"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    app = (
        HTTPSandboxClient(
            base_url,
            timeout=evaluator_http_timeout(runtime.timeout_seconds, runtime.max_retries),
        ) if base_url else import_app(root)
    )
    from sandbox_runtime import AcceptanceScenarioRunner

    trainer_headers = {
        "Authorization": f"Bearer {os.environ['SANDBOX_TRAINER_API_KEY']}"
    }

    def remote_business_snapshot() -> Any:
        status, body, _ = app.handle("GET", "/v1/state", None, trainer_headers)
        if status != 200 or not isinstance(body, Mapping):
            raise RuntimeError("trainer state endpoint failed")
        snapshot = body.get("business_state")
        if not isinstance(snapshot, Mapping):
            raise RuntimeError("trainer state endpoint returned an invalid snapshot")
        return dict(snapshot)

    runner = AcceptanceScenarioRunner(
        app.handle,
        trainer_headers=trainer_headers,
        business_snapshot=(
            remote_business_snapshot
            if base_url
            else getattr(app, "business_snapshot", None)
        ),
        # A production container deliberately has no generic state-mutation
        # endpoint. Counterfactuals must use the public business tools.
        mutate_business_state=(
            None if base_url else getattr(app, "mutate_business_state", None)
        ),
    )

    def execute(name: str, scenario: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
        try:
            result = runner.run(scenario)
            evidence["counterfactuals"][name] = {"reward": reward_of(result), "status": "completed"}
            return result, None
        except Exception as exc:  # rejected counterfactuals are valid evidence
            # Only an actual 4xx response is evidence of environment rejection.
            # A runner error, missing capture or server failure is not a proof.
            details = getattr(exc, "details", None)
            status = details.get("actual_status") if isinstance(details, Mapping) else None
            evidence["counterfactuals"][name] = {
                "reward": None,
                "status": "rejected",
                "http_status": status if isinstance(status, int) else None,
                "error_type": type(exc).__name__,
            }
            if not isinstance(status, int) or not 400 <= status < 500:
                failures.append({
                    "gate": "counterfactual_execution",
                    "message": f"{name}: execution did not produce a client rejection",
                    "error_type": type(exc).__name__,
                })
            return None, type(exc).__name__

    environment_mode = task.get("environment_plan", {}).get("mode")
    reference_baseline = None
    if environment_mode == "reference_data":
        try:
            reset_step = next(
                step for step in success_scenario.get("steps", [])
                if isinstance(step, Mapping) and step.get("operation") == "reset"
            )
            runner.run({"steps": [reset_step], "assertions": []})
            reference_baseline = remote_business_snapshot() if base_url else app.business_snapshot()
        except Exception:
            reference_baseline = None
    success_run, success_error = execute("goal_success", success_scenario)
    if environment_mode == "reference_data":
        try:
            reference_after = remote_business_snapshot() if base_url else app.business_snapshot()
        except Exception:
            reference_after = None
        unchanged = (
            reference_baseline is not None
            and reference_after is not None
            and reference_baseline == reference_after
        )
        evidence["reference_data_immutability"] = {"unchanged": unchanged}
        if not unchanged:
            failures.append({
                "gate": "reference_data_immutability",
                "message": "reference_data success path changed private business state",
            })
    if tool_required and environment_mode in {"reference_data", "stateful"}:
        manifest = task.get("artifacts", {}).get("data_manifest", {})
        table_names = {
            str(item.get("table_name")) for item in manifest.get("tables", [])
            if isinstance(item, Mapping) and isinstance(item.get("table_name"), str)
        }
        replay_status, replay_body, _ = app.handle("GET", "/v1/replay", None, trainer_headers)
        private_reads = (
            business_data_reads_from_replay(replay_body, table_names)
            if replay_status == 200 and success_run else set()
        )
        tool_reads = (
            business_tool_reads_from_replay(replay_body, table_names)
            if replay_status == 200 and success_run else []
        )
        evidence["business_data_access"] = {
            "tables_read": sorted(private_reads), "tool_reads": tool_reads,
        }
        if not private_reads:
            failures.append({
                "gate": "business_data_dependency",
                "message": "success path business tools did not read sandbox business data",
            })
        if (
            len(tool_reads) != len(tool_steps)
            or any(
                item.get("tool_name") != step.get("tool_name")
                for item, step in zip(tool_reads, tool_steps)
            )
        ):
            failures.append({
                "gate": "business_tool_trace_coverage",
                "message": "every success-path business tool must have an ordered runtime event",
            })
        if environment_mode == "reference_data" and private_reads:
            causal_app = app if not base_url else import_app(root)
            causality = probe_reference_data_causality(
                causal_app, trainer_headers, success_scenario, manifest,
            )
            probe_mutation = causality.pop("_probe_mutation", None)
            fallback_mutations = causality.pop("_fallback_mutations", [])
            evidence["business_data_causality"] = causality
            if not causality.get("proved"):
                failures.append({
                    "gate": "business_data_causality",
                    "message": "changing private business data did not change a tool result",
                })
            terminal_weight = terminal_outcome_weight(task)
            answer_sensitivity_needed = terminal_weight > 0
            if (answer_sensitivity_needed and evaluator_mode == "mock"
                    and semantic_terminal_requires_live(task)):
                evidence["business_data_reward_sensitivity"] = {
                    "proved": False,
                    "verification": "requires_live_evaluator",
                    "reason": "offline semantic fixture compares exact answer text only",
                }
            elif answer_sensitivity_needed:
                candidates = ([probe_mutation] if isinstance(probe_mutation, Mapping) else []) + fallback_mutations
                sensitivity = probe_reference_data_reward_candidates(
                    causal_app, trainer_headers, success_scenario, candidates, terminal_weight,
                )
                evidence["business_data_reward_sensitivity"] = sensitivity
                if not sensitivity.get("proved"):
                    failures.append({
                        "gate": "business_data_reward_sensitivity",
                        "message": ("stale answer retained excessive reward in executable private-data probes"
                                    if sensitivity.get("variants_tried", 0) else
                                    "private-data reward sensitivity is inconclusive: no executable probe"),
                    })
    failure_run, _ = execute("goal_failure", scenario_without_assertions(by_kind["goal_failure"]))
    success_reward = reward_of(success_run or {})
    failure_reward = reward_of(failure_run or {})
    if success_error or success_reward is None or success_reward < 1.0 - 1e-9:
        failures.append({"gate": "success_trajectory", "message": "valid goal trajectory must satisfy all reward components (reward 1.0)", "reward": success_reward})
    if failure_reward is None or failure_reward > 0.2:
        failures.append({"gate": "failure_trajectory", "message": "failure trajectory must produce reward <= 0.2", "reward": failure_reward})

    if success_run and tool_required:
        tool_bodies = [item.get("body") for item in success_run.get("history", []) if item.get("operation") == "tool_call"]
        if not tool_bodies or not all(meaningful_result(body) for body in tool_bodies):
            failures.append({"gate": "meaningful_tool_results", "message": "every success-path tool call must return meaningful business evidence"})

    final_step = next((step for step in success_scenario.get("steps", []) if step.get("operation") == "agent_response"), None)
    terminal_weight = terminal_outcome_weight(task)
    if terminal_weight > 0:
        if final_step is None:
            failures.append({"gate": "final_answer_coverage", "message": "terminal outcome has no final answer step"})
        else:
            wrong_answer = copy.deepcopy(success_scenario)
            wrong_step = next(
                step for step in wrong_answer["steps"]
                if step.get("operation") == "agent_response"
            )
            wrong_step["content"] = "此回答与用户请求无关；我没有完成所要求的任务，也不提供结果。"
            wrong_run, wrong_error = execute("wrong_final_answer", wrong_answer)
            wrong_reward = reward_of(wrong_run or {})
            required_drop = 0.5 * terminal_weight
            if (wrong_error is not None or success_reward is None or wrong_reward is None
                    or wrong_reward >= 1.0 - 1e-9
                    or success_reward - wrong_reward < required_drop - 1e-9):
                failures.append({
                    "gate": "final_answer_sensitivity",
                    "message": "wrong final answer must lose reward relative to the same tool trajectory",
                    "success_reward": success_reward, "wrong_reward": wrong_reward,
                    "required_drop": required_drop,
                })
            numeric_answer = numeric_answer_counterfactual(task, final_step.get("content"))
            if numeric_answer is not None:
                numerical = copy.deepcopy(success_scenario)
                next(step for step in numerical["steps"] if step.get("operation") == "agent_response")["content"] = numeric_answer
                numeric_run, numeric_error = execute("wrong_numeric_answer", numerical)
                numeric_reward = reward_of(numeric_run or {})
                if (numeric_error is not None or success_reward is None or numeric_reward is None
                        or numeric_reward >= 1.0 - 1e-9
                        or success_reward - numeric_reward < required_drop - 1e-9):
                    failures.append({
                        "gate": "numeric_answer_sensitivity",
                        "message": "incorrect numerical answer must score below success on the same tool trajectory",
                        "success_reward": success_reward, "wrong_reward": numeric_reward,
                        "required_drop": required_drop,
                    })
    if final_step and tool_required:
        no_tools = {
            "scenario_id": "agentic_no_tools",
            "steps": [
                {"operation": "reset", "body": {"episode_id": "agentic-no-tools", "seed": 1701}},
                copy.deepcopy(final_step),
                {"step_id": "reward", "operation": "reward"},
            ],
            "assertions": [],
        }
        no_tool_run, _ = execute("no_tools", no_tools)
        no_tool_reward = reward_of(no_tool_run or {})
        if no_tool_reward is None or no_tool_reward > 0.2:
            failures.append({"gate": "no_tool_reward_hacking", "message": "answering without tools must produce reward <= 0.2", "reward": no_tool_reward})

    # Use the values observed on the successful trace for all references.
    # Otherwise changing an upstream argument can make a later $ref unresolved
    # inside the test runner, without exercising the environment or reward.
    bound_success = copy.deepcopy(success_scenario)
    if success_run:
        for step in bound_success["steps"]:
            if step.get("operation") == "tool_call":
                step["arguments"] = runner._resolve(
                    step.get("arguments", {}), success_run.get("variables", {})
                )
    for tool_index, original in enumerate(tool_steps if tool_required else []):
        for argument_index, key in enumerate(argument_probe_keys(original, task.get("tools", []))):
            corrupted = copy.deepcopy(bound_success)
            candidate = [step for step in corrupted["steps"]
                         if step.get("operation") == "tool_call"][tool_index]
            arguments = dict(candidate.get("arguments", {}))
            arguments[key] = corrupt(arguments[key])
            candidate["arguments"] = arguments
            name = f"corrupted_arguments_{tool_index + 1}"
            if argument_index:
                name += f"__{argument_index + 1}"
            corrupted_run, corrupted_error = execute(name, corrupted)
            corrupted_reward = reward_of(corrupted_run or {})
            if corrupted_error is None and (corrupted_reward is None or corrupted_reward > 0.2):
                failures.append({
                    "gate": "argument_sensitivity",
                    "message": f"corrupting {original.get('tool_name')}.{key} must be rejected or reward <= 0.2",
                    "reward": corrupted_reward,
                })

    if dependency_required and len(tool_steps) >= 2:
        # Bind known baseline values before removing/reordering producers so
        # the counterfactual reaches the environment rather than failing in $ref.
        for tool_index in range(len(tool_steps)):
            skipped = copy.deepcopy(bound_success)
            positions = [
                index for index, step in enumerate(skipped["steps"])
                if step.get("operation") == "tool_call"
            ]
            del skipped["steps"][positions[tool_index]]
            skipped_run, skipped_error = execute(f"skipped_tool_{tool_index + 1}", skipped)
            skipped_reward = reward_of(skipped_run or {})
            if skipped_error is None and (skipped_reward is None or skipped_reward > 0.2):
                failures.append({
                    "gate": "step_necessity",
                    "message": f"skipping tool step {tool_index} must be rejected or reward <= 0.2",
                    "reward": skipped_reward,
                })
        reordered = copy.deepcopy(bound_success)
        dag = task.get("task_spec", {}).get("capability_dag", {})
        pair = dependency_swap_positions(reordered["steps"], dag if isinstance(dag, Mapping) else {})
        if pair is None:
            failures.append({
                "gate": "dependency_contract",
                "message": "dependency-required task has no ordered tool pair in capability_dag",
            })
        else:
            first_pos, second_pos = pair
            reordered["steps"][first_pos], reordered["steps"][second_pos] = (
                reordered["steps"][second_pos], reordered["steps"][first_pos]
            )
            reordered_run, reordered_error = execute("reordered_tools", reordered)
            reordered_reward = reward_of(reordered_run or {})
            if reordered_error is None and (reordered_reward is None or reordered_reward > 0.2):
                failures.append({
                    "gate": "order_sensitivity",
                    "message": "reordering dependent tool calls must be rejected or reward <= 0.2",
                    "reward": reordered_reward,
                })

    if task.get("noise_tools") and "noise_selection" in by_kind:
        noise_run, _ = execute("noise_selection", scenario_without_assertions(by_kind["noise_selection"]))
        noise_reward = reward_of(noise_run or {})
        if noise_reward is None or noise_reward > 0:
            failures.append({"gate": "noise_selection", "message": "noise tool trajectory must produce reward <= 0", "reward": noise_reward})

    failed_gates = sorted({item["gate"] for item in failures})
    runtime_execution = (
        {
            "version": "1.0",
            "mode": "docker_http",
            "container_image_id": container_image_id,
            "transport": "loopback_http",
            "read_only_root": True,
            "cap_drop": "ALL",
            "no_new_privileges": True,
            "non_root_user": True,
        }
        if base_url
        else {"version": "1.0", "mode": "in_process", "container_image_id": None}
    )
    return {
        "curriculum_training_ready": not failures,
        "agentic_training_ready": not failures,
        "training_category": category,
        "sandbox_profile": contract.get("sandbox_profile", category),
        "agentic_eligible": tool_required,
        "validation_mode": "offline_mock" if evaluator_mode == "mock" else "live_evaluator",
        "evaluator_provider": evaluator_provider,
        "runtime_execution": runtime_execution,
        "hard_gates_passed": not failures,
        "failed_gates": failed_gates,
        "evidence": evidence,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="验证沙箱是否能训练真实 Agentic 行为")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--evaluator-mode", choices=("mock", "live"), default="mock",
        help="live exercises the configured production reward evaluator",
    )
    parser.add_argument("--base-url")
    parser.add_argument("--container-image-id")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        report = validate(
            root,
            evaluator_mode=args.evaluator_mode,
            base_url=args.base_url,
            container_image_id=args.container_image_id,
        )
    except ValueError as exc:
        parser.error(str(exc))
    output = args.output or root / "agentic_training_value.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["curriculum_training_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
