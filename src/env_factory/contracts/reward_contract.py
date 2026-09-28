"""Deterministic checks for reward contracts that invite shortcut policies."""

from __future__ import annotations

from typing import Any


def declared_terminal_outcome_ids(metrics: list[Any], specs: list[Any]) -> set[Any]:
    """Outcome metrics whose contract explicitly reads the final response."""
    outcome_ids = {
        item.get("id") for item in metrics
        if isinstance(item, dict) and item.get("category") == "outcome"
    }
    model_ids = {
        item.get("id") for item in metrics
        if isinstance(item, dict)
        and item.get("type") in {"model-based", "hybrid"}
        and isinstance(item.get("evaluation_inputs"), list)
        and "final_agent_response" in item["evaluation_inputs"]
    }
    rule_ids = {
        spec.get("metric_id") for spec in specs
        if isinstance(spec, dict) and spec.get("source") == "final_agent_response"
    }
    return outcome_ids & (model_ids | rule_ids)


def has_literal_payload_argument(spec: Any) -> bool:
    """Whether a process rule matches one long multiline argument verbatim."""
    if not isinstance(spec, dict) or spec.get("operator") != "contains_tool_call":
        return False
    expected = spec.get("expected")
    if not isinstance(expected, dict):
        return False
    arguments = expected.get("arguments", {})
    if not isinstance(arguments, dict):
        return False

    def contains(value: Any) -> bool:
        if isinstance(value, str):
            return len(value) >= 80 and "\n" in value
        if isinstance(value, dict):
            return any(contains(child) for child in value.values())
        if isinstance(value, list):
            return any(contains(child) for child in value)
        return False

    return contains(arguments)


def terminal_outcome_weight(task: dict[str, Any]) -> float:
    """Weight explicitly assigned to judging the final user-facing answer."""
    if task.get("environment_plan", {}).get("mode") == "stateful":
        return 0.0
    metrics = task.get("metrics")
    specs = task.get("metric_implementations")
    terminal_ids = declared_terminal_outcome_ids(
        metrics if isinstance(metrics, list) else [],
        specs if isinstance(specs, list) else [],
    )
    weight = 0.0
    for metric in metrics if isinstance(metrics, list) else []:
        if not isinstance(metric, dict) or metric.get("category") != "outcome":
            continue
        if metric.get("id") not in terminal_ids:
            continue
        value = metric.get("weight")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            weight += float(value)
    return weight


def reward_contract_issues(task: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = task.get("metrics")
    metrics = metrics if isinstance(metrics, list) else []
    implementations = task.get("metric_implementations")
    implementations = implementations if isinstance(implementations, list) else []
    mode = task.get("environment_plan", {}).get("mode")
    issues: list[dict[str, Any]] = []
    if mode != "stateful":
        outcome_ids = {
            item.get("id") for item in metrics
            if isinstance(item, dict) and item.get("category") == "outcome"
        }
        if not declared_terminal_outcome_ids(metrics, implementations):
            issues.append({
                "code": "TASK_TERMINAL_REWARD_UNDECLARED",
                "owner": "task_generation",
                "message": "outcome reward does not declare a final-agent-response dependency",
                "evidence": {"environment_mode": mode, "outcome_metric_ids": sorted(outcome_ids)},
            })

    for spec in implementations:
        if not has_literal_payload_argument(spec):
            continue
        expected = spec["expected"]
        issues.append({
            "code": "TASK_LITERAL_PROCESS_REWARD",
            "owner": "task_generation",
            "message": "process credit depends on one literal multiline tool argument",
            "evidence": {"metric_id": spec.get("metric_id"), "tool_name": expected.get("tool_name")},
        })
    return issues
