"""Deterministic checks for reward contracts that invite shortcut policies."""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Any


REFERENCE_FACTUALITY_CRITERION = "逐项核对最终回答中的业务事实与实际工具结果及当前业务记录一致。"
STATEFUL_GOAL_CRITERION = "逐项核对当前业务状态满足声明的最终状态断言，并确认成功轨迹完成了所需变更；不得仅凭最终回答或工具调用给分。"


def ambiguous_metric_captures(task: dict[str, Any], preview: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find process rewards that choose a positional row from distinct tool results."""
    issues: list[dict[str, Any]] = []
    for metric in task.get("metric_implementations", []):
        if not isinstance(metric, dict):
            continue
        expected = metric.get("expected")
        if not isinstance(expected, dict):
            continue
        for capture in expected.get("captures", []):
            if not isinstance(capture, dict):
                continue
            path, source = capture.get("path"), capture.get("tool_name")
            if not isinstance(path, str) or not isinstance(source, str):
                continue
            match = re.fullmatch(r"\$\.([A-Za-z_][A-Za-z_0-9]*)\[(\d+)\]\.([A-Za-z_][A-Za-z_0-9]*)", path)
            if not match:
                continue
            result_field, index, value_field = match.group(1), int(match.group(2)), match.group(3)
            for call in preview:
                if call.get("tool_name") != source:
                    continue
                result = call.get("result")
                rows = result.get(result_field) if isinstance(result, dict) else None
                if not isinstance(rows, list) or len(rows) <= 1 or index >= len(rows):
                    continue
                values = [row.get(value_field) for row in rows if isinstance(row, dict)]
                if len(values) <= 1 or all(value == values[0] for value in values[1:]):
                    continue
                issues.append({
                    "code": "AMBIGUOUS_METRIC_CAPTURE",
                    "metric_id": metric.get("metric_id"), "tool_name": source,
                    "path": path, "row_count": len(rows),
                })
                break
    return issues


def contradicts_reference_factuality(value: Any) -> bool:
    """Detect explicit instructions to ignore private evidence in a judge rubric."""
    if not isinstance(value, str):
        return False
    return re.search(
        r"(?:无需|不必|不用|不需要|忽略|禁止|不得).{0,12}"
        r"(?:核对|检查|验证|参考).{0,30}"
        r"(?:工具结果|业务记录|业务数据)",
        value,
    ) is not None


def numeric_answer_counterfactual(task: dict[str, Any], answer: Any) -> str | None:
    """Perturb one final numerical conclusion while preserving the evidence."""
    if task.get("task_intent") not in {"calculate", "estimate"} or not isinstance(answer, str):
        return None
    excluded = [
        item.span() for pattern in (
            r"(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)",
            r"(?<!\d)\d{1,2}:\d{2}(?::\d{2})?(?!\d)",
        ) for item in re.finditer(pattern, answer)
    ]
    matches = [
        item for item in re.finditer(
            r"(?<![A-Za-z0-9_.])\d+(?:\.\d+)?(?![A-Za-z0-9_.])", answer,
        )
        if not any(start <= item.start() < end for start, end in excluded)
    ]
    if not matches:
        return None
    # Numeric outcome contracts identify the actual conclusion field. Perturb
    # that field before considering unrelated trailing numbers (for example a
    # recap sentence or an excluded-record count).
    labeled_matches: list[Any] = []
    section_start = answer.rfind("结论")
    for spec in task.get("metric_implementations", []):
        if not isinstance(spec, dict) or spec.get("operator") != "numeric_targets":
            continue
        expected = spec.get("expected")
        targets = expected.get("targets", []) if isinstance(expected, dict) else []
        for target_spec in targets:
            label = target_spec.get("label") if isinstance(target_spec, dict) else None
            if not isinstance(label, str) or not label:
                continue
            offset = section_start if section_start >= 0 else 0
            region = answer[offset:]
            occurrences = list(re.finditer(re.escape(label), region))
            if not occurrences and section_start >= 0:
                region = answer
                offset = 0
                occurrences = list(re.finditer(re.escape(label), region))[-1:]
            for occurrence in occurrences:
                start = offset + occurrence.end()
                nearby = next((item for item in matches
                               if start <= item.start() <= start + 15), None)
                if nearby is not None:
                    labeled_matches.append(nearby)
    if labeled_matches:
        target = labeled_matches[-1]
        changed = str(Decimal(target.group()) + 1)
        return answer[:target.start()] + changed + answer[target.end():]
    conclusions = list(re.finditer(r"结论|总计|合计|总额|总金额|结果|应付", answer))
    final_candidates = [
        item for item in matches if conclusions and item.start() > conclusions[-1].end()
    ]
    target = (final_candidates or matches)[-1]
    changed = str(Decimal(target.group()) + 1)
    return answer[:target.start()] + changed + answer[target.end():]


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
