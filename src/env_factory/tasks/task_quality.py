"""Deterministic quality scoring for generated Agentic-RL task artifacts."""

from __future__ import annotations

import copy
import json
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from env_factory.contracts.runtime_contract import missing_system_endpoints
from env_factory.contracts.reward_contract import (
    REFERENCE_FACTUALITY_CRITERION,
    STATEFUL_GOAL_CRITERION,
    ambiguous_metric_captures,
    contradicts_reference_factuality,
    reward_contract_issues,
)
from env_factory.contracts.tool_chain_contract import tool_chain_issues


@dataclass(frozen=True)
class QualityDimension:
    score: float
    maximum: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class TaskQualityReport:
    task_id: str
    path: str
    score: float
    passed: bool
    tier: str
    dimensions: dict[str, QualityDimension]
    findings: tuple[str, ...]
    training_category: str = "multi_step_agentic"
    agentic_level: int = 2
    tool_policy_target: str = "follow_dependency_chain"
    eligible: bool = True
    eligibility_failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Preserve the diagnostic quality score when an eligibility gate
        # deliberately forces the effective score to zero.
        value["raw_score"] = round(sum(item.score for item in self.dimensions.values()), 2)
        value["score_kind"] = "gated_weighted_10_point"
        value["score_scope"] = "prebuild_task_qualification"
        value["delivery_verified"] = False
        value["score_interpretation"] = "Prebuild completeness and executability; not final delivery quality or agent success probability"
        value["dimensions"] = {
            name: asdict(dimension) for name, dimension in self.dimensions.items()
        }
        return value


def _reject(report: TaskQualityReport, finding: str) -> TaskQualityReport:
    """Make file-level eligibility failures visible in the effective score."""
    return TaskQualityReport(
        report.task_id, report.path, 0.0, False, "rejected",
        report.dimensions, report.findings + (finding,), report.training_category,
        report.agentic_level, report.tool_policy_target, False,
        report.eligibility_failures + (finding,),
    )


def _manifest_root(task_root: Path, manifest: dict[str, Any]) -> Path:
    """Use the declared task-relative data root, matching sandbox runtime."""
    return task_root / Path(str(manifest.get("root", ".")))


def _objects(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def public_row_sets(materials: list[dict[str, Any]]) -> list[set[str]]:
    """Find complete JSON row collections already delivered to the Agent."""
    collections: list[set[str]] = []

    def visit(value: Any) -> None:
        if isinstance(value, list):
            if value and all(isinstance(item, dict) for item in value):
                collections.append({
                    json.dumps(item, ensure_ascii=False, sort_keys=True)
                    for item in value
                })
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)

    for material in materials:
        content = material.get("content")
        if not isinstance(content, str):
            continue
        try:
            visit(json.loads(content))
        except json.JSONDecodeError:
            continue
    return collections


def _text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).lower()


def _numeric_private_dependencies(value: Any) -> bool:
    if isinstance(value, dict):
        if "lookup" in value or "aggregate" in value:
            return True
        return any(_numeric_private_dependencies(item) for item in value.values())
    if isinstance(value, list):
        return any(_numeric_private_dependencies(item) for item in value)
    return False


def fixed_row_total_targets(description: dict[str, Any], expected: Any) -> bool:
    """Detect totals over an open record set that enumerate only current row IDs."""
    public_input = description.get("public_input")
    scope = " ".join(str(value) for value in (
        description.get("task"), description.get("expected_result"),
        public_input.get("initial_user_message") if isinstance(public_input, dict) else "",
    ))
    if not re.search(r"各条|每条|所有|全部|每笔|各笔|逐条|整批", scope):
        return False
    targets = expected.get("targets") if isinstance(expected, dict) else None
    for target in _objects(targets):
        if not re.search(r"总|合计|汇总", str(target.get("label", ""))):
            continue
        lookups: dict[str, set[str]] = {}
        has_aggregate = False
        def inspect(value: Any) -> None:
            nonlocal has_aggregate
            if isinstance(value, dict):
                if isinstance(value.get("aggregate"), dict):
                    has_aggregate = True
                lookup = value.get("lookup")
                if isinstance(lookup, dict) and isinstance(lookup.get("table"), str):
                    lookups.setdefault(lookup["table"], set()).add(
                        json.dumps(lookup.get("where"), ensure_ascii=False, sort_keys=True)
                    )
                for child in value.values():
                    inspect(child)
            elif isinstance(value, list):
                for child in value:
                    inspect(child)
        inspect(target.get("expression"))
        if not has_aggregate or any(len(rows) > 1 for rows in lookups.values()):
            return True
    return False


def prior_result_only_tool_actions(capabilities: Any) -> list[str]:
    """Find dependent tools whose stated work is only processing known inputs."""
    issues: list[str] = []
    for item in _objects(capabilities):
        if item.get("requires_tool") is not True or not item.get("dependencies"):
            continue
        reason = str(item.get("reason", ""))
        if not re.search(r"(?:前序|上一步|已有|给定).{0,100}(?:计算|模拟|预测|求和|合计|汇总|统计|格式化|生成.{0,8}表格)", reason):
            continue
        if re.search(
            r"(?:再次|另行|进一步|还需|随后|根据.{0,30}(?:标识|编号|结果)).{0,45}"
            r"(?:读取|查询|检索|调用).{0,45}(?:私有|内部|数据表|记录|服务|系统)",
            reason,
        ):
            continue
        name = item.get("action_name")
        if isinstance(name, str) and name:
            issues.append(name)
    return issues


def capability_disclaims_tool(reason: Any) -> bool:
    """Reject a required tool whose own rationale says the Agent can do it."""
    if not isinstance(reason, str):
        return False
    return bool(
        re.search(r"(?:不需要|无需|毋须).{0,8}(?:调用|使用)?(?:额外)?工具", reason)
        or "属于分析、计算和生成" in reason
        or "属于 Agent 推理" in reason
        or "属于agent推理" in reason
    )


def has_private_tool_dependency(scenarios: Any) -> bool:
    """Require a downstream tool to use data beyond a prior selector echo."""
    def refs(value: Any) -> set[str]:
        if isinstance(value, dict):
            if set(value) == {"$ref"} and isinstance(value["$ref"], str):
                return {value["$ref"]}
            return set().union(*(refs(item) for item in value.values())) if value else set()
        if isinstance(value, list):
            return set().union(*(refs(item) for item in value)) if value else set()
        return set()

    for scenario in _objects(scenarios):
        if scenario.get("kind") != "goal_success":
            continue
        captures: dict[str, tuple[str, dict[str, Any]]] = {}
        for step in _objects(scenario.get("steps")):
            if step.get("operation") != "tool_call":
                continue
            for variable in refs(step.get("arguments", {})):
                source = captures.get(variable)
                if source is None:
                    continue
                path, source_arguments = source
                terminal = re.search(r"\.([A-Za-z_][A-Za-z0-9_]*)$", path)
                field = terminal.group(1) if terminal else None
                # Returning the same selector passed into a query reveals no
                # new private fact for the next tool decision.
                if field is not None and isinstance(source_arguments, dict) and field in source_arguments:
                    if not refs(source_arguments[field]):
                        continue
                return True
            step_captures = step.get("capture")
            if not isinstance(step_captures, dict):
                continue
            for name, path in step_captures.items():
                if isinstance(name, str) and isinstance(path, str):
                    captures[name] = (path, step.get("arguments", {}))
    return False


def _dimension(maximum: float, deductions: Iterable[tuple[float, str]]) -> QualityDimension:
    items = list(deductions)
    return QualityDimension(
        score=round(max(0.0, maximum - sum(amount for amount, _ in items)), 2),
        maximum=maximum,
        reasons=tuple(reason for _, reason in items),
    )


def _stateful_goal_tool_issues(task: dict[str, Any]) -> list[str]:
    """Return deterministic goal/mutation closure failures for stateful tasks."""
    if task.get("environment_plan", {}).get("mode") != "stateful":
        return []
    task_spec = task.get("task_spec")
    goal = task_spec.get("goal_contract", {}) if isinstance(task_spec, dict) else {}
    predicates = _objects(goal.get("row_predicates"))
    by_table: dict[str, list[dict[str, Any]]] = {}
    for predicate in predicates:
        table = predicate.get("table")
        if isinstance(table, str) and table:
            by_table.setdefault(table, []).append(predicate)

    issues: list[str] = []
    if not by_table:
        issues.append("状态任务缺少最终行状态断言")
    mutation_tables: set[str] = set()
    called_mutation_tables: set[str] = set()
    success_calls = {
        step.get("tool_name")
        for scenario in _objects(task.get("acceptance_contract", {}).get("executable_scenarios"))
        if scenario.get("kind") == "goal_success"
        for step in _objects(scenario.get("steps"))
        if step.get("operation") == "tool_call"
    }
    for implementation in _objects(task.get("tool_implementations")):
        operation = implementation.get("operation")
        if operation not in {"insert", "update", "delete"}:
            continue
        table = implementation.get("table")
        tool_name = implementation.get("tool_name")
        if isinstance(table, str):
            mutation_tables.add(table)
            if tool_name in success_calls:
                called_mutation_tables.add(table)
        if not isinstance(table, str) or not by_table.get(table):
            issues.append(f"变更工具 {tool_name} 的表 {table} 没有最终状态断言")
    if not mutation_tables:
        issues.append("状态任务缺少声明式变更工具实现")
    for table in sorted(set(by_table) - mutation_tables):
        issues.append(f"目标表 {table} 没有声明式变更工具实现")
    for table in sorted(set(by_table) - called_mutation_tables):
        issues.append(f"成功轨迹未调用目标表 {table} 的变更工具")
    return issues


def _defers_business_truth(task: dict[str, Any]) -> bool:
    if task.get("task_intent") not in {"modify", "execute", "schedule"}:
        return False
    corpus = _text({
        key: task.get(key)
        for key in ("task", "requirements", "public_input", "actions", "environment")
    })
    return any(marker in corpus for marker in (
        "暂时没想好", "先问我", "稍后提供", "之后提供", "待我提供",
        "需要向用户询问", "询问正确", "ask me", "provide later",
        "to be provided",
    ))


def unsourced_real_world_exemplars(
    task: dict[str, Any], *, training_category: str,
) -> bool:
    """Flag public factual examples whose only source is generated prose.

    This intentionally covers explicit real-world exemplar requests, not
    ordinary transformations of supplied text or synthetic business records.
    """
    if training_category != "direct_response":
        return False
    public_input = task.get("public_input") if isinstance(task.get("public_input"), dict) else {}
    request = _text({key: task.get(key) for key in (
        "task", "goal", "expected_result", "requirements",
    )}) + _text(public_input.get("initial_user_message"))
    if not re.search(
        r"(?:代表|经典|著名|真实|史实).{0,8}(?:剧目|作品|唱段|人物|历史事件|案例|实例)"
        r"|(?:剧目|作品|唱段|人物|历史事件|案例|实例).{0,8}(?:代表|经典|著名|真实)",
        request,
    ):
        return False
    if any(marker in request for marker in ("虚构", "模拟案例", "fictional", "hypothetical")):
        return False
    materials = _objects(public_input.get("materials"))
    return not any(
        isinstance(item.get("content"), str)
        and item["content"].strip()
        and (
            re.search(r"https?://[^\s\"'<>]+", item["content"], re.I)
            or isinstance(item.get("source_url"), str)
            and re.match(r"https?://", item["source_url"], re.I)
        )
        for item in materials
    )


def unobservable_tool_preconditions(task: dict[str, Any]) -> list[str]:
    """Find tool preconditions that the declared read cannot enforce or expose."""
    spec = task.get("task_spec") if isinstance(task.get("task_spec"), dict) else {}
    implementations = {
        item.get("tool_name"): item for item in _objects(task.get("tool_implementations"))
        if isinstance(item.get("tool_name"), str)
    }
    issues = []
    for contract in _objects(spec.get("tool_contracts")):
        if contract.get("role") != "business":
            continue
        implementation = implementations.get(contract.get("name"))
        if not isinstance(implementation, dict) or implementation.get("operation") != "select":
            continue
        preconditions = _text(contract.get("preconditions"))
        if not re.search(r"\bstatus\s*(?:为|=|is)\s*['\"]?active\b", preconditions, re.I):
            continue
        projected = set(implementation.get("projection") or [])
        filtered = {item.get("column") for item in _objects(implementation.get("filters"))}
        if "status" not in projected | filtered:
            issues.append(str(contract.get("name")))
    return issues


def score_task(task: dict[str, Any], *, task_id: str = "task", path: str = "", min_score: float = 8.0) -> TaskQualityReport:
    """Score one task artifact on a stable 0-10 training-quality rubric."""
    task_text = str(task.get("task", "")).strip()
    requirements = task.get("requirements") if isinstance(task.get("requirements"), dict) else {}
    semantic_requirements = {
        key: value for key, value in requirements.items()
        if key not in {"runtime_interface", "media_truth_mode"}
    }
    mode = task.get("environment_plan", {}).get("mode")
    public_input = task.get("public_input") if isinstance(task.get("public_input"), dict) else {}
    public_materials = _objects(public_input.get("materials"))
    actions = _objects(task.get("actions"))
    tools = _objects(task.get("tools"))
    noise = _objects(task.get("noise_tools"))
    metrics = _objects(task.get("metrics"))
    implementations = _objects(task.get("metric_implementations"))
    contract = task.get("acceptance_contract") if isinstance(task.get("acceptance_contract"), dict) else {}
    readiness = task.get("task_readiness") if isinstance(task.get("task_readiness"), dict) else {}
    declared_training_category = task.get("training_category")
    category_profiles = {
        "direct_response": (0, "do_not_call"),
        "simple_agentic": (1, "call_required_business_tool"),
        "multi_step_agentic": (2, "follow_dependency_chain"),
    }
    noise_names = {str(item.get("name")) for item in noise}
    tool_names = {
        str(tool.get("function", {}).get("name"))
        for tool in tools if isinstance(tool.get("function"), dict)
    }
    business_names = tool_names - noise_names
    if declared_training_category in category_profiles:
        training_category = str(declared_training_category)
    elif not business_names:
        training_category = "direct_response"
    elif len(business_names) == 1:
        training_category = "simple_agentic"
    else:
        training_category = "multi_step_agentic"
    agentic_level, tool_policy_target = category_profiles[training_category]
    corpus = _text({"task": task_text, "requirements": semantic_requirements})
    contract_deductions: list[tuple[float, str]] = []
    if len(task_text) < 16:
        contract_deductions.append((0.6, "任务描述过短，目标或边界可能不完整"))
    if not requirements:
        contract_deductions.append((0.5, "缺少 requirements"))
    if not actions:
        contract_deductions.append((0.7, "缺少可执行动作分解"))
    if task.get("complexity") not in {"simple", "standard", "complex"}:
        contract_deductions.append((0.4, "complexity 无效"))
    if any(marker in corpus for marker in ("待补充", "自行假设", "相关信息等", "视情况")):
        contract_deductions.append((0.4, "任务契约包含模糊或未决输入"))
    deferred_business_truth = _defers_business_truth(task)
    if deferred_business_truth:
        contract_deductions.append((1.2, "状态任务把关键业务真值推迟到未定义的后续用户回复"))

    challenge_deductions: list[tuple[float, str]] = []
    if task.get("complexity") == "simple" and training_category == "multi_step_agentic":
        challenge_deductions.append((0.35, "任务标记为 simple，长程决策密度有限"))
    if len(actions) < 3 and training_category == "multi_step_agentic":
        challenge_deductions.append((0.65, "少于 3 个语义动作，Agentic 轨迹过短"))
    if not business_names and training_category != "direct_response":
        challenge_deductions.append((0.45, "没有业务工具，只能训练拒绝噪声工具"))
    if len(metrics) < 2:
        challenge_deductions.append((0.35, "奖励信号维度不足"))

    alignment_deductions: list[tuple[float, str]] = []
    if mode not in {"stateless", "reference_data", "stateful", "external_capability"}:
        alignment_deductions.append((0.8, "环境模式缺失或无效"))
    explicit_user_only = any(marker in corpus for marker in ("仅依赖用户", "仅使用用户", "只依赖用户", "只使用用户"))
    complete_input_supplied = any(
        re.search(pattern, corpus) is not None
        for pattern in (
            r"基于用户提供的.{0,8}(?:规格|数据|资料|文本)",
            r"基于以下提供的.{0,8}(?:规格|数据|资料|文本)",
            r"从用户提供的.{0,8}(?:列表|清单|文本|数据)中",
        )
    )
    if mode in {"reference_data", "stateful"} and (explicit_user_only or complete_input_supplied):
        alignment_deductions.append((0.8, "仅依赖用户输入的任务被过度设计为数据环境"))
    if mode in {"reference_data", "stateful"} and not business_names:
        alignment_deductions.append((0.8, "数据环境没有业务数据访问工具"))
    if mode == "reference_data" and any(
        item.get("operation") in {"insert", "update", "delete"}
        for item in _objects(task.get("tool_implementations"))
    ):
        alignment_deductions.append((1.5, "只读数据环境声明了变更工具"))
    if mode == "external_capability" and not business_names:
        alignment_deductions.append((0.8, "外部能力任务没有业务查询工具"))
    references_public_material = re.search(
        r"(?:以下|下列|上述|这段|这些|给定|附上|附件|(?:用户|我|你)(?:已)?提供).{0,12}"
        r"(?:文本|资料|数据|列表|清单|内容|说明|笔记|记录|规格|描述|选项)",
        corpus,
    ) is not None
    concrete_public_materials = [
        item for item in public_materials
        if isinstance(item.get("content"), str) and len(item["content"].strip()) >= 8
    ]
    if references_public_material and not concrete_public_materials:
        alignment_deductions.append((1.2, "任务引用了未交付给 Agent 的公开材料"))
    if len(noise_names) != len(noise) or not noise_names <= tool_names:
        alignment_deductions.append((0.6, "噪声工具元数据与工具定义不一致"))
    stateful_goal_issues = _stateful_goal_tool_issues(task)
    if stateful_goal_issues:
        alignment_deductions.append((1.5, "状态目标与变更工具未形成可验证闭环"))
    reward_deductions: list[tuple[float, str]] = []
    categories = {str(metric.get("category")) for metric in metrics}
    if "outcome" not in categories:
        reward_deductions.append((0.8, "缺少 outcome 指标"))
    if noise and "penalty" not in categories:
        reward_deductions.append((0.5, "存在噪声工具但缺少轨迹惩罚指标"))
    if any(not isinstance(metric.get("evaluator"), dict) for metric in metrics):
        reward_deductions.append((0.6, "指标缺少可执行 evaluator"))
    outcome_metrics = [item for item in metrics if item.get("category") == "outcome"]
    deterministic_outcome_ids = {
        item.get("metric_id") for item in implementations
        if isinstance(item.get("metric_id"), str)
        and (mode != "reference_data" or item.get("operator") in {"numeric_targets", "value_targets"})
    }
    if task.get("task_intent") == "calculate" and outcome_metrics and not any(
        item.get("type") == "rule-based"
        and item.get("id") in deterministic_outcome_ids
        for item in outcome_metrics
    ):
        reward_deductions.append((1.1, "数值计算结果仅由模型判定，尚无确定性结果指标"))
    rule_metric_ids = {
        str(metric.get("id")) for metric in metrics if metric.get("type") == "rule-based"
    }
    process_metric_ids = {
        str(metric.get("id")) for metric in metrics if metric.get("category") == "process"
    }
    implemented_ids = {str(item.get("metric_id")) for item in implementations}
    if rule_metric_ids - implemented_ids:
        reward_deductions.append((0.5, "部分 rule-based 指标缺少声明式实现"))
    unimplemented_process = process_metric_ids - implemented_ids
    if unimplemented_process:
        reward_deductions.append((1.0, "过程奖励依赖运行时猜测，缺少确定性工具调用实现"))
    formula = task.get("reward_formula")
    if not isinstance(formula, dict) or formula.get("score_range") != [-1, 1]:
        reward_deductions.append((0.6, "奖励公式或范围不完整"))

    acceptance_deductions: list[tuple[float, str]] = []
    scenarios = _objects(contract.get("executable_scenarios"))
    scenario_kinds = {str(item.get("kind")) for item in scenarios}
    if not {"goal_success", "goal_failure"} <= scenario_kinds:
        acceptance_deductions.append((0.8, "缺少成功/失败可执行场景"))
    if noise and "noise_selection" not in scenario_kinds:
        acceptance_deductions.append((0.5, "缺少噪声选择场景"))
    if not contract.get("mutation_tests"):
        acceptance_deductions.append((0.4, "缺少 mutation tests"))
    if readiness.get("ready") is not True:
        acceptance_deductions.append((0.5, "任务未声明 task_readiness.ready=true"))
    success_scenarios = [item for item in scenarios if item.get("kind") == "goal_success"]
    success_business_calls = [
        step for scenario in success_scenarios for step in _objects(scenario.get("steps"))
        if step.get("operation") == "tool_call" and step.get("tool_name") in business_names
    ]
    dependency_edge = False
    for scenario in success_scenarios:
        captures: set[str] = set()
        for step in _objects(scenario.get("steps")):
            if step.get("operation") == "tool_call":
                argument_text = _text(step.get("arguments", {}))
                if any(f'"$ref": "{name.lower()}"' in argument_text for name in captures):
                    dependency_edge = True
            capture = step.get("capture")
            if isinstance(capture, dict):
                captures.update(str(name) for name in capture)
    if training_category == "direct_response":
        if business_names or success_business_calls:
            alignment_deductions.append((1.0, "direct_response 不应依赖业务工具"))
        if mode != "stateless":
            alignment_deductions.append((0.8, "direct_response 应使用 stateless 环境"))
    elif training_category == "multi_step_agentic" and len(success_business_calls) < 2:
        acceptance_deductions.append((1.2, "multi_step_agentic 成功轨迹至少需要两个业务工具调用"))
    elif training_category == "multi_step_agentic" and not dependency_edge:
        acceptance_deductions.append((1.0, "multi_step_agentic 缺少 capture/$ref 工具数据依赖"))
    elif training_category == "simple_agentic" and not success_business_calls:
        acceptance_deductions.append((1.2, f"{training_category} 缺少必要业务工具调用"))
    bad_success_fixture = False
    for scenario in success_scenarios:
        for step in _objects(scenario.get("steps")):
            if step.get("operation") != "tool_call":
                continue
            arguments = step.get("arguments")
            argument_text = _text(arguments)
            schema = next((item.get("function", {}).get("parameters", {}) for item in _objects(task.get("tools"))
                           if item.get("function", {}).get("name") == step.get("tool_name")), {})
            if not isinstance(arguments, dict) or not set(schema.get("required", [])) <= set(arguments):
                bad_success_fixture = True
            if any(marker in argument_text for marker in (
                "fixture-value", "placeholder", "sample-value", "test-value"
            )):
                bad_success_fixture = True
            if isinstance(arguments, dict) and any(value in ([], {}) for value in arguments.values()):
                bad_success_fixture = True
    if bad_success_fixture:
        acceptance_deductions.append((1.2, "成功轨迹包含空参数或占位业务值"))
    if not success_business_calls and training_category != "direct_response":
        acceptance_deductions.append((1.2, "成功轨迹未执行任何业务工具，无法验证 Agentic 决策"))
    warnings = readiness.get("warnings", [])
    if isinstance(warnings, list) and warnings:
        acceptance_deductions.append((min(0.5, 0.1 * len(warnings)), "task_readiness 含警告"))

    dimensions = {
        "task_contract": _dimension(2.0, contract_deductions),
        "agentic_challenge": _dimension(2.0, challenge_deductions),
        "environment_tool_alignment": _dimension(2.0, alignment_deductions),
        "reward_evaluability": _dimension(2.0, reward_deductions),
        "acceptance_readiness": _dimension(2.0, acceptance_deductions),
    }
    score = round(sum(item.score for item in dimensions.values()), 2)
    findings_list = [reason for item in dimensions.values() for reason in item.reasons]
    # Eligibility is a hard gate, separate from the descriptive quality score.
    # An invalid training sample must never be represented as merely "7.8".
    category_gate_failed = (
        (training_category == "direct_response" and bool(business_names or success_business_calls or mode != "stateless"))
        or (training_category == "simple_agentic" and not success_business_calls)
        or (training_category == "multi_step_agentic" and (len(success_business_calls) < 2 or not dependency_edge))
    )
    eligibility_failures: list[str] = []
    if declared_training_category not in category_profiles:
        eligibility_failures.append("training_category 缺失或无效")
    artifacts = task.get("artifacts") if isinstance(task.get("artifacts"), dict) else {}
    agent_visible = " ".join([
        task_text,
        str(public_input.get("initial_user_message", "")),
        *(str(item["function"].get("description", ""))
          for item in tools if isinstance(item.get("function"), dict)),
    ])
    if any(marker in agent_visible for marker in (
        "沙箱", "图谱节点", "采样关键词", "场景关键词", "关键词组合", "任务生成",
    )):
        eligibility_failures.append("用户可见任务或工具泄露生成内部术语")
    if unsourced_real_world_exemplars(task, training_category=training_category):
        eligibility_failures.append("直接回答要求现实领域代表例证，但公开材料缺少可追溯来源")
    graph_context = artifacts.get("graph_context")
    if isinstance(graph_context, dict):
        relation = graph_context.get("relation")
        if isinstance(relation, str) and relation and relation.casefold() in agent_visible.casefold():
            eligibility_failures.append("用户可见任务或工具泄露图谱内部关系标识")
        keywords = graph_context.get("keywords", [])
        nodes = graph_context.get("nodes", [])
        terms = [
            unicodedata.normalize("NFKC", value).strip().casefold()
            for value in [*(keywords if isinstance(keywords, list) else []),
                          *(nodes if isinstance(nodes, list) else [])]
            if isinstance(value, str) and len(value.strip()) >= 2
        ]
        user_text = public_input.get("initial_user_message", "")
        prose = unicodedata.normalize("NFKC", task_text + " " + str(user_text)).casefold()
        if not terms or not any(term in prose for term in terms):
            eligibility_failures.append("任务描述脱离采样 Scene 主题")
    manifest = artifacts.get("data_manifest") if isinstance(artifacts.get("data_manifest"), dict) else {}
    governance = manifest.get("data_governance")
    if governance is not None:
        from env_factory.evidence.data_governance import valid_data_origin
        if not valid_data_origin(governance, task):
            eligibility_failures.append("业务数据来源声明不符合合成数据契约")
    missing_endpoints = missing_system_endpoints(requirements.get("runtime_interface"))
    if missing_endpoints:
        eligibility_failures.append(
            "运行接口缺少必需端点："
            + ", ".join(name for name, _, _ in sorted(missing_endpoints))
        )
    if category_gate_failed:
        eligibility_failures.append(f"任务不满足 {training_category} 路由契约")
    if training_category == "multi_step_agentic" and dependency_edge and not has_private_tool_dependency(success_scenarios):
        eligibility_failures.append("多步工具依赖只回传前序查询的公开选择器")
    tool_schemas = {
        item.get("function", {}).get("name"): item.get("function", {}).get("parameters", {})
        for item in tools if isinstance(item.get("function"), dict)
    }
    for spec in _objects(task.get("tool_implementations")):
        parameters = tool_schemas.get(spec.get("tool_name"), {})
        properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
        for rule in _objects(spec.get("filters")):
            argument_schema = properties.get(rule.get("argument"), {}) if isinstance(properties, dict) else {}
            item_schema = argument_schema.get("items", {}) if isinstance(argument_schema, dict) else {}
            if isinstance(item_schema, dict) and item_schema.get("type") == "object":
                item_properties = item_schema.get("properties", {})
                resolver = rule.get("resolve")
                item_field = resolver.get("match_column") if isinstance(resolver, dict) else rule.get("column")
                if (rule.get("operator") != "in" or not isinstance(item_properties, dict)
                        or item_field not in item_properties):
                    eligibility_failures.append(
                        f"对象数组工具筛选无法映射列：{spec.get('tool_name')}.{rule.get('argument')}"
                    )
    from .task_routing import public_calculation_is_self_contained
    if training_category != "direct_response" and public_calculation_is_self_contained(
        task, task.get("task_intent"),
    ):
        eligibility_failures.append("公开输入已足以完成计算或估算，业务工具缺少必要的私有数据依赖")
    if any("过度设计为数据环境" in item for item in findings_list):
        eligibility_failures.append("环境模式与任务输入边界冲突")
    if any("只读数据环境声明了变更工具" in item for item in findings_list):
        eligibility_failures.append("reference_data 工具不得变更业务表")
    if any(item.get("requires_tool") is True
           and capability_disclaims_tool(item.get("reason"))
           for item in _objects(task.get("capability_plan"))):
        eligibility_failures.append("工具能力说明承认该动作可由 Agent 自行完成")
    if mode == "reference_data" and any(
        item.get("requires_tool") is True
        and re.search(
            r"(?:前序|上一步|已有|给定).{0,40}"
            r"(?:求和|合计|统计字符|格式化|生成.{0,8}表格)",
            str(item.get("reason", "")),
        )
        and not any(marker in str(item.get("reason", "")) for marker in (
            "私有", "内部业务", "数据表", "外部系统",
        ))
        for item in _objects(task.get("capability_plan"))
    ):
        eligibility_failures.append("只读数据任务把前序结果的简单加工伪装成业务工具")
    if mode == "reference_data":
        prior_only = prior_result_only_tool_actions(task.get("capability_plan"))
        if prior_only:
            eligibility_failures.append(
                "多步工具只加工前序结果，未声明新的环境访问：" + ", ".join(prior_only)
            )
    if bad_success_fixture:
        eligibility_failures.append("成功轨迹不能作为真实 Agentic 训练正样本")
    if references_public_material and not concrete_public_materials:
        eligibility_failures.append("public_input 缺少题面引用的实际输入材料")
    if deferred_business_truth:
        eligibility_failures.append("关键业务真值依赖未定义的后续用户回复")
    if stateful_goal_issues:
        eligibility_failures.append(
            "状态目标与变更工具不闭合：" + "；".join(stateful_goal_issues)
        )
    if unimplemented_process:
        eligibility_failures.append(
            "过程奖励缺少可复现实现：" + ", ".join(sorted(unimplemented_process))
        )
    if "outcome" not in categories:
        eligibility_failures.append("缺少结果奖励指标")
    semantic_overrides = sorted({
        str(metric.get("id")) for metric in metrics
        if isinstance(metric.get("evaluator"), dict)
        and metric["evaluator"].get("kind") in {"external_llm_judge", "hybrid_outcome"}
        and metric.get("id") in implemented_ids
    })
    if semantic_overrides:
        eligibility_failures.append(
            "语义奖励被声明式实现覆盖：" + ", ".join(semantic_overrides)
        )
    if mode == "reference_data":
        outcome_implementations = {
            item.get("metric_id"): item for item in implementations
            if isinstance(item.get("metric_id"), str)
        }
        for metric in outcome_metrics:
            evaluator = metric.get("evaluator")
            implementation = outcome_implementations.get(metric.get("id"), {})
            if (metric.get("type") == "rule-based"
                    and implementation.get("operator") in {"numeric_targets", "value_targets"}):
                continue
            inputs = metric.get("evaluation_inputs")
            criteria = metric.get("criteria")
            if (metric.get("type") not in {"model-based", "hybrid"}
                    or not isinstance(evaluator, dict)
                    or evaluator.get("kind") not in {"external_llm_judge", "hybrid_outcome"}
                    or not isinstance(inputs, list)
                    or not {"final_agent_response", "tool_results", "business_data"} <= set(
                        value for value in inputs if isinstance(value, str)
                    )
                    or not isinstance(criteria, list)
                    or REFERENCE_FACTUALITY_CRITERION not in criteria
                    or any(contradicts_reference_factuality(item) for item in criteria)
                    or contradicts_reference_factuality(metric.get("rubric"))):
                eligibility_failures.append(
                    f"只读数据结果奖励缺少答案与业务证据核对：{metric.get('id')}"
                )
    if mode == "stateful":
        state_implementations = {
            item.get("metric_id"): item for item in implementations
            if isinstance(item.get("metric_id"), str)
        }
        for metric in outcome_metrics:
            evaluator = metric.get("evaluator")
            implementation = state_implementations.get(metric.get("id"), {})
            if metric.get("type") == "rule-based":
                if (implementation.get("source") != "business_state"
                        and implementation.get("operator") not in {"numeric_targets", "value_targets"}):
                    eligibility_failures.append(
                        f"状态结果奖励未核对业务状态：{metric.get('id')}"
                    )
                continue
            inputs = metric.get("evaluation_inputs")
            criteria = metric.get("criteria")
            if (metric.get("type") not in {"model-based", "hybrid"}
                    or not isinstance(evaluator, dict)
                    or evaluator.get("kind") not in {"external_llm_judge", "hybrid_outcome"}
                    or not isinstance(inputs, list)
                    or not {"final_agent_response", "business_data", "tool_results"} <= set(
                        value for value in inputs if isinstance(value, str)
                    )
                    or not isinstance(criteria, list)
                    or STATEFUL_GOAL_CRITERION not in criteria
                    or any(contradicts_reference_factuality(item) for item in criteria)
                    or contradicts_reference_factuality(metric.get("rubric"))):
                eligibility_failures.append(
                    f"状态结果奖励缺少最终业务状态核对：{metric.get('id')}"
                )
    if any(not isinstance(metric.get("evaluator"), dict) for metric in metrics):
        eligibility_failures.append("奖励指标缺少可执行 evaluator")
    unimplemented_rules = rule_metric_ids - implemented_ids
    if unimplemented_rules:
        eligibility_failures.append(
            "规则奖励缺少可复现实现：" + ", ".join(sorted(unimplemented_rules))
        )
    implementation_ids = [item.get("metric_id") for item in implementations]
    if any(not isinstance(value, str) or not value for value in implementation_ids):
        eligibility_failures.append("奖励指标实现缺少有效 metric_id")
    valid_implementation_ids = [value for value in implementation_ids if isinstance(value, str)]
    if len(valid_implementation_ids) != len(set(valid_implementation_ids)):
        eligibility_failures.append("奖励指标实现存在重复 metric_id")
    from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, SandboxError
    for item in implementations:
        if item.get("operator") not in {"numeric_targets", "value_targets"}:
            continue
        try:
            if item.get("source") != "final_agent_response" or item.get("path") != "$":
                raise SandboxError("METRIC_SPEC_INVALID", "numeric target source is invalid", 500)
            validator = (DeclarativeMetricEvaluator.validate_numeric_targets if item["operator"] == "numeric_targets"
                         else DeclarativeMetricEvaluator.validate_value_targets)
            validator(item.get("expected"))
            if mode != "stateless" and not _numeric_private_dependencies(item.get("expected")):
                raise SandboxError("METRIC_SPEC_INVALID", "numeric target has no private data dependency", 500)
        except SandboxError as exc:
            eligibility_failures.append(f"数值奖励公式无效：{exc}")
    if not isinstance(formula, dict) or formula.get("score_range") != [-1, 1]:
        eligibility_failures.append("奖励公式或范围不完整")
    missing_scenarios = {"goal_success", "goal_failure"} - scenario_kinds
    if missing_scenarios:
        eligibility_failures.append(
            "缺少可执行验收场景：" + ", ".join(sorted(missing_scenarios))
        )
    task_spec = task.get("task_spec")
    if not isinstance(task_spec, dict):
        eligibility_failures.append("缺少可编译的 task_spec IR")
    else:
        spec_training = task_spec.get("training_contract")
        if (isinstance(spec_training, dict)
                and declared_training_category in category_profiles
                and spec_training.get("category") != declared_training_category):
            eligibility_failures.append("training_category 与 task_spec 类别不一致")
        spec_task = task_spec.get("task_contract")
        if isinstance(spec_task, dict) and "task" in spec_task and spec_task["task"] != task.get("task"):
            eligibility_failures.append("任务描述与 task_spec 不一致")
        spec_environment = task_spec.get("environment_contract")
        if (isinstance(spec_environment, dict) and "mode" in spec_environment
                and spec_environment["mode"] != mode):
            eligibility_failures.append("环境模式与 task_spec 不一致")
        if "tool_contracts" in task_spec:
            spec_tools = _objects(task_spec.get("tool_contracts"))
            spec_business = [item.get("name") for item in spec_tools if item.get("role") == "business"]
            spec_noise = [item.get("name") for item in spec_tools if item.get("role") == "noise"]
            if (sorted(spec_business, key=str) != sorted(business_names)
                    or not all(isinstance(name, str) and name in noise_names for name in spec_noise)
                    or len(spec_business) + len(spec_noise) != len(spec_tools)):
                eligibility_failures.append("工具清单与 task_spec 不一致")
        spec_reward = task_spec.get("reward_contract")
        if (isinstance(spec_reward, dict) and "metric_ids" in spec_reward
                and spec_reward["metric_ids"] != [item.get("id") for item in metrics]):
            eligibility_failures.append("奖励指标清单与 task_spec 不一致")
        try:
            from .task_spec import validate_task_spec
            validate_task_spec(task_spec)
        except (ValueError, TypeError) as exc:
            eligibility_failures.append(f"task_spec IR 无效：{exc}")
    for tool_name in unobservable_tool_preconditions(task):
        eligibility_failures.append(
            f"工具 {tool_name} 要求 active 记录，但声明式查询既不筛选也不返回 status"
        )
    if readiness.get("ready") is not True:
        eligibility_failures.append("task_readiness 未通过")
    eligible = not eligibility_failures
    findings_list.extend(f"训练资格失败：{reason}" for reason in eligibility_failures)
    findings = tuple(findings_list)
    # Dimension scores remain diagnostic; a failed hard gate has no passing score.
    if not eligible:
        score = 0.0
    passed = eligible and score >= min_score
    high_value = (
        passed
        and score >= 9.0
        and not category_gate_failed
        and not bad_success_fixture
    )
    tier = "high_value" if high_value else ("usable" if passed else "rejected")
    return TaskQualityReport(
        task_id, path, score, passed, tier, dimensions, findings,
        training_category, agentic_level, tool_policy_target, eligible,
        tuple(eligibility_failures),
    )


def score_file(path: Path, *, min_score: float = 8.0) -> TaskQualityReport:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: task artifact must be a JSON object")
    task_id = path.parent.name if path.name == "task.json" else path.stem
    report = score_task(data, task_id=task_id, path=str(path), min_score=min_score)
    artifacts = data.get("artifacts") if isinstance(data.get("artifacts"), dict) else {}
    manifest = artifacts.get("data_manifest") if isinstance(artifacts, dict) else None
    mode = data.get("environment_plan", {}).get("mode") if isinstance(data.get("environment_plan"), dict) else None
    if mode in {"reference_data", "stateful"} and (
        not isinstance(manifest, dict) or not _objects(manifest.get("tables"))
    ):
        return _reject(report, "数据环境缺少可加载的业务表 manifest")
    if isinstance(manifest, dict):
        declared_root = Path(str(manifest.get("root", ".")))
        if declared_root.is_absolute() or ".." in declared_root.parts:
            return _reject(report, "业务数据 manifest 必须位于任务目录内")
        for table in _objects(manifest.get("tables")):
            for key in ("schema_file", "rows_file"):
                filename = table.get(key)
                if (not isinstance(filename, str) or not filename
                        or Path(filename).is_absolute() or ".." in Path(filename).parts):
                    return _reject(report, "业务数据 manifest 文件路径必须位于任务目录内")
    tables: list[dict[str, Any]] = []
    if isinstance(manifest, dict) and isinstance(manifest.get("tables"), list):
        root = _manifest_root(path.parent, manifest)
        task_directory = path.parent.resolve()
        if (not root.resolve().is_relative_to(task_directory)
                or any(not (root / table[key]).resolve().is_relative_to(task_directory)
                       for table in _objects(manifest["tables"])
                       for key in ("schema_file", "rows_file"))):
            return _reject(report, "业务数据 manifest 指向任务目录外的文件")
        try:
            from env_factory.sandbox_runtime import EpisodeStore, ManifestDataStore, SandboxError
            with tempfile.TemporaryDirectory(prefix="envfactory-quality-") as directory:
                store = ManifestDataStore(
                    manifest, root, EpisodeStore(Path(directory) / "episodes.sqlite3")
                )
                tables = [
                    {**store.schemas[name], "table_name": name, "rows": rows}
                    for name, rows in store.baseline.items()
                ]
                acceptance = data.get("acceptance_contract")
                fixtures = acceptance.get("fixtures") if isinstance(acceptance, dict) else None
                declared_hash = fixtures.get("initial_data_hash") if isinstance(fixtures, dict) else None
                if isinstance(declared_hash, str) and declared_hash != store.data_hash:
                    raise ValueError(
                        f"initial_data_hash mismatch: declared={declared_hash} actual={store.data_hash}"
                    )
        except (SandboxError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            return _reject(report, f"业务数据不满足共享持久化契约：{exc}")
    if mode == "reference_data" and not any(table.get("rows") for table in tables):
        return _reject(report, "只读数据任务缺少基线业务记录")
    if tables:
        from env_factory.task_pipeline import TaskGenerationPipeline
        for spec in _objects(data.get("tool_implementations")):
            missing = TaskGenerationPipeline._missing_insert_columns(spec, tables)
            if missing:
                return _reject(
                    report,
                    f"声明式插入缺少存储列：{spec.get('tool_name')} -> {spec.get('table')} {missing}",
                )
            optional = TaskGenerationPipeline._optional_mutation_arguments(
                spec, _objects(data.get("tools")),
            )
            if optional:
                return _reject(
                    report,
                    f"声明式写入参数未设为必填：{spec.get('tool_name')} {optional}",
                )
    contract_issues = tool_chain_issues(data) + reward_contract_issues(data)
    if contract_issues:
        issue = contract_issues[0]
        return _reject(report, f"任务契约预检失败 [{issue['code']}]：{issue.get('message', '')}")
    if isinstance(data.get("task_spec"), dict):
        from .task_spec import validate_task_spec
        try:
            validate_task_spec(data["task_spec"], data_tables=tables)
        except ValueError as exc:
            return _reject(report, f"任务实际数据不满足 TaskSpec：{exc}")
        environment_contract = data["task_spec"].get("environment_contract")
        initial_fixture = (
            environment_contract.get("initial_fixture")
            if isinstance(environment_contract, dict) else None
        )
        if isinstance(initial_fixture, dict):
            if "manifest" in initial_fixture and initial_fixture["manifest"] != manifest:
                return _reject(report, "TaskSpec 初始数据 manifest 与实际业务数据不一致")
            if "table_names" in initial_fixture and initial_fixture["table_names"] != [
                table["table_name"] for table in tables
            ]:
                return _reject(report, "TaskSpec 初始业务表与实际业务数据不一致")
    if mode == "stateful" and tables:
        from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, SandboxError
        state = {item["table_name"]: item["rows"] for item in tables}
        outcome_ids = {
            item.get("id") for item in _objects(data.get("metrics"))
            if item.get("category") == "outcome" and item.get("type") == "rule-based"
        }
        for spec in _objects(data.get("metric_implementations")):
            if spec.get("metric_id") not in outcome_ids or spec.get("source") != "business_state":
                continue
            try:
                initial_score = DeclarativeMetricEvaluator().evaluate(spec, {
                    "business_state": state, "initial_business_state": state,
                    "trajectory": {"events": []}, "final_agent_response": "",
                })
                initially_satisfied = initial_score == spec.get("score_mapping", {}).get("pass", 1)
            except (SandboxError, ValueError, TypeError, KeyError) as exc:
                return _reject(report, f"状态结果奖励无法核验初始业务状态：{exc}")
            if initially_satisfied:
                return _reject(
                    report,
                    f"状态结果奖励在变更前已满足：{spec.get('metric_id')}",
                )
    if mode in {"reference_data", "stateful"} and tables:
        # Replay the declarative success prefix against the shared data-store
        # rules before admitting the task to the much slower sandbox build.
        from env_factory.task_pipeline import PipelineGenerationError, TaskGenerationPipeline
        from env_factory.sandbox_runtime import SandboxError
        try:
            acceptance = data.get("acceptance_contract")
            scenarios = acceptance.get("executable_scenarios", []) if isinstance(acceptance, dict) else []
            preview = TaskGenerationPipeline._preview_success_tool_results(
                scenarios=scenarios,
                data_tables=tables,
                tool_implementations=_objects(data.get("tool_implementations")),
                environment_mode=mode,
                tools=_objects(data.get("tools")),
            )
        except (PipelineGenerationError, SandboxError, KeyError, TypeError, ValueError) as exc:
            return _reject(report, f"成功工具轨迹预演失败：{exc}")
        ambiguous = ambiguous_metric_captures(data, preview)
        if ambiguous:
            issue = ambiguous[0]
            return _reject(
                report,
                "奖励指标从多条业务结果中按位置取值，无法确认目标记录："
                f"{issue['metric_id']} ({issue['tool_name']} {issue['path']})",
            )
    if data.get("training_category") != "direct_response" and tables:
        public_input = data.get("public_input")
        public_sets = public_row_sets(_objects(
            public_input.get("materials") if isinstance(public_input, dict) else None
        ))
        duplicated = [
            table.get("table_name") for table in tables
            if table.get("rows") and {
                json.dumps(row, ensure_ascii=False, sort_keys=True)
                for row in table["rows"]
            } in public_sets
        ]
        if duplicated:
            return _reject(
                report,
                "私有业务表完整复制了公开材料，工具无独占信息："
                + ", ".join(str(name) for name in duplicated),
            )
    numeric_specs = [item for item in _objects(data.get("metric_implementations"))
                     if item.get("operator") == "numeric_targets"]
    if numeric_specs and tables:
        from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, SandboxError
        task_spec = data.get("task_spec") if isinstance(data.get("task_spec"), dict) else {}
        contract = task_spec.get("task_contract") if isinstance(task_spec.get("task_contract"), dict) else {}
        public_input = data.get("public_input") if isinstance(data.get("public_input"), dict) else {}
        scope_description = {
            "task": data.get("task"), "expected_result": contract.get("expected_result"),
            "public_input": public_input,
        }
        if any(fixed_row_total_targets(scope_description, spec.get("expected"))
               for spec in numeric_specs):
            return _reject(report, "开放记录集合的合计奖励枚举固定行，无法覆盖新增业务记录")
        scenarios = _objects(data.get("acceptance_contract", {}).get("executable_scenarios"))
        success = next((item for item in scenarios if item.get("kind") == "goal_success"), {})
        answer = next((item.get("content") for item in _objects(success.get("steps"))
                       if item.get("operation") == "agent_response"), None)
        state = {item["table_name"]: item["rows"] for item in tables}
        evaluator = DeclarativeMetricEvaluator()
        for spec in numeric_specs:
            try:
                baseline = evaluator.evaluate(spec, {
                    "final_agent_response": answer, "business_state": state,
                })
            except (SandboxError, ValueError, TypeError) as exc:
                return _reject(report, f"数值奖励公式无法执行：{exc}")
            if baseline != 1:
                return _reject(report, "数值奖励公式与成功答案或实际业务数据不一致")
            dependencies: list[dict[str, Any]] = []
            def collect(value: Any) -> None:
                if isinstance(value, dict):
                    for kind in ("lookup", "aggregate"):
                        if isinstance(value.get(kind), dict):
                            dependencies.append(value[kind])
                    for child in value.values():
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)
            collect(spec.get("expected"))
            sensitive = False
            for dependency in dependencies:
                table_name = dependency.get("table")
                field = (dependency.get("fields") or [dependency.get("field")])[0]
                where = dependency.get("where")
                if not isinstance(table_name, str) or not isinstance(field, str) or not isinstance(where, dict):
                    continue
                for index, row in enumerate(state.get(table_name, [])):
                    if not isinstance(row, dict) or not all(row.get(key) == value for key, value in where.items()):
                        continue
                    if dependency.get("op") == "count":
                        altered = copy.deepcopy(state)
                        altered[table_name].pop(index)
                        try:
                            if evaluator.evaluate(spec, {
                                "final_agent_response": answer, "business_state": altered,
                            }) == 0:
                                sensitive = True
                                break
                        except (SandboxError, ValueError, TypeError):
                            continue
                    value = row.get(field)
                    if isinstance(value, bool) or not isinstance(value, (int, float)):
                        continue
                    altered = copy.deepcopy(state)
                    altered[table_name][index][field] = value + 1
                    try:
                        if evaluator.evaluate(spec, {
                            "final_agent_response": answer, "business_state": altered,
                        }) == 0:
                            sensitive = True
                            break
                    except (SandboxError, ValueError, TypeError):
                        continue
                if sensitive:
                    break
            if not sensitive:
                return _reject(report, "数值奖励公式对私有业务记录变动不敏感")
            scope_text = " ".join(str(value) for value in (
                scope_description["task"], scope_description["expected_result"],
                public_input.get("initial_user_message"),
            ))
            if re.search(r"各条|每条|所有|全部|每笔|各笔|逐条|整批", scope_text):
                schema_by_table = {item["table_name"]: item for item in tables}
                for target in _objects(spec.get("expected", {}).get("targets")):
                    if not re.search(r"总|合计|汇总", str(target.get("label", ""))):
                        continue
                    aggregates: list[dict[str, Any]] = []
                    def collect_aggregates(value: Any) -> None:
                        if isinstance(value, dict):
                            if isinstance(value.get("aggregate"), dict):
                                aggregates.append(value["aggregate"])
                            for child in value.values():
                                collect_aggregates(child)
                        elif isinstance(value, list):
                            for child in value:
                                collect_aggregates(child)
                    collect_aggregates(target.get("expression"))
                    target_spec = {**spec, "expected": {"targets": [target]}}
                    added_row_sensitive = False
                    for aggregate in aggregates:
                        table_name = aggregate.get("table")
                        where = aggregate.get("where")
                        schema = schema_by_table.get(table_name)
                        if not isinstance(schema, dict) or not isinstance(where, dict):
                            continue
                        where = evaluator._numeric_where(where, state, depth=0)
                        source_rows = state.get(table_name, [])
                        row = next((item for item in source_rows if isinstance(item, dict)
                                    and all(item.get(key) == value for key, value in where.items())), None)
                        if row is None:
                            continue
                        new_row = copy.deepcopy(row)
                        for key in schema.get("primary_key", []):
                            old = new_row.get(key)
                            if isinstance(old, int) and not isinstance(old, bool):
                                new_row[key] = max((item.get(key) for item in source_rows
                                                    if isinstance(item, dict) and isinstance(item.get(key), int)),
                                                   default=old) + 1
                            elif isinstance(old, str):
                                new_row[key] = old + "__new_row_probe"
                        fields = aggregate.get("fields") if aggregate.get("op") == "sum_product" else [aggregate.get("field")]
                        if isinstance(fields, list) and aggregate.get("op") != "count":
                            for field in fields:
                                if (isinstance(field, str) and field not in where
                                        and isinstance(new_row.get(field), (int, float))
                                        and new_row[field] == 0):
                                    new_row[field] = 1
                        altered = copy.deepcopy(state)
                        altered[table_name].append(new_row)
                        try:
                            if evaluator.evaluate(target_spec, {
                                "final_agent_response": answer, "business_state": altered,
                            }) == 0:
                                added_row_sensitive = True
                                break
                        except (SandboxError, ValueError, TypeError):
                            continue
                    if not added_row_sensitive:
                        return _reject(report, "开放记录集合的合计奖励对新增匹配记录不敏感")
    # New write values and runtime $refs need not occur in the initial rows.
    # Their semantics are verified by execution, not field-name membership.
    return report


def discover_task_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    def key(item: Path) -> tuple[int, str]:
        match = re.fullmatch(r"task-(\d+)", item.parent.name)
        return (int(match.group(1)), item.parent.name) if match else (10**18, item.parent.name)

    return sorted(root.glob("task-*/task.json"), key=key)


def error_report(path: Path, error: Exception, *, min_score: float = 8.0) -> TaskQualityReport:
    reason = f"任务文件无法解析或评分：{error}"
    return TaskQualityReport(
        task_id=path.parent.name if path.name == "task.json" else path.stem,
        path=str(path),
        score=0.0,
        passed=False,
        tier="rejected",
        dimensions={"artifact_integrity": QualityDimension(0.0, 10.0, (reason,))},
        findings=(reason,),
        training_category="unknown",
        agentic_level=0,
        tool_policy_target="unknown",
        eligible=False,
        eligibility_failures=(reason,),
    )
