"""Shared deterministic contract validation and artifact compilation."""

from __future__ import annotations

import copy
import json
import logging
import math
import re
import unicodedata
import hashlib
from decimal import Decimal
from pathlib import Path
from typing import Any

from .pipeline_errors import PipelineGenerationError
from ..contracts.reward_contract import REFERENCE_FACTUALITY_CRITERION, STATEFUL_GOAL_CRITERION, contradicts_reference_factuality, declared_terminal_outcome_ids, has_literal_payload_argument
from .user_simulation_contract import UserSimulationContractMixin
from ..tasks.task_quality import capability_disclaims_tool, prior_result_only_tool_actions

logger = logging.getLogger(__name__)


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False

HIGH_STAKES_MARKERS = (
    "法律", "法规", "法条", "政策文件", "合规", "法律意见",
    "医疗", "诊断", "用药", "食物中毒", "腹痛", "呕吐", "腹泻",
    "肚子不舒服", "手脚颤抖", "震颤", "就医", "疾病", "症状", "癌症", "癌细胞", "器官移植", "移植排斥",
    "mhc", "冠状病毒", "sars", "投资", "证券", "税务", "投资建议",
    "legal", "regulation", "compliance", "legal advice", "medical",
    "diagnosis", "medication", "food poisoning", "investment",
    "investment advice", "securities", "tax",
)

HIGH_STAKES_CHEMICAL_MARKERS = (
    "乙醚", "丙酮", "甲醇", "乙醇", "甲基叔丁基醚", "有机镁化合物",
    "格氏试剂", "偏三甲苯", "溶剂", "萃取", "混合液体",
)
HIGH_STAKES_CHEMICAL_RISK_MARKERS = (
    "安全", "危险", "毒性", "暴露", "防护", "替代", "操作", "事故",
    "收缩", "膨胀", "hazard", "safety", "toxic", "exposure",
)



GENERIC_ACTION_NAMES = frozenset({
    "接收并解析用户输入", "识别用户输入", "检查信息完整性", "请求补充信息",
    "执行诊断处理", "执行处理", "格式化输出结果", "生成最终回答",
    "接收用户输入", "解析用户输入", "处理数据", "分析数据",
})

GENERIC_ACTION_MARKERS = (
    "接收并", "解析用户输入", "检查输入", "检查信息", "补充信息",
    "执行诊断处理", "执行处理", "格式化并输出", "格式化输出", "生成最终",
    "receive input", "parse user input", "check input", "process data", "format output",
)


class ContractCompiler(UserSimulationContractMixin):
    """Compile and validate contracts without invoking a generation model."""

    @staticmethod
    def _list(value: Any, name: str, *, minimum: int = 1) -> list[Any]:
        if not isinstance(value, list) or len(value) < minimum:
            raise PipelineGenerationError(f"{name} must be a list with at least {minimum} item(s)")
        return value

    @staticmethod
    def _fixture_singular_name(name: str) -> str:
        if name.endswith("ies"):
            return f"{name[:-3]}y"
        if name.endswith(("shes", "ches", "sses", "xes")):
            return name[:-2]
        return name[:-1] if name.endswith("s") and not name.endswith("ss") else name

    @staticmethod
    def _schema_fixture(
        schema: dict[str, Any], *, field_name: str = "",
        fixture_values: dict[str, list[Any]] | None = None,
        fixture_rows: dict[str, list[dict[str, Any]]] | None = None,
    ) -> Any:
        """Create a non-empty request fixture, preferring generated business rows."""
        fixture_values = fixture_values or {}
        fixture_rows = fixture_rows or {}
        if "default" in schema:
            return schema["default"]
        enum = schema.get("enum")
        if isinstance(enum, list) and enum:
            return enum[0]
        kind = schema.get("type")
        if isinstance(kind, list):
            kind = next((item for item in kind if item != "null"), None)
        if kind == "object":
            properties = schema.get("properties", {})
            return {
                name: ContractCompiler._schema_fixture(
                    value, field_name=name, fixture_values=fixture_values,
                    fixture_rows=fixture_rows,
                )
                for name, value in properties.items()
                if name in schema.get("required", [])
            }
        if kind == "array":
            items = schema.get("items", {})
            if isinstance(items, dict) and items.get("type") == "object":
                properties = items.get("properties", {})
                required = set(items.get("required", []))
                if isinstance(properties, dict) and required:
                    singular = ContractCompiler._fixture_singular_name(field_name)
                    candidates = [field_name, singular, singular.removesuffix("_data")]
                    matching_tables = [
                        name for name, rows in fixture_rows.items()
                        if rows and required <= rows[0].keys()
                    ]
                    if len(matching_tables) == 1:
                        candidates.extend(matching_tables)
                    for name in dict.fromkeys(candidates):
                        rows = fixture_rows.get(name, [])
                        if rows and all(required <= row.keys() for row in rows):
                            return [
                                {key: row[key] for key in properties if key in row}
                                for row in rows[:3]
                            ]
            values = fixture_values.get(field_name, [])
            singular_name = ContractCompiler._fixture_singular_name(field_name)
            if not values:
                values = fixture_values.get(singular_name, [])
            if not values and singular_name.startswith("target_"):
                values = fixture_values.get(singular_name.removeprefix("target_"), [])
            if not values and field_name.endswith("keywords"):
                values = fixture_values.get("keywords", [])
            if values:
                return values[: min(3, len(values))]
            return [ContractCompiler._schema_fixture(
                schema.get("items", {}), field_name=singular_name,
                fixture_values=fixture_values,
                fixture_rows=fixture_rows,
            )]
        if kind == "integer":
            values = fixture_values.get(field_name, [])
            if values and isinstance(values[0], int) and not isinstance(values[0], bool):
                return values[0]
            return 1
        if kind == "number":
            values = fixture_values.get(field_name, [])
            if values and _finite_number(values[0]):
                return values[0]
            return 1
        if kind == "boolean":
            return True
        values = fixture_values.get(field_name, [])
        if not values and field_name.startswith("selected_"):
            values = fixture_values.get(field_name.removeprefix("selected_"), [])
        if not values and field_name.startswith("target_"):
            values = fixture_values.get(field_name.removeprefix("target_"), [])
        if values:
            return values[0]
        description = str(schema.get("description", ""))
        example_match = re.search(r"(?:例如|示例|如)\s*[:：]?\s*([^，。；,;]+)", description)
        if example_match:
            return example_match.group(1).strip(" '\"`“”‘’「」『』")
        return f"任务输入中的{field_name or '值'}"

    @staticmethod
    def _business_fixture_values(data_tables: list[dict[str, Any]]) -> dict[str, list[Any]]:
        """Index real generated row values for deterministic acceptance fixtures."""
        values: dict[str, list[Any]] = {}
        table_names: list[str] = []
        observed_dates: list[str] = []
        for table in data_tables:
            table_name = table.get("table_name")
            if isinstance(table_name, str) and table_name:
                table_names.append(table_name)
                values.setdefault(f"{table_name}_table", []).append(table_name)
            for row in table.get("rows", []):
                if not isinstance(row, dict):
                    continue
                for name, value in row.items():
                    if value in (None, "", [], {}):
                        continue
                    if (isinstance(value, str)
                            and re.match(r"^\d{4}-\d{2}-\d{2}(?:$|[ T])", value)
                            and (str(name).lower().endswith("_at") or any(
                                marker in str(name).lower() for marker in (
                                "date", "time", "日期", "时间",
                            )))):
                        observed_dates.append(value[:10])
                    bucket = values.setdefault(str(name), [])
                    if value not in bucket:
                        bucket.append(value)
                    if str(name).endswith("_name"):
                        stem = str(name).removesuffix("_name")
                        alias_bucket = values.setdefault(stem, [])
                        if value not in alias_bucket:
                            alias_bucket.append(value)
                    # Tool schemas often use localized business labels while
                    # generated tables use storage-oriented snake_case names
                    # (for example 撮口呼样本计数 vs table_a_撮口呼_count).
                    # Index conservative aliases so executable positive paths
                    # use a real correlated row instead of the numeric fallback 1.
                    alias = re.sub(r"^(?:table|表)[_-]?[a-z0-9]+[_-]", "", str(name), flags=re.I)
                    alias = alias.replace("_count", "样本计数").replace("count", "样本计数")
                    alias = alias.replace("_", "")
                    if alias and alias != str(name):
                        alias_bucket = values.setdefault(alias, [])
                        if value not in alias_bucket:
                            alias_bucket.append(value)
        if table_names:
            values["reference_data"] = table_names
            values["data_source"] = table_names
        if observed_dates:
            values.setdefault("start_date", [min(observed_dates)])
            values.setdefault("end_date", [max(observed_dates)])
        return values

    @classmethod
    def _business_fixture_context(
        cls, data_tables: list[dict[str, Any]], task_description: dict[str, Any],
    ) -> tuple[dict[str, list[Any]], dict[str, list[dict[str, Any]]]]:
        """Index generated rows and concrete public inputs for executable calls."""
        values = cls._business_fixture_values(data_tables)
        scene_plan = task_description.get("_scene_business_plan", {})
        anchor = scene_plan.get("anchor_term") if isinstance(scene_plan, dict) else None
        visible_text = " ".join((
            str(task_description.get("task", "")),
            str(task_description.get("public_input", {}).get("initial_user_message", ""))
            if isinstance(task_description.get("public_input"), dict) else "",
        ))
        if isinstance(anchor, str) and anchor.strip() and anchor in visible_text:
            values.setdefault("keywords", [anchor])
        rows = {
            str(table.get("table_name")): [row for row in table.get("rows", []) if isinstance(row, dict)]
            for table in data_tables if isinstance(table, dict) and table.get("table_name")
        }

        def add(name: str, value: Any) -> None:
            if value in (None, "", [], {}):
                return
            bucket = values.setdefault(name, [])
            if value not in bucket:
                bucket.append(value)

        public_input = task_description.get("public_input", {})
        materials = public_input.get("materials", []) if isinstance(public_input, dict) else []
        material_rows: list[dict[str, Any]] = []
        for material in materials if isinstance(materials, list) else []:
            if not isinstance(material, dict):
                continue
            material_rows.append({
                key: value for key, value in material.items()
                if key in {"name", "mime_type", "content"}
                and value not in (None, "", [], {})
            })
            name = material.get("name")
            content = material.get("content")
            if isinstance(name, str) and name.strip():
                add("material_name", name)
            if isinstance(content, str) and content.strip():
                add("corpus_text", content)
            if material.get("mime_type") != "application/json":
                continue
            try:
                document = json.loads(content)
            except (TypeError, json.JSONDecodeError):
                continue
            collections = (
                document.items() if isinstance(document, dict)
                else [(str(name or "records"), document)] if isinstance(document, list) else []
            )
            for collection_name, collection_rows in collections:
                if not (isinstance(collection_name, str) and isinstance(collection_rows, list)
                        and collection_rows and all(isinstance(row, dict) for row in collection_rows)):
                    continue
                rows.setdefault(collection_name, collection_rows)
                for row in collection_rows:
                    for field, value in row.items():
                        add(str(field), value)
        if material_rows:
            rows.setdefault("materials", material_rows)
        return values, rows

    @staticmethod
    def _acceptance_requires_model(
        *, tools: list[dict[str, Any]], noise_tools: list[dict[str, Any]],
        tool_implementations: list[dict[str, Any]], metrics: list[dict[str, Any]],
        metric_implementations: list[dict[str, Any]], environment_mode: str,
    ) -> bool:
        """Use model review only where EnvFactory cannot execute the contract itself."""
        if environment_mode == "external_capability":
            return True
        compiled_tools = {
            item.get("tool_name") for item in tool_implementations
            if isinstance(item, dict)
        } | {
            item.get("name") for item in noise_tools if isinstance(item, dict)
        }
        if any(
            tool.get("function", {}).get("name") not in compiled_tools
            for tool in tools if isinstance(tool, dict)
        ):
            return True
        compiled_metrics = {
            item.get("metric_id") for item in metric_implementations
            if isinstance(item, dict)
        }
        return any(
            metric.get("id") not in compiled_metrics
            and metric.get("evaluator", {}).get("kind") not in {
                "external_llm_judge", "hybrid_outcome",
            }
            for metric in metrics if isinstance(metric, dict)
        )

    @classmethod
    def _build_acceptance_contract(
        cls,
        *,
        task_description: dict[str, Any],
        data_manifest: dict[str, Any],
        data_tables: list[dict[str, Any]],
        actions: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        key_steps: list[dict[str, Any]],
        metrics: list[dict[str, Any]],
        reward_formula: dict[str, Any],
        tool_implementations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        tool_cases: list[dict[str, Any]] = []
        fixture_values, fixture_rows = cls._business_fixture_context(
            data_tables, task_description,
        )
        sales_rows = fixture_rows.get("sales", [])
        if sales_rows and all(
            isinstance(row.get("item"), str)
            and isinstance(row.get("quantity"), (int, float))
            and not isinstance(row.get("quantity"), bool)
            for row in sales_rows
        ):
            totals: dict[str, list[float]] = {}
            for row in sales_rows:
                totals.setdefault(row["item"], []).append(float(row["quantity"]))
            fixture_rows.setdefault("average_daily_sales", [
                {"item": item, "average_daily_sales": sum(values) / len(values)}
                for item, values in totals.items()
            ])
        implementation_by_tool = {
            item.get("tool_name"): item for item in (tool_implementations or [])
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }
        for tool in tools:
            function = tool["function"]
            parameters = function["parameters"]
            required = list(parameters.get("required", []))
            arguments_template = cls._schema_fixture(
                parameters, fixture_values=fixture_values, fixture_rows=fixture_rows,
            )
            implementation = implementation_by_tool.get(function["name"], {})
            for mapping_name in ("selector", "changes", "values"):
                mapping = implementation.get(mapping_name)
                if not isinstance(mapping, dict):
                    continue
                for argument_name, column_name in mapping.items():
                    if (isinstance(arguments_template, dict)
                            and isinstance(arguments_template.get(argument_name), str)
                            and arguments_template[argument_name].startswith("任务输入中的")
                            and isinstance(column_name, str) and fixture_values.get(column_name)):
                        arguments_template[argument_name] = fixture_values[column_name][0]
            tool_cases.append({
                "case_id": f"{function['name']}.valid_shape",
                "tool_name": function["name"],
                "kind": "schema_and_business_smoke",
                "arguments_template": arguments_template,
                "expected": {"status_class": [200], "must_not_return": ["observation", "reward"]},
            })
            if required:
                tool_cases.append({
                    "case_id": f"{function['name']}.missing_required",
                    "tool_name": function["name"], "kind": "invalid_input",
                    "arguments": {}, "expected": {"status_class": [400, 422]},
                })
            if parameters.get("additionalProperties") is False:
                tool_cases.append({
                    "case_id": f"{function['name']}.unexpected_property",
                    "tool_name": function["name"], "kind": "invalid_input",
                    "arguments": {"__outer_invalid_property__": True}, "expected": {"status_class": [400, 422]},
                })
        table_names = [table.get("table_name") for table in data_tables]
        critical_fields = {
            table.get("table_name"): [column.get("name") for column in table.get("columns", [])]
            for table in data_tables
        }
        reward_cases = []
        for metric in metrics:
            mapping = metric.get("evaluator", {}).get("score_mapping", {})
            values = [value for value in mapping.values() if isinstance(value, (int, float)) and not isinstance(value, bool)]
            if not values:
                continue
            reward_cases.append({
                "case_id": f"{metric['id']}.pass",
                "metric_id": metric["id"],
                "expected_score": max(values),
                "fixture": "success",
            })
            reward_cases.append({
                "case_id": f"{metric['id']}.fail",
                "metric_id": metric["id"],
                "expected_score": min(values),
                "fixture": "failure",
            })
        argument_probes = [
            {"probe_id": f"{case['case_id']}.argument_sensitivity", "tool_name": case["tool_name"], "arguments": case["arguments_template"]}
            for case in tool_cases
            if case.get("kind") == "schema_and_business_smoke"
            and isinstance(case.get("arguments_template"), dict) and case["arguments_template"]
        ]
        executable_scenarios = [{
            "scenario_id": "runtime_reset_replay",
            "steps": [
                {"operation": "reset", "body": {"episode_id": "acceptance-episode", "seed": 17}, "expected_status": 200},
                {"operation": "observation", "expected_status": 200},
                {"operation": "replay", "expected_status": 200},
            ],
            "assertions": [{"path": "$.episode_id", "operator": "eq", "expected": "acceptance-episode"}],
        }, *[
            {
                "scenario_id": f"invalid_{case['case_id']}",
                "steps": [{"step_id": "invalid_call", "operation": "tool_call", "tool_name": case["tool_name"], "arguments": case.get("arguments", {}), "expected_status": case["expected"]["status_class"][0]}],
                "assertions": [{"source": "step:invalid_call", "path": "$.error", "operator": "exists", "expected": True}],
            }
            for case in tool_cases if case.get("kind") == "invalid_input"
        ]]
        # Match ManifestDataStore exactly: the runtime hashes the loaded table
        # mapping, not the richer generation-time schema/table documents.
        initial_tables = {
            str(table.get("table_name")): table.get("rows", [])
            for table in data_tables if isinstance(table, dict) and table.get("table_name")
        }
        initial_data_hash = hashlib.sha256(
            json.dumps(
                initial_tables, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
        return {
            "version": "1.0",
            "authority": "env_factory_outer_workflow",
            "task_goal": task_description.get("goal") or task_description.get("task"),
            "fixtures": {
                "data_manifest": data_manifest,
                "tables": table_names,
                "critical_fields": critical_fields,
                "initial_data_hash": initial_data_hash,
            },
            "scenarios": [
                {"scenario_id": "reset_isolation", "kind": "lifecycle", "steps": ["reset(seed=17)", "write_episode_a", "reset(seed=17)", "assert_episode_a_not_visible"]},
                {"scenario_id": "replay_integrity", "kind": "replay", "steps": ["reset(seed=17)", "execute_tool_cases", "get_replay", "assert_trace_hash"]},
                *[{
                    "scenario_id": f"key_step_{step['step_id']}",
                    "kind": "goal_critical",
                    "action_name": step["action_name"],
                    "required_for_goal": step["required_for_goal"],
                    "steps": [
                        "reset(seed=17)",
                        f"execute declared action {step['action_name']}",
                        "assert the executable success trajectory covers this action",
                    ],
                } for step in key_steps],
            ],
            "tool_cases": tool_cases,
            "argument_probes": argument_probes,
            "executable_scenarios": executable_scenarios,
            "invariants": [
                "tool response never contains observation or reward",
                "failed tool call does not partially mutate business data",
                "hidden truth is absent from public observation",
                "reward is computed only by explicit reward endpoint",
            ],
            "mutations": [
                {"mutation_id": f"mutate_{table}", "table": table, "strategy": "change_one_critical_field", "fields": critical_fields.get(table, [])}
                for table in table_names
            ],
            "reward_cases": reward_cases,
            "reward_formula": reward_formula,
            "mutation_tests": [
                "replace_tool_result_with_constant",
                "skip_business_write",
                "return_constant_reward",
                "ignore_tool_arguments",
                "bypass_trainer_auth",
            ],
            "actions": [{"name": action["name"], "preconditions": action.get("preconditions", []), "effects": action.get("effects", [])} for action in actions],
        }

    @staticmethod
    def _ground_acceptance_probes_from_success(
        contract: dict[str, Any], scenarios: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Reuse validated success-call values without carrying runtime refs into smoke probes."""
        success_calls: dict[str, dict[str, Any]] = {}
        for scenario in scenarios:
            if not isinstance(scenario, dict) or scenario.get("kind") != "goal_success":
                continue
            for step in scenario.get("steps", []):
                if (isinstance(step, dict) and step.get("operation") == "tool_call"
                        and isinstance(step.get("tool_name"), str)
                        and isinstance(step.get("arguments"), dict)):
                    success_calls.setdefault(step["tool_name"], step["arguments"])

        def concrete(value: Any, fallback: Any) -> Any:
            if isinstance(value, dict) and set(value) == {"$ref"}:
                return copy.deepcopy(fallback)
            if isinstance(value, dict) and isinstance(fallback, dict):
                return {
                    key: concrete(value[key], item) if key in value else copy.deepcopy(item)
                    for key, item in fallback.items()
                }
            if isinstance(value, list):
                if not value:
                    return copy.deepcopy(fallback)
                template = fallback if isinstance(fallback, list) else []
                return [
                    concrete(item, template[min(index, len(template) - 1)] if template else None)
                    for index, item in enumerate(value)
                ]
            return copy.deepcopy(value)

        grounded = copy.deepcopy(contract)
        for case in grounded.get("tool_cases", []):
            if (isinstance(case, dict) and case.get("kind") == "schema_and_business_smoke"
                    and case.get("tool_name") in success_calls):
                case["arguments_template"] = concrete(
                    success_calls[case["tool_name"]], case.get("arguments_template", {})
                )
        for probe in grounded.get("argument_probes", []):
            if isinstance(probe, dict) and probe.get("tool_name") in success_calls:
                probe["arguments"] = concrete(
                    success_calls[probe["tool_name"]], probe.get("arguments", {})
                )
        return grounded

    @staticmethod
    def _merge_acceptance_contract(candidate: Any, baseline: dict[str, Any]) -> dict[str, Any]:
        """Merge an LLM proposal onto the deterministic valid contract.

        The baseline supplies structural completeness.  Candidate fields are
        retained, so the LLM may add richer business assertions without being
        allowed to remove required scenario/tool/reward cases accidentally.
        """
        if not isinstance(candidate, dict):
            return dict(baseline)
        merged = dict(baseline)
        for key, value in candidate.items():
            if key not in {"scenarios", "tool_cases", "reward_cases", "argument_probes", "executable_scenarios"}:
                if isinstance(merged.get(key), dict) and isinstance(value, dict):
                    merged[key] = {**merged[key], **value}
                else:
                    merged[key] = value

        identity = {
            "scenarios": "scenario_id",
            "tool_cases": "case_id",
            "reward_cases": "case_id",
        }
        allowed_actions = {
            action.get("name") for action in baseline.get("actions", [])
            if isinstance(action, dict) and isinstance(action.get("name"), str)
        }
        allowed_actions.update(
            scenario.get("action_name") for scenario in baseline.get("scenarios", [])
            if isinstance(scenario, dict)
            and isinstance(scenario.get("action_name"), str)
        )

        def safe_overlay(base_item: Any, candidate_item: Any) -> dict[str, Any]:
            if not isinstance(base_item, dict):
                return dict(candidate_item) if isinstance(candidate_item, dict) else {}
            if not isinstance(candidate_item, dict):
                return dict(base_item)
            result = dict(base_item)
            for field, value in candidate_item.items():
                if value is None or value == "":
                    continue
                if field == "steps" and not isinstance(value, list):
                    continue
                if field == "expected" and not isinstance(value, dict):
                    continue
                result[field] = value
            return result

        for key, id_field in identity.items():
            base_items = baseline.get(key, [])
            candidate_items = candidate.get(key, [])
            if not isinstance(base_items, list):
                base_items = []
            if not isinstance(candidate_items, list):
                candidate_items = []
            by_id = {
                item.get(id_field): item
                for item in candidate_items
                if isinstance(item, dict) and isinstance(item.get(id_field), str) and item.get(id_field)
            }
            result: list[Any] = []
            used: set[int] = set()
            for index, base_item in enumerate(base_items):
                overlay = by_id.get(base_item.get(id_field)) if isinstance(base_item, dict) else None
                if overlay is not None:
                    for candidate_index, candidate_item in enumerate(candidate_items):
                        if candidate_item is overlay:
                            used.add(candidate_index)
                            break
                if overlay is None and index < len(candidate_items) and isinstance(candidate_items[index], dict):
                    overlay = candidate_items[index]
                    used.add(index)
                if isinstance(base_item, dict):
                    combined = safe_overlay(base_item, overlay)
                    # Positional enrichment must never replace the stable
                    # identity of an EnvFactory-owned baseline case.
                    combined[id_field] = base_item[id_field]
                    if key == "scenarios":
                        for field in ("kind", "action_name", "required_for_goal"):
                            if field in base_item:
                                combined[field] = base_item[field]
                    if key == "tool_cases":
                        # The executable request and status expectation are
                        # derived from validated tools and business fixtures.
                        # Model prose may enrich a case but cannot replace
                        # its authoritative input with a placeholder.
                        for field in ("tool_name", "kind", "arguments_template", "arguments", "expected"):
                            if field in base_item:
                                combined[field] = base_item[field]
                    result.append(combined)
                else:
                    result.append(overlay or base_item)
            for index, item in enumerate(candidate_items):
                if index in used or not isinstance(item, dict) or item in result:
                    continue
                if key == "scenarios" and not (
                    isinstance(item.get("scenario_id"), str) and item["scenario_id"]
                    and isinstance(item.get("kind"), str) and item["kind"]
                    and isinstance(item.get("steps"), list)
                ):
                    continue
                if (key == "scenarios" and item.get("kind") == "goal_critical"
                        and item.get("action_name") not in allowed_actions):
                    continue
                if key == "tool_cases" and not (
                    isinstance(item.get("case_id"), str) and item["case_id"]
                    and isinstance(item.get("tool_name"), str)
                    and isinstance(item.get("expected"), dict)
                ):
                    continue
                if key == "reward_cases" and not (
                    isinstance(item.get("case_id"), str) and item["case_id"]
                    and isinstance(item.get("metric_id"), str)
                    and isinstance(item.get("expected_score"), (int, float))
                    and not isinstance(item["expected_score"], bool)
                ):
                    continue
                if item.get(id_field) not in {
                    existing.get(id_field) for existing in result if isinstance(existing, dict)
                }:
                    result.append(item)
            merged[key] = result
        return merged

    @staticmethod
    def _validate_acceptance_contract(contract: Any, *, actions: list[dict[str, Any]], tools: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
        if not isinstance(contract, dict) or contract.get("authority") != "env_factory_outer_workflow":
            raise PipelineGenerationError("acceptance_contract must be EnvFactory-owned")
        for key in ("fixtures", "scenarios", "tool_cases", "argument_probes", "executable_scenarios", "invariants", "mutations", "reward_cases", "mutation_tests"):
            if not isinstance(contract.get(key), list | dict):
                raise PipelineGenerationError(f"acceptance_contract.{key} is invalid")
        if not contract["scenarios"] or (tools and not contract["tool_cases"]) or not contract["reward_cases"]:
            raise PipelineGenerationError("acceptance_contract requires scenarios, tool_cases and reward_cases")
        action_names = {action.get("name") for action in actions}
        tool_names = {tool.get("function", {}).get("name") for tool in tools}
        metric_by_id = {metric.get("id"): metric for metric in metrics}
        metric_ids = set(metric_by_id)
        for case in contract["tool_cases"]:
            if not isinstance(case, dict) or not case.get("case_id") or case.get("tool_name") not in tool_names or not isinstance(case.get("expected"), dict):
                raise PipelineGenerationError("acceptance_contract references unknown tool")
        for case in contract["reward_cases"]:
            if (
                not isinstance(case, dict) or not case.get("case_id")
                or case.get("metric_id") not in metric_ids
                or isinstance(case.get("expected_score"), bool)
                or not isinstance(case.get("expected_score"), (int, float))
            ):
                raise PipelineGenerationError("acceptance_contract references unknown metric")
            metric = metric_by_id[case["metric_id"]]
            mapping = metric.get("evaluator", {}).get("score_mapping", {})
            values = [
                value for value in mapping.values()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
            case_id = str(case["case_id"])
            if case_id.endswith(".pass"):
                allowed_scores = {max(values)} if values else set()
            elif case_id.endswith(".partial"):
                allowed_scores = set(values)
            else:
                allowed_scores = {min(values)} if values else set()
            if values and case["expected_score"] not in allowed_scores:
                raise PipelineGenerationError(
                    f"acceptance_contract reward case {case['case_id']} has reversed expected score"
                )
        for scenario in contract["scenarios"]:
            if not isinstance(scenario, dict) or not scenario.get("scenario_id") or not scenario.get("kind") or not isinstance(scenario.get("steps"), list):
                raise PipelineGenerationError("acceptance_contract scenario is incomplete")
            if scenario.get("kind") == "goal_critical" and scenario.get("action_name") not in action_names:
                raise PipelineGenerationError("acceptance_contract references unknown action")
        if not contract.get("invariants") or not contract.get("mutation_tests"):
            raise PipelineGenerationError("acceptance_contract requires invariants and mutation_tests")

    @staticmethod
    def _validate_executable_scenarios(
        scenarios: Any, *, tools: list[dict[str, Any]], noise_tools: list[dict[str, Any]],
        training_category: str | None = None,
        tool_implementations: list[dict[str, Any]] | None = None,
        semantic_goal: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(scenarios, list) or not scenarios:
            raise PipelineGenerationError("executable business scenarios must be a non-empty list")
        tool_schemas = {
            tool["function"]["name"]: tool["function"]["parameters"] for tool in tools
        }
        noise_names = {item.get("name") for item in noise_tools if isinstance(item, dict)}
        allowed_operations = {"dialogue_turn",
            "reset", "tool_call", "agent_response", "observation", "reward", "replay",
            "business_snapshot", "mutate_business_state",
        }
        allowed_operators = {
            "eq", "ne", "gt", "gte", "lt", "lte", "contains", "exists", "count_gte",
            "count_eq", "changed", "unchanged", "subset",
        }
        kinds: set[str] = set()
        ids: set[str] = set()
        implementation_by_tool = {
            item.get("tool_name"): item for item in (tool_implementations or [])
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }

        def validate_value(
            value: Any, schema: dict[str, Any], path: str, *, require_content: bool = True
        ) -> None:
            if isinstance(value, dict) and set(value) == {"$ref"}:
                if not isinstance(value["$ref"], str) or not value["$ref"].strip():
                    raise PipelineGenerationError(f"{path} has an invalid reference")
                return
            declared_types = schema.get("type")
            if isinstance(declared_types, list):
                if value is None and "null" in declared_types:
                    return
                kind = next((item for item in declared_types if item != "null"), None)
            else:
                kind = declared_types
            if kind == "array":
                if not isinstance(value, list) or (require_content and not value):
                    raise PipelineGenerationError(f"{path} must be a non-empty array")
                for item_index, item in enumerate(value):
                    validate_value(
                        item, schema.get("items", {}), f"{path}[{item_index}]",
                        require_content=require_content,
                    )
            elif kind == "object":
                if not isinstance(value, dict) or (require_content and not value):
                    raise PipelineGenerationError(f"{path} must be a non-empty object")
                properties = schema.get("properties", {})
                for required_name in schema.get("required", []):
                    if required_name not in value:
                        raise PipelineGenerationError(f"{path} misses required field {required_name}")
                for child_name, child_value in value.items():
                    if child_name in properties:
                        validate_value(
                            child_value, properties[child_name], f"{path}.{child_name}",
                            require_content=child_name in schema.get("required", []),
                        )
            elif kind == "string":
                if not isinstance(value, str) or (require_content and not value.strip()):
                    raise PipelineGenerationError(f"{path} must be a non-empty string")
                lowered = value.strip().lower()
                if any(marker in lowered for marker in (
                    "fixture-value", "placeholder", "todo", "sample-value", "test-value",
                    "任务输入中", "待提供", "请填写", "示例值",
                )):
                    raise PipelineGenerationError(f"{path} uses a placeholder value")
            elif kind == "integer":
                if isinstance(value, bool) or not isinstance(value, int):
                    raise PipelineGenerationError(f"{path} must be an integer")
            elif kind == "number":
                if not _finite_number(value):
                    raise PipelineGenerationError(f"{path} must be a finite number")
            elif kind == "boolean":
                if not isinstance(value, bool):
                    raise PipelineGenerationError(f"{path} must be a boolean")
        def references(value: Any) -> set[str]:
            if isinstance(value, dict):
                if set(value) == {"$ref"} and isinstance(value.get("$ref"), str):
                    return {value["$ref"]}
                return set().union(*(references(item) for item in value.values())) if value else set()
            if isinstance(value, list):
                return set().union(*(references(item) for item in value)) if value else set()
            return set()
        for index, scenario in enumerate(scenarios):
            if not isinstance(scenario, dict) or not isinstance(scenario.get("scenario_id"), str):
                raise PipelineGenerationError(f"executable_scenarios[{index}] is invalid")
            if scenario["scenario_id"] in ids or scenario.get("kind") not in {"goal_success", "goal_failure", "noise_selection", "counterfactual"}:
                raise PipelineGenerationError(f"executable_scenarios[{index}] id/kind is invalid")
            ids.add(scenario["scenario_id"])
            kinds.add(scenario["kind"])
            steps = scenario.get("steps")
            if not isinstance(steps, list) or not steps:
                raise PipelineGenerationError(f"executable_scenarios[{index}] requires steps")
            captures: set[str] = set()
            capture_shapes: dict[str, set[str]] = {}
            success_business_calls = 0
            has_dependency_edge = False
            for step in steps:
                if not isinstance(step, dict) or step.get("operation") not in allowed_operations:
                    raise PipelineGenerationError(f"executable_scenarios[{index}] has invalid operation")
                if step["operation"] == "tool_call":
                    name, arguments = step.get("tool_name"), step.get("arguments", {})
                    if name not in tool_schemas or not isinstance(arguments, dict):
                        raise PipelineGenerationError(f"executable_scenarios[{index}] references invalid tool")
                    if scenario["kind"] == "goal_success" and name in noise_names:
                        raise PipelineGenerationError("goal_success scenario cannot use noise tools")
                    if scenario["kind"] == "goal_success":
                        success_business_calls += 1
                        step_refs = references(arguments)
                        if step_refs:
                            if not step_refs <= captures:
                                raise PipelineGenerationError(
                                    f"executable_scenarios[{index}] references an uncaptured value"
                                )
                            has_dependency_edge = True
                    schema = tool_schemas[name]
                    if set(arguments) - set(schema.get("properties", {})):
                        raise PipelineGenerationError(f"executable_scenarios[{index}] has unknown tool arguments")
                    if set(schema.get("required", [])) - set(arguments):
                        raise PipelineGenerationError(f"executable_scenarios[{index}] misses required tool arguments")
                    if scenario["kind"] == "goal_success":
                        for argument_name, argument in arguments.items():
                            required_argument = argument_name in schema.get("required", [])
                            validate_value(
                                argument,
                                schema.get("properties", {}).get(argument_name, {}),
                                f"executable_scenarios[{index}].{name}.{argument_name}",
                                require_content=required_argument,
                            )
                            argument_schema = schema.get("properties", {}).get(argument_name, {})
                            if isinstance(argument, dict) and set(argument) == {"$ref"}:
                                projected = capture_shapes.get(argument["$ref"])
                                item_schema = argument_schema.get("items", {})
                                required_fields = set(item_schema.get("required", [])) if (
                                    argument_schema.get("type") == "array"
                                    and isinstance(item_schema, dict)
                                ) else set()
                                if projected is not None and not required_fields <= projected:
                                    missing = sorted(required_fields - projected)
                                    raise PipelineGenerationError(
                                        f"executable_scenarios[{index}] dependency into "
                                        f"{name}.{argument_name} misses projected fields: {missing}"
                                    )
                if step["operation"] == "agent_response" and (
                    not isinstance(step.get("content"), str) or not step["content"].strip()
                ):
                    raise PipelineGenerationError(f"executable_scenarios[{index}] has invalid agent response")
                capture = step.get("capture") or {}
                if not isinstance(capture, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in capture.items()):
                    raise PipelineGenerationError(f"executable_scenarios[{index}] capture is invalid")
                if step.get("operation") == "tool_call":
                    implementation = implementation_by_tool.get(step.get("tool_name"), {})
                    result_field = implementation.get("result_field")
                    projection = implementation.get("projection")
                    if isinstance(result_field, str) and isinstance(projection, list):
                        aliases = implementation.get("projection_aliases", {})
                        for variable, json_path in capture.items():
                            if json_path == f"$.{result_field}":
                                capture_shapes[variable] = {
                                    field for field in projection if isinstance(field, str)
                                }
                                if isinstance(aliases, dict):
                                    capture_shapes[variable].update(
                                        alias for alias in aliases if isinstance(alias, str)
                                    )
                captures.update(capture)
            if scenario["kind"] == "goal_success":
                if training_category == "direct_response" and success_business_calls:
                    raise PipelineGenerationError("direct_response success must not call business tools")
                if training_category == "simple_agentic" and success_business_calls != 1:
                    raise PipelineGenerationError("simple_agentic success requires exactly one business tool call")
                if training_category == "multi_step_agentic" and (
                    success_business_calls < 2 or not has_dependency_edge
                ):
                    raise PipelineGenerationError(
                        "multi_step_agentic success requires two business calls and a capture/$ref dependency"
                    )
                ContractCompiler._validate_success_scenario_goal_coverage(
                    scenario,
                    semantic_goal=semantic_goal,
                    tool_implementations=tool_implementations or [],
                )
            assertions = scenario.get("assertions", [])
            if not isinstance(assertions, list) or not assertions:
                raise PipelineGenerationError(f"executable_scenarios[{index}] requires assertions")
            if any(not isinstance(item, dict) or item.get("operator") not in allowed_operators for item in assertions):
                raise PipelineGenerationError(f"executable_scenarios[{index}] assertion is invalid")
            reward_assertions = [
                item for item in assertions
                if isinstance(item, dict) and item.get("path") == "$.reward"
            ]
            if scenario["kind"] == "goal_success" and not any(
                item.get("operator") == "gte"
                and isinstance(item.get("expected"), (int, float))
                and not isinstance(item.get("expected"), bool)
                and float(item["expected"]) >= 0.5
                for item in reward_assertions
            ):
                raise PipelineGenerationError("goal_success must assert reward >= 0.5")
            if scenario["kind"] == "goal_failure" and not any(
                item.get("operator") == "lte"
                and isinstance(item.get("expected"), (int, float))
                and not isinstance(item.get("expected"), bool)
                and float(item["expected"]) <= 0.2
                for item in reward_assertions
            ):
                raise PipelineGenerationError("goal_failure must assert reward <= 0.2")
            if scenario["kind"] == "noise_selection" and not any(
                item.get("operator") == "lte"
                and isinstance(item.get("expected"), (int, float))
                and not isinstance(item.get("expected"), bool)
                and float(item["expected"]) <= 0
                for item in reward_assertions
            ):
                raise PipelineGenerationError("noise_selection must assert non-positive reward")
        if not {"goal_success", "goal_failure"} <= kinds:
            raise PipelineGenerationError("executable scenarios require goal_success and goal_failure")
        if noise_names and "noise_selection" not in kinds:
            raise PipelineGenerationError("tasks with noise tools require a noise_selection scenario")

    @staticmethod
    def _validate_success_scenario_goal_coverage(
        scenario: dict[str, Any], *, semantic_goal: dict[str, Any] | None,
        tool_implementations: list[dict[str, Any]],
    ) -> None:
        """Prove that the success trace can establish every declared state goal."""
        predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        if not predicates:
            return
        implementations = {
            item.get("tool_name"): item for item in tool_implementations
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }
        calls = [
            step for step in scenario.get("steps", [])
            if isinstance(step, dict) and step.get("operation") == "tool_call"
        ]
        used: set[int] = set()

        def matches(call: dict[str, Any], predicate: dict[str, Any]) -> bool:
            implementation = implementations.get(call.get("tool_name"), {})
            operation = implementation.get("operation")
            expected_operation = "delete" if predicate.get("count") == 0 else None
            if implementation.get("table") != predicate.get("table"):
                return False
            if expected_operation and operation != expected_operation:
                return False
            if not expected_operation and operation not in {"update", "insert"}:
                return False
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict):
                return False
            selector = implementation.get("selector", {})
            changes = implementation.get("changes", {})
            values = implementation.get("values", {})

            def establishes(expected: dict[str, Any], mappings: list[Any], *, allow_ref: bool) -> bool:
                for column, value in expected.items():
                    candidates = [
                        argument for mapping in mappings if isinstance(mapping, dict)
                        for argument, mapped_column in mapping.items()
                        if mapped_column == column
                    ]
                    if not candidates:
                        return False
                    actual = arguments.get(candidates[0])
                    if actual == value:
                        continue
                    if allow_ref and isinstance(actual, dict) and set(actual) == {"$ref"}:
                        continue
                    return False
                return True

            return (
                establishes(predicate.get("where", {}), [selector, values], allow_ref=True)
                and establishes(predicate.get("values", {}), [changes, values], allow_ref=False)
            )

        for index, predicate in enumerate(predicates):
            if not isinstance(predicate, dict):
                continue
            match = next((
                call_index for call_index, call in enumerate(calls)
                if call_index not in used and matches(call, predicate)
            ), None)
            if match is None:
                raise PipelineGenerationError(
                    f"goal_success does not establish row_predicates[{index}]"
                )
            used.add(match)

    @classmethod
    def _build_business_scenario_baseline(
        cls,
        *,
        task_description: dict[str, Any],
        tools: list[dict[str, Any]],
        noise_tools: list[dict[str, Any]],
        data_tables: list[dict[str, Any]] | None = None,
        tool_implementations: list[dict[str, Any]] | None = None,
        semantic_goal: dict[str, Any] | None = None,
        training_category: str | None = None,
        success_content: str | None = None,
    ) -> list[dict[str, Any]]:
        noise_names = {
            item.get("name") for item in noise_tools if isinstance(item, dict)
        }
        business_tools = [
            tool for tool in tools if tool.get("function", {}).get("name") not in noise_names
        ]
        fixture_values, fixture_rows = cls._business_fixture_context(
            data_tables or [], task_description,
        )
        public_text = json.dumps({
            key: task_description.get(key)
            for key in ("task", "goal", "public_input", "requirements")
        }, ensure_ascii=False)
        public_numbers = {
            Decimal(token)
            for token in re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", public_text)
        }
        success_steps: list[dict[str, Any]] = [
            {"operation": "reset", "body": {"episode_id": "goal-success", "seed": 17}, "expected_status": 200}
        ]
        implementation_by_tool = {
            item.get("tool_name"): item for item in (tool_implementations or [])
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }
        prior_outputs: list[tuple[str, list[str]]] = []
        dependency_added = False
        tool_step_index = 0
        goal_predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        for tool in business_tools:
            function = tool["function"]
            fixture_arguments = cls._schema_fixture(
                function["parameters"], fixture_values=fixture_values,
                fixture_rows=fixture_rows,
            )
            properties = function["parameters"].get("properties", {})
            if isinstance(properties, dict):
                for argument_name, argument_schema in properties.items():
                    if (
                        argument_name in fixture_values
                        or not isinstance(argument_schema, dict)
                        or argument_schema.get("type") not in ("integer", "number")
                        or fixture_arguments.get(argument_name) != 1
                    ):
                        continue
                    description_text = str(argument_schema.get("description", ""))
                    examples = {
                        Decimal(token) for token in re.findall(
                            r"(?:例如|示例|如)\s*[:：]?\s*(-?\d+(?:\.\d+)?)",
                            description_text,
                        )
                    }
                    grounded = examples & public_numbers
                    if len(grounded) == 1:
                        value = grounded.pop()
                        if argument_schema["type"] == "number" or value == int(value):
                            fixture_arguments[argument_name] = (
                                float(value) if argument_schema["type"] == "number"
                                else int(value)
                            )
            implementation = implementation_by_tool.get(function["name"], {})
            # Business tool argument names often differ from source table columns.
            # Bind schema placeholders through the declared selector before asking
            # the model to invent a value for an executable acceptance trace.
            selector_mapping = implementation.get("selector", {})
            for argument_name, column_name in (selector_mapping.items()
                                               if isinstance(selector_mapping, dict) else []):
                if (argument_name in fixture_arguments and isinstance(column_name, str)
                        and isinstance(fixture_arguments[argument_name], str)
                        and fixture_arguments[argument_name].startswith("任务输入中的")
                        and fixture_values.get(column_name)):
                    fixture_arguments[argument_name] = fixture_values[column_name][0]
            predicates = [
                item for item in goal_predicates
                if isinstance(item, dict) and item.get("table") == implementation.get("table")
            ]
            argument_variants = [fixture_arguments]
            if implementation.get("operation") in {"insert", "update", "delete"} and predicates:
                argument_variants = []
                for predicate in predicates:
                    arguments = copy.deepcopy(fixture_arguments)
                    for mapping_name, source in (
                        ("selector", predicate.get("where", {})),
                        ("changes", predicate.get("values", {})),
                        ("values", {**predicate.get("where", {}), **predicate.get("values", {})}),
                    ):
                        mapping = implementation.get(mapping_name, {})
                        if isinstance(mapping, dict) and isinstance(source, dict):
                            for argument, column in mapping.items():
                                if column in source:
                                    arguments[argument] = source[column]
                    argument_variants.append(arguments)
            for arguments in argument_variants:
                if training_category == "multi_step_agentic" and prior_outputs and not dependency_added:
                    required = function["parameters"].get("required", [])
                    properties = function["parameters"].get("properties", {})
                    argument_fields = {
                        argument: field
                        for mapping_name in ("selector", "values", "changes")
                        for argument, field in (
                            implementation.get(mapping_name, {}).items()
                            if isinstance(implementation.get(mapping_name), dict) else ()
                        )
                        if isinstance(argument, str) and isinstance(field, str)
                    }
                    dependency = cls._match_baseline_dependency(
                        required=required, properties=properties, prior_outputs=prior_outputs,
                        argument_fields=argument_fields,
                    )
                    if dependency is not None:
                        argument_name, variable_name, json_path = dependency
                        arguments[argument_name] = {"$ref": variable_name}
                        prior_step = next(
                            step for step in reversed(success_steps)
                            if step.get("operation") == "tool_call"
                        )
                        prior_step.setdefault("capture", {})[variable_name] = json_path
                        dependency_added = True
                tool_step_index += 1
                success_steps.append({
                    "step_id": f"business_tool_{tool_step_index}",
                    "operation": "tool_call",
                    "tool_name": function["name"],
                    "arguments": arguments,
                    "expected_status": 200,
                })
            result_field = implementation.get("result_field")
            projection = implementation.get("projection", [])
            if isinstance(result_field, str) and result_field.strip():
                fields = [item for item in projection if isinstance(item, str)]
                aliases = implementation.get("projection_aliases", {})
                if isinstance(aliases, dict):
                    fields.extend(alias for alias in aliases if isinstance(alias, str))
                prior_outputs.append((result_field, fields))
        success_steps.extend([
            {
                "step_id": "final_answer",
                "operation": "agent_response",
                "content": str(success_content or task_description.get("expected_result") or task_description.get("goal") or "已完成任务并给出依据。"),
                "expected_status": 200,
            },
            {"step_id": "success_reward", "operation": "reward", "expected_status": 200},
        ])
        scenarios: list[dict[str, Any]] = [{
            "scenario_id": "deterministic_goal_success",
            "kind": "goal_success",
            "steps": success_steps,
            "assertions": [
                {"source": "step:success_reward", "path": "$.reward", "operator": "gte", "expected": 0.6}
            ],
        }, {
            "scenario_id": "deterministic_goal_failure",
            "kind": "goal_failure",
            "steps": [
                {"operation": "reset", "body": {"episode_id": "goal-failure", "seed": 17}, "expected_status": 200},
                {"step_id": "failure_reward", "operation": "reward", "expected_status": 200},
            ],
            "assertions": [
                {"source": "step:failure_reward", "path": "$.reward", "operator": "lte", "expected": 0.2}
            ],
        }]
        if noise_tools:
            noise_name = str(noise_tools[0]["name"])
            noise_tool = next(tool for tool in tools if tool.get("function", {}).get("name") == noise_name)
            scenarios.append({
                "scenario_id": "deterministic_noise_selection",
                "kind": "noise_selection",
                "steps": [
                    {"operation": "reset", "body": {"episode_id": "noise-selection", "seed": 17}, "expected_status": 200},
                    {
                        "step_id": "noise_call", "operation": "tool_call", "tool_name": noise_name,
                        "arguments": cls._schema_fixture(noise_tool["function"]["parameters"]), "expected_status": 200,
                    },
                    {"step_id": "noise_reward", "operation": "reward", "expected_status": 200},
                ],
                "assertions": [
                    {"source": "step:noise_reward", "path": "$.reward", "operator": "lte", "expected": 0}
                ],
            })
        return scenarios

    @staticmethod
    def _match_baseline_dependency(
        *, required: list[Any], properties: dict[str, Any],
        prior_outputs: list[tuple[str, list[str]]],
        argument_fields: dict[str, str] | None = None,
    ) -> tuple[str, str, str] | None:
        """Find a schema-compatible, fixture-backed edge for a multi-tool trace."""
        for result_field, projection in reversed(prior_outputs):
            for argument_name in required:
                schema = properties.get(argument_name, {})
                if not isinstance(argument_name, str) or schema.get("type") != "array":
                    continue
                item_schema = schema.get("items", {})
                if isinstance(item_schema, dict) and item_schema.get("type") in {
                    "string", "integer", "number",
                }:
                    candidates = [
                        (argument_fields or {}).get(argument_name),
                        ContractCompiler._fixture_singular_name(argument_name),
                    ]
                    matched = next(
                        (field for field in candidates
                         if isinstance(field, str) and field in projection), None
                    )
                    if matched is not None:
                        variable = f"upstream_{argument_name}"
                        return argument_name, variable, f"$.{result_field}[*].{matched}"
                if not isinstance(item_schema, dict) or item_schema.get("type") != "object":
                    continue
                item_required = item_schema.get("required", [])
                if (not isinstance(item_required, list)
                        or any(not isinstance(field, str) for field in item_required)
                        or not set(item_required) <= set(projection)):
                    continue
                variable = f"upstream_{result_field}"
                return argument_name, variable, f"$.{result_field}"
            for argument_name in required:
                if not isinstance(argument_name, str):
                    continue
                if argument_name in projection:
                    variable = f"upstream_{argument_name}"
                    return argument_name, variable, f"$.{result_field}[0].{argument_name}"
                source_field = (argument_fields or {}).get(argument_name)
                schema = properties.get(argument_name, {})
                if (source_field in projection and isinstance(schema, dict)
                        and schema.get("type") in {"string", "integer", "number", "boolean"}):
                    variable = f"upstream_{argument_name}"
                    return argument_name, variable, f"$.{result_field}[0].{source_field}"
        return None

    @staticmethod
    def _select_success_response_fixture(
        *, task_description: dict[str, Any]
    ) -> str:
        return str(
            task_description.get("expected_result")
            or task_description.get("goal")
            or "已完成任务并给出依据。"
        )

    @staticmethod
    def _compile_business_scenario_structure(
        proposals: list[dict[str, Any]], baseline: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Models fill business calls; platform owns lifecycle and assertions."""
        import copy
        result = copy.deepcopy(baseline)
        success = next((item for item in proposals if item.get("kind") == "goal_success"), None)
        if success is None:
            return result
        calls = [copy.deepcopy(step) for step in success.get("steps", [])
                 if isinstance(step, dict) and step.get("operation") == "tool_call"]
        def references(value: Any) -> set[str]:
            if isinstance(value, dict):
                if set(value) == {"$ref"} and isinstance(value["$ref"], str):
                    return {value["$ref"]}
                return set().union(*(references(item) for item in value.values()))
            if isinstance(value, list):
                return set().union(*(references(item) for item in value))
            return set()
        for index, step in enumerate(calls):
            capture = step.get("capture")
            if not isinstance(capture, dict):
                continue
            used = set().union(*(references(later.get("arguments", {}))
                                 for later in calls[index + 1:]))
            retained = {name: path for name, path in capture.items() if name in used}
            if retained:
                step["capture"] = retained
            else:
                step.pop("capture", None)
        compiled = next(item for item in result if item.get("kind") == "goal_success")
        for index, step in enumerate(calls):
            step["step_id"] = f"business_tool_{index + 1}"
            step["expected_status"] = 200
        compiled["steps"] = [compiled["steps"][0], *calls, *compiled["steps"][-2:]]
        return result

    @staticmethod
    def _apply_baseline_dependency(
        scenarios: list[dict[str, Any]], baseline: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fill a missing read-only data edge when the two tool sequences agree."""
        import copy

        result = copy.deepcopy(scenarios)
        success = next((item for item in result if item.get("kind") == "goal_success"), None)
        baseline_success = next((item for item in baseline if item.get("kind") == "goal_success"), None)
        if success is None or baseline_success is None:
            return result
        calls = [step for step in success.get("steps", []) if step.get("operation") == "tool_call"]
        baseline_calls = [step for step in baseline_success.get("steps", [])
                          if step.get("operation") == "tool_call"]
        if ([step.get("tool_name") for step in calls]
                != [step.get("tool_name") for step in baseline_calls]):
            return result
        for index, (call, original) in enumerate(zip(calls, baseline_calls)):
            baseline_arguments = original.get("arguments", {})
            arguments = call.get("arguments", {})
            if not isinstance(arguments, dict) or not isinstance(baseline_arguments, dict):
                continue
            for name, value in baseline_arguments.items():
                if not (isinstance(value, dict) and set(value) == {"$ref"}
                        and isinstance(value["$ref"], str)):
                    continue
                if isinstance(arguments.get(name), dict) and set(arguments[name]) == {"$ref"}:
                    continue
                variable = value["$ref"]
                source = next((source for source, prior in zip(calls[:index], baseline_calls[:index])
                               if variable in prior.get("capture", {})), None)
                original_source = next((prior for prior in baseline_calls[:index]
                                        if variable in prior.get("capture", {})), None)
                if source is None or original_source is None:
                    continue
                path = original_source["capture"][variable]
                existing = source.setdefault("capture", {})
                if variable in existing and existing[variable] != path:
                    continue
                existing[variable] = path
                arguments[name] = copy.deepcopy(value)
        return result

    @staticmethod
    def _repair_business_scenario_arguments(
        proposals: list[dict[str, Any]], *, baseline: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Absorb malformed argument serialization without changing a valid plan.

        Tool schemas and fixture-backed baseline calls are already validated by
        the platform.  We use them only to fill missing, empty, or wrong-typed
        values.  Valid model values and capture references remain authoritative.
        """
        import copy

        schemas = {
            item.get("function", {}).get("name"): item.get("function", {}).get("parameters", {})
            for item in tools if isinstance(item, dict)
        }
        baseline_success = next(
            (item for item in baseline if isinstance(item, dict) and item.get("kind") == "goal_success"),
            {},
        )
        baseline_calls = [
            step for step in baseline_success.get("steps", []) if isinstance(step, dict)
            and step.get("operation") == "tool_call"
        ]
        fallback_by_name: dict[str, dict[str, Any]] = {}
        repeated_names: set[str] = set()
        for step in baseline_calls:
            name = step.get("tool_name")
            if not isinstance(name, str):
                continue
            if name in fallback_by_name:
                repeated_names.add(name)
            elif isinstance(step.get("arguments"), dict):
                fallback_by_name[name] = step["arguments"]
        for name in repeated_names:
            fallback_by_name.pop(name, None)

        def repair(value: Any, schema: dict[str, Any], fallback: Any, field_name: str = "") -> Any:
            if isinstance(value, dict) and set(value) == {"$ref"}:
                reference = value.get("$ref")
                if isinstance(reference, str) and reference.strip():
                    return value
            kind = schema.get("type")
            if isinstance(kind, list):
                if value is None and "null" in kind:
                    return None
                kind = next((item for item in kind if item != "null"), None)
            if kind == "object":
                properties = schema.get("properties", {})
                current = value if isinstance(value, dict) else {}
                base = fallback if isinstance(fallback, dict) else {}
                result = {
                    name: repair(child, properties.get(name, {}), base.get(name), name)
                    for name, child in current.items() if name in properties
                }
                for name in schema.get("required", []):
                    if name not in result:
                        result[name] = repair(None, properties.get(name, {}), base.get(name), name)
                return result
            if kind == "array":
                if not isinstance(value, list) or not value:
                    if not isinstance(fallback, list) or not fallback:
                        fallback = ContractCompiler._schema_fixture(
                            schema, field_name=field_name
                        )
                    return copy.deepcopy(fallback)
                item_schema = schema.get("items", {})
                fallback_items = fallback if isinstance(fallback, list) else []
                return [
                    repair(item, item_schema, fallback_items[min(index, len(fallback_items) - 1)] if fallback_items else None, field_name)
                    for index, item in enumerate(value)
                ]
            if kind == "string":
                if isinstance(value, str) and value.strip() and not any(
                    marker in value.strip().lower() for marker in (
                        "fixture-value", "placeholder", "todo", "sample-value", "test-value",
                        "任务输入中", "待提供", "请填写", "示例值",
                    )
                ):
                    return value
                if not isinstance(fallback, str) or not fallback.strip():
                    fallback = ContractCompiler._schema_fixture(
                        schema, field_name=field_name
                    )
                return copy.deepcopy(fallback)
            if kind in {"integer", "number"}:
                valid = (
                    isinstance(value, int) and not isinstance(value, bool)
                    if kind == "integer" else _finite_number(value)
                )
                if valid:
                    return value
                if isinstance(fallback, dict) and set(fallback) == {"$ref"}:
                    return copy.deepcopy(fallback)
                valid_fallback = (
                    isinstance(fallback, int) and not isinstance(fallback, bool)
                    if kind == "integer" else _finite_number(fallback)
                )
                if not valid_fallback:
                    fallback = ContractCompiler._schema_fixture(
                        schema, field_name=field_name,
                    )
                return copy.deepcopy(fallback)
            if kind == "boolean":
                return value if isinstance(value, bool) else copy.deepcopy(fallback)
            return value if value is not None else copy.deepcopy(fallback)

        repaired = copy.deepcopy(proposals)
        for scenario in repaired:
            if not isinstance(scenario, dict) or scenario.get("kind") != "goal_success":
                continue
            call_index = 0
            for step in scenario.get("steps", []):
                if not isinstance(step, dict) or step.get("operation") != "tool_call":
                    continue
                name = step.get("tool_name")
                schema = schemas.get(name)
                if not isinstance(schema, dict):
                    call_index += 1
                    continue
                baseline_step = baseline_calls[call_index] if call_index < len(baseline_calls) else {}
                fallback = (
                    baseline_step.get("arguments", {})
                    if baseline_step.get("tool_name") == name
                    else fallback_by_name.get(name, {})
                )
                step["arguments"] = repair(
                    step.get("arguments"), schema, fallback
                )
                call_index += 1
        return repaired

    @staticmethod
    def _normalize_executable_scenarios(
        scenarios: Any, *, success_content: str = ""
    ) -> list[dict[str, Any]]:
        """Canonicalize business scenario identity without changing its semantics."""
        if not isinstance(scenarios, list):
            return []
        aliases = {
            "success": "goal_success",
            "successful": "goal_success",
            "goal-success": "goal_success",
            "failure": "goal_failure",
            "failed": "goal_failure",
            "goal-failure": "goal_failure",
            "noise": "noise_selection",
            "noise-selection": "noise_selection",
            "counter-factual": "counterfactual",
        }
        allowed = {"goal_success", "goal_failure", "noise_selection", "counterfactual"}
        normalized: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        counts: dict[str, int] = {}
        for raw in scenarios:
            if not isinstance(raw, dict):
                continue
            item = dict(raw)
            raw_kind = item.get("kind")
            kind = aliases.get(str(raw_kind).strip().lower(), raw_kind)
            scenario_id = item.get("scenario_id")
            identity = str(scenario_id or "").lower()
            if kind not in allowed:
                if "success" in identity:
                    kind = "goal_success"
                elif "fail" in identity:
                    kind = "goal_failure"
                elif "noise" in identity:
                    kind = "noise_selection"
                elif "counter" in identity or "mutation" in identity:
                    kind = "counterfactual"
                else:
                    # Models sometimes echo lifecycle/replay entries from the
                    # baseline. They are already retained by the outer
                    # contract and are not business success/failure scenarios.
                    continue
            item["kind"] = kind
            counts[kind] = counts.get(kind, 0) + 1
            base_id = str(scenario_id).strip() if isinstance(scenario_id, str) else ""
            if not base_id:
                base_id = f"{kind}_{counts[kind]}"
            unique_id = base_id
            suffix = 2
            while unique_id in used_ids:
                unique_id = f"{base_id}_{suffix}"
                suffix += 1
            item["scenario_id"] = unique_id
            if kind == "goal_success" and success_content.strip():
                steps = item.get("steps")
                if isinstance(steps, list):
                    for step in steps:
                        if (
                            isinstance(step, dict)
                            and step.get("operation") == "agent_response"
                            and (
                                not isinstance(step.get("content"), str)
                                or not step["content"].strip()
                            )
                        ):
                            step["content"] = success_content.strip()
            used_ids.add(unique_id)
            normalized.append(item)
        return normalized

    @staticmethod
    def _fill_tool_schema_descriptions(tools: list[Any]) -> None:
        """Fill cosmetic JSON Schema descriptions without another LLM call."""
        def fill_schema(schema: Any, label: str) -> None:
            if not isinstance(schema, dict):
                return
            if not isinstance(schema.get("description"), str) or not schema["description"].strip():
                schema["description"] = f"{label} 的输入值。"
            properties = schema.get("properties")
            if isinstance(properties, dict):
                for name, child in properties.items():
                    fill_schema(child, str(name))
            items = schema.get("items")
            if isinstance(items, dict):
                fill_schema(items, f"{label} 数组元素")

        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict):
                continue
            name = function.get("name") if isinstance(function.get("name"), str) else "工具"
            if not isinstance(function.get("description"), str) or not function["description"].strip():
                function["description"] = f"执行 {name} 操作。"
            fill_schema(function.get("parameters"), f"{name} 参数")

    @staticmethod
    def _validate_noise_tool_audit(decisions: Any, *, noise_names: set[str]) -> set[str]:
        if not isinstance(decisions, list):
            raise PipelineGenerationError("noise tool audit must return decisions")
        seen: set[str] = set()
        unsafe: set[str] = set()
        for index, decision in enumerate(decisions):
            if not isinstance(decision, dict):
                raise PipelineGenerationError(f"noise audit decisions[{index}] must be an object")
            name = decision.get("name")
            if name not in noise_names or name in seen:
                raise PipelineGenerationError("noise audit references an unknown or duplicate tool")
            if not isinstance(decision.get("is_safe_noise"), bool):
                raise PipelineGenerationError("noise audit requires boolean is_safe_noise")
            if not isinstance(decision.get("reason"), str) or not decision["reason"].strip():
                raise PipelineGenerationError("noise audit requires reason")
            if not decision["is_safe_noise"]:
                unsafe.add(str(name))
            seen.add(str(name))
        if seen != noise_names:
            raise PipelineGenerationError(
                f"noise audit must cover every noise tool; missing={sorted(noise_names - seen)}"
            )
        return unsafe

    @staticmethod
    def _build_noise_fixture(function: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Build a minimal exact-match fixture from a validated noise schema."""
        properties = function.get("parameters", {}).get("properties", {})
        mapping = {name: name for name in properties}
        rows: list[dict[str, Any]] = []
        for index in range(2):
            row: dict[str, Any] = {}
            for name, schema in properties.items():
                value = ContractCompiler._schema_fixture(schema, field_name=name)
                if isinstance(value, str) and not schema.get("enum"):
                    value = f"{value}-{index + 1}"
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    value += index
                row[name] = value
            row["result"] = f"{function.get('name', 'noise_lookup')} 的合成参考结果 {index + 1}"
            rows.append(row)
        return rows, mapping

    @staticmethod
    def _render_data_document(tables: list[dict[str, Any]]) -> str:
        """Describe the actual reset fixture without a second source of truth."""
        lines = [
            "# 业务数据说明", "",
            "以下各表及行是每个 episode reset 后、Agent 执行前的基线状态；目标终态由任务契约与实际操作决定。",
            "运行时按 manifest 加载 schemas/<表名>.json 和 rows/<表名>.jsonl，并为每个 episode 保持独立副本。",
            "",
        ]
        for table in tables:
            name = table["table_name"]
            lines.extend([f"## {name}", "", str(table.get("description") or "业务数据表"), ""])
            lines.extend([
                f"- 初始行数：{len(table.get('rows', []))}",
                f"- 主键：{json.dumps(table.get('primary_key', []), ensure_ascii=False)}",
                f"- 外键：{json.dumps(table.get('foreign_keys', []), ensure_ascii=False)}",
                f"- 索引：{json.dumps(table.get('indexes', []), ensure_ascii=False)}",
                f"- 约束：{json.dumps(table.get('constraints', []), ensure_ascii=False)}",
                "", "| 字段 | 类型 | 可空 | 说明 |", "| --- | --- | --- | --- |",
            ])
            for column in table.get("columns", []):
                if not isinstance(column, dict):
                    continue
                description = str(column.get("description") or "").replace("|", "\\|").replace("\n", " ")
                lines.append(
                    f"| {column.get('name', '')} | {column.get('type', '')} | "
                    f"{'是' if column.get('nullable') else '否'} | {description} |"
                )
            lines.extend(["", "初始记录以对应的 `rows/<表名>.jsonl` 文件为准。", ""])
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _validate_noise_tool_safety(tools: list[Any]) -> None:
        """Noise tools must not claim real-world writes the shared runtime cannot perform."""
        mutation_name_prefixes = (
            "book_", "reserve_", "create_", "update_", "delete_", "submit_", "send_",
            "purchase_", "order_",
        )
        read_only_name_prefixes = (
            "search_", "lookup_", "get_", "list_", "find_", "estimate_",
            "calculate_", "check_", "convert_", "preview_", "inspect_",
        )
        embedded_mutation_name = re.compile(
            r"(?:^|_)(?:book|reserve|create|update|delete|submit|send|purchase|order)(?:_|$)"
        )
        # Descriptions often mention read-only fields such as "booking status" or
        # "预订信息".  Match an explicit operation claim instead of rejecting any
        # occurrence of a mutation-related noun.
        mutation_description_patterns = (
            r"\b(?:book|reserve|create|update|delete|submit|send|purchase|order)s?\b"
            r"\s+(?:a|an|the|new|user|customer|external|real-world)?\s*"
            r"(?:booking|reservation|record|request|message|email|order|purchase|resource)",
            r"(?:执行|进行|发起|完成)(?:实际|外部|真实)?"
            r"(?:预订|预约|创建|修改|更新|删除|提交|发送|购买|下单)",
            r"(?:预订|预约|创建|修改|更新|删除|提交|发送|购买|下单)"
            r"(?:场地|座位|记录|请求|消息|邮件|订单|商品|资源)",
        )
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = str(function.get("name", "")).lower()
            description = str(function.get("description", "")).lower()
            if name.startswith(read_only_name_prefixes) and not embedded_mutation_name.search(name):
                claims_mutation = False
            else:
                claims_mutation = name.startswith(mutation_name_prefixes) or any(
                    re.search(pattern, description, re.I)
                    for pattern in mutation_description_patterns
                )
            if claims_mutation:
                raise PipelineGenerationError(
                    f"noise tool {function.get('name')} claims a side effect; use a read-only distractor"
                )

    @staticmethod
    def _validate_tools(tools: list[Any]) -> None:
        names: set[str] = set()
        for index, tool in enumerate(tools):
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(function, dict):
                raise PipelineGenerationError(f"tools[{index}] is not an OpenAI function tool")
            name = function.get("name")
            if not isinstance(name, str) or not name or name in names:
                raise PipelineGenerationError(f"tools[{index}].function.name is invalid or duplicated")
            if re.fullmatch(
                r"(?:noise_)?tool_?\d*|function_?\d*|semantic_function_name_?\d*",
                name,
                flags=re.IGNORECASE,
            ):
                raise PipelineGenerationError(
                    f"tool {name} uses a placeholder name; use a concrete semantic function name"
                )
            if not isinstance(function.get("description"), str) or not function["description"].strip():
                raise PipelineGenerationError(f"tool {name} has no description")
            parameters = function.get("parameters")
            if not isinstance(parameters, dict) or parameters.get("type") != "object":
                raise PipelineGenerationError(f"tool {name} parameters must be an object schema")
            required = parameters.get("required", [])
            properties = parameters.get("properties", {})
            if not isinstance(properties, dict) or not isinstance(required, list) or any(item not in properties for item in required):
                raise PipelineGenerationError(f"tool {name} has invalid properties/required")
            forbidden = {"acceptance_evidence_id", "acceptance_message_id", "ground_truth", "user_accepted"}
            if forbidden & set(properties):
                raise PipelineGenerationError(f"tool {name} exposes forbidden internal input")
            def check_properties(items: dict[str, Any], prefix: str) -> None:
                for property_name, schema in items.items():
                    if not isinstance(schema, dict) or not isinstance(schema.get("description"), str) or not schema["description"].strip():
                        raise PipelineGenerationError(f"tool {name}.{prefix}{property_name} requires a description")
                    if schema.get("type") == "object":
                        nested = schema.get("properties", {})
                        if not isinstance(nested, dict):
                            raise PipelineGenerationError(f"tool {name}.{prefix}{property_name} has invalid nested properties")
                        check_properties(nested, prefix + property_name + ".")
                    if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
                        check_properties({"[]": schema["items"]}, prefix + property_name)
            check_properties(properties, "")
            names.add(name)

    @staticmethod
    def _missing_insert_columns(spec: dict[str, Any], tables: list[dict[str, Any]]) -> list[str]:
        """Only Agent-sourced columns must have declarative insert mappings."""
        from ..sandbox_runtime import ManifestDataStore

        if spec.get("operation") != "insert":
            return []
        table = next((item for item in tables if item.get("table_name") == spec.get("table")), None)
        if table is None:
            return []
        columns = {item.get("name") for item in table.get("columns", []) if isinstance(item, dict)}
        values = spec.get("values")
        supplied = set(values.values()) if isinstance(values, dict) else set()
        platform_fields = ManifestDataStore.platform_insert_fields(table)
        return sorted(name for name in columns - supplied - platform_fields if isinstance(name, str))

    @staticmethod
    def _complete_insert_values_from_tool(
        spec: Any, *, tools: list[dict[str, Any]], tables: list[dict[str, Any]],
    ) -> Any:
        """Fill exact-name insert mappings; reject an impossible tool surface early."""
        if not isinstance(spec, dict) or spec.get("operation") != "insert":
            return spec
        missing = ContractCompiler._missing_insert_columns(spec, tables)
        if not missing:
            return spec
        function = next((
            item.get("function") for item in tools
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
            and item["function"].get("name") == spec.get("tool_name")
        ), {})
        parameter_schema = function.get("parameters")
        parameters = parameter_schema.get("properties") if isinstance(parameter_schema, dict) else None
        values = spec.get("values")
        if not isinstance(parameters, dict) or not isinstance(values, dict):
            return spec
        raw_required = parameter_schema.get("required")
        required_arguments = set(raw_required) & set(parameters) if isinstance(raw_required, list) else set()
        unused_arguments = required_arguments - set(values)
        same_name = set(missing) & unused_arguments
        completed = copy.deepcopy(spec)
        completed["values"].update({name: name for name in same_name})
        missing = ContractCompiler._missing_insert_columns(completed, tables)
        if missing and not (unused_arguments - same_name):
            raise PipelineGenerationError(
                "declarative insert cannot supply storage columns from tool inputs: "
                f"tool={spec.get('tool_name')} table={spec.get('table')} missing={missing}"
            )
        return completed

    @staticmethod
    def _optional_mutation_arguments(
        spec: dict[str, Any], tools: list[dict[str, Any]],
    ) -> list[str]:
        """Mutation mappings index their arguments unconditionally at runtime."""
        if spec.get("operation") not in {"insert", "update", "delete"}:
            return []
        function = next((
            item.get("function") for item in tools
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
            and item["function"].get("name") == spec.get("tool_name")
        ), None)
        if not isinstance(function, dict):
            return []
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            return []
        raw_required = parameters.get("required")
        required = set(raw_required) if isinstance(raw_required, list) else set()
        mapped = set()
        for field in ("values", "selector", "changes"):
            mapping = spec.get(field)
            if isinstance(mapping, dict):
                mapped.update(mapping)
        return sorted(mapped - required)

    @staticmethod
    def _validate_tool_implementations(
        specs: list[Any], *, tools: list[dict[str, Any]], tables: list[dict[str, Any]]
    ) -> None:
        tool_parameters = {
            tool["function"]["name"]: set(tool["function"]["parameters"].get("properties", {}))
            for tool in tools
        }
        tool_parameter_schemas = {
            tool["function"]["name"]: tool["function"]["parameters"].get("properties", {})
            for tool in tools
        }
        table_columns = {
            table["table_name"]: {column["name"] for column in table.get("columns", [])}
            for table in tables
        }
        seen: set[str] = set()
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise PipelineGenerationError(f"tool_implementations[{index}] must be an object")
            name, table = spec.get("tool_name"), spec.get("table")
            if not isinstance(name, str) or name not in tool_parameters or name in seen:
                raise PipelineGenerationError(f"tool_implementations[{index}] references an invalid tool")
            operation = spec.get("operation")
            if (
                operation not in {"select", "aggregate_count", "insert", "update", "delete"}
                or not isinstance(table, str)
                or table not in table_columns
            ):
                raise PipelineGenerationError(f"tool_implementations[{index}] operation/table is invalid")
            if not isinstance(spec.get("result_field"), str) or not spec["result_field"]:
                raise PipelineGenerationError(f"tool_implementations[{index}] requires result_field")
            filters = spec.get("filters", [])
            if not isinstance(filters, list):
                raise PipelineGenerationError(f"tool_implementations[{index}].filters must be a list")
            for rule in filters:
                if (
                    not isinstance(rule, dict)
                    or not isinstance(rule.get("argument"), str)
                    or not isinstance(rule.get("column"), str)
                    or rule.get("argument") not in tool_parameters[name]
                    or rule.get("column") not in table_columns[table]
                    or rule.get("operator") not in {"eq", "in", "contains", "gte", "lte"}
                ):
                    raise PipelineGenerationError(f"tool_implementations[{index}] has an invalid filter")
                argument_schema = tool_parameter_schemas[name].get(rule["argument"], {})
                item_schema = argument_schema.get("items", {}) if isinstance(argument_schema, dict) else {}
                if isinstance(item_schema, dict) and item_schema.get("type") == "object":
                    item_properties = item_schema.get("properties", {})
                    resolver = rule.get("resolve")
                    item_field = resolver.get("match_column") if isinstance(resolver, dict) else rule["column"]
                    if (
                        rule["operator"] != "in"
                        or not isinstance(item_properties, dict)
                        or item_field not in item_properties
                    ):
                        raise PipelineGenerationError(
                            f"tool_implementations[{index}] object-array filter must use in "
                            f"and expose matching field {item_field}"
                        )
                resolver = rule.get("resolve")
                if resolver is not None:
                    if not isinstance(resolver, dict):
                        raise PipelineGenerationError(
                            f"tool_implementations[{index}] has an invalid filter resolver"
                        )
                    resolver_table = resolver.get("table")
                    match_column = resolver.get("match_column")
                    value_column = resolver.get("value_column")
                    if (
                        resolver_table not in table_columns
                        or match_column not in table_columns[resolver_table]
                        or value_column not in table_columns[resolver_table]
                    ):
                        raise PipelineGenerationError(
                            f"tool_implementations[{index}] resolver references an invalid table column"
                        )
                    target_table = next(
                        item for item in tables if item.get("table_name") == table
                    )
                    foreign_keys = target_table.get("foreign_keys", [])
                    if not any(
                        isinstance(foreign, dict)
                        and foreign.get("column") == rule.get("column")
                        and (foreign.get("ref_table") or foreign.get("references_table"))
                            == resolver_table
                        and (foreign.get("ref_column") or foreign.get("references_column"))
                            == value_column
                        for foreign in foreign_keys
                    ):
                        raise PipelineGenerationError(
                            f"tool_implementations[{index}] resolver is not backed by a foreign key"
                        )
            for field in ("projection",):
                values = spec.get(field, [])
                if not isinstance(values, list) or any(
                    not isinstance(value, str) or value not in table_columns[table]
                    for value in values
                ):
                    raise PipelineGenerationError(f"tool_implementations[{index}].{field} is invalid")
            from ..sandbox_runtime import normalize_record_order, SandboxError
            try:
                normalize_record_order(spec.get('order_by', []), columns=table_columns[table])
            except SandboxError as exc:
                raise PipelineGenerationError(f"tool_implementations[{index}].order_by is invalid: {exc}") from exc
            projection_aliases = spec.get("projection_aliases", {})
            if not isinstance(projection_aliases, dict) or any(
                not isinstance(alias, str) or not alias
                or not isinstance(column, str) or column not in table_columns[table]
                or alias in spec.get("projection", [])
                for alias, column in projection_aliases.items()
            ):
                raise PipelineGenerationError(
                    f"tool_implementations[{index}].projection_aliases is invalid"
                )
            for field in ("selector", "values", "changes"):
                mapping = spec.get(field, {})
                if not isinstance(mapping, dict) or any(
                    not isinstance(argument, str)
                    or not isinstance(column, str)
                    or argument not in tool_parameters[name]
                    or column not in table_columns[table]
                    for argument, column in mapping.items()
                ):
                    raise PipelineGenerationError(f"tool_implementations[{index}].{field} is invalid")
            if operation in {"update", "delete"} and not spec.get("selector"):
                raise PipelineGenerationError(f"tool_implementations[{index}] requires selector")
            if operation == "insert" and not spec.get("values"):
                raise PipelineGenerationError(f"tool_implementations[{index}] requires values")
            missing_insert_columns = ContractCompiler._missing_insert_columns(spec, tables)
            if missing_insert_columns:
                raise PipelineGenerationError(
                    f"tool_implementations[{index}] insert row omits storage columns "
                    f"{missing_insert_columns}; use a custom handler for generated fields"
                )
            derived = spec.get('derived_changes', {})
            if (not isinstance(derived, dict) or set(derived) - table_columns[table]
                    or set(derived) & set(spec.get('changes', {}).values())
                    or (derived and operation != 'update')):
                raise PipelineGenerationError(f'tool_implementations[{index}].derived_changes is invalid')
            if derived:
                from env_factory.tasks.task_spec import validate_goal_contract
                validate_goal_contract({'row_predicates': [{'table': table, 'where': {}, 'values': {},
                    'value_expressions': derived, 'count': 1}]}, tables, require_change=False)
            if operation == "update" and not (spec.get("changes") or derived):
                raise PipelineGenerationError(f"tool_implementations[{index}] requires changes")
            optional_mutation_arguments = ContractCompiler._optional_mutation_arguments(spec, tools)
            if optional_mutation_arguments:
                raise PipelineGenerationError(
                    f"tool_implementations[{index}] mutation arguments must be required: "
                    f"{optional_mutation_arguments}"
                )
            seen.add(name)

    @staticmethod
    def _complete_filter_resolvers(
        *, implementations: list[dict[str, Any]], tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Resolve human-readable related-entity arguments through declared FKs."""
        table_map = {
            table.get("table_name"): table for table in tables
            if isinstance(table, dict) and isinstance(table.get("table_name"), str)
        }
        completed = copy.deepcopy(implementations)
        for spec in completed:
            target = table_map.get(spec.get("table"), {})
            foreign_keys = target.get("foreign_keys", []) if isinstance(target, dict) else []
            for rule in spec.get("filters", []):
                if not isinstance(rule, dict) or rule.get("resolve") is not None:
                    continue
                foreign = next((
                    item for item in foreign_keys
                    if isinstance(item, dict) and item.get("column") == rule.get("column")
                ), None)
                if not foreign:
                    continue
                related_name = foreign.get("ref_table") or foreign.get("references_table")
                value_column = foreign.get("ref_column") or foreign.get("references_column")
                related = table_map.get(related_name)
                argument = rule.get("argument")
                if not isinstance(related, dict) or not isinstance(argument, str):
                    continue
                related_columns = {
                    column.get("name") for column in related.get("columns", [])
                    if isinstance(column, dict)
                }
                candidates = [argument]
                prefix = f"{related_name}_"
                if isinstance(related_name, str) and argument.startswith(prefix):
                    candidates.append(argument[len(prefix):])
                match_column = next(
                    (candidate for candidate in candidates if candidate in related_columns), None
                )
                if match_column and value_column in related_columns:
                    rule["resolve"] = {
                        "table": related_name,
                        "match_column": match_column,
                        "value_column": value_column,
                    }
        return completed

    @staticmethod
    def _validate_business_tool_semantics(
        *, tools: list[dict[str, Any]], implementations: list[dict[str, Any]],
        interaction_scripts=(), metrics=(), metric_implementations=(), business_lifecycle=None,
    ) -> None:
        """Reject declarative tools whose public API overclaims their behavior."""
        from .parameter_consumption import disclosed_parameter_uses, lifecycle_parameter_uses
        disclosed = disclosed_parameter_uses(interaction_scripts, metrics, metric_implementations)
        by_name = {
            item.get("tool_name"): item for item in implementations
            if isinstance(item, dict) and isinstance(item.get("tool_name"), str)
        }
        read_tokens = {"get", "read", "retrieve", "query", "search", "lookup", "list", "find", "fetch", "inspect"}
        semantic_tokens = {
            "calculate", "validate", "verify", "compare", "recommend", "classify", "summarize",
            "estimate", "score", "evaluate",
        }
        explicit_write_verbs = {"create", "insert", "add", "write", "update", "modify",
                                "delete", "remove", "set", "save"}
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = function.get("name")
            spec = by_name.get(name)
            # No declarative spec means the sandbox must supply a real custom
            # handler; the independent semantic review owns that path.
            if not isinstance(name, str) or not isinstance(spec, dict):
                continue
            parameters = function.get("parameters", {}).get("properties", {})
            parameter_names = set(parameters) if isinstance(parameters, dict) else set()
            consumed = {
                rule.get("argument") for rule in spec.get("filters", [])
                if isinstance(rule, dict) and isinstance(rule.get("argument"), str)
            }
            for field in ("selector", "values", "changes"):
                mapping = spec.get(field, {})
                if isinstance(mapping, dict):
                    consumed.update(mapping)
            consumed.update(argument for tool_name, argument in disclosed if tool_name == name)
            consumed.update(lifecycle_parameter_uses((business_lifecycle or {}).get('tools', {}).get(name, {})))
            unused = sorted(parameter_names - consumed)
            if unused:
                raise PipelineGenerationError(
                    f"business tool {name} exposes parameters ignored by its implementation: {unused}"
                )
            operation = spec.get("operation")
            lowered = name.lower()
            name_tokens = set(lowered.split("_"))
            if operation == "select" and lowered.split('_')[0] in semantic_tokens | explicit_write_verbs:
                raise PipelineGenerationError(
                    f"business tool {name} overclaims a plain select implementation"
                )
            if operation == "aggregate_count" and "count" not in name_tokens:
                raise PipelineGenerationError(
                    f"business tool {name} must expose count semantics"
                )
            # Business verbs (reserve, dispatch, approve, cancel, etc.) are not
            # a closed vocabulary. Reject an explicit read-only verb hiding a
            # write, not the absence of one of a few English mutation words.
            if (operation in {"insert", "update", "delete"}
                    and lowered.split('_')[0] in read_tokens):
                raise PipelineGenerationError(
                    f"business tool {name} advertises a read but its implementation mutates state"
                )

    @staticmethod
    def _validate_goal_tool_coverage(
        *, semantic_goal: dict[str, Any] | None,
        tool_implementations: list[Any],
    ) -> None:
        """Require every compiled mutation to implement an asserted goal delta."""
        predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        by_table: dict[str, list[dict[str, Any]]] = {}
        for predicate in predicates:
            if isinstance(predicate, dict) and isinstance(predicate.get("table"), str):
                by_table.setdefault(predicate["table"], []).append(predicate)

        mutations = [
            item for item in tool_implementations
            if isinstance(item, dict)
            and item.get("operation") in {"insert", "update", "delete"}
        ]
        mutation_tables = {
            item.get("table") for item in mutations if isinstance(item.get("table"), str)
        }
        uncovered_goal_tables = sorted(set(by_table) - mutation_tables)
        if uncovered_goal_tables:
            raise PipelineGenerationError(
                "stateful goal_contract has no declarative mutation for tables: "
                f"{uncovered_goal_tables}"
            )
        for implementation in mutations:
            table = str(implementation.get("table", ""))
            tool_name = implementation.get("tool_name")
            table_predicates = by_table.get(table, [])
            if not table_predicates:
                raise PipelineGenerationError(
                    f"stateful tool {tool_name} mutates table {table} but goal_contract "
                    "does not assert its final state"
                )
            if implementation.get("operation") != "update":
                continue
            changed_columns = {
                column for column in implementation.get("changes", {}).values()
                if isinstance(column, str)
            }
            changed_columns.update(implementation.get('derived_changes', {}))
            asserted_values = {
                field for predicate in table_predicates
                for field in predicate.get("values", {})
            }
            missing = asserted_values - changed_columns
            if missing:
                raise PipelineGenerationError(
                    f"stateful tool {tool_name} cannot establish goal fields: {sorted(missing)}"
                )

    @staticmethod
    def _complete_stateful_tool_schemas(
        tools: list[dict[str, Any]], plans: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compile required mutation arguments when the target tool is unambiguous."""
        completed = copy.deepcopy(tools)
        mutation_markers = {
            "insert": ("create", "insert", "add", "write", "book", "booking", "reserve", "place", "submit",
                       "创建", "新增", "添加", "预订", "预约", "提交"),
            "update": ("update", "modify", "correct", "write", "更新", "修改", "修正"),
            "delete": ("delete", "remove", "删除", "移除"),
        }
        used_tools: set[str] = set()
        for plan in plans:
            if not isinstance(plan, dict) or plan.get("operation") not in mutation_markers:
                continue
            table = plan.get("table")
            if not isinstance(table, str):
                continue
            candidates: list[tuple[int, dict[str, Any]]] = []
            for tool in completed:
                function = tool.get("function", {}) if isinstance(tool, dict) else {}
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if not isinstance(name, str) or name in used_tools:
                    continue
                label = f"{function.get('name', '')} {function.get('description', '')}".lower()
                parameters = function.get("parameters")
                properties = parameters.get("properties") if isinstance(parameters, dict) else None
                if not isinstance(properties, dict):
                    continue
                field_names = {
                    field["name"] for field in plan.get("required_fields", [])
                    if isinstance(field, dict) and isinstance(field.get("name"), str)
                }
                overlap = sum(any(
                    argument == field or argument.endswith(f"_{field}")
                    or argument.removeprefix("new_") == field
                    or argument.removeprefix("target_") == field
                    for argument in properties if isinstance(argument, str)
                ) for field in field_names)
                table_match = table.lower() in label
                operation_label = label.replace(table.lower(), "")
                operation_match = any(
                    (marker in operation_label if not marker.isascii() else bool(re.search(
                        rf"(?<![a-z]){re.escape(marker)}(?![a-z])", operation_label,
                    )))
                    for marker in mutation_markers[plan["operation"]]
                )
                if not operation_match or not (table_match or overlap or len(plans) == 1):
                    continue
                candidates.append((100 * table_match + 10 * operation_match + overlap,
                                   function))
            if not candidates:
                continue
            candidates.sort(key=lambda item: item[0], reverse=True)
            if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
                continue
            function = candidates[0][1]
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                continue
            properties = parameters.get("properties")
            required = parameters.get("required")
            if not isinstance(properties, dict) or not isinstance(required, list):
                continue
            for field in plan.get("required_fields", []):
                if not isinstance(field, dict) or not isinstance(field.get("name"), str):
                    continue
                column = field["name"]
                aliases = [name for name in properties if isinstance(name, str) and (
                    name == column or name.endswith(f"_{column}")
                    or name.removeprefix("new_") == column
                    or name.removeprefix("target_") == column
                )]
                if len(aliases) > 1:
                    continue
                argument = aliases[0] if aliases else column
                if not aliases:
                    properties[argument] = {
                        "type": field.get("json_type", "string"),
                        "description": f"{table}.{column} 的业务变更值",
                    }
                elif isinstance(properties[argument], dict):
                    properties[argument]["type"] = field.get("json_type", "string")
                if argument not in required:
                    required.append(argument)
            used_tools.add(function["name"])
        return completed

    @staticmethod
    def _stateful_mutation_surface_plan(
        semantic_goal: dict[str, Any] | None, tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expose exact mutation parameter requirements before tool generation."""
        from ..sandbox_runtime import ManifestDataStore

        goals = semantic_goal.get("row_predicates", []) if isinstance(semantic_goal, dict) else []
        by_name = {table.get("table_name"): table for table in tables if isinstance(table, dict)}
        plan: list[dict[str, Any]] = []
        for goal in goals if isinstance(goals, list) else []:
            if not isinstance(goal, dict):
                continue
            table_name = goal.get("table")
            table = by_name.get(table_name)
            if not isinstance(table, dict):
                continue
            where, values = goal.get("where", {}), goal.get("values", {})
            if not isinstance(where, dict) or not isinstance(values, dict):
                continue
            rows = table.get("rows", [])
            matching = [row for row in rows if isinstance(row, dict)
                        and all(row.get(field) == value for field, value in where.items())] if isinstance(rows, list) else []
            operation = "delete" if goal.get("count") == 0 else ("update" if matching else "insert")
            columns = {column.get("name"): column for column in table.get("columns", [])
                       if isinstance(column, dict) and isinstance(column.get("name"), str)}
            required = set(where) | set(values)
            platform_fields: set[str] = set()
            if operation == "insert":
                required.update(columns)
                platform_fields = ManifestDataStore.platform_insert_fields(table)
                platform_fields.difference_update(where)
                platform_fields.difference_update(
                    field for field, value in values.items() if value is not None
                )
                required.difference_update(platform_fields)
            elif operation == "delete":
                required = set(where)
            fields = []
            for name in sorted(required):
                sql_type = str(columns.get(name, {}).get("type", "")).upper()
                json_type = ("boolean" if sql_type.startswith(("BOOL", "BOOLEAN"))
                             else "integer" if "INT" in sql_type
                             else "number" if any(part in sql_type for part in ("DECIMAL", "NUMERIC", "REAL", "FLOAT", "DOUBLE"))
                             else "string")
                fields.append({"name": name, "json_type": json_type})
            plan.append({"table": table_name, "operation": operation,
                         "required_fields": fields,
                         "platform_filled_fields": sorted(platform_fields)})
        return plan

    @staticmethod
    def _validate_stateful_tool_surface(
        tools: list[Any], *, semantic_goal: dict[str, Any] | None,
        tables: list[dict[str, Any]] | None = None,
    ) -> None:
        """Require a compilable mutation surface before accepting tool schemas."""
        predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        mutation_markers = (
            "create", "insert", "add", "write", "update", "modify", "correct", "delete", "remove",
            "创建", "新增", "添加", "更新", "修改", "修正", "删除", "移除",
        )
        surfaces: list[tuple[set[str], set[str]]] = []
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            label = f"{function.get('name', '')} {function.get('description', '')}".lower()
            if not any(marker in label for marker in mutation_markers):
                continue
            parameters = function.get("parameters", {})
            properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
            required = parameters.get("required", []) if isinstance(parameters, dict) else []
            if isinstance(properties, dict) and isinstance(required, list):
                # Declarative mutations index mapped arguments unconditionally.
                # An optional field cannot establish a goal predicate at runtime.
                surfaces.append((set(properties), set(properties) & {
                    name for name in required if isinstance(name, str)
                }))
        columns_by_table = {
            table.get("table_name"): {
                column.get("name") for column in table.get("columns", [])
                if isinstance(column, dict) and isinstance(column.get("name"), str)
            }
            for table in (tables or []) if isinstance(table, dict)
        }

        def maps_to_column(argument: str, column: str) -> bool:
            return (argument == column or argument.endswith(f"_{column}")
                    or argument.removeprefix("new_") == column
                    or argument.removeprefix("target_") == column)

        for predicate in predicates if isinstance(predicates, list) else []:
            if not isinstance(predicate, dict):
                continue
            required = set(predicate.get("where", {})) | set(predicate.get("values", {}))
            if tables:
                matching_plan = next((item for item in ContractCompiler._stateful_mutation_surface_plan(
                    {"row_predicates": [predicate]}, tables,
                ) if item.get("table") == predicate.get("table")), None)
                if matching_plan is not None:
                    required = {item["name"] for item in matching_plan["required_fields"]}
            columns = columns_by_table.get(predicate.get("table"))
            def covers(surface: tuple[set[str], set[str]]) -> bool:
                properties, required_arguments = surface
                return all(
                    len([
                        argument for argument in required_arguments
                        if maps_to_column(argument, field)
                    ]) == 1
                    for field in required
                ) and (
                    columns is None or all(
                        any(maps_to_column(argument, column) for column in columns)
                        for argument in properties
                    )
                )
            if not required or not any(covers(surface) for surface in surfaces):
                raise PipelineGenerationError(
                    "stateful tool schema has no compilable mutation parameters for "
                    f"table {predicate.get('table')}: {sorted(required)}"
                )

    @staticmethod
    def _validate_compiled_tool_chain(
        task_spec: dict[str, Any], scenarios: list[dict[str, Any]],
    ) -> None:
        from ..contracts.tool_chain_contract import tool_chain_issues

        issues = tool_chain_issues({
            "task_spec": task_spec,
            "acceptance_contract": {"executable_scenarios": scenarios},
        })
        if issues:
            details = [{"code": issue["code"], **issue["evidence"]} for issue in issues]
            raise PipelineGenerationError(
                f"compiled tool-chain schema conflicts: {details}"
            )

    @staticmethod
    def _preview_success_tool_results(
        *, scenarios: list[dict[str, Any]], data_tables: list[dict[str, Any]],
        tool_implementations: list[dict[str, Any]], environment_mode: str,
        tools: list[dict[str, Any]] | None = None,
        semantic_goal: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute the declarative success prefix against shared data-store rules."""
        if environment_mode not in {"reference_data", "stateful"}:
            return []
        success = next((item for item in scenarios if item.get("kind") == "goal_success"), None)
        if success is None:
            return []
        by_name = {item.get("tool_name"): item for item in tool_implementations}
        # A custom handler ends the executable prefix. Earlier declarative
        # results still provide concrete evidence to the observability audit.
        preview_steps = []
        for step in success.get("steps", []):
            if step.get("operation") == "reset":
                preview_steps.append(step)
            elif step.get("operation") == "tool_call":
                spec = by_name.get(step.get("tool_name"), {})
                allowed = ({"select", "aggregate_count"} if environment_mode == "reference_data"
                           else {"select", "aggregate_count", "insert", "update", "delete"})
                if spec.get("operation") not in allowed:
                    break
                preview_steps.append(step)
        steps = [step for step in preview_steps if step.get("operation") == "tool_call"]
        if not steps:
            return []
        from ..sandbox_runtime import (
            AcceptanceScenarioRunner, DeclarativeToolCompiler, ManifestDataStore,
            SandboxError, validate_json_schema,
        )
        tool_schemas = {
            function["name"]: function.get("parameters", {})
            for item in (tools or []) if isinstance(item, dict)
            for function in [item.get("function")]
            if isinstance(function, dict) and isinstance(function.get("name"), str)
            and isinstance(function.get("parameters", {}), dict)
        }

        class PreviewState:
            def __init__(self) -> None:
                self.values: dict[str, Any] = {}

            def get_state(self, key: str, default=None) -> Any:
                return copy.deepcopy(self.values.get(key, default))

            def set_state(self, key: str, value: Any) -> None:
                self.values[key] = copy.deepcopy(value)

        class PreviewData(ManifestDataStore):
            def __init__(self) -> None:
                self.baseline = {
                    item["table_name"]: [dict(row) for row in item.get("rows", [])]
                    for item in data_tables
                }
                self.schemas = {}
                for item in data_tables:
                    schema = copy.deepcopy(item)
                    if not schema.get("columns") and schema.get("rows"):
                        # Unit callers may provide only rows. Production data
                        # tables carry the complete validated schema.
                        schema["columns"] = [
                            {"name": name, "nullable": True}
                            for name in schema["rows"][0]
                        ]
                    self.schemas[item["table_name"]] = schema
                self.episode_store = PreviewState()
                self._validate_tables(self.baseline)
                self.reset()

        try:
            preview_data = PreviewData()
            handlers = DeclarativeToolCompiler(preview_data).compile_all(
                [by_name[name] for name in dict.fromkeys(step["tool_name"] for step in steps)]
            )
        except (SandboxError, KeyError) as exc:
            raise PipelineGenerationError(f"success tool preview cannot compile: {exc}") from exc

        executed_arguments: list[dict[str, Any]] = []

        def call(method: str, path: str, body: Any, headers: dict[str, str]) -> tuple[int, Any, dict[str, str]]:
            if path == "/v1/reset":
                preview_data.reset()
                return 200, {}, {}
            name = path.removeprefix("/v1/tools/")
            if tools is not None:
                if name not in tool_schemas:
                    raise SandboxError("TOOL_NOT_FOUND", f"tool schema is missing: {name}", 400)
                validate_json_schema(tool_schemas[name], body)
            executed_arguments.append(copy.deepcopy(body))
            return 200, handlers[name](body), {}

        try:
            replay = AcceptanceScenarioRunner(call).run({"steps": preview_steps, "assertions": []})
        except (SandboxError, KeyError, TypeError, ValueError) as exc:
            raise PipelineGenerationError(f"success tool preview failed: {exc}") from exc
        if (semantic_goal and isinstance(semantic_goal.get("row_predicates"), list)
                and len(steps) == sum(step.get("operation") == "tool_call"
                                      for step in success.get("steps", []))):
            from ..sandbox_runtime import BusinessGoalEvaluator

            final_state = {name: preview_data.table(name) for name in preview_data.baseline}
            predicates = semantic_goal["row_predicates"]
            if not BusinessGoalEvaluator.evaluate(predicates, final_state, preview_data.baseline):
                raise PipelineGenerationError(
                    "success tool preview does not establish the stateful goal"
                )
            if semantic_goal.get("requires_state_change") and final_state == preview_data.baseline:
                raise PipelineGenerationError(
                    "success tool preview leaves business state unchanged"
                )
        observed = []
        tool_index = 0
        for step, result in zip(preview_steps, replay["history"]):
            if step.get("operation") != "tool_call":
                continue
            observed.append({
                "tool_name": step["tool_name"],
                "arguments": executed_arguments[tool_index],
                "result": result["body"],
            })
            tool_index += 1
            for variable, path in step.get("capture", {}).items():
                if replay["variables"].get(variable) in (None, [], {}):
                    raise PipelineGenerationError(
                        f"success tool preview capture is empty: {step['tool_name']}.{path}"
                    )
        return observed

    @staticmethod
    def _tool_chain_projection_gaps(
        *, route_plan: dict[str, Any], tool_bindings: list[dict[str, Any]],
        tools: list[dict[str, Any]], implementations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find unambiguous route edges whose required record shape is unavailable."""
        bound: dict[str, list[str]] = {}
        for binding in tool_bindings:
            if not isinstance(binding, dict):
                continue
            action, name = binding.get("action_name"), binding.get("tool_name")
            if isinstance(action, str) and isinstance(name, str):
                bound.setdefault(action, []).append(name)
        schemas = {
            tool.get("function", {}).get("name"): tool.get("function", {}).get("parameters", {})
            for tool in tools if isinstance(tool, dict)
        }
        specs = {
            spec.get("tool_name"): spec for spec in implementations
            if isinstance(spec, dict)
        }
        gaps: list[dict[str, Any]] = []
        operations = route_plan.get("environment_operations", [])
        for operation in operations if isinstance(operations, list) else []:
            if not isinstance(operation, dict):
                continue
            dependencies = operation.get("dependencies", [])
            downstream_names = bound.get(operation.get("action_name"), [])
            if not isinstance(dependencies, list) or len(dependencies) != 1 or len(downstream_names) != 1:
                continue
            upstream_names = bound.get(dependencies[0], [])
            if len(upstream_names) != 1:
                continue
            upstream_name, downstream_name = upstream_names[0], downstream_names[0]
            upstream = specs.get(upstream_name, {})
            if upstream.get("operation") != "select" or not isinstance(upstream.get("result_field"), str):
                continue
            projected = {
                field for field in upstream.get("projection", []) if isinstance(field, str)
            }
            aliases = upstream.get("projection_aliases", {})
            if isinstance(aliases, dict):
                projected.update(field for field in aliases if isinstance(field, str))
            parameters = schemas.get(downstream_name, {})
            if not isinstance(parameters, dict):
                continue
            properties = parameters.get("properties", {})
            required = parameters.get("required", [])
            if not isinstance(properties, dict) or not isinstance(required, list):
                continue
            # A matching scalar can carry the dependency even when a separate
            # record array is supplied from public input.
            if any(
                name in projected
                and isinstance(properties.get(name), dict)
                and properties[name].get("type") != "array"
                for name in required if isinstance(name, str)
            ):
                continue
            arrays: list[tuple[str, set[str]]] = []
            for name in required:
                schema = properties.get(name, {})
                if not isinstance(name, str) or not isinstance(schema, dict) or schema.get("type") != "array":
                    continue
                item = schema.get("items", {})
                fields = item.get("required", []) if isinstance(item, dict) and item.get("type") == "object" else []
                if isinstance(fields, list) and fields and all(isinstance(field, str) for field in fields):
                    arrays.append((name, set(fields)))
            if len(arrays) != 1:
                continue
            argument, fields = arrays[0]
            missing = sorted(fields - projected)
            if missing:
                gaps.append({
                    "upstream_tool": upstream_name,
                    "upstream_table": upstream.get("table"),
                    "downstream_tool": downstream_name,
                    "downstream_argument": argument,
                    "missing_fields": missing,
                    "projected_fields": sorted(projected),
                })
        return gaps

    @staticmethod
    def _complete_dependency_projections(
        *, implementations: list[dict[str, Any]], tools: list[dict[str, Any]],
        tables: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Expose downstream fields and labels for selected business entities."""
        import copy

        schema_fields: set[str] = set()
        def collect(schema: Any) -> None:
            if not isinstance(schema, dict):
                return
            properties = schema.get("properties", {})
            if isinstance(properties, dict):
                schema_fields.update(str(name) for name in properties)
                for child in properties.values():
                    collect(child)
            collect(schema.get("items"))
        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            collect(function.get("parameters"))

        table_columns = {
            table.get("table_name"): [
                str(column.get("name")) for column in table.get("columns", [])
                if isinstance(column, dict) and isinstance(column.get("name"), str)
            ]
            for table in tables if isinstance(table, dict)
        }
        table_primary_keys = {
            table.get("table_name"): {
                key for key in table.get("primary_key", []) if isinstance(key, str)
            }
            for table in tables if isinstance(table, dict)
            and isinstance(table.get("primary_key", []), list)
        }
        completed = copy.deepcopy(implementations)
        for spec in completed:
            if not isinstance(spec, dict) or spec.get("operation") != "select":
                continue
            available = table_columns.get(spec.get("table"), [])
            required = [name for name in available if name in schema_fields]
            projection = spec.get("projection", [])
            if not isinstance(projection, list):
                projection = []
            table_name = str(spec.get("table", ""))
            # A record identified only by its id is hard to use in a named
            # report. Include a same-entity display label when the select
            # already exposes that entity's primary key. This does not add
            # unrelated table columns or infer hidden business truth.
            label_names = {
                "name", "title", "label", f"{table_name}_name",
                f"{table_name.removesuffix('s')}_name",
            }
            primary_keys = table_primary_keys.get(table_name, set())
            projected_names = {name for name in projection if isinstance(name, str)}
            labels = [
                name for name in available if name in label_names
                and primary_keys and primary_keys <= projected_names
            ]
            projection = list(dict.fromkeys([*projection, *required, *labels]))
            if projection:
                spec["projection"] = projection
            aliases = spec.get("projection_aliases", {})
            if not isinstance(aliases, dict):
                aliases = {}
            for public_name in sorted(schema_fields - set(available)):
                candidates = [
                    column for column in available
                    if public_name == f"{table_name}_{column}"
                ]
                if len(candidates) == 1 and public_name not in projection:
                    aliases.setdefault(public_name, candidates[0])
            if aliases:
                spec["projection_aliases"] = aliases
        return completed

    @staticmethod
    def _complete_stateful_tool_implementations(
        *, implementations: list[dict[str, Any]], tools: list[dict[str, Any]],
        tables: list[dict[str, Any]], semantic_goal: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Compile obvious one-table updates when a model omits a valid spec.

        This is intentionally conservative: argument and column names must be
        identical, selector and changed fields must both be present, and their
        primitive types must agree. More complex semantics remain custom-handler
        work and therefore fail the stateful buildability gate.
        """
        completed = [dict(item) for item in implementations]
        raw_predicates = (
            semantic_goal.get("row_predicates", [])
            if isinstance(semantic_goal, dict) else []
        )
        predicates = [
            item for item in raw_predicates if isinstance(item, dict)
        ] if isinstance(raw_predicates, list) else []
        goals_by_table: dict[str, list[dict[str, Any]]] = {}
        for predicate in predicates:
            table = predicate.get("table")
            if isinstance(table, str):
                goals_by_table.setdefault(table, []).append(predicate)

        table_by_name = {
            table.get("table_name"): table for table in tables
            if isinstance(table, dict) and isinstance(table.get("table_name"), str)
        }
        numeric_types = ("INT", "DECIMAL", "NUMERIC", "REAL", "FLOAT", "DOUBLE")

        def compatible(schema: dict[str, Any], sql_type: str) -> bool:
            json_type = schema.get("type")
            if sql_type.upper().startswith(("BOOL", "BOOLEAN")):
                return json_type == "boolean"
            is_numeric = any(token in sql_type.upper() for token in numeric_types)
            return json_type in ({"number", "integer"} if is_numeric else {"string"})

        def argument_for(field: str, properties: dict[str, Any], columns: dict[str, Any]) -> str | None:
            candidates = [
                argument for argument, schema in properties.items()
                if isinstance(argument, str)
                and (
                    argument == field
                    or argument.endswith(f"_{field}")
                    or argument.removeprefix("new_") == field
                    or argument.removeprefix("target_") == field
                )
                and field in columns
                and isinstance(schema, dict)
                and compatible(schema, str(columns[field].get("type", "")))
            ]
            return candidates[0] if len(candidates) == 1 else None

        for tool in tools:
            function = tool.get("function", {}) if isinstance(tool, dict) else {}
            name = function.get("name")
            if not isinstance(name, str):
                continue
            label = f"{name} {function.get('description', '')}".lower()
            if not any(marker in label for marker in ("update", "modify", "更新", "修改", "修正")):
                continue
            existing = next(
                (item for item in completed if item.get("tool_name") == name), None
            )
            if isinstance(existing, dict) and existing.get("operation") in {
                "insert", "update", "delete",
            }:
                continue
            properties = function.get("parameters", {}).get("properties", {})
            if not isinstance(properties, dict):
                continue
            candidates: list[dict[str, Any]] = []
            for table_name, table_predicates in goals_by_table.items():
                table = table_by_name.get(table_name, {})
                columns = {
                    column.get("name"): column
                    for column in table.get("columns", [])
                    if isinstance(column, dict)
                    if isinstance(column.get("name"), str)
                }
                where_fields = {
                    field for predicate in table_predicates
                    for field in predicate.get("where", {})
                }
                value_fields = {
                    field for predicate in table_predicates
                    for field in predicate.get("values", {})
                }
                selector = {
                    argument: field for field in where_fields
                    if (argument := argument_for(field, properties, columns)) is not None
                }
                changes = {
                    argument: field for field in value_fields
                    if (argument := argument_for(field, properties, columns)) is not None
                }
                if (
                    selector and changes
                    and where_fields <= set(selector.values())
                    and value_fields <= set(changes.values())
                ):
                    candidates.append({
                        "tool_name": name, "operation": "update", "table": table_name,
                        "selector": selector, "changes": changes, "result_field": "records",
                    })
            if len(candidates) == 1:
                self_spec = candidates[0]
                completed = [
                    item for item in completed if item.get("tool_name") != name
                ]
                ContractCompiler._validate_tool_implementations(
                    [*completed, self_spec], tools=tools, tables=tables
                )
                completed.append(self_spec)
                logger.warning(
                    "compiled deterministic stateful implementation: tool=%s table=%s",
                    name, self_spec["table"],
                )
        return completed

    @staticmethod
    def _build_runtime_interface(
        tools: list[dict[str, Any]], reward_formula: dict[str, Any]
    ) -> dict[str, Any]:
        """Build the HTTP contract consumed by sandbox builders and trainers.

        This is deliberately derived from validated artifacts rather than
        requested from the LLM.  Every generated LLM tool is listed with its
        exact request schema, and the single runtime reward function has an
        explicit endpoint.
        """
        tool_endpoints = []
        for tool in tools:
            function = tool["function"]
            tool_endpoints.append({
                "name": function["name"],
                "kind": "llm_tool",
                "method": "POST",
                "path": "/v1/tools/{tool_name}",
                "request_schema": function["parameters"],
                "response_schema": {
                    "type": "object",
                    "description": "本次工具调用的业务结果；不包含 observation 或 reward。",
                },
            })
        from ..sandbox_runtime import VirtualClock
        return {
            "protocol": "http",
            "base_path": "/v1",
            "version": "1.0",
            "shared_runtime": {
                "module": "sandbox_runtime.py",
                "version": "1.3",
                "required_components": [
                    "EpisodeStore",
                    "ManifestDataStore",
                    "DeclarativeToolCompiler",
                    "SandboxApplication",
                    "VirtualClock",
                    "ContractToolRegistry",
                    "ContractRewardAggregator",
                    "ContractRewardGate",
                    "DeclarativeMetricEvaluator",
                    "AcceptanceScenarioRunner",
                    "ContractUserSimulator",
                    "validate_json_schema",
                ],
            },
            "errors": {
                "content_type": "application/json",
                "schema": {
                    "type": "object",
                    "required": ["error"],
                    "error_required": ["code", "message", "request_id"],
                },
                "status_codes": {"invalid_argument": 400, "unauthorized": 401, "forbidden": 403, "not_found": 404, "conflict": 409, "server_error": 500},
            },
            "observability": {
                "request_id_header": "X-Request-ID",
                "required_fields": ["request_id", "episode_id", "tool_call_id", "timestamp", "duration_ms"],
                "credential_redaction": True,
                "structured_format": "json",
            },
            "security": {
                "trainer": {
                    "scheme": "bearer",
                    "environment_variable": "SANDBOX_TRAINER_API_KEY",
                    "required_for": ["reset", "observation", "state", "user_simulator", "agent_response", "reward", "replay"],
                },
                "agent": {
                    "scheme": "none",
                    "allowed_kinds": ["llm_tool"],
                    "cannot_access": ["state", "user_simulator", "agent_response", "reward", "replay"],
                },
            },
            "episode": {
                "isolation": "per_episode",
                "reset_accepts_seed": True,
                "deterministic_replay": True,
                "idempotency_header": "Idempotency-Key",
                "persist_trace": True,
            },
            "llm_runtime": {
                "provider": "openai_compatible_chat_completions",
                "api_key_environment_variable": "SANDBOX_LLM_API_KEY",
                "base_url_environment_variable": "SANDBOX_LLM_BASE_URL",
                "model_environment_variable": "SANDBOX_LLM_MODEL",
                "timeout_environment_variable": "SANDBOX_LLM_TIMEOUT_SECONDS",
                "max_retries_environment_variable": "SANDBOX_LLM_MAX_RETRIES",
                "never_persist_credentials": True,
            },
            "evaluator_runtime": {
                "external_llm": "SANDBOX_LLM_API_KEY",
                "mock_mode_environment_variable": "SANDBOX_EVALUATOR_MOCK",
                "record_call_trace": True,
                "cache_by_context_hash": True,
            },
            "mutation_testing": {
                "environment_variable": "SANDBOX_MUTATION_MODE",
                "modes": ["constant_tool_result", "skip_business_write", "constant_reward", "ignore_tool_arguments", "bypass_trainer_auth"],
                "production_default": "disabled",
                "must_be_observable_in_test_only": True,
            },
            "launcher": {
                "command": ["python", "app.py", "--port", "{port}"],
                "health_path": "/health",
            },
            "endpoints": [
                {"name": "health", "kind": "system", "method": "GET", "path": "/health"},
                {
                    "name": "reset", "kind": "system", "method": "POST", "path": "/v1/reset",
                    "access": "rl_trainer_only",
                    "request_schema": {
                        "type": "object", "description": "创建或重置独立 episode。",
                        "properties": {
                            "episode_id": {"type": "string", "description": "可选的 episode 标识。"},
                            "seed": {"type": "integer", "description": "可选的确定性随机种子。"},
                        }, "required": [], "additionalProperties": False,
                    },
                },
                {"name": "observation", "kind": "system", "access": "rl_trainer_only", "method": "GET", "path": "/v1/observation"},
                {
                    "name": "state", "kind": "trainer_evidence",
                    "access": "rl_trainer_only", "method": "GET", "path": "/v1/state",
                    "response_schema": {
                        "type": "object",
                        "description": "仅供 Trainer 记录初末业务状态；不得进入 policy observation。",
                    },
                },
                {"name": "tools", "kind": "system", "method": "GET", "path": "/v1/tools"},
                {"name": "time", "kind": "system", "access": "agent",
                 "method": "GET", "path": "/v1/time",
                 "response_schema": copy.deepcopy(VirtualClock.RESPONSE_SCHEMA)},
                {
                    "name": "user_simulator",
                    "kind": "user_simulator",
                    "access": "rl_trainer_only",
                    "method": "POST",
                    "path": "/v1/user_simulator",
                    "request_schema": {
                        "type": "object",
                        "description": "RL Trainer 传入的完整对话上下文。该接口不暴露给待训练 Agent。",
                        "properties": {
                            "messages": {
                                "type": "array",
                                "description": "从当前会话开始的完整对话消息序列，按时间顺序排列。",
                                "items": {
                                    "type": "object",
                                    "description": "一条对话消息。",
                                    "properties": {
                                        "role": {"type": "string", "description": "消息角色，例如 user 或 assistant。"},
                                        "content": {"type": "string", "description": "消息文本内容。"},
                                    },
                                    "required": ["role", "content"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["messages"],
                        "additionalProperties": False,
                    },
                    "response_schema": {
                        "type": "object",
                        "description": "User Simulator 生成的用户查询和终止标志。",
                        "properties": {
                            "user_query": {"type": "string", "description": "模拟用户生成的自然语言查询。"},
                            "should_end": {"type": "boolean", "description": "用户是否停止继续提问。"},
                            "match_status": {"type": "string", "description": "实时对话为 matched、unmatched 或 ambiguous。"},
                            "outcome_category": {"type": "string", "description": "固定的八类对话结果之一。"},
                            "reason_code": {"type": "string", "description": "分支匹配或恢复原因。"},
                            "interaction_branch": {"type": ["string", "null"], "description": "Trainer-only：按实际环境证据选择的用户分支。"},
                            "interaction_stage": {"type": ["string", "null"], "description": "Trainer-only：本轮完成的任务专属交互阶段。"},
                            "fsm_script_id": {"type": "string", "description": "Trainer-only：本轮使用的 FSM 脚本身份。"},
                            "fsm_transition_id": {"type": ["string", "null"], "description": "Trainer-only：matched 时实际应用的迁移；恢复结果为空。"},
                            "fsm_state_before": {"type": "string", "description": "Trainer-only：迁移前状态。"},
                            "fsm_state_after": {"type": "string", "description": "Trainer-only：迁移后状态。"},
                            "fsm_transition_applied": {"type": "boolean", "description": "Trainer-only：是否应用了声明式迁移。"},
                            "fsm_recovery_count": {"type": "integer", "minimum": 0, "description": "Trainer-only：连续恢复次数。"},
                            "attachments": {"type": "array", "items": {"type": "object", "description": "附件对象（当前不支持）"}, "maxItems": 0,
                                            "description": "当前文本/结构化环境不接受模型生成附件，固定为空。"},
                            "termination_reason": {"type": "string", "description": "结束时为 completed 或 unresolved_dialogue。"},
                        },
                        "required": [
                            "user_query", "should_end", "match_status",
                            "outcome_category", "reason_code", "attachments",
                            "fsm_script_id", "fsm_transition_id",
                            "fsm_state_before", "fsm_state_after",
                            "fsm_transition_applied", "fsm_recovery_count",
                        ],
                        "additionalProperties": False,
                    },
                },
                {
                    "name": "agent_response",
                    "kind": "agent_response",
                    "access": "rl_trainer_only",
                    "method": "POST",
                    "path": "/v1/agent_response",
                    "request_schema": {
                        "type": "object",
                        "description": "RL Trainer 提交待训练 Agent 的最终自然语言回答。",
                        "properties": {
                            "content": {"type": "string", "description": "Agent 的非空最终回答。"},
                        },
                        "required": ["content"],
                        "additionalProperties": False,
                    },
                    "response_schema": {
                        "type": "object",
                        "description": "确认最终回答已写入当前 episode。",
                    },
                },
                *tool_endpoints,
                {
                    "name": "reward",
                    "kind": "reward_function",
                    "access": "rl_trainer_only",
                    "method": "GET",
                    "path": "/v1/reward",
                    "formula": reward_formula,
                    "response_schema": {
                        "type": "object",
                        "description": "调用该接口时统一计算并返回当前会话奖励和各指标得分。",
                    },
                },
                {
                    "name": "replay",
                    "kind": "replay",
                    "access": "rl_trainer_only",
                    "method": "GET",
                    "path": "/v1/replay",
                    "response_schema": {
                        "type": "object",
                        "description": "返回当前 episode 的可回放轨迹、seed、版本和数据哈希。",
                    },
                },
            ],
            "llm_tools": [endpoint["name"] for endpoint in tool_endpoints],
            "reward_functions": [{"name": "reward", "endpoint": "/v1/reward"}],
        }

    @staticmethod
    def _validate_runtime_interface(
        interface: Any, tools: list[dict[str, Any]], reward_formula: dict[str, Any]
    ) -> None:
        if not isinstance(interface, dict) or interface.get("protocol") != "http":
            raise PipelineGenerationError("requirements.runtime_interface must declare http")
        shared_runtime = interface.get("shared_runtime")
        required_shared = {
            "EpisodeStore", "ManifestDataStore", "DeclarativeToolCompiler", "SandboxApplication", "ContractToolRegistry",
            "VirtualClock",
            "ContractRewardAggregator", "ContractRewardGate", "DeclarativeMetricEvaluator", "AcceptanceScenarioRunner", "ContractUserSimulator",
            "validate_json_schema"
        }
        if (
            not isinstance(shared_runtime, dict)
            or shared_runtime.get("module") != "sandbox_runtime.py"
            or not required_shared <= set(shared_runtime.get("required_components", []))
        ):
            raise PipelineGenerationError("runtime_interface shared runtime contract is incomplete")
        endpoints = interface.get("endpoints")
        if not isinstance(endpoints, list) or not endpoints:
            raise PipelineGenerationError("runtime_interface.endpoints must be non-empty")
        endpoint_keys = {(item.get("name"), item.get("method"), item.get("path"))
                         for item in endpoints if isinstance(item, dict)}
        for name, method, path in (
            ("health", "GET", "/health"),
            ("reset", "POST", "/v1/reset"),
            ("observation", "GET", "/v1/observation"),
            ("state", "GET", "/v1/state"),
            ("tools", "GET", "/v1/tools"),
            ("time", "GET", "/v1/time"),
            ("user_simulator", "POST", "/v1/user_simulator"),
            ("agent_response", "POST", "/v1/agent_response"),
            ("reward", "GET", "/v1/reward"),
            ("replay", "GET", "/v1/replay"),
        ):
            if (name, method, path) not in endpoint_keys:
                raise PipelineGenerationError(f"runtime_interface missing {method} {path}")
        from ..sandbox_runtime import VirtualClock
        time_endpoint = next(item for item in endpoints if item.get('name') == 'time')
        if time_endpoint.get('response_schema') != VirtualClock.RESPONSE_SCHEMA or time_endpoint.get('access') != 'agent':
            raise PipelineGenerationError('runtime_interface time endpoint must publish the shared clock schema and agent access')
        expected_tools = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
        declared_tools = [
            item for item in endpoints
            if isinstance(item, dict) and item.get("kind") == "llm_tool"
        ]
        if len(declared_tools) != len(expected_tools):
            raise PipelineGenerationError("runtime_interface must list every LLM tool exactly once")
        for item in declared_tools:
            name = item.get("name")
            if name not in expected_tools or item.get("method") != "POST" or item.get("path") != "/v1/tools/{tool_name}":
                raise PipelineGenerationError(f"invalid runtime endpoint for tool {name!r}")
            if item.get("request_schema") != expected_tools[name]:
                raise PipelineGenerationError(f"runtime schema does not match tool {name}")
        if interface.get("llm_tools") != list(expected_tools):
            raise PipelineGenerationError("runtime_interface.llm_tools is inconsistent with tools")
        user_simulator_endpoint = next(
            item for item in endpoints if item.get("name") == "user_simulator"
        )
        if user_simulator_endpoint.get("kind") != "user_simulator" or user_simulator_endpoint.get("access") != "rl_trainer_only":
            raise PipelineGenerationError("runtime_interface user_simulator endpoint has invalid kind")
        expected_user_request = {
            "type": "object",
            "description": "RL Trainer 传入的完整对话上下文。该接口不暴露给待训练 Agent。",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "从当前会话开始的完整对话消息序列，按时间顺序排列。",
                    "items": {
                        "type": "object",
                        "description": "一条对话消息。",
                        "properties": {
                            "role": {"type": "string", "description": "消息角色，例如 user 或 assistant。"},
                            "content": {"type": "string", "description": "消息文本内容。"},
                        },
                        "required": ["role", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["messages"],
            "additionalProperties": False,
        }
        if user_simulator_endpoint.get("request_schema") != expected_user_request:
            raise PipelineGenerationError("user_simulator request schema must use complete messages")
        reward_endpoint = next(item for item in endpoints if item.get("name") == "reward")
        if reward_endpoint.get("access") != "rl_trainer_only":
            raise PipelineGenerationError("reward endpoint must be Trainer-only")
        reward_endpoint = next(item for item in endpoints if item.get("name") == "reward")
        if reward_endpoint.get("formula") != reward_formula:
            raise PipelineGenerationError("runtime reward endpoint formula is inconsistent")
        replay_endpoint = next(item for item in endpoints if item.get("name") == "replay")
        if replay_endpoint.get("access") != "rl_trainer_only":
            raise PipelineGenerationError("runtime replay endpoint must be Trainer-only")
        security = interface.get("security")
        if not isinstance(security, dict):
            raise PipelineGenerationError("runtime_interface.security is required")
        trainer_security = security.get("trainer")
        if not isinstance(trainer_security, dict) or trainer_security.get("scheme") != "bearer" \
                or trainer_security.get("environment_variable") != "SANDBOX_TRAINER_API_KEY":
            raise PipelineGenerationError("runtime_interface trainer authentication is invalid")
        episode = interface.get("episode")
        if not isinstance(episode, dict) or episode.get("isolation") != "per_episode" \
                or episode.get("reset_accepts_seed") is not True \
                or episode.get("deterministic_replay") is not True:
            raise PipelineGenerationError("runtime_interface episode isolation/replay contract is invalid")
        llm_runtime = interface.get("llm_runtime")
        if not isinstance(llm_runtime, dict) or llm_runtime.get("api_key_environment_variable") != "SANDBOX_LLM_API_KEY":
            raise PipelineGenerationError("runtime_interface LLM credential boundary is invalid")
        evaluator_runtime = interface.get("evaluator_runtime")
        if not isinstance(evaluator_runtime, dict) or evaluator_runtime.get("mock_mode_environment_variable") != "SANDBOX_EVALUATOR_MOCK":
            raise PipelineGenerationError("runtime_interface evaluator mock contract is invalid")
        mutation_testing = interface.get("mutation_testing")
        if not isinstance(mutation_testing, dict) or mutation_testing.get("environment_variable") != "SANDBOX_MUTATION_MODE" \
                or not isinstance(mutation_testing.get("modes"), list) or not mutation_testing["modes"]:
            raise PipelineGenerationError("runtime_interface mutation testing contract is invalid")
        launcher = interface.get("launcher")
        if not isinstance(launcher, dict) or launcher.get("command") != ["python", "app.py", "--port", "{port}"]:
            raise PipelineGenerationError("runtime_interface launcher contract is invalid")
        errors = interface.get("errors")
        if not isinstance(errors, dict) or errors.get("content_type") != "application/json" \
                or not isinstance(errors.get("schema"), dict):
            raise PipelineGenerationError("runtime_interface error protocol is invalid")
        observability = interface.get("observability")
        if not isinstance(observability, dict) or observability.get("request_id_header") != "X-Request-ID" \
                or observability.get("credential_redaction") is not True:
            raise PipelineGenerationError("runtime_interface observability contract is invalid")

    @staticmethod
    def _validate_actions(actions: list[Any]) -> None:
        """Validate the action decomposition before tools are generated."""
        names: set[str] = set()
        for index, action in enumerate(actions):
            if not isinstance(action, dict):
                raise PipelineGenerationError(f"agent_actions[{index}] must be an object")
            name = action.get("name") or action.get("action")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise PipelineGenerationError(f"agent_actions[{index}] has invalid or duplicate name")
            if not isinstance(action.get("atomicity_rationale"), str) or not action["atomicity_rationale"].strip():
                raise PipelineGenerationError(f"agent action {name} requires atomicity_rationale")
            for field in ("inputs", "outputs"):
                values = action.get(field)
                if not isinstance(values, list):
                    raise PipelineGenerationError(f"agent action {name}.{field} must be a list")
                for item_index, item in enumerate(values):
                    if not isinstance(item, dict):
                        raise PipelineGenerationError(f"agent action {name}.{field}[{item_index}] must be an object")
                    for item_field in ("name", "description"):
                        if not isinstance(item.get(item_field), str) or not item[item_field].strip():
                            raise PipelineGenerationError(
                                f"agent action {name}.{field}[{item_index}] requires {item_field}"
                            )
            for field in ("preconditions", "effects"):
                values = action.get(field)
                if not isinstance(values, list) or any(not isinstance(value, str) or not value.strip() for value in values):
                    raise PipelineGenerationError(f"agent action {name}.{field} must be a list of non-empty strings")
            names.add(name)

    @staticmethod
    def _deterministic_actions(
        *, task_description: dict[str, Any], keywords: list[str], environment_mode: str,
        route_plan: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Build a minimal grounded plan after repeated action-model drift."""
        relevant = ContractCompiler._task_relevant_keywords(
            task_description, keywords
        )
        subject = relevant[0] if relevant else str(task_description.get("task", "任务"))[:16]
        actions: list[dict[str, Any]] = []

        def action(name: str, description: str, output: str) -> dict[str, Any]:
            return {
                "name": name,
                "description": description,
                "atomicity_rationale": "该动作只完成一个明确阶段，不合并后续判断或表达。",
                "inputs": [{"name": "task_input", "description": "用户提供的任务输入与约束。"}],
                "outputs": [{"name": output, "description": f"完成{name}后得到的可验证结果。"}],
                "preconditions": ["已获得当前阶段所需输入。"],
                "effects": [f"产生{output}，不引入任务外事实。"],
            }

        route_operations = (
            route_plan.get("environment_operations", [])
            if isinstance(route_plan, dict) else []
        )
        if route_operations:
            for index, operation in enumerate(route_operations, start=1):
                actions.append(action(
                    str(operation["action_name"]),
                    str(operation.get("purpose", "执行任务所需的环境操作。")),
                    f"environment_result_{index}",
                ))
        elif environment_mode in {"reference_data", "stateful"}:
            actions.append(action(
                f"读取{subject}任务数据", f"读取沙箱中与{subject}任务直接相关的记录。", "task_records"
            ))
        elif environment_mode == "external_capability":
            actions.append(action(
                f"查询{subject}外部信息", f"通过声明的外部能力获取与{subject}直接相关的信息。", "external_facts"
            ))
        actions.extend([
            action(f"分析{subject}任务输入", f"依据任务约束分析与{subject}相关的输入。", "analysis_result"),
            action(f"生成{subject}任务结果", f"根据已验证信息完成{subject}任务并遵守输出格式。", "final_result"),
        ])
        return actions

    @staticmethod
    def _ensure_data_access_action(
        actions: list[Any], *, environment_mode: str, table_names: list[str]
    ) -> list[Any]:
        """Add a stable read boundary when a data-backed task omitted one."""
        if environment_mode not in {"reference_data", "stateful"} or not table_names:
            return actions
        chinese_markers = ("读取", "查询", "检索", "查找", "获取")
        english_pattern = re.compile(r"\b(?:read|query|search|lookup|get)\b", re.I)
        if any(
            isinstance(action, dict)
            and (
                any(marker in f"{action.get('name', '')} {action.get('description', '')}" for marker in chinese_markers)
                or english_pattern.search(f"{action.get('name', '')} {action.get('description', '')}")
            )
            for action in actions
        ):
            return actions
        name = "读取任务参考数据"
        used = {str(action.get("name")) for action in actions if isinstance(action, dict)}
        suffix = 2
        while name in used:
            name = f"读取任务参考数据{suffix}"
            suffix += 1
        return [
            {
                "name": name,
                "description": f"从沙箱只读数据表中获取完成任务所需的记录：{', '.join(table_names)}。",
                "atomicity_rationale": "该动作仅负责读取环境事实，不进行比较、决策或最终回答。",
                "inputs": [{"name": "query", "description": "用户目标和筛选条件"}],
                "outputs": [{"name": "records", "description": "与任务相关的只读业务记录"}],
                "preconditions": ["沙箱参考数据已初始化"],
                "effects": ["向 Agent 提供可验证的环境事实，不修改业务状态"],
            },
            *actions,
        ]

    @staticmethod
    def _normalize_data_access_capabilities(
        capabilities: list[Any], *, actions: list[Any], environment_mode: str
    ) -> list[Any]:
        if environment_mode not in {"reference_data", "stateful"}:
            return capabilities
        action_text = {
            str(action.get("name")): f"{action.get('name', '')} {action.get('description', '')}"
            for action in actions if isinstance(action, dict)
        }
        chinese_markers = ("读取", "查询", "检索", "查找", "获取")
        english_pattern = re.compile(r"\b(?:read|query|search|lookup|get)\b", re.I)
        normalized: list[Any] = []
        for capability in capabilities:
            if not isinstance(capability, dict):
                normalized.append(capability)
                continue
            item = dict(capability)
            text = action_text.get(str(item.get("action_name")), "")
            if any(marker in text for marker in chinese_markers) or english_pattern.search(text):
                item["kind"] = "environment_operation"
                item["requires_tool"] = True
                if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                    item["reason"] = "该动作必须读取沙箱中的参考或业务数据。"
            normalized.append(item)
        return normalized

    @staticmethod
    def _normalize_capability_dependencies(
        capabilities: list[Any], *, actions: list[Any]
    ) -> list[Any]:
        """Translate common ordinal dependency aliases to canonical action names.

        Dependency topology belongs to the action/capability contract, while
        weaker models often serialize an intended edge as ``step-1`` or
        ``action_1``.  Canonicalizing those identifiers preserves the edge;
        unknown values remain untouched so semantic validation still rejects
        invented dependencies.
        """
        action_names = [
            str(action["name"]) for action in actions
            if isinstance(action, dict) and isinstance(action.get("name"), str)
        ]

        def key(value: str) -> str:
            return re.sub(r"[\s_-]+", "-", value.strip().lower())

        aliases: dict[str, str] = {}
        for index, action_name in enumerate(action_names, start=1):
            aliases[key(action_name)] = action_name
            for alias in (str(index), f"step-{index}", f"action-{index}"):
                aliases[key(alias)] = action_name
        for action in actions:
            if not isinstance(action, dict) or action.get("name") not in action_names:
                continue
            for field in ("id", "action_id", "step_id"):
                value = action.get(field)
                if isinstance(value, str) and value.strip():
                    aliases[key(value)] = str(action["name"])

        normalized: list[Any] = []
        for capability in capabilities:
            if not isinstance(capability, dict):
                normalized.append(capability)
                continue
            item = dict(capability)
            dependencies = item.get("dependencies", [])
            if isinstance(dependencies, list):
                canonical: list[Any] = []
                for dependency in dependencies:
                    value = aliases.get(key(dependency), dependency) if isinstance(dependency, str) else dependency
                    if value not in canonical:
                        canonical.append(value)
                item["dependencies"] = canonical
            normalized.append(item)
        return normalized

    @staticmethod
    def _apply_route_capability_contract(
        capabilities: list[Any], *, route_plan: dict[str, Any]
    ) -> list[Any]:
        """Project the validated route skeleton onto capability classification."""
        operations = {
            item["action_name"]: item
            for item in route_plan.get("environment_operations", [])
            if isinstance(item, dict) and isinstance(item.get("action_name"), str)
        }
        normalized: list[Any] = []
        for capability in capabilities:
            if not isinstance(capability, dict):
                normalized.append(capability)
                continue
            item = dict(capability)
            route_operation = operations.get(item.get("action_name"))
            if route_operation is not None:
                reason = str(item.get("reason", ""))
                if capability_disclaims_tool(reason) or (
                    re.search(
                        r"(?:前序|上一步|已有|给定).{0,40}"
                        r"(?:求和|合计|统计字符|格式化|生成.{0,8}表格)",
                        reason,
                    )
                    and not any(marker in reason for marker in (
                        "私有", "内部业务", "数据表", "外部系统",
                    ))
                ):
                    raise PipelineGenerationError(
                        "route operation is agent reasoning or response composition: "
                        f"{item.get('action_name')}"
                    )
                item["kind"] = "environment_operation"
                item["requires_tool"] = True
                item["dependencies"] = list(route_operation.get("dependencies", []))
                if not isinstance(item.get("reason"), str) or not item["reason"].strip():
                    item["reason"] = str(route_operation.get("purpose", "环境操作"))
            normalized.append(item)
        prior_only = prior_result_only_tool_actions(normalized)
        if prior_only:
            raise PipelineGenerationError(
                "route operation only processes prior results without new environment access: "
                + ", ".join(prior_only)
            )
        return normalized

    @staticmethod
    def _validate_capability_plan(
        capabilities: list[Any],
        actions: list[Any],
        *,
        environment_mode: str | None = None,
        has_business_data: bool = False,
        training_category: str | None = None,
    ) -> None:
        action_names = {
            action.get("name") for action in actions
            if isinstance(action, dict) and isinstance(action.get("name"), str)
        }
        seen: set[str] = set()
        response_markers = (
            "生成markdown", "生成表格", "格式化", "输出最终", "生成最终",
            "生成推荐", "生成说明", "撰写", "总结", "generate markdown",
            "format output", "final response", "write summary",
        )
        for index, capability in enumerate(capabilities):
            if not isinstance(capability, dict):
                raise PipelineGenerationError(f"capabilities[{index}] must be an object")
            action_name = capability.get("action_name")
            kind = capability.get("kind")
            requires_tool = capability.get("requires_tool")
            if action_name not in action_names or action_name in seen:
                raise PipelineGenerationError(f"capabilities[{index}] has unknown or duplicate action_name")
            if kind not in {"environment_operation", "agent_reasoning", "agent_response"}:
                raise PipelineGenerationError(f"capabilities[{index}] has invalid kind")
            if not isinstance(requires_tool, bool):
                raise PipelineGenerationError(f"capabilities[{index}] requires boolean requires_tool")
            if requires_tool != (kind == "environment_operation"):
                raise PipelineGenerationError(f"capabilities[{index}] kind/requires_tool conflict")
            if requires_tool and capability_disclaims_tool(capability.get("reason")):
                raise PipelineGenerationError(
                    f"capabilities[{index}] declares a tool unnecessary"
                )
            if requires_tool and any(marker in str(action_name).lower() for marker in response_markers):
                raise PipelineGenerationError(
                    f"capabilities[{index}] delegates response composition to a tool"
                )
            if not isinstance(capability.get("reason"), str) or not capability["reason"].strip():
                raise PipelineGenerationError(f"capabilities[{index}] requires reason")
            seen.add(action_name)
        if seen != action_names:
            missing = sorted(action_names - seen)
            raise PipelineGenerationError(f"capability plan must cover every action; missing={missing}")
        environment_actions = {item["action_name"] for item in capabilities if item["requires_tool"]}
        dependency_graph = {}
        for item in capabilities:
            dependencies = item.get("dependencies", [])
            if not isinstance(dependencies, list) or any(not isinstance(name, str) or name not in action_names for name in dependencies):
                raise PipelineGenerationError("capability dependencies must reference known action names")
            dependency_graph[item["action_name"]] = set(dependencies)
        remaining = set(dependency_graph)
        while remaining:
            roots = {name for name in remaining if not dependency_graph[name] & remaining}
            if not roots:
                raise PipelineGenerationError("capability dependency graph contains a cycle")
            remaining -= roots
        if training_category == "multi_step_agentic" and not any(
            dependency_graph[name] & environment_actions for name in environment_actions
        ):
            raise PipelineGenerationError("multi_step_agentic capability plan requires a real environment-action dependency; declare dependencies using action names, not reward step IDs")
        if environment_mode in {"reference_data", "stateful"} and has_business_data and not any(
            isinstance(item, dict) and item.get("requires_tool") is True
            for item in capabilities
        ):
            raise PipelineGenerationError(
                "capability plan for a data-backed environment must expose at least one data access operation"
            )

    @staticmethod
    def _validate_data_grounding_audit(audit: Any) -> None:
        if not isinstance(audit, dict):
            raise PipelineGenerationError("environment data grounding audit must be an object")
        fields = ("task_supported", "decision_determinate", "facts_consistent")
        if any(not isinstance(audit.get(field), bool) for field in fields):
            raise PipelineGenerationError(
                "environment data grounding audit requires boolean verdicts"
            )
        issues = audit.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) or not issue.strip() for issue in issues
        ):
            raise PipelineGenerationError(
                "environment data grounding audit requires a list of issue strings"
            )
        failed = [field for field in fields if audit[field] is False]
        if failed:
            detail = "; ".join(issues) if issues else "no details supplied"
            raise PipelineGenerationError(
                f"environment data does not ground the task: failed={failed}; issues={detail}"
            )

    @staticmethod
    def _validate_task_grounding_audit(audit: Any) -> None:
        if not isinstance(audit, dict):
            raise PipelineGenerationError("task grounding audit must be an object")
        fields = (
            "self_contained", "no_unprovided_facts",
            "expected_result_derivable", "internally_consistent",
        )
        if any(not isinstance(audit.get(field), bool) for field in fields):
            raise PipelineGenerationError("task grounding audit requires boolean verdicts")
        issues = audit.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) or not issue.strip() for issue in issues
        ):
            raise PipelineGenerationError("task grounding audit requires issue strings")
        failed = [field for field in fields if audit[field] is False]
        if failed:
            detail = "; ".join(issues) if issues else "no details supplied"
            raise PipelineGenerationError(
                f"task description is not grounded: failed={failed}; issues={detail}"
            )

    @staticmethod
    def _validate_task_observability_audit(audit: Any) -> None:
        if not isinstance(audit, dict):
            raise PipelineGenerationError("task observability audit must be an object")
        if any(not isinstance(audit.get(field), bool)
               for field in ("answer_observable", "goal_covered")):
            raise PipelineGenerationError("task observability audit requires boolean verdicts")
        issues = audit.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(item, str) or not item.strip() for item in issues
        ):
            raise PipelineGenerationError("task observability audit requires issue strings")
        failed = [field for field in ("answer_observable", "goal_covered")
                  if audit[field] is False]
        if failed:
            raise PipelineGenerationError(
                f"task success is not observable: failed={failed}; "
                f"issues={'; '.join(issues) if issues else 'no details supplied'}"
            )

    @staticmethod
    def _validate_observability_repair_progress(
        previous: list[Any], candidate: list[Any],
    ) -> None:
        """A second audit must examine a materially changed success trace."""
        def business_calls(scenarios: list[Any]) -> list[dict[str, Any]]:
            success = next(
                (item for item in scenarios
                 if isinstance(item, dict) and item.get("kind") == "goal_success"),
                {},
            )
            return [
                {key: step.get(key) for key in ("tool_name", "arguments", "capture")}
                for step in success.get("steps", [])
                if isinstance(step, dict) and step.get("operation") == "tool_call"
            ]

        if business_calls(previous) == business_calls(candidate):
            raise PipelineGenerationError(
                "observability repair left business tool calls unchanged"
            )

    @staticmethod
    def _replace_success_tool_steps(
        scenarios: list[Any], tool_steps: Any,
    ) -> list[Any]:
        """Apply a focused tool trace patch without rewriting answer or assertions."""
        if (not isinstance(tool_steps, list) or not tool_steps
                or any(not isinstance(step, dict) or step.get("operation") != "tool_call"
                       for step in tool_steps)):
            raise PipelineGenerationError("observability repair requires non-empty tool_call steps")
        candidate = copy.deepcopy(scenarios)
        success = next((item for item in candidate
                        if isinstance(item, dict) and item.get("kind") == "goal_success"), None)
        if success is None or not isinstance(success.get("steps"), list):
            raise PipelineGenerationError("observability repair lacks goal_success steps")
        steps = success["steps"]
        positions = [index for index, step in enumerate(steps)
                     if isinstance(step, dict) and step.get("operation") == "tool_call"]
        if positions and positions != list(range(positions[0], positions[-1] + 1)):
            raise PipelineGenerationError("observability repair requires contiguous business calls")
        insertion = positions[0] if positions else next(
            (index for index, step in enumerate(steps)
             if isinstance(step, dict) and step.get("operation") == "agent_response"),
            len(steps),
        )
        retained = [step for step in steps
                    if not isinstance(step, dict) or step.get("operation") != "tool_call"]
        success["steps"] = [
            *retained[:insertion], *copy.deepcopy(tool_steps), *retained[insertion:],
        ]
        return candidate

    @staticmethod
    def _validate_task_description_consistency(description: Any) -> None:
        """Check structure only; task-specific semantics belong to grounding audits."""
        if not isinstance(description, dict):
            raise PipelineGenerationError("task description must be an object")
        if not isinstance(description.get("task"), str) or not description["task"].strip():
            raise PipelineGenerationError("task description requires non-empty task")
        if not isinstance(description.get("requirements", {}), dict):
            raise PipelineGenerationError("task requirements must be an object")

    @staticmethod
    def _validate_scene_business_plan(
        plan: Any, *, keywords: list[str], graph_context: dict[str, Any],
        training_category: str,
    ) -> None:
        if not isinstance(plan, dict):
            raise PipelineGenerationError("scene business plan must be an object")
        terms = [*keywords, *graph_context.get("nodes", [])]
        canonical = {
            unicodedata.normalize("NFKC", item).strip().casefold()
            for item in terms if isinstance(item, str) and item.strip()
        }
        anchor = plan.get("anchor_term")
        if (not isinstance(anchor, str)
                or unicodedata.normalize("NFKC", anchor).strip().casefold() not in canonical):
            raise PipelineGenerationError("scene business plan anchor must come from sampled Scene")
        for field in ("business_context", "user_role", "user_need"):
            value = plan.get(field)
            if not isinstance(value, str) or len(value.strip()) < 2:
                raise PipelineGenerationError(f"scene business plan lacks {field}")
        records = plan.get("source_records")
        if training_category != "direct_response" and (
            not isinstance(records, list) or not records
            or any(not isinstance(item, str) or not item.strip() for item in records)
        ):
            raise PipelineGenerationError("agentic scene business plan lacks source records")
        dependency = plan.get("tool_dependency")
        if training_category == "multi_step_agentic" and (
            not isinstance(dependency, str) or len(dependency.strip()) < 8
        ):
            raise PipelineGenerationError("multi-step scene business plan lacks tool dependency")
        visible = " ".join(str(plan.get(field, "")) for field in (
            "business_context", "user_role", "user_need",
        ))
        if any(marker in visible for marker in (
            "沙箱", "图谱节点", "采样关键词", "任务生成",
        )):
            raise PipelineGenerationError("scene business plan exposes generation internals")

    @staticmethod
    def _validate_scene_anchor_in_task(
        description: dict[str, Any], plan: dict[str, Any],
    ) -> None:
        public = description.get("public_input")
        initial = public.get("initial_user_message", "") if isinstance(public, dict) else ""
        prose = unicodedata.normalize(
            "NFKC", str(description.get("task", "")) + " " + str(initial),
        ).casefold()
        anchor = unicodedata.normalize("NFKC", plan["anchor_term"]).strip().casefold()
        if anchor not in prose:
            raise PipelineGenerationError("task description omits the selected Scene anchor")

    @staticmethod
    def _validate_graph_keyword_alignment(
        description: dict[str, Any], keywords: list[str], graph_context: dict[str, Any],
    ) -> None:
        """Keep the user-facing task anchored to the sampled Scene path."""
        nodes = graph_context.get("nodes", []) if isinstance(graph_context, dict) else []
        terms = [
            unicodedata.normalize("NFKC", item).strip().casefold()
            for item in [*keywords, *(nodes if isinstance(nodes, list) else [])]
            if isinstance(item, str) and len(item.strip()) >= 2
        ]
        if not terms:
            raise PipelineGenerationError("sampled Scene path has no usable keywords")
        public = description.get("public_input")
        initial = public.get("initial_user_message", "") if isinstance(public, dict) else ""
        prose = unicodedata.normalize(
            "NFKC", str(description.get("task", "")) + " " + str(initial),
        ).casefold()
        relation = graph_context.get("relation")
        if isinstance(relation, str) and relation and relation.casefold() in prose:
            raise PipelineGenerationError("task description exposes internal graph relation metadata")
        if any(marker in prose for marker in (
            "沙箱", "图谱节点", "采样关键词", "场景关键词", "关键词组合", "任务生成",
        )):
            raise PipelineGenerationError(
                "task description exposes generation or sandbox internals"
            )
        if not any(term in prose for term in terms):
            raise PipelineGenerationError(
                "task description does not use any sampled Scene keyword"
            )

    @staticmethod
    def _validate_route_plan(route_plan: Any, training_category: str) -> None:
        if not isinstance(route_plan, dict):
            raise PipelineGenerationError("task route_plan must be an object")
        operations = route_plan.get("environment_operations")
        if not isinstance(operations, list):
            raise PipelineGenerationError(
                "task route_plan.environment_operations must be a list"
            )
        names: list[str] = []
        dependencies: dict[str, list[str]] = {}
        for index, operation in enumerate(operations):
            if not isinstance(operation, dict):
                raise PipelineGenerationError(
                    f"route_plan.environment_operations[{index}] must be an object"
                )
            name = operation.get("action_name")
            purpose = operation.get("purpose")
            declared = operation.get("dependencies")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise PipelineGenerationError(
                    f"route_plan.environment_operations[{index}] has invalid action_name"
                )
            if (
                re.search(
                    r"^(?:generate|format|summarize|compose|render)_.*"
                    r"(?:table|report|summary|answer|response)$",
                    name, re.I,
                )
                or re.search(r"(?:生成|整理|撰写|格式化).{0,16}(?:表格|报告|清单|最终回答)", name)
            ):
                raise PipelineGenerationError(
                    f"route_plan.environment_operations[{index}] delegates final response composition"
                )
            if not isinstance(purpose, str) or not purpose.strip():
                raise PipelineGenerationError(
                    f"route_plan.environment_operations[{index}] requires purpose"
                )
            if not isinstance(declared, list) or any(
                not isinstance(item, str) or item not in names for item in declared
            ):
                raise PipelineGenerationError(
                    f"route_plan.environment_operations[{index}] dependencies must reference earlier actions"
                )
            names.append(name)
            dependencies[name] = list(dict.fromkeys(declared))
        if training_category == "direct_response" and operations:
            raise PipelineGenerationError(
                "direct_response route_plan must not contain environment operations"
            )
        if training_category == "simple_agentic" and len(operations) != 1:
            raise PipelineGenerationError(
                "simple_agentic route_plan requires exactly one environment operation"
            )
        if training_category == "multi_step_agentic" and (
            len(operations) < 2 or not any(dependencies.values())
        ):
            raise PipelineGenerationError(
                "multi_step_agentic route_plan requires dependent environment operations"
            )

    @staticmethod
    def _validate_route_input_boundary(
        description: dict[str, Any], training_category: str, task_intent: str | None = None,
    ) -> None:
        """Reject tool routes whose complete truth is already in public input."""
        if training_category == "direct_response":
            return
        public_input = description.get("public_input", {})
        corpus = " ".join(
            str(description.get(key, ""))
            for key in ("task", "goal", "context", "expected_result", "requirements")
        )
        if isinstance(public_input, dict):
            corpus += " " + str(public_input.get("initial_user_message", ""))
        complete_input_supplied = any(
            re.search(pattern, corpus) is not None
            for pattern in (
                r"(?:基于|仅依赖|仅使用|只依赖|只使用)用户提供的.{0,12}(?:规格|数据|资料|文本|列表|清单)",
                r"基于以下提供的.{0,8}(?:规格|数据|资料|文本)",
                r"从用户提供的.{0,8}(?:列表|清单|文本|数据)中",
            )
        )
        operations = description.get("route_plan", {}).get("environment_operations", [])
        route_text = json.dumps(operations, ensure_ascii=False).lower()
        private_markers = (
            "内部", "私有", "沙箱", "数据库", "业务数据", "库存", "目录", "历史记录",
            "系统记录", "internal", "private", "sandbox", "database", "inventory",
            "catalog", "system record",
        )
        has_private_route = any(marker in route_text for marker in private_markers)
        from ..tasks.task_routing import public_calculation_is_self_contained
        if public_calculation_is_self_contained(description, task_intent):
            complete_input_supplied = True
            has_private_route = False
        if complete_input_supplied and not has_private_route:
            raise PipelineGenerationError(
                f"{training_category} task is fully solvable from public user input; "
                "make at least one route operation depend on sandbox-private business data"
            )

    @staticmethod
    def _validate_no_deferred_business_truth(
        description: dict[str, Any], task_intent: str
    ) -> None:
        """Do not make a future simulator invent authoritative mutation inputs.

        Clarification dialogue is still useful for preferences and presentation
        details.  It cannot supply a value that determines the sandbox's target
        business state unless that value (or a deterministic derivation rule) is
        already part of the immutable task contract.
        """
        if task_intent not in {"modify", "execute", "schedule"}:
            return
        corpus = json.dumps({
            key: description.get(key)
            for key in ("task", "goal", "expected_result")
        } | {
            "initial_user_message": (
                description.get("public_input", {}).get("initial_user_message")
                if isinstance(description.get("public_input"), dict) else None
            )
        }, ensure_ascii=False).lower()
        unresolved_markers = (
            "暂时没想好", "先问我", "稍后提供", "之后提供", "待我提供",
            "需要向用户询问", "询问正确", "ask me", "provide later",
            "to be provided",
        )
        if any(marker in corpus for marker in unresolved_markers):
            raise PipelineGenerationError(
                "stateful task defers required business truth to an unspecified "
                "future user reply"
            )

    @staticmethod
    def _validate_route_action_coverage(
        actions: list[Any], *, route_plan: dict[str, Any]
    ) -> None:
        action_names = {
            action.get("name") for action in actions if isinstance(action, dict)
        }
        required = {
            operation.get("action_name")
            for operation in route_plan.get("environment_operations", [])
            if isinstance(operation, dict)
        }
        missing = sorted(required - action_names)
        if missing:
            raise PipelineGenerationError(
                f"agent actions do not implement route_plan operations: {missing}"
            )

    @staticmethod
    def _validate_action_alignment_audit(audit: Any) -> None:
        if not isinstance(audit, dict):
            raise PipelineGenerationError("agent action audit must be an object")
        fields = ("aligned", "complete", "atomic")
        if any(not isinstance(audit.get(field), bool) for field in fields):
            raise PipelineGenerationError("agent action audit requires boolean verdicts")
        issues = audit.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) or not issue.strip() for issue in issues
        ):
            raise PipelineGenerationError("agent action audit requires issue strings")
        failed = [field for field in fields if audit[field] is False]
        if failed:
            detail = "; ".join(issues) if issues else "no details supplied"
            raise PipelineGenerationError(
                f"agent actions are not task-aligned: failed={failed}; issues={detail}"
            )

    @staticmethod
    def _validate_action_grounding(
        actions: list[Any], *, task_description: dict[str, Any], keywords: list[str]
    ) -> None:
        """Reject action plans that import entities or numbers from another task."""
        public_input = task_description.get("public_input", {})
        public_message = (public_input.get("initial_user_message", "")
                          if isinstance(public_input, dict) else "")
        task_text = json.dumps(
            {"task": task_description.get("task", ""), "initial_user_message": public_message},
            ensure_ascii=False,
        ).lower()
        action_core = []
        for item in actions:
            if not isinstance(item, dict):
                continue
            action_core.append({
                "name": item.get("name"),
                "description": item.get("description"),
                "outputs": item.get("outputs", []),
                "effects": item.get("effects", []),
            })
        action_text = json.dumps(action_core, ensure_ascii=False).lower()
        relevant_keywords = [
            str(keyword).strip().lower() for keyword in keywords
            if len(str(keyword).strip()) >= 2 and str(keyword).strip().lower() in task_text
        ]
        if relevant_keywords and not any(keyword in action_text for keyword in relevant_keywords):
            raise PipelineGenerationError(
                "agent actions omit every task-specific keyword and may belong to another task"
            )

        action_names = [
            str(item.get("name", "")).strip() for item in actions if isinstance(item, dict)
        ]
        generic_count = sum(
            name in GENERIC_ACTION_NAMES
            or any(marker.lower() in name.lower() for marker in GENERIC_ACTION_MARKERS)
            for name in action_names
        )
        if action_names and generic_count * 2 >= len(action_names):
            raise PipelineGenerationError(
                "agent actions are predominantly generic placeholders instead of task-specific steps"
            )
        allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", task_text))
        # Numeric suffixes in stable action identifiers (for example
        # lookup_stage_1) are graph identity, not user-facing constraints.
        numeric_semantics = json.dumps([
            {
                "description": item.get("description"),
                "inputs": item.get("inputs", []),
                "outputs": item.get("outputs", []),
                "preconditions": item.get("preconditions", []),
                "effects": item.get("effects", []),
            }
            for item in actions if isinstance(item, dict)
        ], ensure_ascii=False).lower()
        action_numbers = set(re.findall(r"\d+(?:\.\d+)?", numeric_semantics))
        invented = sorted(action_numbers - allowed_numbers)
        if invented:
            raise PipelineGenerationError(
                f"agent actions invent numeric constraints absent from the task: {invented}"
            )

    @staticmethod
    def _normalize_action_numeric_examples(
        actions: list[Any], *, task_description: dict[str, Any],
    ) -> list[Any]:
        """Remove model-invented example numbers without changing action structure."""
        public_input = task_description.get("public_input", {})
        public_message = (public_input.get("initial_user_message", "")
                          if isinstance(public_input, dict) else "")
        allowed = set(re.findall(
            r"\d+(?:\.\d+)?",
            json.dumps(
                {"task": task_description.get("task", ""),
                 "initial_user_message": public_message},
                ensure_ascii=False,
            ),
        ))

        def clean(value: Any) -> Any:
            if isinstance(value, str):
                # Only examples can be discarded safely. Replacing a numeric
                # business rule (for example ">0" or "+2 days") with a
                # placeholder silently changes the requested operation.
                if not any(marker in value.lower() for marker in (
                    "例如", "比如", "示例", "举例", "e.g.", "for example",
                )) or re.search(r"(?:>=|<=|[><=]|\+\s*\d)", value):
                    return value
                return re.sub(
                    r"\d+(?:\.\d+)?",
                    lambda match: match.group(0) if match.group(0) in allowed else "用户提供值",
                    value,
                )
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, dict):
                return {key: (item if key == "name" else clean(item)) for key, item in value.items()}
            return value

        return [clean(action) for action in actions]

    @staticmethod
    def _task_relevant_keywords(
        task_description: dict[str, Any], keywords: list[str]
    ) -> list[str]:
        task_text = json.dumps(task_description, ensure_ascii=False).lower()
        return [
            str(keyword).strip().lower() for keyword in keywords
            if len(str(keyword).strip()) >= 2 and str(keyword).strip().lower() in task_text
        ]

    @staticmethod
    def _validate_data_keyword_alignment(
        data_tables: list[Any], *, task_description: dict[str, Any], keywords: list[str]
    ) -> None:
        """Keep generated business fixtures in the task's semantic domain."""
        relevant = ContractCompiler._task_relevant_keywords(
            task_description, keywords
        )
        if not relevant or not data_tables:
            return
        data_text = json.dumps(data_tables, ensure_ascii=False).lower()
        if not any(keyword in data_text for keyword in relevant):
            raise PipelineGenerationError(
                "environment data omits every task-specific keyword and may belong to another task"
            )

    @staticmethod
    def _validate_stateful_preconditions(
        data_tables: list[Any], *, task_description: dict[str, Any], environment_mode: str
    ) -> None:
        """Reject baselines that already contain an explicitly requested end state."""
        if environment_mode != "stateful":
            return
        task_text = json.dumps(task_description, ensure_ascii=False)
        changes = re.findall(
            r"从\s*[“\"]([^”\"]+)[”\"]\s*(?:更正|修改|更新|调整|改|变更)?\s*为\s*[“\"]([^”\"]+)[”\"]",
            task_text,
        )
        if not changes:
            return
        rows_text = json.dumps(data_tables, ensure_ascii=False)
        missing_old = [old for old, _new in changes if old not in rows_text]
        if missing_old:
            raise PipelineGenerationError(
                "stateful baseline omits explicit precondition value(s): "
                + ", ".join(sorted(set(missing_old)))
            )

    @staticmethod
    def _validate_training_contract_consistency(
        *,
        task_description: dict[str, Any],
        training_contract: dict[str, Any],
        keywords: list[str],
        environment_mode: str,
        data_tables: list[Any],
        actions: list[Any],
        tools: list[Any],
        tool_bindings: list[Any],
        metrics: list[Any],
        success_fixture: str,
    ) -> None:
        """Final cross-stage gate before a task is accepted for sandbox construction."""
        category = training_contract.get("category")
        allowed_modes = training_contract.get("allowed_environment_modes", [])
        if environment_mode not in allowed_modes:
            raise PipelineGenerationError(
                f"final consistency: {category} does not allow environment mode {environment_mode}"
            )
        ContractCompiler._validate_task_description_consistency(task_description)
        task_text = json.dumps(task_description, ensure_ascii=False).lower()
        action_text = json.dumps(actions, ensure_ascii=False).lower()
        if environment_mode == "stateless" and any(
            marker in task_text for marker in ("用户提供", "给定的", "provided")
        ) and any(
            marker in action_text for marker in ("生成候选列表", "创建候选列表", "invent candidates")
        ):
            raise PipelineGenerationError(
                "final consistency: stateless actions invent candidates instead of using user input"
            )
        ContractCompiler._validate_action_grounding(
            actions, task_description=task_description, keywords=keywords
        )
        if environment_mode in {"reference_data", "stateful"}:
            ContractCompiler._validate_data_keyword_alignment(
                data_tables, task_description=task_description, keywords=keywords
            )
        tool_names = {
            item.get("function", {}).get("name") for item in tools
            if isinstance(item, dict) and isinstance(item.get("function"), dict)
        }
        action_names = {
            item.get("name") for item in actions if isinstance(item, dict)
        }
        for binding in tool_bindings:
            if not isinstance(binding, dict):
                raise PipelineGenerationError("final consistency: invalid tool binding")
            if binding.get("tool_name") not in tool_names or binding.get("action_name") not in action_names:
                raise PipelineGenerationError(
                    "final consistency: tool binding references an unknown tool or action"
                )
        bounds = training_contract.get("business_tools", {})
        count = len(tool_bindings)
        minimum, maximum = bounds.get("min", 0), bounds.get("max")
        if count < minimum or (maximum is not None and count > maximum):
            raise PipelineGenerationError(
                f"final consistency: {category} business tool count {count} violates [{minimum}, {maximum}]"
            )
        if not isinstance(metrics, list) or not metrics:
            raise PipelineGenerationError("final consistency: task has no reward metrics")
        ContractCompiler._validate_success_fixture_constraints(
            success_fixture, task_description=task_description
        )

    @staticmethod
    def _validate_task_generation_scope(
        task_description: dict[str, Any], *, graph_context: dict[str, Any]
    ) -> None:
        text = json.dumps(task_description, ensure_ascii=False).lower()
        chain_of_thought_markers = (
            "思考过程", "推理过程", "内心推理", "逐步思考",
            "chain of thought", "hidden reasoning", "show your reasoning",
        )
        if any(marker in text for marker in chain_of_thought_markers):
            raise PipelineGenerationError(
                "task must request concise rationale, not hidden chain-of-thought"
            )
        if ContractCompiler._is_high_stakes_task(task_description):
            evidence = json.dumps(graph_context, ensure_ascii=False)
            if not re.search(r"https?://[^\s\"']+", evidence, re.I):
                raise PipelineGenerationError(
                    "high-stakes task requires authoritative source URLs in graph_context"
                )

    @staticmethod
    def _is_high_stakes_task(task_description: dict[str, Any]) -> bool:
        """Classify the user-facing decision, not incidental generated metadata."""
        text = json.dumps({
            key: task_description.get(key)
            for key in ("task", "goal", "context", "requirements")
        }, ensure_ascii=False).lower()
        if any(marker in text for marker in HIGH_STAKES_MARKERS):
            return True
        return (
            any(marker in text for marker in HIGH_STAKES_CHEMICAL_MARKERS)
            and any(marker in text for marker in HIGH_STAKES_CHEMICAL_RISK_MARKERS)
        )

    @staticmethod
    def _validate_scene_source_scope(
        plan: dict[str, Any], *, has_source_urls: bool,
    ) -> None:
        if has_source_urls:
            return
        projected_task = {
            "task": plan.get("user_need"),
            "goal": plan.get("tool_dependency"),
            "context": plan.get("business_context"),
        }
        if ContractCompiler._is_high_stakes_task(projected_task):
            raise PipelineGenerationError(
                "unsourced high-stakes scene business plan"
            )

    @staticmethod
    def _normalize_reasoning_request(value: Any) -> Any:
        """Rewrite requests for hidden reasoning into an observable short rationale."""
        replacements = (
            ("逐步思考", "给出简要依据"), ("思考过程", "简要依据"),
            ("推理过程", "简要依据"), ("内心推理", "简要依据"),
            ("chain of thought", "concise rationale"),
            ("hidden reasoning", "concise rationale"),
            ("show your reasoning", "give a concise rationale"),
        )
        if isinstance(value, str):
            # A prohibition on exposing private reasoning is already a valid
            # constraint. Rewriting its object to "简要依据" reverses the
            # meaning and contradicts requests for a short conclusion basis.
            result = re.sub(
                r"(?:不要|不得|禁止|无需|无须|不必|不应)\s*"
                r"(?:输出|展示|提供|写出|呈现|包含)?\s*"
                r"(?:你的|完整的)?\s*"
                r"(?:思考过程|推理过程|内心推理|逐步思考)",
                "不要输出内部思维链", value,
            )
            for old, new in replacements:
                result = re.sub(re.escape(old), new, result, flags=re.I)
            return result
        if isinstance(value, list):
            return [ContractCompiler._normalize_reasoning_request(item) for item in value]
        if isinstance(value, dict):
            return {
                key: ContractCompiler._normalize_reasoning_request(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _normalize_unsourced_governance_task(value: Any) -> Any:
        """Turn unsourced governance wording into fixture-owned rule checks.

        Medical, financial, tax and hazardous-chemical decisions are not
        rewritten here; without authoritative sources they remain rejected.
        """
        replacements = (
            ("法律法规", "内部业务规则"), ("法律规定", "内部业务规则"),
            ("法规要求", "内部规则要求"), ("政策文件", "内部规则文档"),
            ("监管要求", "内部审核要求"), ("合规性", "内部规则一致性"),
            ("合规", "符合内部规则"),
            ("legal compliance", "internal rule consistency"),
            ("regulatory compliance", "internal rule consistency"),
        )
        if isinstance(value, str):
            result = value
            for old, new in replacements:
                result = re.sub(re.escape(old), new, result, flags=re.I)
            return result
        if isinstance(value, list):
            return [
                ContractCompiler._normalize_unsourced_governance_task(item)
                for item in value
            ]
        if isinstance(value, dict):
            return {
                key: ContractCompiler._normalize_unsourced_governance_task(item)
                for key, item in value.items()
            }
        return value

    @staticmethod
    def _unwrap_task_description(value: dict[str, Any]) -> dict[str, Any]:
        """Accept common repair envelopes while keeping one canonical shape."""
        if isinstance(value.get("output"), dict) and isinstance(value["output"].get("task"), str):
            return value["output"]
        if isinstance(value.get("task_description"), dict) and isinstance(
            value["task_description"].get("task"), str
        ):
            return value["task_description"]
        return value

    @staticmethod
    def _unwrap_structured_output(
        value: Any, *, expected_fields: set[str]
    ) -> Any:
        """Unwrap common model response envelopes without weakening validation."""
        current = value
        for _ in range(3):
            if not isinstance(current, dict):
                break
            if expected_fields.issubset(current):
                return current
            nested = current.get("output")
            if isinstance(nested, str):
                try:
                    nested = json.loads(nested)
                except (TypeError, ValueError, json.JSONDecodeError):
                    break
            if not isinstance(nested, dict):
                break
            current = nested
        return current

    @staticmethod
    def _resolve_environment_plan(
        candidate: Any, *, task_description: dict[str, Any], task_intent: str
    ) -> dict[str, Any]:
        modes = {"stateless", "reference_data", "stateful", "external_capability"}
        value = dict(candidate) if isinstance(candidate, dict) else {}
        mode = value.get("mode")
        fallback_reason: str | None = None
        aliases = {
            "none": "stateless", "no_data": "stateless",
            "no_business_data": "stateless", "text_only": "stateless",
            "pure_text": "stateless", "无状态": "stateless", "纯文本": "stateless",
            "reference": "reference_data", "read_only": "reference_data",
            "readonly": "reference_data", "read_only_data": "reference_data",
            "只读数据": "reference_data", "参考数据": "reference_data",
            "persistent": "stateful", "persistence": "stateful",
            "business_state": "stateful", "有状态": "stateful", "持久化": "stateful",
            "external": "external_capability", "external_tool": "external_capability",
            "external_api": "external_capability", "外部能力": "external_capability",
        }
        if isinstance(mode, str):
            normalized_mode = re.sub(r"[\s-]+", "_", mode.strip().lower())
            mode = aliases.get(normalized_mode, normalized_mode)
        if mode not in modes:
            mode = ContractCompiler._infer_environment_mode(
                task_description=task_description, task_intent=task_intent
            )
            fallback_reason = (
                "模型返回了未知环境模式，已根据任务中明确的外部能力、持久化和只读资料需求"
                "确定性选择最小必要环境。"
            )
            logger.warning(
                "environment_plan.mode is invalid; using deterministic fallback: raw=%r resolved=%s",
                value.get("mode"), mode,
            )
        inferred_mode = ContractCompiler._infer_environment_mode(
            task_description=task_description, task_intent=task_intent
        )
        if mode == "stateless" and inferred_mode in {"external_capability", "reference_data"}:
            mode = inferred_mode
            fallback_reason = (
                "任务需要可验证的外部或参考事实，已确定性升级为相应的数据环境。"
            )
            logger.warning(
                "environment_plan understated external dependency; upgraded to external_capability"
            )
        task_text = " ".join(
            str(task_description.get(key, ""))
            for key in ("task", "goal", "expected_result", "context", "requirements")
        )
        persistence_markers = (
            "写入", "保存", "持久化", "更新", "修改", "删除", "创建记录", "提交订单",
            "数据库", "审批", "排程", "预订", "write", "save", "persist", "update",
            "delete", "create record", "database", "approve", "schedule", "book",
        )
        external_markers = (
            "天气", "实时", "当前市场", "目前市场", "最新", "联网", "搜索", "汇率",
            "预估总价", "估算价格", "查询价格",
            "weather", "current market", "latest", "search", "exchange rate",
        )
        explicitly_persistent = any(marker in task_text.lower() for marker in persistence_markers)
        explicitly_external = any(marker in task_text.lower() for marker in external_markers)
        runtime_candidates_supplied = any(
            marker in task_text.lower()
            for marker in (
                "提供的规格", "提供的候选", "给定的候选", "给定选项", "两个方案",
                "两款", "候选清单", "provided specifications", "given candidates",
                "provided options", "two options",
            )
        ) or re.search(
            r"(?:用户)?(?:明确)?提供的.{0,8}(?:候选|选项|列表|清单|规格|属性)",
            task_text,
        ) is not None
        stateless_intents = {"extract", "summarize", "classify", "transform", "explain"}
        common_knowledge_explanation = (
            task_intent == "explain"
            and any(
                marker in task_text.lower()
                for marker in (
                    "常见用途", "基本用途", "日常用途", "基本构造", "基本结构",
                    "一般原理", "基础原理", "常见特点", "common uses",
                    "basic structure", "basic principle",
                )
            )
            and not any(
                marker in task_text.lower()
                for marker in (
                    "历史", "来源", "出处", "资料", "记录", "数据", "文献",
                    "权威", "给定", "提供的", "history", "source", "reference",
                    "record", "data", "provided",
                )
            )
        )
        if (
            task_intent in stateless_intents
            and (inferred_mode == "stateless" or common_knowledge_explanation)
            and not explicitly_persistent
            and not explicitly_external
        ):
            mode = "stateless"
        if (
            task_intent in {"compare", "recommend", "decide"}
            and runtime_candidates_supplied
            and not explicitly_persistent
            and not explicitly_external
        ):
            mode = "stateless"
        requires_business_data = mode in {"reference_data", "stateful"}
        requires_persistence = mode == "stateful"
        reason = fallback_reason or value.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            reason = "由任务明确目标和预期结果推导运行时环境模式。"
        return {
            "mode": mode,
            "requires_business_data": requires_business_data,
            "requires_persistence": requires_persistence,
            "reason": reason.strip(),
        }

    @staticmethod
    def _align_environment_plan_with_route(
        environment_plan: dict[str, Any], *, route_plan: dict[str, Any],
        task_intent: str, supported_modes: set[str],
    ) -> dict[str, Any]:
        """Make the validated tool route authoritative over a noisy mode label.

        A non-empty route plan represents operations that must cross the sandbox
        boundary.  Treating such a task as stateless contradicts that contract.
        The correction is deterministic and chooses the smallest environment
        capable of implementing the route; it does not invent an unavailable
        external service.
        """
        operations = route_plan.get("environment_operations", [])
        if not isinstance(operations, list) or not operations:
            raise PipelineGenerationError(
                "agentic route requires at least one environment operation"
            )
        if environment_plan.get("mode") != "stateless":
            return environment_plan

        mutating_intents = {"modify", "execute", "schedule"}
        required_mode = "stateful" if task_intent in mutating_intents else "reference_data"
        if required_mode not in supported_modes:
            raise PipelineGenerationError(
                "task buildability: route requires environment mode "
                f"{required_mode!r}, but supported={sorted(supported_modes)}"
            )
        logger.warning(
            "environment_plan conflicts with non-empty route; aligned mode from "
            "stateless to %s for intent=%s",
            required_mode, task_intent,
        )
        return {
            "mode": required_mode,
            "requires_business_data": True,
            "requires_persistence": required_mode == "stateful",
            "reason": (
                "已验证的训练路由包含必须跨越沙箱边界的业务操作；"
                "按任务意图确定性选择最小可实现环境。"
            ),
        }

    @staticmethod
    def _infer_environment_mode(
        *, task_description: dict[str, Any], task_intent: str
    ) -> str:
        """Conservatively infer a mode when the model emits an unknown label."""
        text = " ".join(
            str(task_description.get(key, ""))
            for key in ("task", "goal", "expected_result", "context", "requirements")
        ).lower()
        external_markers = (
            "实时天气", "实时汇率", "实时价格", "联网搜索", "网络搜索", "外部 api",
            "当前市场", "目前市场", "最新", "最新列表", "最新名单", "公开查询", "公开可查", "官方网站", "官网查询",
            "所属省级行政区", "行政区归属", "地名归属",
            "预估总价", "估算价格", "查询价格",
            "现行法律", "当前开放", "开放时间", "公共交通", "权威来源",
            "weather forecast", "exchange rate", "web search", "external api",
            "latest list", "current list", "official website", "publicly available",
        )
        persistence_markers = (
            "写入数据库", "保存到", "持久化", "更新记录", "修改记录", "删除记录",
            "创建记录", "提交订单", "审批申请", "预订", "写入日历",
            "write to", "save to", "persist", "update record", "delete record",
            "create record", "submit order", "book ",
        )
        reference_markers = (
            "资料库", "知识库", "档案库", "商品目录", "内部文档", "业务记录中",
            "训练计划", "锻炼计划", "运动计划", "康复计划", "疼痛", "伤病",
            "reference data", "knowledge base", "catalog", "internal document",
            "training plan", "exercise plan", "rehabilitation plan", "pain", "injury",
        )
        if any(marker in text for marker in external_markers):
            return "external_capability"
        if any(marker in text for marker in persistence_markers):
            return "stateful"
        if any(marker in text for marker in reference_markers):
            return "reference_data"
        if task_intent in {"recommend", "compare", "explain", "troubleshoot", "decide"}:
            return "reference_data"
        # A schedule/create/plan intent describes what the Agent produces, not
        # whether the result must be persisted. Without an explicit state
        # boundary, the minimal environment is stateless.
        return "stateless"

    @staticmethod
    def _validate_success_fixture_audit(audit: Any) -> None:
        if not isinstance(audit, dict):
            raise PipelineGenerationError("success fixture audit must be an object")
        fields = ("latest_instructions_satisfied", "grounded", "goal_completed")
        if any(not isinstance(audit.get(field), bool) for field in fields):
            raise PipelineGenerationError("success fixture audit requires boolean verdicts")
        issues = audit.get("issues")
        if not isinstance(issues, list) or any(
            not isinstance(issue, str) or not issue.strip() for issue in issues
        ):
            raise PipelineGenerationError("success fixture audit requires issue strings")
        failed = [field for field in fields if audit[field] is False]
        if failed:
            detail = "; ".join(issues) if issues else "no details supplied"
            raise PipelineGenerationError(
                f"success fixture is inconsistent: failed={failed}; issues={detail}"
            )

    @staticmethod
    def _has_conditional_noop_branch(
        task_description: dict[str, Any], public_input: dict[str, Any],
    ) -> bool:
        """Spot explicit idempotent branches that the single-delta goal cannot score."""
        text = "\n".join((
            str(task_description.get("task", "")),
            str(task_description.get("expected_result", "")),
            str(public_input.get("initial_user_message", "")),
        ))
        chinese = re.search(
            r"(?:若|如果|当|如)[^。；\n]{0,80}(?:已|已经|原本|当前)[^。；\n]{0,80}"
            r"(?:则|就)[^。；\n]{0,50}(?:无需|不用|不必|保持|不做|不再|原样)",
            text,
        )
        english = re.search(
            r"\bif\b[^.!?\n]{0,90}\balready\b[^.!?\n]{0,90}"
            r"\b(?:do not|don't|leave unchanged|keep as is|no update)\b",
            text, flags=re.I,
        )
        return bool(chinese or english)

    @staticmethod
    def _apply_reward_metric_repairs(
        metrics: list[dict[str, Any]], repairs: Any,
    ) -> None:
        """Only prose for semantic outcome judges may be repaired after compilation."""
        if not isinstance(repairs, list) or not repairs:
            raise PipelineGenerationError("reward contract audit supplied no metric repairs")
        eligible = {
            metric["id"]: metric for metric in metrics
            if isinstance(metric, dict) and metric.get("category") == "outcome"
            and metric.get("type") == "model-based"
            and isinstance(metric.get("id"), str)
        }
        seen: set[str] = set()
        for repair in repairs:
            if not isinstance(repair, dict) or repair.get("id") not in eligible:
                raise PipelineGenerationError("reward contract repair targets a non-semantic outcome metric")
            metric_id = repair["id"]
            if metric_id in seen:
                raise PipelineGenerationError("reward contract repair duplicates a metric")
            seen.add(metric_id)
            rubric = repair.get("rubric")
            criteria = repair.get("criteria")
            if (not isinstance(rubric, str) or not rubric.strip()
                    or not isinstance(criteria, list) or not criteria
                    or any(not isinstance(item, str) or not item.strip() for item in criteria)):
                raise PipelineGenerationError("reward contract repair needs rubric and criteria")
        for repair in repairs:
            metric = eligible[repair["id"]]
            metric["rubric"] = repair["rubric"].strip()
            metric["criteria"] = [item.strip() for item in repair["criteria"]]

    @staticmethod
    def _strip_unrequested_identifier_rows(
        content: str, *, public_input: dict[str, Any]
    ) -> str:
        """Keep generated answers from making gratuitous private ID claims.

        Such claims can pass a numeric outcome oracle while a later sandbox
        mutation correctly detects that the stale answer still gets reward.
        Only remove a complete Markdown table row when its identifier label
        was never requested in the user's public message.
        """
        if not isinstance(content, str) or not isinstance(public_input, dict):
            return content
        request = str(public_input.get("initial_user_message", ""))
        if not request:
            return content
        kept: list[str] = []
        for line in content.splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if (line.lstrip().startswith("|") and len(cells) == 2
                    and re.search(r"(?:编号|代码|标识|\bID\b)", cells[0], re.I)
                    and cells[0] not in request):
                logger.info("removed unrequested identifier from success fixture: %s", cells[0])
                continue
            kept.append(line)
        return "\n".join(kept).strip()

    @staticmethod
    def _validate_success_fixture_constraints(
        content: Any, *, task_description: dict[str, Any]
    ) -> None:
        """Enforce simple objective constraints without trusting an LLM judge."""
        if not isinstance(content, str) or not content.strip():
            raise PipelineGenerationError("success fixture content must be non-empty")
        source = json.dumps(task_description, ensure_ascii=False)
        compact = re.sub(r"\s+", "", content)
        ranges = re.findall(r"(\d+)\s*[-–—~～至到]\s*(\d+)\s*(?:个)?字", source)
        for lower_text, upper_text in ranges:
            lower, upper = int(lower_text), int(upper_text)
            if lower <= upper and not lower <= len(compact) <= upper:
                raise PipelineGenerationError(
                    f"success fixture length {len(compact)} violates required {lower}-{upper} characters"
                )

    @staticmethod
    def _normalize_success_fixture_length(
        content: str, *, task_description: dict[str, Any]
    ) -> str:
        """Deterministically trim an overlong fixture to an explicit char limit."""
        if not isinstance(content, str):
            return content
        source = json.dumps(task_description, ensure_ascii=False)
        ranges = re.findall(r"(\d+)\s*[-–—~～至到]\s*(\d+)\s*(?:个)?字", source)
        upper_bounds = [int(upper) for lower, upper in ranges if int(lower) <= int(upper)]
        if not upper_bounds:
            return content.strip()
        upper = min(upper_bounds)
        if len(re.sub(r"\s+", "", content)) <= upper:
            return content.strip()
        kept: list[str] = []
        count = 0
        for char in content.strip():
            if not char.isspace():
                if count >= upper:
                    break
                count += 1
            kept.append(char)
        normalized = "".join(kept).rstrip()
        logger.info(
            "trimmed success fixture to explicit character limit: upper=%d", upper
        )
        return normalized

    @staticmethod
    def _validate_metric_constraint_grounding(
        metrics: list[Any], *, task_description: dict[str, Any]
    ) -> None:
        """Reject thresholds invented by reward generation.

        We inspect only human-facing metric semantics, deliberately excluding
        weights, score ranges and score mappings which contain evaluator
        mechanics rather than task requirements.
        """
        task_text = json.dumps(task_description, ensure_ascii=False)
        allowed = {
            Decimal(number)
            for number in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", task_text)
        }
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                continue
            evaluator = metric.get("evaluator", {})
            semantic_parts: list[str] = []
            for key in ("rubric", "condition"):
                if isinstance(metric.get(key), str):
                    semantic_parts.append(metric[key])
            criteria = metric.get("criteria")
            if isinstance(criteria, list):
                semantic_parts.extend(item for item in criteria if isinstance(item, str))
            if isinstance(evaluator, dict):
                for key in ("assertion", "criteria"):
                    value = evaluator.get(key)
                    if isinstance(value, str):
                        semantic_parts.append(value)
                    elif isinstance(value, list):
                        semantic_parts.extend(item for item in value if isinstance(item, str))
            mentioned = set(re.findall(
                r"(?<![\w.])\d+(?:\.\d+)?", " ".join(semantic_parts)
            ))
            # Zero and one frequently express boolean/control-state assertions
            # (for example count == 0 or completed == 1), not user-facing
            # quantitative requirements. Treat them as evaluator mechanics.
            invented = sorted(
                number for number in mentioned
                if Decimal(number) not in allowed and Decimal(number) not in {0, 1}
            )
            if invented:
                raise PipelineGenerationError(
                    f"metrics[{index}] invents numeric constraints absent from the task: {invented}"
                )

    @classmethod
    def _drop_ungrounded_numeric_metrics(
        cls, metrics: list[Any], *, task_description: dict[str, Any]
    ) -> list[Any]:
        """Discard isolated reward thresholds invented by the model."""
        retained: list[Any] = []
        for metric in metrics:
            try:
                cls._validate_metric_constraint_grounding(
                    [metric], task_description=task_description
                )
            except PipelineGenerationError as exc:
                logger.warning(
                    "discarding metric with ungrounded numeric constraint: metric=%s error=%s",
                    metric.get("id") if isinstance(metric, dict) else None, exc,
                )
                continue
            retained.append(metric)
        return retained

    @staticmethod
    def _derive_complexity(
        *, business_tool_count: int, metric_count: int, key_step_count: int
    ) -> str:
        if business_tool_count <= 1 and metric_count <= 4 and key_step_count <= 1:
            return "simple"
        if business_tool_count <= 4 and metric_count <= 8 and key_step_count <= 4:
            return "standard"
        return "complex"

    @staticmethod
    def _derive_training_profile(
        *, business_tool_count: int, noise_tool_count: int, key_step_count: int
    ) -> str:
        if business_tool_count == 0:
            return "tool_abstention" if noise_tool_count else "language_only"
        if key_step_count > 1 or business_tool_count > 1:
            return "multi_step_tool_use"
        return "single_step_tool_use"

    @staticmethod
    def _requires_file_deliverable(task_description: dict[str, Any]) -> bool:
        """Detect tasks whose goal is an actual file, not merely textual content."""
        requirements = task_description.get("requirements")
        requirements = requirements if isinstance(requirements, dict) else {}
        output_format = str(requirements.get("output_format", "")).lower()
        text = " ".join(
            str(task_description.get(key, ""))
            for key in ("task", "goal", "expected_result")
        ).lower()
        text = f"{text} {output_format}"
        file_markers = (
            "pdf", "png", "jpg", "jpeg", "svg", "docx", "xlsx", "pptx",
            "音频文件", "视频文件", "图片文件", "可打印文件", "生成文件",
        )
        return any(marker in text for marker in file_markers)

    @staticmethod
    def _validate_media_generation(
        media_generation: dict[str, Any],
        *,
        input_media_required: bool,
        deliverable_required: bool,
        task_description: dict[str, Any],
    ) -> None:
        code = str(media_generation.get("code", ""))
        lower = code.lower()
        if deliverable_required and not media_generation.get("output_dir"):
            raise PipelineGenerationError(
                "file deliverable requires a non-empty media_generation.output_dir"
            )
        if deliverable_required:
            requirements = task_description.get("requirements")
            requirements = requirements if isinstance(requirements, dict) else {}
            expected = str(requirements.get("output_format", "")).lower()
            extensions = {
                "pdf": ".pdf", "png": ".png", "jpg": ".jpg", "jpeg": ".jpeg",
                "svg": ".svg", "docx": ".docx", "xlsx": ".xlsx", "pptx": ".pptx",
            }
            required_extensions = {
                extension for label, extension in extensions.items() if label in expected
            }
            if required_extensions and not any(extension in lower for extension in required_extensions):
                raise PipelineGenerationError(
                    "media_generation code does not create the declared file format"
                )
        if input_media_required:
            fabricated_markers = (
                "模拟数据", "生成一个模拟", "mock data", "dummy data",
                "placeholder data", "synthetic user", "假设最大", "示例数据：",
            )
            if any(marker in lower for marker in fabricated_markers):
                raise PipelineGenerationError(
                    "media_generation fabricates runtime user input; read inputs from arguments instead"
                )
            input_markers = (
                "argparse", "sys.argv", "input_path", "input_file", "source_path",
                "image_path", "file_path", "json.load", "csv.", "read_csv",
                "image.open", "soundfile.read", "wave.open",
            )
            if not any(marker in lower for marker in input_markers):
                raise PipelineGenerationError(
                    "media_generation for user-provided media must read an explicit runtime input"
                )

    @staticmethod
    def _validate_authoritative_source_schema(
        *, task_description: dict[str, Any], table_definitions: list[Any]
    ) -> None:
        """Require traceable provenance for generated high-stakes reference facts."""
        if not ContractCompiler._is_high_stakes_task(task_description):
            return
        column_names = {
            str(column.get("name", "")).lower()
            for table in table_definitions if isinstance(table, dict)
            for column in table.get("columns", []) if isinstance(column, dict)
        }
        aliases = {
            "source_url": {"source_url", "official_url", "url"},
            "retrieved_at": {"retrieved_at", "accessed_at", "fetched_at"},
            "content_hash": {"content_hash", "source_hash", "sha256"},
        }
        missing = [
            canonical for canonical, names in aliases.items()
            if not column_names.intersection(names)
        ]
        if missing:
            raise PipelineGenerationError(
                "authoritative reference data requires provenance columns: "
                + ", ".join(missing)
            )

    @staticmethod
    def _normalize_public_input(task_description: dict[str, Any]) -> dict[str, Any]:
        """Compile the public episode payload without copying hidden truth."""
        task = str(task_description.get("task") or "").strip()
        candidate = task_description.get("public_input")
        candidate = candidate if isinstance(candidate, dict) else {}
        message = candidate.get("initial_user_message")
        if not isinstance(message, str) or not message.strip():
            message = task
        materials: list[dict[str, str]] = []
        raw_materials = candidate.get("materials")
        if isinstance(raw_materials, list):
            for index, item in enumerate(raw_materials, start=1):
                if isinstance(item, str):
                    item = {"name": f"material-{index}", "mime_type": "text/plain", "content": item}
                if not isinstance(item, dict):
                    continue
                content = item.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                name = item.get("name")
                mime_type = item.get("mime_type")
                materials.append({
                    "name": name.strip() if isinstance(name, str) and name.strip() else f"material-{index}",
                    "mime_type": mime_type.strip() if isinstance(mime_type, str) and mime_type.strip() else "text/plain",
                    "content": content.strip(),
                })
        return {"initial_user_message": message.strip(), "materials": materials}

    @staticmethod
    def _validate_public_input(
        *, task_description: dict[str, Any], public_input: dict[str, Any], environment_mode: str,
    ) -> None:
        message = public_input.get("initial_user_message")
        if not isinstance(message, str) or not message.strip():
            raise PipelineGenerationError("public_input.initial_user_message must be non-empty")
        materials = public_input.get("materials")
        if not isinstance(materials, list):
            raise PipelineGenerationError("public_input.materials must be a list")
        text = json.dumps({
            "task": task_description.get("task"),
            "context": task_description.get("context"),
            "requirements": task_description.get("requirements"),
        }, ensure_ascii=False).lower()
        references_runtime_material = re.search(
            r"(?:以下|下列|上述|这段|这些|给定|附上|附件|(?:用户|我|你)(?:已)?提供).{0,12}"
            r"(?:文本|资料|数据|列表|清单|内容|说明|笔记|记录|规格|描述|选项)",
            text,
        ) is not None
        concrete = [
            item for item in materials
            if isinstance(item, dict)
            and isinstance(item.get("content"), str)
            and len(item["content"].strip()) >= 8
            and not re.fullmatch(r"(?:用户)?(?:已)?提供(?:的)?(?:文本|资料|数据|内容|说明|笔记|记录|规格|描述)", item["content"].strip())
        ]
        if references_runtime_material and not concrete:
            raise PipelineGenerationError(
                "public_input is missing the concrete material referenced by the task"
            )
        if environment_mode == "stateless" and not message.strip():
            raise PipelineGenerationError("stateless task requires a complete public user message")

    @staticmethod
    def _validate_authoritative_task_input(
        *,
        task_description: dict[str, Any],
        environment_mode: str,
        graph_context: dict[str, Any],
    ) -> None:
        """Do not let an LLM invent the source corpus for high-stakes lookup tasks."""
        if environment_mode not in {"reference_data", "stateful"}:
            return
        if not ContractCompiler._is_high_stakes_task(task_description):
            return
        evidence = json.dumps(graph_context, ensure_ascii=False)
        if not re.search(r"https?://[^\s\"']+", evidence, re.I):
            raise PipelineGenerationError(
                "authoritative reference task requires source URLs in graph_context; "
                "LLM-generated authority data is not allowed"
            )

    @staticmethod
    def _validate_task_readiness(
        *,
        capability_plan: list[Any],
        tool_bindings: list[Any],
        noise_tools: list[Any],
        metrics: list[Any],
        metric_implementations: list[Any],
        business_scenarios: list[Any],
        environment_mode: str = "stateless",
        has_business_data: bool = False,
        task_description: dict[str, Any] | None = None,
        data_tables: list[Any] | None = None,
        media_generation: dict[str, Any] | None = None,
        success_fixture: str = "",
        require_noise: bool = True,
    ) -> None:
        eligible = {
            item.get("action_name") for item in capability_plan
            if isinstance(item, dict) and item.get("requires_tool") is True
        }
        bound = {
            item.get("action_name") for item in tool_bindings if isinstance(item, dict)
        }
        if not bound <= eligible:
            raise PipelineGenerationError("task readiness: a tool delegates agent reasoning or response composition")
        if environment_mode == "external_capability" and not bound:
            raise PipelineGenerationError(
                "task readiness: external-capability task exposes no business tool"
            )
        if environment_mode in {"reference_data", "stateful"} and has_business_data and not bound:
            raise PipelineGenerationError(
                "task readiness: data-backed task exposes no business data access tool"
            )
        if environment_mode == "reference_data":
            leaked = [
                metric.get("id") for metric in metrics
                if isinstance(metric, dict)
                and metric.get("category") == "outcome"
                and isinstance(metric.get("evaluator"), dict)
                and metric["evaluator"].get("kind") == "business_state_rule"
            ]
            if leaked:
                raise PipelineGenerationError(
                    "task readiness: read-only reference data cannot by itself prove agent outcome; "
                    f"metrics={leaked}"
                )
        if require_noise and not noise_tools:
            raise PipelineGenerationError("task readiness: tool-selection tasks require at least one noise tool")
        rule_ids = {
            item.get("id") for item in metrics
            if isinstance(item, dict) and item.get("type") == "rule-based"
        }
        implemented = {
            item.get("metric_id") for item in metric_implementations if isinstance(item, dict)
        }
        if not rule_ids <= implemented:
            raise PipelineGenerationError("task readiness: rule-based metrics lack executable implementations")
        kinds = {
            item.get("kind") for item in business_scenarios if isinstance(item, dict)
        }
        if not {"goal_success", "goal_failure"} <= kinds:
            raise PipelineGenerationError("task readiness: success and failure trajectories are required")
        if noise_tools and "noise_selection" not in kinds:
            raise PipelineGenerationError("task readiness: a noise-selection trajectory is required")
        ContractCompiler._validate_metric_runtime_semantics(
            metrics=metrics,
            metric_implementations=metric_implementations,
            noise_tools=noise_tools,
        )
        description = task_description or {}
        if ContractCompiler._requires_file_deliverable(description):
            media = media_generation or {}
            if media.get("required") is not True or not str(media.get("code", "")).strip():
                raise PipelineGenerationError(
                    "task readiness: file deliverable has no executable artifact generation"
                )
            file_metric_text = json.dumps(metrics, ensure_ascii=False).lower()
            if any(marker in file_metric_text for marker in ("file_generation", "pdf_file", "生成文件", "pdf文件")):
                model_only = [
                    metric.get("id") for metric in metrics
                    if isinstance(metric, dict)
                    and any(marker in json.dumps(metric, ensure_ascii=False).lower()
                            for marker in ("file_generation", "pdf_file", "生成文件", "pdf文件"))
                    and isinstance(metric.get("evaluator"), dict)
                    and metric["evaluator"].get("kind") == "external_llm_judge"
                ]
                if model_only:
                    raise PipelineGenerationError(
                        "task readiness: file existence cannot be evaluated only by an LLM judge; "
                        f"metrics={model_only}"
                    )
            if re.search(r"(?:已|已经).{0,12}(?:生成|创建|提供).{0,30}\.(?:pdf|png|jpg|docx|xlsx|pptx)", success_fixture, re.I):
                # A claimed artifact is acceptable only because executable media
                # generation was required and validated above. Keep this branch
                # explicit so future artifact-manifest support can tighten it.
                pass

    @staticmethod
    def _normalize_metric_weights(metrics: list[Any]) -> list[Any]:
        """Normalize positive and negative metric weights independently.

        LLMs often emit proportional weights such as 0.4/0.5/1.0 instead of
        already-normalized weights. Since the ratio is the meaningful part,
        normalize each sign group before strict contract validation.
        """
        positive = 0.0
        negative = 0.0
        for metric in metrics:
            if not isinstance(metric, dict) or not isinstance(metric.get("weight"), (int, float)):
                continue
            category = metric.get("category")
            if category in {"process", "outcome"} and metric["weight"] >= 0:
                positive += float(metric["weight"])
            elif category == "penalty" and metric["weight"] >= 0:
                negative += float(metric["weight"])
        if positive > 0 and not math.isclose(positive, 1.0, abs_tol=1e-6):
            logger.info("normalized positive metric weights: original_sum=%s", positive)
            for metric in metrics:
                if isinstance(metric, dict) and metric.get("category") in {"process", "outcome"} and isinstance(metric.get("weight"), (int, float)):
                    metric["weight"] = float(metric["weight"]) / positive
        if negative > 0 and not math.isclose(negative, 1.0, abs_tol=1e-6):
            logger.info("normalized penalty metric weights: original_sum=%s", negative)
            for metric in metrics:
                if isinstance(metric, dict) and metric.get("category") == "penalty" and isinstance(metric.get("weight"), (int, float)):
                    metric["weight"] = float(metric["weight"]) / negative
        return metrics

    @staticmethod
    def _canonicalize_process_metrics(metrics: list[Any]) -> list[Any]:
        """Process reward semantics are fixed by the platform tool-call contract."""
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "process":
                continue
            metric["type"] = "hybrid"
            metric["scope"] = "step"
            metric["condition"] = "llm_expected_tool_call_exact_match"
            metric["evaluation_inputs"] = [
                "recent_conversation", "public_observation", "tool_call", "tool_arguments",
            ]
            metric["evaluator"] = {
                "kind": "hybrid_tool_call", "source": "external_llm",
                "comparison": "exact_tool_name_and_canonical_arguments",
                "score_mapping": {"match": 1, "mismatch": 0},
            }
        return metrics

    @staticmethod
    def _normalize_metric_types(metrics: list[Any]) -> list[Any]:
        """Derive the canonical metric type from an unambiguous evaluator."""
        evaluator_to_type = {
            "business_state_rule": "rule-based",
            "document_rule": "rule-based",
            "trajectory_rule": "rule-based",
            "external_llm_judge": "model-based",
            "hybrid_tool_call": "hybrid",
            "hybrid_outcome": "hybrid",
        }
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") in {"rule-based", "model-based", "hybrid"}:
                continue
            evaluator = metric.get("evaluator")
            kind = evaluator.get("kind") if isinstance(evaluator, dict) else None
            canonical = evaluator_to_type.get(kind)
            if canonical:
                logger.info(
                    "normalized metric type from evaluator: metric=%s original=%r normalized=%s",
                    metric.get("id"), metric.get("type"), canonical,
                )
                metric["type"] = canonical
        return metrics

    @staticmethod
    def _normalize_metric_scopes(metrics: list[Any]) -> list[Any]:
        """Accept unambiguous scope names without weakening the scope gate."""
        aliases = {
            "final": "terminal", "final_answer": "terminal",
            "final_response": "terminal", "terminal_response": "terminal",
            "tool_call": "step", "action": "step", "per_step": "step",
            "business_state": "state", "database_state": "state",
            "final_state": "state", "full_trajectory": "trajectory",
            "episode_trajectory": "trajectory", "whole_trajectory": "trajectory",
        }
        for metric in metrics:
            if not isinstance(metric, dict) or not isinstance(metric.get("scope"), str):
                continue
            original = metric["scope"]
            normalized = aliases.get(original.strip().lower())
            if normalized:
                metric["scope"] = normalized
                logger.info(
                    "normalized metric scope: metric=%s original=%r normalized=%s",
                    metric.get("id"), original, normalized,
                )
        return metrics

    @staticmethod
    def _normalize_metric_evaluators(metrics: list[Any]) -> list[Any]:
        """Canonicalize duplicated rule text emitted at either contract level."""
        rule_kinds = {"business_state_rule", "document_rule", "trajectory_rule"}
        for metric in metrics:
            if not isinstance(metric, dict):
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict):
                continue
            condition = metric.get("condition")
            assertion = evaluator.get("assertion")
            if evaluator.get("kind") in rule_kinds:
                if not isinstance(assertion, str) or not assertion.strip():
                    if isinstance(condition, str) and condition.strip():
                        evaluator["assertion"] = condition.strip()
                        logger.info(
                            "normalized rule assertion from condition: metric=%s",
                            metric.get("id"),
                        )
                elif not isinstance(condition, str) or not condition.strip():
                    metric["condition"] = assertion.strip()
            elif evaluator.get("kind") == "hybrid_outcome":
                rule = evaluator.get("rule")
                if isinstance(rule, dict):
                    rule_assertion = rule.get("assertion")
                    if (not isinstance(rule_assertion, str) or not rule_assertion.strip()) and isinstance(condition, str) and condition.strip():
                        rule["assertion"] = condition.strip()
                # A prose assertion is not a deterministic program. Avoid a
                # hybrid declaration that the runtime cannot honestly execute
                # as two independent branches; retain all criteria in one
                # explicit semantic judge instead.
                criteria: list[str] = []
                for value in metric.get("criteria", []):
                    if isinstance(value, str) and value.strip() and value.strip() not in criteria:
                        criteria.append(value.strip())
                if isinstance(rule, dict) and isinstance(rule.get("assertion"), str):
                    value = rule["assertion"].strip()
                    if value and value not in criteria:
                        criteria.append(value)
                semantic = evaluator.get("external_llm")
                if isinstance(semantic, dict):
                    values = semantic.get("criteria", [])
                    if isinstance(values, str):
                        values = [values]
                    judge = semantic.get("judge_criteria")
                    if isinstance(judge, str):
                        values = [*values, judge]
                    for value in values:
                        if isinstance(value, str) and value.strip() and value.strip() not in criteria:
                            criteria.append(value.strip())
                metric["type"] = "model-based"
                metric.pop("condition", None)
                metric["criteria"] = criteria or [str(metric.get("rubric") or "任务结果满足目标")]
                metric["evaluator"] = {
                    "kind": "external_llm_judge",
                    "source": "external_llm",
                    "criteria": list(metric["criteria"]),
                    "score_mapping": copy.deepcopy(evaluator.get("score_mapping", {"pass": 1, "fail": 0})),
                }
                logger.info(
                    "normalized non-executable hybrid outcome to semantic judge: metric=%s",
                    metric.get("id"),
                )
        return metrics

    @staticmethod
    def _normalize_metrics_for_environment(
        metrics: list[Any], *, environment_mode: str
    ) -> list[Any]:
        """Prevent read-only source data from being treated as an Agent outcome."""
        if environment_mode != "reference_data":
            return metrics
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "outcome":
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict) or evaluator.get("kind") not in {
                "business_state_rule", "document_rule", "trajectory_rule",
            }:
                continue
            criterion = (
                metric.get("rubric") or evaluator.get("assertion")
                or "最终回答是否基于参考数据完成用户目标"
            )
            metric["type"] = "model-based"
            metric["scope"] = "terminal"
            metric.pop("condition", None)
            metric["evaluation_inputs"] = ["final_agent_response", "business_data"]
            metric["criteria"] = [str(criterion)]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "criteria": [str(criterion)],
                "score_mapping": {"pass": 1, "fail": 0},
            }
            logger.info(
                "promoted reference-data outcome to response judge: metric=%s",
                metric.get("id"),
            )
        return metrics

    @staticmethod
    def _ensure_deterministic_noise_penalty(
        metrics: list[Any], *, noise_names: list[str]
    ) -> None:
        retained: list[Any] = []
        for metric in metrics:
            if not isinstance(metric, dict):
                retained.append(metric)
                continue
            text = json.dumps(metric, ensure_ascii=False, sort_keys=True).lower()
            if metric.get("category") == "penalty" and (
                "noise" in text
                or "噪声" in text
                or "tool_call" in text
                or "tool use" in text
                or "工具调用" in text
                or "工具使用" in text
                or any(name.lower() in text for name in noise_names)
            ):
                continue
            retained.append(metric)
        retained.append({
            "id": "penalty_noise_tool_usage",
            "category": "penalty",
            "type": "rule-based",
            "scope": "trajectory",
            "condition": "trajectory contains no calls to declared noise tools",
            "evaluator": {
                "kind": "trajectory_rule",
                "source": "runtime_rule",
                "assertion": "no declared noise tool appears in trajectory events",
                "score_mapping": {"pass": 0, "fail": -1},
            },
            "score_range": [-1, 0],
            "weight": 1,
            "rubric": "调用任一声明的噪声工具时施加确定性惩罚。",
        })
        metrics[:] = retained

    @staticmethod
    def _promote_unstructured_document_rules(metrics: list[Any]) -> None:
        """Use a semantic judge for facts that only exist inside document text."""
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") != "rule-based":
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict) or evaluator.get("kind") != "document_rule":
                continue
            criterion = evaluator.get("assertion") or metric.get("rubric") or metric.get("condition")
            mapping = evaluator.get("score_mapping", {"pass": 1, "fail": 0})
            metric["type"] = "model-based"
            metric["evaluation_inputs"] = ["conversation", "final_agent_response"]
            metric["criteria"] = [str(criterion or "判断最终文档是否满足该指标。")]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "score_mapping": dict(mapping) if isinstance(mapping, dict) else {"pass": 1, "fail": 0},
            }
            metric.pop("condition", None)
            logger.info(
                "promoted document rule over raw final_agent_response to external judge: metric=%s",
                metric.get("id"),
            )

    @staticmethod
    def _promote_mixed_response_business_rules(metrics: list[Any]) -> None:
        """A textual answer cannot be joined to business tables by the path DSL."""
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") != "rule-based":
                continue
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict) or evaluator.get("kind") != "business_state_rule":
                continue
            source_fields = evaluator.get("source_fields", [])
            assertion = str(evaluator.get("assertion", ""))
            mixes_response = (
                isinstance(source_fields, list) and "final_agent_response" in source_fields
            ) or "final_agent_response" in assertion
            if not mixes_response:
                continue
            criterion = assertion or metric.get("rubric") or metric.get("condition")
            mapping = evaluator.get("score_mapping", {"pass": 1, "fail": 0})
            metric["type"] = "model-based"
            metric["evaluation_inputs"] = ["business_data", "final_agent_response"]
            metric["criteria"] = [str(criterion or "判断最终回答是否与业务数据一致。")]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "score_mapping": dict(mapping) if isinstance(mapping, dict) else {"pass": 1, "fail": 0},
            }
            metric.pop("condition", None)
            logger.info(
                "promoted mixed response/business rule to external judge: metric=%s",
                metric.get("id"),
            )

    @staticmethod
    def _canonical_observation_schema(candidate: Any) -> dict[str, Any]:
        """Return one stable observation shape for every generated task."""
        descriptions = {
            "episode_id": "当前 episode 标识。",
            "conversation": "当前 episode 的完整对话消息。",
            "public_observation": "沙箱允许 Agent 或 Trainer 观察的公开环境信息。",
            "available_tools": "当前提供给 Agent 的 OpenAI Function Tool 定义。",
            "tool_call": "最近一次工具调用；未调用时为空。",
            "tool_results": "当前 episode 的工具调用结果列表。",
            "final_agent_response": "Agent 提交的最终自然语言回答。",
        }
        properties: dict[str, Any] = {}
        if isinstance(candidate, dict) and isinstance(candidate.get("properties"), dict):
            properties.update(candidate["properties"])
        field_types = {
            "episode_id": "string", "conversation": "array", "public_observation": "object",
            "available_tools": "array", "tool_call": "object",
            "tool_results": "array", "final_agent_response": "string",
        }
        for name, description in descriptions.items():
            # These fields are emitted by the shared runtime protocol.  Model
            # proposals may add task-specific observation fields, but must not
            # redefine platform-owned types (for example tool_results as an
            # object when the runtime always emits an array).
            properties[name] = {
                "type": field_types[name],
                "description": description,
            }
        return {
            "type": "object",
            "properties": properties,
            "required": [
                "episode_id", "conversation", "available_tools", "tool_results",
                "public_observation", "final_agent_response",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _validate_metric_runtime_semantics(
        *, metrics: list[Any], metric_implementations: list[Any], noise_tools: list[Any]
    ) -> None:
        """Reject contracts whose valid-looking DSL would produce reversed rewards."""
        metric_by_id = {
            item.get("id"): item for item in metrics
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
        noise_names = {
            item.get("name") for item in noise_tools
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        }
        noise_penalties = 0
        for index, spec in enumerate(metric_implementations):
            if not isinstance(spec, dict):
                continue
            metric = metric_by_id.get(spec.get("metric_id"), {})
            mapping = spec.get("score_mapping", {})
            if metric.get("category") == "penalty":
                if mapping.get("pass") != 0 or not isinstance(mapping.get("fail"), (int, float)) or mapping.get("fail") >= 0:
                    raise PipelineGenerationError(
                        f"task readiness: penalty metric {spec.get('metric_id')} has reversed score mapping"
                    )
            elif isinstance(mapping, dict) and mapping.get("pass", 0) < mapping.get("fail", 0):
                raise PipelineGenerationError(
                    f"task readiness: positive metric {spec.get('metric_id')} has reversed score mapping"
                )
            if spec.get("source") == "final_agent_response" and spec.get("path") not in {"$", ""}:
                raise PipelineGenerationError(
                    f"task readiness: metric_implementations[{index}] addresses fields on raw final_agent_response"
                )
            if spec.get("source") == "business_state" and "final_agent_response" in str(spec.get("path")):
                raise PipelineGenerationError(
                    f"task readiness: metric_implementations[{index}] mixes response data into business_state"
                )
            expected = spec.get("expected")
            targets_noise = (
                spec.get("operator") == "none_tool_calls"
                or (isinstance(expected, str) and expected in noise_names)
                or (isinstance(expected, list) and bool(set(expected) & noise_names))
            )
            if metric.get("category") == "penalty" and targets_noise:
                noise_penalties += 1
        if noise_names and noise_penalties != 1:
            raise PipelineGenerationError(
                f"task readiness: expected exactly one executable noise penalty, found {noise_penalties}"
            )

    @staticmethod
    def _validate_key_steps(key_steps: list[Any], actions: list[Any]) -> None:
        action_names = {
            str(action.get("name") or action.get("action"))
            for action in actions
            if isinstance(action, dict)
        }
        step_ids: set[str] = set()
        declared_dependencies: dict[str, list[str]] = {}
        for index, step in enumerate(key_steps):
            if not isinstance(step, dict):
                raise PipelineGenerationError(f"key_steps[{index}] must be an object")
            step_id = step.get("step_id")
            action_name = step.get("action_name")
            if not isinstance(step_id, str) or not step_id.strip() or step_id in step_ids:
                raise PipelineGenerationError(f"key_steps[{index}] has invalid or duplicate step_id")
            if not isinstance(action_name, str) or action_name not in action_names:
                raise PipelineGenerationError(f"key_steps[{index}] references an unknown action")
            if not isinstance(step.get("rationale"), str) or not step["rationale"].strip():
                raise PipelineGenerationError(f"key_steps[{index}] requires rationale")
            if not isinstance(step.get("required_for_goal"), bool):
                raise PipelineGenerationError(f"key_steps[{index}] requires boolean required_for_goal")
            dependencies = step.get("dependencies", [])
            if not isinstance(dependencies, list) or any(
                not isinstance(item, str) or not item.strip() for item in dependencies
            ):
                raise PipelineGenerationError(f"key_steps[{index}].dependencies must be a list")
            unknown = [item for item in dependencies if item not in step_ids]
            if unknown:
                raise PipelineGenerationError(
                    f"key_steps[{index}] dependencies must reference earlier steps: {unknown}"
                )
            declared_dependencies[step_id] = list(dependencies)
            step_ids.add(step_id)

    @staticmethod
    def _validate_metrics(metrics: list[Any], *, key_steps: list[Any] | None = None) -> None:
        ids: set[str] = set()
        category_weights = {"process": 0.0, "outcome": 0.0, "penalty": 0.0}
        key_action_names = {
            str(step.get("action_name"))
            for step in (key_steps or [])
            if isinstance(step, dict) and isinstance(step.get("action_name"), str)
        }
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                raise PipelineGenerationError(f"metrics[{index}] must be an object")
            metric_id = metric.get("id")
            if not isinstance(metric_id, str) or not metric_id.strip() or metric_id in ids:
                raise PipelineGenerationError(f"metrics[{index}] has invalid or duplicate id")
            category = metric.get("category")
            if category not in category_weights:
                raise PipelineGenerationError(
                    f"metrics[{index}] has invalid category: {category!r}; "
                    "expected one of 'process', 'outcome', 'penalty'"
                )
            metric_type = metric.get("type")
            if metric_type not in {"rule-based", "model-based", "hybrid"}:
                raise PipelineGenerationError(f"metrics[{index}] has invalid type")
            if metric.get("scope") not in {"step", "state", "terminal", "trajectory"}:
                raise PipelineGenerationError(
                    f"metrics[{index}] has invalid scope: {metric.get('scope')!r}"
                )
            if not isinstance(metric.get("rubric"), str) or not metric["rubric"].strip():
                raise PipelineGenerationError(f"metrics[{index}] requires rubric")
            weight = metric.get("weight")
            if not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
                raise PipelineGenerationError(f"metrics[{index}] weight must be a finite non-negative number")
            score_range = metric.get("score_range")
            expected_range = [-1, 0] if category == "penalty" else [0, 1]
            if score_range != expected_range:
                raise PipelineGenerationError(f"metrics[{index}] score_range must be {expected_range}")
            evaluator = metric.get("evaluator")
            if not isinstance(evaluator, dict):
                raise PipelineGenerationError(f"metrics[{index}] requires executable evaluator")
            if not isinstance(evaluator.get("kind"), str) or not evaluator["kind"].strip():
                raise PipelineGenerationError(f"metrics[{index}].evaluator requires kind")
            if evaluator.get("source") not in {"runtime_rule", "external_llm"}:
                raise PipelineGenerationError(
                    f"metrics[{index}].evaluator.source must be runtime_rule or external_llm"
                )
            score_mapping = evaluator.get("score_mapping")
            if not isinstance(score_mapping, dict) or not score_mapping:
                raise PipelineGenerationError(f"metrics[{index}].evaluator requires score_mapping")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < expected_range[0]
                or float(value) > expected_range[1]
                for value in score_mapping.values()
            ):
                raise PipelineGenerationError(
                    f"metrics[{index}].evaluator.score_mapping values must be within {expected_range}"
                )
            if category == "process":
                if (
                    evaluator.get("kind") != "hybrid_tool_call"
                    or evaluator.get("source") != "external_llm"
                    or evaluator.get("comparison") != "exact_tool_name_and_canonical_arguments"
                ):
                    raise PipelineGenerationError(
                        f"metrics[{index}] process evaluator must use external exact tool-call comparison"
                    )
                if set(evaluator.get("score_mapping", {})) != {"match", "mismatch"}:
                    raise PipelineGenerationError(
                        f"metrics[{index}] process evaluator score_mapping must contain match and mismatch"
                    )
            elif metric_type == "rule-based":
                if evaluator.get("kind") not in {
                    "business_state_rule", "document_rule", "trajectory_rule"
                } or evaluator.get("source") != "runtime_rule":
                    raise PipelineGenerationError(
                        f"metrics[{index}] rule-based evaluator must be an executable runtime rule"
                    )
                if not isinstance(evaluator.get("assertion"), str) or not evaluator["assertion"].strip():
                    raise PipelineGenerationError(f"metrics[{index}] rule-based evaluator requires assertion")
            elif metric_type == "model-based":
                if evaluator.get("kind") != "external_llm_judge" or evaluator.get("source") != "external_llm":
                    raise PipelineGenerationError(
                        f"metrics[{index}] model-based evaluator must use external_llm_judge"
                    )
            elif metric_type == "hybrid":
                if evaluator.get("kind") != "hybrid_outcome" or evaluator.get("source") != "external_llm":
                    raise PipelineGenerationError(
                        f"metrics[{index}] hybrid non-process evaluator must use hybrid_outcome"
                    )
                if not isinstance(evaluator.get("rule"), dict) or not isinstance(evaluator.get("external_llm"), dict):
                    raise PipelineGenerationError(
                        f"metrics[{index}] hybrid evaluator requires rule and external_llm sections"
                    )
            evaluation_inputs = metric.get("evaluation_inputs")
            if isinstance(evaluation_inputs, list):
                simulated_inputs = {
                    "user_profiles",
                    "user_scripts",
                    "simulated_dialogue",
                    "user_simulation",
                }
                forbidden = sorted(
                    str(item) for item in evaluation_inputs if str(item) in simulated_inputs
                )
                if forbidden:
                    raise PipelineGenerationError(
                        f"metrics[{index}] cannot depend on generated simulation inputs: {forbidden}"
                    )
            if category == "process":
                if metric_type != "hybrid":
                    raise PipelineGenerationError("process metrics must be hybrid")
                if not isinstance(metric.get("target_action"), str) or not metric["target_action"].strip():
                    raise PipelineGenerationError(f"metrics[{index}] process metric requires target_action")
                if key_steps is not None and metric["target_action"] not in key_action_names:
                    raise PipelineGenerationError(
                        f"metrics[{index}] process target_action must reference a key step"
                    )
                if metric.get("condition") != "llm_expected_tool_call_exact_match":
                    raise PipelineGenerationError(
                        f"metrics[{index}] process condition must be llm_expected_tool_call_exact_match"
                    )
            if metric_type in {"model-based", "hybrid"}:
                if not isinstance(metric.get("evaluation_inputs"), list) or not metric["evaluation_inputs"]:
                    raise PipelineGenerationError(f"metrics[{index}] requires evaluation_inputs")
                if not isinstance(metric.get("criteria"), list) or not metric["criteria"]:
                    raise PipelineGenerationError(f"metrics[{index}] requires criteria")
            if metric_type in {"rule-based", "hybrid"}:
                if not isinstance(metric.get("condition"), str) or not metric["condition"].strip():
                    raise PipelineGenerationError(f"metrics[{index}] requires condition")
            category_weights[category] += float(weight)
            ids.add(metric_id)
        if not category_weights["outcome"]:
            raise PipelineGenerationError("metrics must include an outcome metric")
        if not category_weights["penalty"]:
            raise PipelineGenerationError("metrics must include a penalty metric")
        positive_weight_sum = category_weights["process"] + category_weights["outcome"]
        if not math.isclose(positive_weight_sum, 1.0, abs_tol=1e-6):
            raise PipelineGenerationError(f"positive metric weights must sum to 1, got {positive_weight_sum}")
        if not math.isclose(category_weights["penalty"], 1.0, abs_tol=1e-6):
            raise PipelineGenerationError(f"penalty metric weights must sum to 1, got {category_weights['penalty']}")
        if category_weights["outcome"] <= category_weights["process"]:
            raise PipelineGenerationError("outcome metric weight must exceed process metric weight")

    @staticmethod
    def _ground_reference_outcome_metrics(
        metrics: list[Any], *, environment_mode: str,
    ) -> None:
        """Give semantic outcome judges the evidence needed to check business goals."""
        if environment_mode not in {"reference_data", "stateful"}:
            return
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "outcome":
                continue
            evaluator = metric.get("evaluator")
            if (not isinstance(evaluator, dict)
                    or evaluator.get("kind") not in {"external_llm_judge", "hybrid_outcome"}):
                continue
            inputs = metric.get("evaluation_inputs")
            inputs = [value for value in inputs if isinstance(value, str)] if isinstance(inputs, list) else []
            for source in ("final_agent_response", "tool_results", "business_data"):
                if source not in inputs:
                    inputs.append(source)
            metric["evaluation_inputs"] = inputs
            criteria = metric.get("criteria")
            criteria = [value for value in criteria if isinstance(value, str) and value.strip()] if isinstance(criteria, list) else []
            if any(contradicts_reference_factuality(value) for value in criteria) or contradicts_reference_factuality(metric.get("rubric")):
                raise PipelineGenerationError(
                    f"business outcome metric contradicts business evidence checking: {metric.get('id')}"
                )
            required_criterion = (
                REFERENCE_FACTUALITY_CRITERION if environment_mode == "reference_data"
                else STATEFUL_GOAL_CRITERION
            )
            if required_criterion not in criteria:
                criteria.append(required_criterion)
            metric["criteria"] = criteria

    @staticmethod
    def _validate_metrics_for_environment(
        metrics: list[Any], *, environment_mode: str
    ) -> None:
        if environment_mode != "stateless":
            return
        for index, metric in enumerate(metrics):
            if not isinstance(metric, dict):
                continue
            if metric.get("category") == "process":
                raise PipelineGenerationError(
                    f"metrics[{index}] stateless tasks must not reward tool processes"
                )
            evaluator = metric.get("evaluator", {})
            if isinstance(evaluator, dict) and evaluator.get("kind") == "business_state_rule":
                raise PipelineGenerationError(
                    f"metrics[{index}] stateless tasks cannot depend on business state"
                )
            inputs = metric.get("evaluation_inputs", [])
            if isinstance(inputs, list) and "business_data" in inputs:
                raise PipelineGenerationError(
                    f"metrics[{index}] stateless tasks cannot evaluate business_data"
                )

    @staticmethod
    def _compile_row_product_total(
        expected: dict[str, Any], public_input: dict[str, Any],
        data_tables: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Compile a verified fixed-row product sum into a batch-wide aggregate."""
        compiled = copy.deepcopy(expected)
        public_text = json.dumps(public_input, ensure_ascii=False)
        tables = {table.get("table_name"): table for table in data_tables if isinstance(table, dict)}
        for target in compiled.get("targets", []):
            expression = target.get("expression") if isinstance(target, dict) else None
            if not isinstance(expression, dict) or expression.get("op") not in ("sum", "add"):
                continue
            terms = expression.get("args")
            if not isinstance(terms, list) or len(terms) < 2:
                continue
            table_name: str | None = None
            fields: list[str] | None = None
            product_mode: bool | None = None
            selected_ids: set[str] = set()
            selected_rows: list[dict[str, Any]] = []
            for term in terms:
                is_product = isinstance(term, dict) and term.get("op") in ("mul", "multiply")
                operands = term.get("args") if is_product else [term]
                if (not isinstance(operands, list) or not 1 <= len(operands) <= 4
                        or (is_product and len(operands) < 2)
                        or (product_mode is not None and is_product != product_mode)):
                    break
                lookups = [operand.get("lookup") if isinstance(operand, dict) else None for operand in operands]
                if any(not isinstance(item, dict) for item in lookups):
                    break
                candidate_table = lookups[0].get("table")
                table = tables.get(candidate_table)
                if not isinstance(table, dict) or any(item.get("table") != candidate_table for item in lookups):
                    break
                primary_key = table.get("primary_key")
                where = lookups[0].get("where")
                if (not isinstance(primary_key, list) or not primary_key
                        or not isinstance(where, dict) or set(where) != set(primary_key)
                        or any(item.get("where") != where for item in lookups)):
                    break
                term_fields = [item.get("field") for item in lookups]
                if any(not isinstance(field, str) or not field for field in term_fields):
                    break
                rows = [row for row in table.get("rows", []) if isinstance(row, dict)
                        and all(row.get(key) == value for key, value in where.items())]
                identity = json.dumps(where, ensure_ascii=False, sort_keys=True)
                if len(rows) != 1 or identity in selected_ids:
                    break
                if table_name is not None and (candidate_table != table_name or term_fields != fields):
                    break
                table_name, fields, product_mode = candidate_table, term_fields, is_product
                selected_ids.add(identity)
                selected_rows.append(rows[0])
            else:
                assert table_name is not None and fields is not None
                table = tables[table_name]
                group_selectors = []
                for key, value in selected_rows[0].items():
                    if (key in table["primary_key"] or not isinstance(value, str)
                            or len(value) < 2 or value not in public_text
                            or not re.search(r"batch|group|category|period|month|date|cohort|lot", key, re.I)
                            or any(row.get(key) != value for row in selected_rows)):
                        continue
                    matching = [row for row in table["rows"] if isinstance(row, dict)
                                and row.get(key) == value]
                    if len(matching) == len(selected_rows) and all(
                        row in selected_rows for row in matching
                    ):
                        group_selectors.append((key, value))
                if len(group_selectors) == 1:
                    key, value = group_selectors[0]
                    aggregate = {"table": table_name, "where": {key: value}}
                    if product_mode:
                        aggregate.update({"fields": fields, "op": "sum_product"})
                    else:
                        aggregate.update({"field": fields[0], "op": "sum"})
                    target["expression"] = {"aggregate": aggregate}
        return compiled

    @staticmethod
    def _validate_metric_implementations(
        specs: list[Any], metrics: list[Any], *, require_process: bool = False
    ) -> None:
        from ..sandbox_runtime import DeclarativeMetricEvaluator, SandboxError

        by_id = {
            metric.get("id"): metric for metric in metrics
            if isinstance(metric, dict) and isinstance(metric.get("id"), str)
        }
        seen: set[str] = set()
        allowed_sources = {"business_state", "trajectory", "final_agent_response", "observation"}
        allowed_operators = {
            "eq", "ne", "gt", "gte", "lt", "lte", "contains", "exists", "count_gte",
            "count_eq", "none_tool_calls", "contains_tool_call", "numeric_targets", "value_targets", "state_predicates",
        }
        for index, spec in enumerate(specs):
            if not isinstance(spec, dict):
                raise PipelineGenerationError(f"metric_implementations[{index}] must be an object")
            metric_id = spec.get("metric_id")
            metric = by_id.get(metric_id)
            is_process_call = (
                isinstance(metric, dict)
                and metric.get("category") == "process"
                and spec.get("operator") == "contains_tool_call"
            )
            if metric is None or metric_id in seen or (
                metric.get("type") != "rule-based" and not is_process_call
            ):
                raise PipelineGenerationError(
                    f"metric_implementations[{index}] references an unsupported metric"
                )
            if spec.get("source") not in allowed_sources or spec.get("operator") not in allowed_operators:
                raise PipelineGenerationError(f"metric_implementations[{index}] source/operator is invalid")
            if not isinstance(spec.get("path"), str) or not spec["path"]:
                raise PipelineGenerationError(f"metric_implementations[{index}] path is invalid")
            try:
                DeclarativeMetricEvaluator.path_tokens(spec["path"])
            except (SandboxError, ValueError, SyntaxError) as exc:
                raise PipelineGenerationError(
                    f"metric_implementations[{index}] unsupported path {spec['path']!r}; "
                    "use fields, integer indices or literal equality filters"
                ) from exc
            if spec.get("source") == "final_agent_response" and spec.get("path") not in {"$", ""}:
                raise PipelineGenerationError(
                    f"metric_implementations[{index}] cannot address fields on raw final_agent_response"
                )
            if spec.get("source") == "business_state" and "final_agent_response" in str(spec.get("path")):
                raise PipelineGenerationError(
                    f"metric_implementations[{index}] cannot address final_agent_response through business_state"
                )
            if spec.get("operator") == "state_predicates":
                if spec.get("source") != "business_state" or spec.get("path") != "$":
                    raise PipelineGenerationError("state_predicates requires business_state at $")
                try:
                    DeclarativeMetricEvaluator.validate_state_predicates(spec.get("expected"))
                except SandboxError as exc:
                    raise PipelineGenerationError(f"invalid state reward: {exc}") from exc
            if spec.get("operator") in {"numeric_targets", "value_targets"}:
                if spec.get("source") != "final_agent_response" or spec.get("path") != "$":
                    raise PipelineGenerationError(
                        f"metric_implementations[{index}] numeric target source is invalid"
                    )
                try:
                    validator = (DeclarativeMetricEvaluator.validate_numeric_targets if spec["operator"] == "numeric_targets"
                                 else DeclarativeMetricEvaluator.validate_value_targets)
                    validator(spec.get("expected"))
                except SandboxError as exc:
                    raise PipelineGenerationError(
                        f"metric_implementations[{index}] numeric target spec is invalid: {exc}"
                    ) from exc
            if is_process_call:
                expected = spec.get("expected")
                if (
                    spec.get("source") != "trajectory"
                    or spec.get("path") != "$.events"
                    or not isinstance(expected, dict)
                    or not isinstance(expected.get("tool_name"), str)
                    or not isinstance(expected.get("arguments"), dict)
                    or not isinstance(expected.get("captures", []), list)
                ):
                    raise PipelineGenerationError(
                        f"metric_implementations[{index}] process call contract is invalid"
                    )
            mapping = spec.get("score_mapping")
            low, high = metric.get("score_range", [0, 1])
            if (
                not isinstance(mapping, dict) or set(mapping) != {"pass", "fail"}
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high for value in mapping.values())
            ):
                raise PipelineGenerationError(f"metric_implementations[{index}] score_mapping is invalid")
            seen.add(metric_id)
        required = {
            metric_id for metric_id, metric in by_id.items()
            if metric.get("type") == "rule-based"
            or (require_process and metric.get("category") == "process")
        }
        if seen != required:
            raise PipelineGenerationError(
                "rule-based and process metrics require executable implementations; "
                f"missing={sorted(required - seen)}"
            )

    @staticmethod
    def _compile_process_metric_implementations(
        *,
        metrics: list[Any],
        metric_implementations: list[Any],
        business_scenarios: list[Any],
        tool_bindings: list[Any],
    ) -> list[Any]:
        """Compile process rewards from the accepted success trajectory.

        The success scenario is already the executable source of truth for tool
        names and canonical arguments. Reusing it removes a redundant runtime
        LLM judgment while retaining support for arguments captured from an
        earlier tool result.
        """
        successes = [case for case in business_scenarios
                     if isinstance(case, dict) and case.get('kind') == 'goal_success']
        if any('user_script_indices' in case for case in successes):
            combined = {}
            scoped = {}
            action_tools = {item['action_name']: item['tool_name'] for item in tool_bindings}
            for case in successes:
                plain = {key: value for key, value in case.items() if key != 'user_script_indices'}
                called = {step.get('tool_name') for step in case.get('steps', []) if step.get('operation') == 'tool_call'}
                applicable_metrics = [metric for metric in metrics if metric.get('category') != 'process'
                    or not case.get('execution_route')
                    or action_tools.get(metric.get('target_action'), metric.get('target_action')) in called]
                rules = ContractCompiler._compile_process_metric_implementations(
                    metrics=applicable_metrics, metric_implementations=[rule for rule in metric_implementations
                        if rule.get('metric_id') in {metric['id'] for metric in applicable_metrics}],
                    business_scenarios=[plain], tool_bindings=tool_bindings)
                for rule in rules:
                    combined.setdefault(rule['metric_id'], rule)
                script_ids = ([f'script-{index + 1}' for index in case['user_script_indices']]
                              if 'user_script_indices' in case else ['*'])
                for rule in rules:
                    if rule.get('operator') != 'contains_tool_call':
                        continue
                    per_script = scoped.setdefault(rule['metric_id'], {})
                    for identity in script_ids:
                        per_script.setdefault(identity, []).append(copy.deepcopy(rule['expected']))
            if any(metric['id'] not in combined for metric in metrics if metric.get('category') == 'process'):
                raise PipelineGenerationError('process metric has no matching success tool call in any reference')
            for rule in combined.values():
                if rule['metric_id'] in scoped:
                    rule['expected']['by_user_script'] = scoped[rule['metric_id']]
            return list(combined.values())
        result = [dict(item) for item in metric_implementations if isinstance(item, dict)]
        result = [
            item for item in result
            if not any(
                isinstance(metric, dict)
                and metric.get("id") == item.get("metric_id")
                and metric.get("category") == "process"
                for metric in metrics
            )
        ]
        action_to_tool = {
            item.get("action_name"): item.get("tool_name")
            for item in tool_bindings
            if isinstance(item, dict)
            and isinstance(item.get("action_name"), str)
            and isinstance(item.get("tool_name"), str)
        }
        success = next((
            scenario for scenario in business_scenarios
            if isinstance(scenario, dict) and scenario.get("kind") == "goal_success"
        ), None)
        steps = success.get("steps", []) if isinstance(success, dict) else []
        if not isinstance(steps, list):
            steps = []
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "process":
                continue
            target = metric.get("target_action")
            tool_name = action_to_tool.get(target, target)
            target_index = next((
                index for index, step in enumerate(steps)
                if isinstance(step, dict)
                and step.get("operation") == "tool_call"
                and step.get("tool_name") == tool_name
            ), None)
            if target_index is None:
                raise PipelineGenerationError(
                    f"process metric {metric.get('id')} has no matching success tool call"
                )
            target_step = steps[target_index]
            arguments = target_step.get("arguments", {})
            if not isinstance(arguments, dict):
                raise PipelineGenerationError(
                    f"process metric {metric.get('id')} success arguments are invalid"
                )
            referenced = {
                value.get("$ref")
                for value in ContractCompiler._walk_values(arguments)
                if isinstance(value, dict) and set(value) == {"$ref"}
                and isinstance(value.get("$ref"), str)
            }
            captures: list[dict[str, str]] = []
            for step in steps[:target_index]:
                if not isinstance(step, dict) or step.get("operation") != "tool_call":
                    continue
                declared = step.get("capture", {})
                if not isinstance(declared, dict):
                    continue
                for name, path in declared.items():
                    if name in referenced and isinstance(path, str):
                        captures.append({
                            "name": name,
                            "tool_name": str(step.get("tool_name")),
                            "path": path,
                        })
            if referenced != {item["name"] for item in captures}:
                raise PipelineGenerationError(
                    f"process metric {metric.get('id')} has unresolved success references"
                )
            result.append({
                "metric_id": metric["id"],
                "source": "trajectory",
                "path": "$.events",
                "operator": "contains_tool_call",
                "expected": {
                    "tool_name": tool_name,
                    "arguments": copy.deepcopy(arguments),
                    "captures": captures,
                },
                "score_mapping": {"pass": 1, "fail": 0},
            })
        return result

    @staticmethod
    def _normalize_compiled_process_metrics(
        metrics: list[Any], metric_implementations: list[Any]
    ) -> None:
        """Make the metric declaration agree with its compiled runtime rule."""
        compiled = {
            item.get("metric_id"): item
            for item in metric_implementations
            if isinstance(item, dict) and item.get("operator") == "contains_tool_call"
        }
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "process":
                continue
            spec = compiled.get(metric.get("id"))
            if not isinstance(spec, dict):
                continue
            expected = spec.get("expected", {})
            metric["type"] = "rule-based"
            metric["condition"] = "trajectory_contains_exact_compiled_tool_call"
            metric["evaluator"] = {
                "kind": "trajectory_rule",
                "source": "runtime_rule",
                "assertion": (
                    "trajectory contains the accepted success-path call to "
                    f"{expected.get('tool_name')} with exact canonical arguments"
                ),
                "score_mapping": {"pass": 1, "fail": 0},
            }
            metric.pop("evaluation_inputs", None)
            metric.pop("criteria", None)

    @staticmethod
    def _walk_values(value: Any) -> list[Any]:
        values = [value]
        if isinstance(value, dict):
            for item in value.values():
                values.extend(ContractCompiler._walk_values(item))
        elif isinstance(value, list):
            for item in value:
                values.extend(ContractCompiler._walk_values(item))
        return values

    @staticmethod
    def _drop_initially_satisfied_stateful_outcome_specs(
        metrics: list[Any], specs: list[Any], *, data_tables: list[dict[str, Any]],
        environment_mode: str,
    ) -> list[Any]:
        """Promote a state outcome rule when it is true before any Agent action."""
        if environment_mode != "stateful" or not data_tables:
            return specs
        from ..sandbox_runtime import DeclarativeMetricEvaluator, SandboxError

        state = {
            table["table_name"]: table.get("rows", [])
            for table in data_tables if isinstance(table, dict)
            and isinstance(table.get("table_name"), str)
        }
        outcome_ids = {
            metric.get("id") for metric in metrics if isinstance(metric, dict)
            and metric.get("category") == "outcome"
            and metric.get("type") == "rule-based"
        }
        retained: list[Any] = []
        for spec in specs:
            if (not isinstance(spec, dict) or spec.get("metric_id") not in outcome_ids
                    or spec.get("source") != "business_state"):
                retained.append(spec)
                continue
            try:
                initial_score = DeclarativeMetricEvaluator().evaluate(spec, {
                    "business_state": state, "initial_business_state": state,
                    "trajectory": {"events": []}, "final_agent_response": "",
                })
                satisfied = initial_score == spec.get("score_mapping", {}).get("pass", 1)
            except (SandboxError, ValueError, TypeError, KeyError):
                retained.append(spec)
                continue
            if satisfied:
                logger.warning(
                    "promoting initially satisfied stateful outcome rule: metric=%s",
                    spec.get("metric_id"),
                )
            else:
                retained.append(spec)
        return retained

    @staticmethod
    def _promote_unimplemented_rule_metrics(
        metrics: list[Any], specs: list[Any], *, environment_mode: str | None = None
    ) -> None:
        """Never leave a declared rule metric without executable semantics.

        A rule that could not be compiled to the constrained DSL becomes an
        external judge.  This is slower, but preserves evaluability and is
        safer than asking the sandbox builder to interpret free-form text.
        """
        implemented = {
            spec.get("metric_id") for spec in specs if isinstance(spec, dict)
        }
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("type") != "rule-based":
                continue
            metric_id = metric.get("id")
            if metric_id in implemented:
                continue
            evaluator = metric.get("evaluator")
            assertion = evaluator.get("assertion") if isinstance(evaluator, dict) else None
            criterion = assertion if isinstance(assertion, str) and assertion.strip() else metric.get("rubric")
            scope = metric.get("scope")
            if scope == "trajectory":
                inputs = ["conversation", "tool_call", "tool_results", "final_agent_response"]
            elif scope == "terminal":
                inputs = ["final_agent_response", "terminal_observation"]
                if environment_mode != "stateless":
                    inputs.insert(1, "business_data")
            else:
                inputs = ["public_observation", "tool_results"]
                if environment_mode != "stateless":
                    inputs.insert(1, "business_data")
            metric["type"] = "model-based"
            metric["evaluation_inputs"] = inputs
            metric["criteria"] = [str(criterion or "判断该指标是否满足。")]
            metric["evaluator"] = {
                "kind": "external_llm_judge",
                "source": "external_llm",
                "score_mapping": dict(evaluator.get("score_mapping", {}))
                if isinstance(evaluator, dict) else {"pass": 1, "fail": 0},
            }
            logger.info("promoted uncompiled rule metric to external judge: metric=%s", metric_id)

    @staticmethod
    def _ensure_terminal_outcome_metrics(
        metrics: list[Any], specs: list[Any], *, environment_mode: str,
    ) -> None:
        """Make user-facing answers part of the declared reward contract."""
        if environment_mode == "stateful":
            return
        outcome_ids = {
            item.get("id") for item in metrics
            if isinstance(item, dict) and item.get("category") == "outcome"
        }
        if declared_terminal_outcome_ids(metrics, specs):
            return
        for metric in metrics:
            if not isinstance(metric, dict) or metric.get("category") != "outcome":
                continue
            rubric = str(metric.get("rubric") or "完成用户目标")
            metric["type"] = "model-based"
            metric["scope"] = "terminal"
            metric["evaluation_inputs"] = [
                "conversation", "tool_results", "final_agent_response",
            ]
            metric["criteria"] = [
                "最终回答必须完成用户当前目标；仅调用工具或检索到记录不能代替回答。",
                "回答中的结论必须由公开输入或工具结果支持，不得编造事实。",
                f"业务目标：{rubric}",
            ]
            metric["evaluator"] = {
                "kind": "external_llm_judge", "source": "external_llm",
                "score_mapping": {"full": 1, "partial": 0.5, "missing": 0},
            }
            metric["rubric"] = f"最终回答完成目标并有证据支持：{rubric}"
            metric.pop("condition", None)
        specs[:] = [
            spec for spec in specs
            if not isinstance(spec, dict) or spec.get("metric_id") not in outcome_ids
        ]
        logger.warning("promoted outcome reward to declared terminal evaluator: metrics=%s", sorted(outcome_ids))

    @staticmethod
    def _prune_literal_payload_process_metrics(
        metrics: list[Any], specs: list[Any],
    ) -> None:
        """Avoid process credit tied to one serialized free-form payload.

        A later process call that captures the parser's result already proves
        the parsing step occurred and supplies the causal dependency.  Only in
        that case can the brittle exact-input metric be removed safely.
        """
        brittle: set[str] = set()
        for spec in specs:
            if not has_literal_payload_argument(spec):
                continue
            tool_name = spec["expected"].get("tool_name")
            downstream = any(
                other is not spec and isinstance(other, dict)
                and isinstance(other.get("expected"), dict)
                and any(
                    isinstance(capture, dict)
                    and capture.get("tool_name") == tool_name
                    for capture in other["expected"].get("captures", [])
                )
                for other in specs
            )
            if not downstream:
                raise PipelineGenerationError(
                    f"process metric {spec.get('metric_id')} matches a literal payload "
                    "without downstream capture evidence"
                )
            brittle.add(str(spec.get("metric_id")))
        if not brittle:
            return
        metrics[:] = [
            metric for metric in metrics
            if not isinstance(metric, dict) or metric.get("id") not in brittle
        ]
        specs[:] = [
            spec for spec in specs
            if not isinstance(spec, dict) or spec.get("metric_id") not in brittle
        ]
        ContractCompiler._normalize_metric_weights(metrics)
        logger.warning("removed literal-payload process metrics with downstream capture proof: %s", sorted(brittle))

    @staticmethod
    def _validate_reward_formula(formula: Any, metrics: list[Any]) -> None:
        if not isinstance(formula, dict):
            raise PipelineGenerationError("reward_formula must be an object")
        if formula.get("type") != "separate_sign_weighted_sum":
            raise PipelineGenerationError("reward_formula.type must be separate_sign_weighted_sum")
        if not isinstance(formula.get("formula"), str) or not formula["formula"].strip():
            raise PipelineGenerationError("reward_formula.formula must be non-empty")
        if formula.get("score_range") != [-1, 1]:
            raise PipelineGenerationError("reward_formula.score_range must be [-1, 1]")
        if formula.get("positive_weight_sum") != 1 or formula.get("negative_weight_sum") != 1:
            raise PipelineGenerationError("reward_formula positive and negative weights must each sum to 1")

    @staticmethod
    def _canonical_reward_formula(metrics: list[Any]) -> dict[str, Any]:
        """Return a bounded formula derived from the validated metric weights.

        Process and outcome are one positive feedback pool. Penalty is the
        independent negative feedback pool. Dividing process and outcome into
        separate averages would make the positive contribution reach 2, so
        the canonical formula uses the already-normalized metric weights once.
        """
        positive_terms: list[str] = []
        negative_terms: list[str] = []
        for metric in metrics:
            metric_id = str(metric["id"])
            term = f"{float(metric['weight']):.12g}*score({metric_id})"
            if metric["category"] in {"process", "outcome"}:
                positive_terms.append(term)
            else:
                negative_terms.append(term)
        positive_expression = " + ".join(positive_terms) or "0"
        negative_expression = " + ".join(negative_terms) or "0"
        return {
            "type": "separate_sign_weighted_sum",
            "formula": f"R = clip(({positive_expression}) + ({negative_expression}), -1, 1)",
            "positive_weight_sum": 1,
            "negative_weight_sum": 1,
            "score_range": [-1, 1],
            "positive_categories": ["process", "outcome"],
            "negative_categories": ["penalty"],
            "normalization": "metric weights are normalized once across all positive categories and independently across penalty metrics",
        }

    @staticmethod
    def _materialize_tools(tools: list[dict[str, Any]], artifact_dir: Path) -> dict[str, Any]:
        """Persist only standard OpenAI tool definitions.

        Bindings, Trainer actions, and implementation details deliberately stay
        outside this file. Code Agent consumes this schema and implements the
        corresponding sandbox behavior separately.
        """
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / "tools.json"
        path.write_text(
            json.dumps(tools, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return {
            "version": "1.0",
            "file": str(path.relative_to(artifact_dir)),
            "format": "openai_function_tools",
            "tool_count": len(tools),
        }

    @staticmethod
    def _table_parent_names(table: dict[str, Any]) -> list[str]:
        return list(dict.fromkeys(
            name for foreign in table.get("foreign_keys", [])
            if isinstance(foreign, dict)
            if isinstance((name := foreign.get("ref_table") or foreign.get("references_table")), str)
            and name != table.get("table_name")
        ))

    @classmethod
    def _table_generation_layers(cls, tables: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Generate parents before children while retaining independent parallelism."""
        pending = {table["table_name"]: table for table in tables}
        layers: list[list[dict[str, Any]]] = []
        completed: set[str] = set()
        while pending:
            ready = [table for table in pending.values()
                     if set(cls._table_parent_names(table)) <= completed]
            if not ready:
                # Cyclic schemas still receive the existing global consistency
                # repair; row generation cannot order a cycle.
                ready = list(pending.values())
            layers.append(ready)
            for table in ready:
                name = table["table_name"]
                completed.add(name)
                del pending[name]
        return layers

    @staticmethod
    def _validate_parent_references(
        table: dict[str, Any], rows: list[dict[str, Any]],
        parent_tables: list[dict[str, Any]],
    ) -> None:
        parents = {item["table_name"]: item for item in parent_tables}
        for foreign in table.get("foreign_keys", []):
            if not isinstance(foreign, dict):
                continue
            parent = parents.get(foreign.get("ref_table") or foreign.get("references_table"))
            if parent is None:
                continue
            column = foreign.get("column")
            parent_column = foreign.get("ref_column") or foreign.get("references_column")
            allowed = {row.get(parent_column) for row in parent.get("rows", [])}
            missing = sorted({row.get(column) for row in rows if row.get(column) is not None} - allowed,
                             key=str)
            if missing:
                raise PipelineGenerationError(
                    f"data table {table['table_name']}.{column} has missing foreign values: {missing[:5]}"
                )

    @staticmethod
    def _validate_generated_table_rows(
        table: dict[str, Any], rows: Any, *, check_values: bool = True,
    ) -> None:
        from ..sandbox_runtime import ManifestDataStore, SandboxError

        name = str(table["table_name"])
        if not isinstance(rows, list) or not rows:
            raise PipelineGenerationError(f"data table {name} requires complete non-empty rows")
        columns = {str(column["name"]): column for column in table["columns"]}
        primary = table["primary_key"]
        keys: set[tuple[Any, ...]] = set()
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or set(row) != set(columns):
                raise PipelineGenerationError(f"data table {name}.rows[{index}] is not a complete row")
            if not check_values:
                continue
            key = tuple(row[field] for field in primary)
            try:
                duplicate = key in keys
            except TypeError as exc:
                raise PipelineGenerationError(f"data table {name} has a non-scalar primary key") from exc
            if any(value is None for value in key) or duplicate:
                raise PipelineGenerationError(f"data table {name} has null or duplicate primary keys")
            keys.add(key)
            for column_name, column in columns.items():
                if row[column_name] is None and column.get("nullable") is False:
                    raise PipelineGenerationError(
                        f"data table {name}.rows[{index}].{column_name} violates NOT NULL"
                    )
            for constraint in table.get("constraints", []):
                expression = constraint if isinstance(constraint, str) else constraint.get("expression")
                try:
                    valid = ManifestDataStore._check_constraint(row, expression)
                except (SandboxError, TypeError, ValueError) as exc:
                    raise PipelineGenerationError(
                        f"data table {name} has unsupported CHECK constraint: {expression}"
                    ) from exc
                if not valid:
                    raise PipelineGenerationError(
                        f"data table {name}.rows[{index}] violates CHECK constraint: {expression}"
                    )

    @staticmethod
    def _validate_data_tables(tables: Any) -> None:
        if not isinstance(tables, list) or not tables:
            raise PipelineGenerationError("business data tables must be a non-empty list")
        ContractCompiler._validate_table_definitions(tables)
        for table in tables:
            ContractCompiler._validate_generated_table_rows(
                table, table.get("rows"), check_values=False,
            )

    @staticmethod
    def _validate_relational_data(tables: list[dict[str, Any]]) -> None:
        """Validate generated rows with the same relational invariants as runtime."""
        ContractCompiler._validate_data_tables(tables)
        by_name = {str(table["table_name"]): table for table in tables}
        for table in tables:
            name = str(table["table_name"])
            rows = table["rows"]
            ContractCompiler._validate_generated_table_rows(table, rows)
            for foreign in table.get("foreign_keys", []):
                if not isinstance(foreign, dict):
                    raise PipelineGenerationError(f"data table {name} has an invalid foreign key")
                field = foreign.get("column")
                parent_name = foreign.get("ref_table") or foreign.get("references_table")
                target = foreign.get("ref_column") or foreign.get("references_column")
                parent = by_name.get(str(parent_name))
                parent_columns = {
                    item.get("name") for item in parent.get("columns", [])
                } if isinstance(parent, dict) else set()
                own_columns = {item.get("name") for item in table.get("columns", [])}
                if field not in own_columns or target not in parent_columns:
                    raise PipelineGenerationError(
                        f"data table {name} foreign key declaration is invalid"
                    )
                available = {row.get(target) for row in parent["rows"]}
                missing = sorted({row.get(field) for row in rows if row.get(field) is not None} - available, key=str)
                if missing:
                    raise PipelineGenerationError(
                        f"data table {name}.{field} has missing foreign values: {missing[:5]}"
                    )

    @staticmethod
    def _validate_entities(entities: Any) -> None:
        if not isinstance(entities, list) or not entities:
            raise PipelineGenerationError("environment_entities.entities must be a non-empty list")
        ids: set[str] = set()
        for index, entity in enumerate(entities):
            if not isinstance(entity, dict):
                raise PipelineGenerationError(f"entities[{index}] must be an object")
            entity_id = entity.get("entity_id") or entity.get("name")
            if not isinstance(entity_id, str) or not entity_id.strip() or entity_id in ids:
                raise PipelineGenerationError(f"entities[{index}] has invalid or duplicate entity_id/name")
            if not isinstance(entity.get("description"), str) or not entity["description"].strip():
                raise PipelineGenerationError(f"entities[{index}] requires a description")
            if not isinstance(entity.get("required_facts", []), list):
                raise PipelineGenerationError(f"entities[{index}].required_facts must be a list")
            ids.add(entity_id)

    @staticmethod
    def _validate_table_definitions(tables: Any) -> None:
        from ..sandbox_runtime import ManifestDataStore, SandboxError

        if not isinstance(tables, list) or not tables:
            raise PipelineGenerationError("environment_table_design.tables must be a non-empty list")
        declared_names = [
            table.get("table_name") for table in tables if isinstance(table, dict)
        ]
        if any(not isinstance(name, str) or not name.strip() for name in declared_names):
            raise PipelineGenerationError("table definitions contain an invalid table_name")
        if len(declared_names) != len(set(declared_names)):
            raise PipelineGenerationError("table definitions contain duplicate table_name values")
        known_tables = set(declared_names)
        names: set[str] = set()
        for index, table in enumerate(tables):
            if not isinstance(table, dict):
                raise PipelineGenerationError(f"table definitions[{index}] must be an object")
            name = table.get("table_name")
            columns = table.get("columns")
            primary_key = table.get("primary_key")
            if not isinstance(name, str) or not name.strip() or name in names:
                raise PipelineGenerationError(f"table definitions[{index}] has invalid or duplicate table_name")
            if not isinstance(columns, list) or not columns:
                raise PipelineGenerationError(f"table definitions[{index}] requires columns")
            if not isinstance(primary_key, list) or not primary_key:
                raise PipelineGenerationError(f"table definitions[{index}] requires primary_key")
            column_names: set[str] = set()
            for column in columns:
                if not isinstance(column, dict) or not isinstance(column.get("name"), str) or not column["name"].strip():
                    raise PipelineGenerationError(f"table definitions[{index}] contains an invalid column")
                if column["name"] in column_names:
                    raise PipelineGenerationError(f"table definitions[{index}] contains duplicate columns")
                column_names.add(column["name"])
            if any(key not in column_names for key in primary_key):
                raise PipelineGenerationError(f"table definitions[{index}] primary_key references an unknown column")
            for field in ("foreign_keys", "indexes", "constraints"):
                if field in table and not isinstance(table[field], list):
                    raise PipelineGenerationError(f"table definitions[{index}].{field} must be a list")
            for foreign in table.get("foreign_keys", []):
                ref_table = foreign.get("ref_table") or foreign.get("references_table") if isinstance(foreign, dict) else None
                ref_column = foreign.get("ref_column") or foreign.get("references_column") if isinstance(foreign, dict) else None
                if (
                    not isinstance(foreign, dict)
                    or not isinstance(foreign.get("column"), str)
                    or foreign.get("column") not in column_names
                    or ref_table not in known_tables
                    or not isinstance(ref_column, str)
                    or not ref_column.strip()
                ):
                    raise PipelineGenerationError(
                        f"table definitions[{index}] contains an invalid foreign key"
                    )
            for constraint in table.get("constraints", []):
                if isinstance(constraint, str):
                    expression = constraint.strip()
                elif isinstance(constraint, dict) and str(constraint.get("type", "CHECK")).upper() == "CHECK":
                    expression = constraint.get("expression")
                else:
                    expression = None
                if not isinstance(expression, str) or not expression.strip():
                    raise PipelineGenerationError(
                        f"table definitions[{index}] contains an invalid CHECK constraint"
                    )
                for clause in re.split(r"\s+AND\s+", expression.strip(), flags=re.I):
                    nullable_comparison = re.fullmatch(
                        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+IS\s+NULL\s+OR\s+\1\s*"
                        r"(>=|<=|<>|!=|=|>|<)\s*(.*?)\s*",
                        clause, flags=re.I,
                    )
                    if nullable_comparison:
                        field, operator, rhs = nullable_comparison.groups()
                        clause = f"{field} {operator} {rhs}"
                    nullable_in_match = re.fullmatch(
                        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\((.*?)\)\s+OR\s+\1\s+IS\s+NULL\s*",
                        clause, flags=re.I,
                    )
                    in_match = re.fullmatch(
                        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s+IN\s*\((.*)\)\s*",
                        clause, flags=re.I,
                    )
                    comparison = re.fullmatch(
                        r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|<>|!=|=|>|<)\s*(.*?)\s*",
                        clause,
                    )
                    match = nullable_in_match or in_match or comparison
                    if not match or match.group(1) not in column_names:
                        raise PipelineGenerationError(
                            f"table definitions[{index}] has unsupported CHECK constraint: {expression}"
                        )
                    try:
                        if nullable_in_match or in_match:
                            ManifestDataStore._constraint_in_values(match.group(2))
                        else:
                            rhs = comparison.group(3).strip()
                            product = re.fullmatch(
                                r"([A-Za-z_][A-Za-z0-9_]*)\s*\*\s*([A-Za-z_][A-Za-z0-9_]*)",
                                rhs,
                            )
                            if product and set(product.groups()) <= column_names:
                                numeric_columns = {
                                    item["name"] for item in columns
                                    if ManifestDataStore._column_json_type(item.get("type"))
                                    in {"integer", "number"}
                                }
                                if set(product.groups()) <= numeric_columns:
                                    continue
                            if rhs not in column_names:
                                ManifestDataStore._constraint_literal(rhs)
                    except (ValueError, SyntaxError, SandboxError) as exc:
                        raise PipelineGenerationError(
                            f"table definitions[{index}] has unsupported CHECK constraint: {expression}"
                        ) from exc
            names.add(name)

    @staticmethod
    def _normalize_structural_constraints(tables: Any) -> Any:
        """Compile equivalent structural forms into the supported schema.

        Mid-tier models occasionally put structural constraints such as
        ``name 唯一且非空`` in CHECK constraints.  Those are not executable
        CHECK expressions, but their meaning is unambiguous and already has a
        canonical representation in the schema.
        """
        if not isinstance(tables, list):
            return tables
        normalized = copy.deepcopy(tables)
        for table in normalized:
            if not isinstance(table, dict):
                continue
            for foreign in table.get("foreign_keys", []):
                if not isinstance(foreign, dict):
                    continue
                local_columns = foreign.get("columns")
                if ("column" not in foreign and isinstance(local_columns, list)
                        and len(local_columns) == 1
                        and isinstance(local_columns[0], str)):
                    foreign["column"] = local_columns[0]
                ref_columns = foreign.get("references_columns") or foreign.get("ref_columns")
                if ("ref_column" not in foreign and "references_column" not in foreign
                        and isinstance(ref_columns, list) and len(ref_columns) == 1
                        and isinstance(ref_columns[0], str)):
                    foreign["ref_column"] = ref_columns[0]
            columns = {
                column.get("name"): column for column in table.get("columns", [])
                if isinstance(column, dict) and isinstance(column.get("name"), str)
            }
            indexes = table.setdefault("indexes", [])
            kept = []
            for constraint in table.get("constraints", []):
                if not isinstance(constraint, str):
                    kept.append(constraint)
                    continue
                lowered = constraint.casefold()
                unique = "唯一" in constraint or bool(re.search(r"\bunique\b", lowered))
                non_null = (
                    "非空" in constraint
                    or bool(re.search(r"\bnot[ _-]?null\b", lowered))
                    or "不能为空" in constraint
                )
                mentioned = [
                    name for name in columns
                    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", constraint)
                ]
                if (unique or non_null) and len(mentioned) == 1:
                    column_name = mentioned[0]
                    if unique and not any(
                        isinstance(index, dict)
                        and index.get("unique") is True
                        and index.get("columns") == [column_name]
                        for index in indexes
                    ):
                        indexes.append({
                            "name": f"uq_{table.get('table_name', 'table')}_{column_name}",
                            "columns": [column_name], "unique": True,
                        })
                    if non_null:
                        columns[column_name]["nullable"] = False
                    continue
                kept.append(constraint)
            table["constraints"] = kept
        return normalized

    @staticmethod
    def _materialize_business_data(
        tables: list[dict[str, Any]],
        data_document: str,
        artifact_dir: Path,
        *,
        environment_mode: str = "business_data",
        manifest_root: str | None = None,
    ) -> dict[str, Any]:
        """Write table schemas/rows to files and return the compact task manifest."""
        artifact_dir.mkdir(parents=True, exist_ok=True)
        schema_dir = artifact_dir / "schemas"
        rows_dir = artifact_dir / "rows"
        schema_dir.mkdir(exist_ok=True)
        rows_dir.mkdir(exist_ok=True)
        used_names: set[str] = set()
        manifest_tables: list[dict[str, Any]] = []

        for index, table in enumerate(tables):
            raw_name = str(table["table_name"])
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("._") or f"table_{index + 1}"
            if safe_name in used_names:
                safe_name = f"{safe_name}_{index + 1}"
            used_names.add(safe_name)
            schema = {key: value for key, value in table.items() if key != "rows"}
            schema_path = schema_dir / f"{safe_name}.json"
            rows_path = rows_dir / f"{safe_name}.jsonl"
            schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            with rows_path.open("w", encoding="utf-8") as handle:
                for row in table["rows"]:
                    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            manifest_tables.append({
                "table_name": raw_name,
                "schema_file": str(schema_path.relative_to(artifact_dir)),
                "rows_file": str(rows_path.relative_to(artifact_dir)),
                "row_count": len(table["rows"]),
                "primary_key": table["primary_key"],
            })

        document_path = artifact_dir / "data_document.md"
        document_path.write_text(data_document.rstrip() + "\n", encoding="utf-8")
        return {
            "version": "1.0",
            "environment_mode": environment_mode,
            "data_governance": {
                "origin": "model_generated_synthetic",
                "contains_real_user_data": False,
                "intended_use": "agentic_rl_training_material",
            },
            "root": manifest_root or str(artifact_dir),
            "document_file": str(document_path.relative_to(artifact_dir)),
            "tables": manifest_tables,
        }

    @staticmethod
    def _environment_records(environment: dict[str, Any], actions: list[Any]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in environment.get("records", []):
            if isinstance(item, dict) and {"type", "field", "description", "value"} <= set(item):
                records.append({k: item[k] for k in ("type", "field", "description", "value")})
        for table in environment.get("data_tables", []):
            if isinstance(table, dict) and isinstance(table.get("table_name"), str):
                records.append({
                    "type": "business_table",
                    "field": table["table_name"],
                    "description": str(table.get("description") or "任务业务数据表"),
                    "value": table,
                })
        for action in actions:
            if isinstance(action, dict):
                name = action.get("name") or action.get("action")
                if isinstance(name, str) and name.strip():
                    records.append({"type": "action", "name": name, "description": action.get("description", name)})
        return records
