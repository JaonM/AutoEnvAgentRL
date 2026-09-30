"""Versioned intermediate representation for compiling RL task environments.

The IR separates model-authored semantics from EnvFactory-owned executable
contracts.  It is deliberately compact: large row fixtures remain in the data
manifest, while predicates, tool effects and dependency edges are explicit.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


TASK_SPEC_VERSION = "1.0"
ENVIRONMENT_ARCHETYPES = frozenset({
    "text_only", "single_read", "single_mutation", "multi_read_join",
    "multi_step_mutation", "external_lookup", "external_then_mutate",
})


class TaskSpecError(ValueError):
    """Raised when semantic inputs cannot compile into a trainable contract."""


def validate_goal_contract(goal: Mapping[str, Any], data_tables: Sequence[Mapping[str, Any]], *, require_change: bool = True) -> None:
    """Require typed, addressable predicates rather than prose matching."""
    if "allow_noop" in goal and not isinstance(goal["allow_noop"], bool):
        raise TaskSpecError("goal allow_noop must be boolean")
    require_change = require_change and goal.get("allow_noop") is not True
    tables = {item["table_name"]: item for item in data_tables}
    predicates = goal.get("row_predicates")
    if not isinstance(predicates, list) or not predicates:
        raise TaskSpecError("stateful goal requires non-empty row_predicates")
    for predicate in predicates:
        if not isinstance(predicate, Mapping) or predicate.get("table") not in tables:
            raise TaskSpecError("goal predicate references unknown table")
        table = tables[predicate["table"]]
        columns = {column["name"] for column in table.get("columns", [])}
        if not columns and table.get("rows"):
            columns = set(table["rows"][0])
        for field in ("where", "values"):
            value = predicate.get(field)
            if not isinstance(value, Mapping) or not set(value) <= columns:
                raise TaskSpecError(f"goal predicate {field} references invalid columns")
        expressions = predicate.get("value_expressions", {})
        if (not isinstance(expressions, Mapping) or not set(expressions) <= columns
                or set(expressions) & set(predicate["values"])):
            raise TaskSpecError("goal value_expressions references invalid or duplicate columns")
        from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, SandboxError
        for expression in expressions.values():
            try:
                DeclarativeMetricEvaluator._value_expression(expression, {})
            except SandboxError as exc:
                raise TaskSpecError(f"invalid goal value expression: {exc}") from exc
        if not predicate["where"] and not predicate["values"] and not expressions:
            raise TaskSpecError("goal predicate must identify target rows or values")
        count = predicate.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise TaskSpecError("goal predicate count must be a non-negative integer")
    from env_factory.sandbox_runtime import BusinessGoalEvaluator
    baseline = {name: table.get("rows", []) for name, table in tables.items()}
    if require_change and BusinessGoalEvaluator.evaluate(predicates, baseline, baseline):
        raise TaskSpecError("initial fixture already satisfies the stateful goal")


def _objects(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, Mapping)] if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _refs(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        if set(value) == {"$ref"} and isinstance(value.get("$ref"), str):
            return {str(value["$ref"])}
        result: set[str] = set()
        for item in value.values():
            result.update(_refs(item))
        return result
    if isinstance(value, list):
        result: set[str] = set()
        for item in value:
            result.update(_refs(item))
        return result
    return set()


def _ref_paths(value: Any, path: str = "$") -> list[tuple[str, str, Any]]:
    """Locate each captured value at its actual nested tool argument path."""
    if isinstance(value, Mapping):
        if set(value) == {"$ref"} and isinstance(value.get("$ref"), str):
            return [(value["$ref"], path, None)]
        if set(value) == {"$expr"}:
            return [(reference, path, value) for reference in sorted(_refs(value))]
        paths: list[tuple[str, str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
                if _refs(item):
                    raise TaskSpecError("captured dependency uses an unsupported argument path")
                continue
            paths.extend(_ref_paths(item, f"{path}.{key}"))
        return paths
    if isinstance(value, list):
        return [entry for index, item in enumerate(value)
                for entry in _ref_paths(item, f"{path}[{index}]")]
    return []


def derive_environment_archetype(
    *, mode: str, training_category: str, tool_bindings: Sequence[Mapping[str, Any]],
) -> str:
    count = len(tool_bindings)
    if mode == "stateless":
        return "text_only"
    if mode == "external_capability":
        return "external_lookup"
    if mode == "reference_data":
        return "single_read" if count <= 1 else "multi_read_join"
    if mode == "stateful":
        return "single_mutation" if count <= 1 else "multi_step_mutation"
    raise TaskSpecError(f"unsupported environment mode: {mode}")


def _explicit_changes(task_description: Mapping[str, Any]) -> list[dict[str, str]]:
    text = json.dumps(task_description, ensure_ascii=False)
    return [
        {"before": before.strip(), "after": after.strip()}
        for before, after in re.findall(
            r"从\s*[“\"]([^”\"]+)[”\"]\s*(?:更正|修改|更新|调整|改|变更)?\s*为\s*[“\"]([^”\"]+)[”\"]",
            text,
        )
    ]


def _capability_edges(scenarios: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for scenario in scenarios:
        if scenario.get("kind") != "goal_success":
            continue
        producers: dict[str, tuple[str, str]] = {}
        for step in _objects(scenario.get("steps")):
            if step.get("operation") != "tool_call":
                continue
            tool_name = str(step.get("tool_name", ""))
            for reference, argument_path, expression in _ref_paths(step.get("arguments", {})):
                producer = producers.get(reference)
                if producer and producer[0] != tool_name:
                    edge = {"from_tool": producer[0], "to_tool": tool_name, "via": reference,
                            "result_path": producer[1], "argument_path": argument_path}
                    if expression is not None:
                        refs = sorted(_refs(expression))
                        if any(name not in producers for name in refs):
                            raise TaskSpecError("argument expression has an unavailable capture")
                        edge["argument_expression"] = expression
                        edge["expression_captures"] = [{"name": name, "tool_name": producers[name][0],
                            "path": producers[name][1]} for name in refs]
                    if edge not in edges:
                        edges.append(edge)
            capture = step.get("capture")
            if isinstance(capture, Mapping):
                for name in capture:
                    producers[str(name)] = (tool_name, str(capture[name]))
    return edges


def _key_step_edges(
    key_steps: Sequence[Mapping[str, Any]],
    tool_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Compile reviewed action dependencies into tool-level DAG edges."""
    action_to_tool = {
        str(item.get("action_name")): str(item.get("tool_name"))
        for item in _objects(tool_bindings)
        if item.get("action_name") and item.get("tool_name")
    }
    step_to_action = {
        str(item.get("step_id")): str(item.get("action_name"))
        for item in _objects(key_steps)
        if item.get("step_id") and item.get("action_name")
    }
    edges: list[dict[str, str]] = []
    for step in _objects(key_steps):
        target_action = str(step.get("action_name", ""))
        target_tool = action_to_tool.get(target_action)
        if not target_tool:
            continue
        for dependency in step.get("dependencies", []):
            source_action = step_to_action.get(str(dependency))
            source_tool = action_to_tool.get(str(source_action))
            if source_tool and source_tool != target_tool:
                edge = {
                    "from_tool": source_tool,
                    "to_tool": target_tool,
                    "via": f"step_dependency:{dependency}",
                }
                if edge not in edges:
                    edges.append(edge)
    return edges


def _declarative_output_schema(implementation: Mapping[str, Any]) -> dict[str, Any] | None:
    operation = implementation.get("operation")
    result_field = implementation.get("result_field")
    if operation == "select" and isinstance(result_field, str) and result_field:
        projection = implementation.get("projection", [])
        aliases = implementation.get("projection_aliases", {})
        fields = [field for field in projection if isinstance(field, str) and field] if isinstance(projection, list) else []
        if isinstance(aliases, Mapping):
            fields.extend(alias for alias in aliases if isinstance(alias, str) and alias)
        fields = list(dict.fromkeys(fields))
        item_schema: dict[str, Any] = {"type": "object"}
        if fields:
            item_schema["properties"] = {field: {} for field in fields}
            item_schema["required"] = fields
        return {
            "type": "object",
            "properties": {
                result_field: {"type": "array", "items": item_schema},
                "count": {"type": "integer"},
            },
            "required": [result_field, "count"],
        }
    if operation == "aggregate_count" and isinstance(result_field, str) and result_field:
        return {
            "type": "object", "properties": {result_field: {"type": "integer"}},
            "required": [result_field],
        }
    field = {"insert": "record", "update": "updated_count", "delete": "deleted_count"}.get(str(operation))
    if field:
        value_schema = {"type": "object"} if operation == "insert" else {"type": "integer"}
        properties = {field: value_schema}
        required = [field]
        if operation == "insert":
            properties["count"] = {"type": "integer"}
            required.append("count")
        return {"type": "object", "properties": properties, "required": required}
    return None


def compile_task_spec(
    *,
    task_description: Mapping[str, Any],
    training_category: str,
    environment_plan: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    data_tables: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    noise_tools: Sequence[Mapping[str, Any]],
    tool_bindings: Sequence[Mapping[str, Any]],
    tool_implementations: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    key_steps: Sequence[Mapping[str, Any]],
    metrics: Sequence[Mapping[str, Any]],
    executable_scenarios: Sequence[Mapping[str, Any]],
    semantic_goal: Mapping[str, Any] | None = None,
    capability_plan: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compile validated pipeline artifacts into a single semantic IR."""
    mode = str(environment_plan.get("mode", ""))
    bindings = _objects(tool_bindings)
    archetype = derive_environment_archetype(
        mode=mode, training_category=training_category, tool_bindings=bindings,
    )
    noise_names = {str(item.get("name")) for item in _objects(noise_tools)}
    implementation_by_tool = {
        str(item.get("tool_name")): item for item in _objects(tool_implementations)
    }
    action_by_name = {str(item.get("name")): item for item in _objects(actions)}
    binding_by_tool = {str(item.get("tool_name")): item for item in bindings}
    tool_contracts: list[dict[str, Any]] = []
    for tool in _objects(tools):
        function = tool.get("function") if isinstance(tool.get("function"), Mapping) else {}
        name = str(function.get("name", ""))
        implementation = implementation_by_tool.get(name, {})
        binding = binding_by_tool.get(name, {})
        action = action_by_name.get(str(binding.get("action_name", "")), {})
        output_schema = implementation.get("output_schema")
        if not isinstance(output_schema, Mapping):
            output_schema = _declarative_output_schema(implementation)
        tool_contracts.append({
            "name": name,
            "role": "noise" if name in noise_names else "business",
            "input_schema": function.get("parameters", {}),
            "output_contract": {
                "result_field": implementation.get("result_field"),
                "schema": output_schema,
            },
            "execution": {
                "operation": implementation.get("operation") or ("noise" if name in noise_names else "custom"),
                "table": implementation.get("table"),
            },
            "preconditions": list(action.get("preconditions", [])) if isinstance(action.get("preconditions"), list) else [],
            "effects": list(action.get("effects", action.get("expected_effects", [])))
            if isinstance(action.get("effects", action.get("expected_effects", [])), list) else [],
        })
    # A typed goal is authoritative. Inferring additional changes from repeated
    # task/goal/public-input prose reintroduces a second, contradictory truth.
    changes = [] if semantic_goal is not None else _explicit_changes(task_description)
    goal_contract = {
        "initial_predicates": [
            {"kind": "contains_business_value", "value": item["before"]} for item in changes
        ],
        "success_predicates": [
            {"kind": "contains_business_value", "value": item["after"]} for item in changes
        ],
        "expected_delta": changes,
        "forbidden_deltas": [],
        "reset_invariant": "business_state_equals_manifest_baseline",
    }
    if semantic_goal is not None:
        validate_goal_contract(semantic_goal, data_tables)
        goal_contract.update(semantic_goal)
        goal_contract["forbidden_deltas"] = [{
            "kind": "outside_goal_rows_or_fields",
            "scope": "row_predicates",
            "enforcement": "BusinessGoalEvaluator.preserves_unrelated",
        }]
        goal_contract["requires_state_change"] = mode == "stateful" and semantic_goal.get("allow_noop") is not True
        goal_contract["table_primary_keys"] = {table["table_name"]: table.get("primary_key", []) for table in data_tables}
    scenarios = _objects(executable_scenarios)
    edges = _key_step_edges(key_steps, bindings)
    action_tools = {item["action_name"]: item["tool_name"] for item in bindings}
    for capability in capability_plan:
        target = action_tools.get(capability.get("action_name"))
        for dependency in capability.get("dependencies", []):
            source = action_tools.get(dependency)
            if source and target and source != target:
                edges.append({"from_tool": source, "to_tool": target, "via": "capability_dependency"})
    for edge in _capability_edges(scenarios):
        if edge not in edges:
            edges.append(edge)
    spec = {
        "version": TASK_SPEC_VERSION,
        "task_contract": {
            "task": task_description.get("task"),
            "intent": task_description.get("task_intent"),
            "goal": task_description.get("goal"),
            "expected_result": task_description.get("expected_result"),
            "requirements": task_description.get("requirements", {}),
            "public_input": task_description.get("public_input", {}),
        },
        "training_contract": {
            "category": training_category,
            "environment_archetype": archetype,
        },
        "environment_contract": {
            "mode": mode,
            "archetype": archetype,
            "persistence_required": environment_plan.get("requires_persistence") is True,
            "initial_fixture": {
                "manifest": data_manifest,
                "table_names": [str(item.get("table_name")) for item in _objects(data_tables)],
            },
        },
        "tool_contracts": tool_contracts,
        "capability_dag": {
            "nodes": [item["name"] for item in tool_contracts if item["role"] == "business"],
            "edges": edges,
            "success_paths": [list(dict.fromkeys(step["tool_name"] for step in _objects(scenario.get("steps"))
                              if step.get("operation") == "tool_call" and step.get("tool_name") in binding_by_tool))
                              for scenario in scenarios if scenario.get("kind") == "goal_success"],
        },
        "goal_contract": goal_contract,
        "reward_contract": {
            "metric_ids": [str(item.get("id")) for item in _objects(metrics)],
            "primary_truth": "business_state" if mode == "stateful" else "tool_result_or_response",
        },
        "acceptance_contract": {
            "scenario_ids": [str(item.get("scenario_id")) for item in scenarios],
        },
    }
    validate_task_spec(spec, data_tables=data_tables)
    return spec


def validate_task_spec(spec: Mapping[str, Any], *, data_tables: Sequence[Mapping[str, Any]] | None = None) -> None:
    if spec.get("version") != TASK_SPEC_VERSION:
        raise TaskSpecError("task_spec version is invalid")
    environment = spec.get("environment_contract")
    if not isinstance(environment, Mapping) or environment.get("archetype") not in ENVIRONMENT_ARCHETYPES:
        raise TaskSpecError("task_spec environment archetype is invalid")
    training = spec.get("training_contract")
    if not isinstance(training, Mapping):
        raise TaskSpecError("task_spec training contract is missing")
    tools = _objects(spec.get("tool_contracts"))
    business_tools = [item for item in tools if item.get("role") == "business"]
    category = training.get("category")
    if category not in {"direct_response", "simple_agentic", "multi_step_agentic"}:
        raise TaskSpecError("task_spec training category is invalid")
    names = [item.get("name") for item in business_tools]
    if len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
        raise TaskSpecError("task_spec business tool names are invalid")
    dag = spec.get("capability_dag", {})
    edges = dag.get("edges", [])
    graph = {name: set() for name in names}
    for edge in edges:
        if not isinstance(edge, Mapping) or edge.get("from_tool") not in graph or edge.get("to_tool") not in graph:
            raise TaskSpecError("capability edge references unknown tool")
        if bool(edge.get("result_path")) != bool(edge.get("argument_path")):
            raise TaskSpecError("captured dependency requires both result_path and argument_path")
        if "argument_expression" in edge:
            captures = edge.get("expression_captures")
            if (not isinstance(captures, list) or not captures
                    or any(not isinstance(item, Mapping) or item.get("tool_name") not in graph
                           or not isinstance(item.get("name"), str) or not isinstance(item.get("path"), str)
                           for item in captures)
                    or {item["name"] for item in captures} != _refs(edge["argument_expression"])):
                raise TaskSpecError("argument expression requires complete capture bindings")
        graph[edge["to_tool"]].add(edge["from_tool"])
    remaining = set(graph)
    while remaining:
        ready = {name for name in remaining if not graph[name] & remaining}
        if not ready:
            raise TaskSpecError("capability DAG contains a cycle")
        remaining -= ready
    if category == "direct_response" and business_tools:
        raise TaskSpecError("direct_response task_spec cannot contain business tools")
    if category == "simple_agentic" and len(business_tools) != 1:
        raise TaskSpecError("simple_agentic task_spec requires exactly one business tool")
    if category == "multi_step_agentic":
        edges = spec.get("capability_dag", {}).get("edges", [])
        if len(business_tools) < 2 or not isinstance(edges, list) or not edges:
            raise TaskSpecError("multi_step_agentic task_spec requires a real capability dependency edge")
    goal = spec.get("goal_contract")
    if not isinstance(goal, Mapping):
        raise TaskSpecError("task_spec goal contract is missing")
    if environment.get("mode") == "stateful" and data_tables is not None:
        if goal.get("row_predicates"):
            validate_goal_contract(goal, data_tables)
        rows_text = json.dumps(data_tables, ensure_ascii=False)
        for delta in _objects(goal.get("expected_delta")):
            before = delta.get("before")
            if isinstance(before, str) and before and before not in rows_text:
                raise TaskSpecError(f"stateful initial fixture omits precondition value: {before}")
