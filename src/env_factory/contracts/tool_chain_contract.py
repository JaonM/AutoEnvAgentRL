"""Check declared tool-output captures against downstream input schemas."""

from __future__ import annotations

from typing import Any

from ..sandbox_runtime import DeclarativeMetricEvaluator, SandboxError


def captured_schema(schema: Any, path: str) -> dict[str, Any] | None:
    if not isinstance(schema, dict):
        return None
    try:
        tokens = DeclarativeMetricEvaluator.path_tokens(path)
    except (SandboxError, ValueError, SyntaxError):
        return None
    current = schema
    for kind, value in tokens:
        if kind == "field":
            current = current.get("properties", {}).get(value)
        elif kind == "index":
            current = current.get("items")
        else:
            return None
        if not isinstance(current, dict):
            return None
    return current


def schema_conflict(source: Any, target: Any, path: str = "") -> str | None:
    """Report only shape conflicts guaranteed by declared schemas."""
    if not isinstance(source, dict) or not isinstance(target, dict):
        return None
    source_kind, target_kind = source.get("type"), target.get("type")
    if source_kind and target_kind and source_kind != target_kind:
        if {source_kind, target_kind} != {"integer", "number"}:
            return f"{path or '$'}: {source_kind} cannot satisfy {target_kind}"
    if source_kind == target_kind == "array":
        return schema_conflict(source.get("items"), target.get("items"), f"{path}[]")
    if source_kind == target_kind == "object":
        source_properties = source.get("properties", {})
        target_properties = target.get("properties", {})
        if not isinstance(source_properties, dict) or not isinstance(target_properties, dict):
            return None
        source_required = set(source.get("required", []))
        target_required = set(target.get("required", []))
        if target.get("additionalProperties") is False:
            extras = sorted(source_required - target_properties.keys())
            if extras:
                return f"{path or '$'}: required upstream fields are forbidden downstream: {extras}"
        if source.get("additionalProperties") is False:
            missing = sorted(target_required - source_properties.keys())
            if missing:
                return f"{path or '$'}: required downstream fields cannot appear upstream: {missing}"
        for name in sorted(source_required & target_required):
            conflict = schema_conflict(
                source_properties.get(name), target_properties.get(name),
                f"{path}.{name}" if path else name,
            )
            if conflict:
                return conflict
    return None


def tool_chain_issues(task: dict[str, Any]) -> list[dict[str, Any]]:
    task_spec = task.get("task_spec")
    task_spec = task_spec if isinstance(task_spec, dict) else {}
    tool_contracts = task_spec.get("tool_contracts")
    tool_contracts = tool_contracts if isinstance(tool_contracts, list) else []
    contracts = {
        item.get("name"): item for item in tool_contracts
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    issues: list[dict[str, Any]] = []
    acceptance = task.get("acceptance_contract")
    acceptance = acceptance if isinstance(acceptance, dict) else {}
    scenarios = acceptance.get("executable_scenarios")
    scenarios = scenarios if isinstance(scenarios, list) else []
    for scenario in scenarios:
        if not isinstance(scenario, dict) or scenario.get("kind") != "goal_success":
            continue
        captures: dict[str, tuple[str, dict[str, Any]]] = {}
        steps = scenario.get("steps")
        steps = steps if isinstance(steps, list) else []
        for step in steps:
            if not isinstance(step, dict) or step.get("operation") != "tool_call":
                continue
            tool_name = step.get("tool_name")
            target_schema = contracts.get(tool_name, {}).get("input_schema", {})

            def inspect(value: Any, schema: Any, argument_path: str) -> None:
                if isinstance(value, dict) and set(value) == {"$ref"}:
                    reference = value["$ref"]
                    captured = captures.get(reference) if isinstance(reference, str) else None
                    if captured:
                        producer, source_schema = captured
                        conflict = schema_conflict(source_schema, schema, argument_path)
                        if conflict:
                            issues.append({
                                "code": "TASK_TOOL_CHAIN_SCHEMA",
                                "owner": "task_generation",
                                "message": "captured tool output cannot satisfy downstream input schema",
                                "evidence": {
                                    "scenario_id": scenario.get("scenario_id"),
                                    "from_tool": producer,
                                    "to_tool": tool_name,
                                    "argument_path": argument_path,
                                    "conflict": conflict,
                                },
                            })
                    return
                if isinstance(value, dict) and isinstance(schema, dict):
                    properties = schema.get("properties", {})
                    if isinstance(properties, dict):
                        for name, child in value.items():
                            inspect(child, properties.get(name), f"{argument_path}.{name}")
                elif isinstance(value, list) and isinstance(schema, dict):
                    for index, child in enumerate(value):
                        inspect(child, schema.get("items"), f"{argument_path}[{index}]")

            inspect(step.get("arguments", {}), target_schema, "arguments")
            output_contract = contracts.get(tool_name, {}).get("output_contract")
            output_contract = output_contract if isinstance(output_contract, dict) else {}
            output_schema = output_contract.get("schema")
            capture = step.get("capture")
            capture = capture if isinstance(capture, dict) else {}
            for variable, path in capture.items():
                if isinstance(variable, str) and isinstance(path, str):
                    source_schema = captured_schema(output_schema, path)
                    if source_schema is not None:
                        captures[variable] = (tool_name, source_schema)
                    elif isinstance(output_schema, dict):
                        issues.append({
                            "code": "TASK_TOOL_CAPTURE_PATH",
                            "owner": "task_generation",
                            "message": "capture path is absent from the declared upstream output schema",
                            "evidence": {
                                "scenario_id": scenario.get("scenario_id"),
                                "from_tool": tool_name,
                                "variable": variable,
                                "capture_path": path,
                            },
                        })
    return issues
