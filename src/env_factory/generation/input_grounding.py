"""Reject reference queries that require guessing an undisclosed exact value."""
from __future__ import annotations

import json
import re


def validate_stateful_value_origins(source: dict) -> None:
    """Reject copied private numeric fixture values in successful writes/goals.

    This is an origin check, not a proof of arbitrary business arithmetic.
    Expressions are checked by execution and independent semantic review.
    """
    if source.get("environment_plan", {}).get("mode") != "stateful":
        return
    from decimal import Decimal
    def numbers(value):
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return {Decimal(str(value))}
        if isinstance(value, dict):
            return set().union(*(numbers(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(numbers(item) for item in value))
        return set()
    private = numbers([table.get("rows", []) for table in source.get("tables", [])])
    public_text = json.dumps(source["description"]["public_input"], ensure_ascii=False)
    public = {Decimal(value.replace(",", "")) for value in re.findall(
        r"(?<![\d.])-?\d+(?:,\d{3})*(?:\.\d+)?(?![\d.])", public_text)}
    words = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    public.update(Decimal(index) for index, word in enumerate(words)
                  if re.search(r"\b" + word + r"\b", public_text, re.I))
    def check(value, path, captures):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                if value["$ref"] not in captures:
                    raise ValueError(f"STATE_VALUE_UNGROUNDED: {path} uses an unavailable capture")
                return
            for key, item in value.items():
                check(item, f"{path}.{key}", captures)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                check(item, f"{path}[{index}]", captures)
        elif numbers(value) & (private - public):
            raise ValueError(f"STATE_VALUE_UNGROUNDED: {path} copies a private numeric fixture value; "
                "derive the write from preceding tool captures ($ref/$expr) and use "
                "semantic_goal value_expressions for a data-dependent target")
    implementations = {item["tool_name"]: item for item in source["tool_implementations"]}
    for scenario in source["scenarios"]:
        if scenario.get("kind") != "goal_success":
            continue
        captures = set()
        for step in scenario["steps"]:
            if step.get("operation") != "tool_call":
                continue
            if implementations[step["tool_name"]].get("operation") in {"insert", "update", "delete"}:
                check(step.get("arguments", {}), step["tool_name"], captures)
            captures.update(step.get("capture", {}))
    for index, predicate in enumerate(source.get("semantic_goal", {}).get("row_predicates", [])):
        check(predicate.get("values", {}), f"semantic_goal.row_predicates[{index}].values", set())


def validate_query_inputs(source: dict) -> list[dict]:
    public = source["description"]["public_input"]
    public_text = json.dumps(public, ensure_ascii=False)
    tools = {item["function"]["name"]: item["function"] for item in source["tools"]}
    implementations = {item["tool_name"]: item for item in source["tool_implementations"]}
    evidence = []
    for scenario in source["scenarios"]:
        if scenario.get("kind") != "goal_success":
            continue
        captures = set()
        for step in scenario["steps"]:
            if step.get("operation") != "tool_call":
                continue
            name = step["tool_name"]
            impl, tool = implementations[name], tools[name]
            for rule in impl.get("filters", []) if impl.get("operation") == "select" else []:
                argument = rule["argument"]
                if rule.get("operator") != "eq" or argument not in step.get("arguments", {}):
                    continue
                value = step["arguments"][argument]
                if isinstance(value, dict) and set(value) == {"$ref"}:
                    if value["$ref"] not in captures:
                        raise ValueError(f"QUERY_INPUT_UNDISCOVERABLE: {name}.{argument} uses an unavailable capture")
                    origin = "previous_tool_capture"
                elif isinstance(value, str) and value:
                    parameter = tool.get("parameters", {}).get("properties", {}).get(argument, {})
                    visible = public_text + " " + tool.get("description", "") + " " + json.dumps(parameter, ensure_ascii=False)
                    # Match whole ASCII tokens: ID 'A1' is not disclosed by A10.
                    pattern = r"(?<!\w)" + re.escape(value) + r"(?!\w)" if value.isascii() else re.escape(value)
                    variants = source.get('interaction_contract', {}).get('variants', [])
                    disclosed = bool(variants) and all(any(
                        stage.get('private_fact') == value and stage.get('before_tool') == name
                        for stage in variant.get('stages', [])) for variant in variants)
                    if not re.search(pattern, visible) and not disclosed:
                        raise ValueError(f"QUERY_INPUT_UNDISCOVERABLE: {name}.{argument} requires exact value {value!r}, "
                            "which is absent from public input and its tool parameter contract. "
                            "Publish a meaningful domain enum, obtain the value via a preceding tool capture, "
                            "or use a broader query with observable selection. Do not publish private answer IDs.")
                    origin = "required_user_disclosure" if disclosed else "public_input_or_tool_contract"
                else:
                    # Numeric bounds may be calculated or written as words; this
                    # check does not pretend to prove arbitrary arithmetic inputs.
                    continue
                evidence.append({"scenario": scenario.get("scenario_id"), "tool": name,
                                 "argument": argument, "origin": origin})
            captures.update(step.get("capture", {}))
    return evidence
