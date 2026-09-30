"""Bind authored answer schemas to public instructions and executable rewards."""
from __future__ import annotations

import copy
import json

from env_factory.sandbox_runtime import SandboxError, validate_json_schema


def validate_direct_text_targets(source: dict) -> None:
    """Require public evidence for exact literal text, before publishing schemas.

    This is a necessary origin check, not a semantic equivalence evaluator.
    Computed expressions and public categorical choices retain their own checks.
    """
    if source.get("environment_plan", {}).get("mode") != "stateless":
        return
    public = source["description"]["public_input"]
    def texts(value):
        if isinstance(value, str):
            return [" ".join(value.split())]
        if isinstance(value, dict):
            return [text for child in value.values() for text in texts(child)]
        if isinstance(value, list):
            return [text for child in value for text in texts(child)]
        return []
    evidence = texts(public.get("initial_user_message", "")) + texts(public.get("materials", []))
    properties = source.get("answer_contract", {}).get("schema", {}).get("properties", {})
    def check(expression, schema, path):
        if not isinstance(expression, dict):
            return
        if set(expression) == {"literal"} and isinstance(expression["literal"], str):
            if schema.get("type") not in {None, "string"}:
                return  # schema/reference validation owns type mismatches
            value = " ".join(expression["literal"].split())
            choices = schema.get("enum", []) if isinstance(schema, dict) else []
            if value and not any(value in text for text in evidence) and expression["literal"] not in choices:
                raise ValueError(f"DIRECT_EXACT_TEXT_UNDISCLOSED: {path} requires private literal text "
                    "not present in public evidence or a public categorical enum. "
                    "Use a publicly identifiable verbatim fact, or express the actual transformation; "
                    "do not invent a private canonical wording for an open-ended summary.")
        elif set(expression) == {"literal"} and isinstance(expression["literal"], list):
            for index, item in enumerate(expression["literal"]):
                check({"literal": item}, schema.get("items", {}), f"{path}[{index}]")
        elif set(expression) == {"array"}:
            for index, item in enumerate(expression["array"]):
                check(item, schema.get("items", {}), f"{path}[{index}]")
    for rule in source["metric_implementations"]:
        if rule.get("source") == "final_agent_response" and rule.get("operator") == "value_targets":
            for target in rule["expected"]["targets"]:
                check(target["expression"], properties.get(target["key"], {}), target["key"])


def _finite_labels(expression: object, depth: int = 0) -> set[str] | None:
    """Expose all literal decision alternatives, never the selected gold value."""
    if not isinstance(expression, dict) or depth > 12:
        return None
    if set(expression) == {"literal"} and isinstance(expression["literal"], str):
        return {expression["literal"]}
    if set(expression) == {"if"} and isinstance(expression["if"], dict):
        branch = expression["if"]
        yes = _finite_labels(branch.get("then"), depth + 1)
        no = _finite_labels(branch.get("else"), depth + 1)
        return yes | no if yes is not None and no is not None else None
    return None


def _schema(schema: object, depth: int = 0) -> None:
    if not isinstance(schema, dict) or depth > 8:
        raise ValueError("ANSWER_CONTRACT_INVALID: schema must be a bounded object")
    kind = schema.get("type")
    allowed = {"type", "description", "enum"}
    if "enum" in schema:
        values = schema["enum"]
        if not isinstance(values, list) or not 2 <= len(values) <= 64:
            raise ValueError("ANSWER_CONTRACT_INVALID: public enums require 2..64 choices, not a gold answer")
        if len({json.dumps(value, sort_keys=True) for value in values}) != len(values):
            raise ValueError("ANSWER_CONTRACT_INVALID: enum choices must be distinct")
    if kind == "object":
        allowed |= {"properties", "required", "additionalProperties"}
        properties = schema.get("properties")
        if (not isinstance(properties, dict) or not 1 <= len(properties) <= 64
                or schema.get("additionalProperties") is not False
                or not isinstance(schema.get("required"), list)
                or set(schema["required"]) != set(properties)
                or len(schema["required"]) != len(properties)):
            raise ValueError("ANSWER_CONTRACT_INVALID: objects require exactly their declared properties")
        for key, child in properties.items():
            if not isinstance(key, str) or not key:
                raise ValueError("ANSWER_CONTRACT_INVALID: property names must be nonempty strings")
            _schema(child, depth + 1)
    elif kind == "array":
        allowed |= {"items", "minItems", "maxItems"}
        _schema(schema.get("items"), depth + 1)
        for key in ("minItems", "maxItems"):
            if key in schema and (type(schema[key]) is not int or not 0 <= schema[key] <= 64):
                raise ValueError("ANSWER_CONTRACT_INVALID: array bounds must be integers within 0..64")
        if schema.get("minItems", 0) > schema.get("maxItems", 64):
            raise ValueError("ANSWER_CONTRACT_INVALID: invalid array bounds")
    elif kind not in {"string", "boolean", "integer", "number"}:
        raise ValueError("ANSWER_CONTRACT_INVALID: unsupported answer type")
    if set(schema) - allowed:
        raise ValueError("ANSWER_CONTRACT_INVALID: unsupported schema keywords: " +
                         ", ".join(sorted(set(schema) - allowed)))
    if "description" in schema and not isinstance(schema["description"], str):
        raise ValueError("ANSWER_CONTRACT_INVALID: description must be text")
    for value in schema.get("enum", []):
        validate_json_schema({key: child for key, child in schema.items() if key != "enum"}, value, "answer_enum")


def bind_answer_contract(source: dict) -> dict | None:
    """Reject hidden JSON formats and render one public, validated contract.

    Schemas describe shape, never contain reference values or private selectors.
    Semantic fidelity of business facts is still checked by actual execution and
    independent review; JSON shape checks do not replace either.
    """
    rules = [rule for rule in source["metric_implementations"]
             if rule.get("source") == "final_agent_response"]
    structured = [rule for rule in rules if rule.get("operator") == "value_targets"]
    semantic = [metric for metric in source.get("metrics", []) if metric.get("semantic_fields")]
    for metric in semantic:
        if (metric.get("type") != "model-based" or metric.get("category") != "outcome"
                or metric.get("evaluator", {}).get("kind") != "external_llm_judge"):
            raise ValueError("ANSWER_CONTRACT_SEMANTIC: semantic fields require a platform model outcome")
    contract = source.get("answer_contract")
    if not structured and not semantic:
        if contract is not None:
            raise ValueError("ANSWER_CONTRACT_INVALID: JSON contract requires value_targets rewards")
        numeric = [rule for rule in rules if rule.get("operator") == "numeric_targets"]
        if numeric:
            if (len(numeric) != 1 or len(rules) != 1 or numeric[0].get("path") != "$"
                    or numeric[0].get("expected", {}).get("answer_format") != "single_labeled_number"
                    or len(numeric[0]["expected"].get("targets", [])) != 1):
                raise ValueError("ANSWER_CONTRACT_NUMERIC: use a single_labeled_number target, "
                                 "or value_targets with a public JSON schema for multiple fields")
            target = numeric[0]["expected"]["targets"][0]
            contract = {"format": "single_labeled_number", "label": target["label"], "unit": target["unit"]}
            public = source["description"]["public_input"]
            if "answer_contract" in public:
                raise ValueError("ANSWER_CONTRACT_INVALID: numeric format is derived by the compiler")
            public["answer_contract"] = contract
            public["initial_user_message"] = public["initial_user_message"].rstrip() + (
                "\n\nRequired answer format: exactly " + target["label"] + ": <number>" + target["unit"]
                + ". Replace <number> with the calculated value; include the exact label and unit, "
                  "with no additional prose or numbers."
            )
            return copy.deepcopy(contract)
        return None
    if (not isinstance(contract, dict) or set(contract) != {"format", "schema"}
            or contract.get("format") != "json_object"):
        raise ValueError("ANSWER_CONTRACT_REQUIRED: value_targets require answer_contract with "
                         "format=json_object and a public schema; do not hide the output format")
    schema = contract["schema"]
    _schema(schema)
    if schema.get("type") != "object":
        raise ValueError("ANSWER_CONTRACT_INVALID: top-level answer must be an object")
    if len(structured) != len(rules) or any(rule.get("path") != "$" for rule in structured):
        raise ValueError("ANSWER_CONTRACT_INVALID: use value_targets at $ for every JSON answer metric")
    keys = [target["key"] for rule in structured for target in rule["expected"]["targets"]]
    for metric in semantic:
        fields = metric["semantic_fields"]
        if not isinstance(fields, list) or not fields or any(not isinstance(key, str) for key in fields):
            raise ValueError("ANSWER_CONTRACT_SEMANTIC: invalid semantic fields")
        for key in fields:
            field = schema["properties"].get(key, {})
            if field.get("type") != "string" or "enum" in field:
                raise ValueError("ANSWER_CONTRACT_SEMANTIC: semantic fields must be open strings")
        keys.extend(fields)
    if len(keys) != len(set(keys)) or set(keys) != set(schema["properties"]):
        raise ValueError("ANSWER_CONTRACT_COVERAGE: public fields must each have exactly one reward target")
    for rule in structured:
        for target in rule["expected"]["targets"]:
            labels = _finite_labels(target["expression"])
            field = schema["properties"][target["key"]]
            if labels and len(labels) > 1:
                if field.get("type") != "string":
                    raise ValueError("ANSWER_CONTRACT_TYPES: decision labels require a string field")
                if "enum" in field and not labels <= set(field["enum"]):
                    raise ValueError("ANSWER_CONTRACT_ENUM: public enum omits a reachable decision label")
                field.setdefault("enum", sorted(labels))
    _schema(schema)
    if semantic:
        for rule in structured:
            rule["expected"]["answer_schema"] = copy.deepcopy(schema)
    for scenario in source["scenarios"]:
        if scenario.get("kind") != "goal_success":
            continue
        answers = [step.get("content") for step in scenario["steps"]
                   if step.get("operation") == "agent_response"]
        if not answers:
            raise ValueError("ANSWER_CONTRACT_REFERENCE: successful scenario requires a final answer")
        try:
            answer = json.loads(answers[-1])
            validate_json_schema(schema, answer, "reference_answer")
        except (ValueError, TypeError, SandboxError) as exc:
            raise ValueError(f"ANSWER_CONTRACT_REFERENCE: {exc}") from exc
    public = source["description"]["public_input"]
    if "answer_contract" in public:
        raise ValueError("ANSWER_CONTRACT_INVALID: define answer_contract once at source root")
    public["answer_contract"] = copy.deepcopy(contract)
    public["initial_user_message"] = public["initial_user_message"].rstrip() + (
        "\n\nRequired answer format: return only a JSON object matching this schema. "
        "Use exactly the declared keys and JSON types; no Markdown fences or extra prose. "
        "Copy requested textual facts accurately from the supplied evidence; preserve requested array order.\n"
        + json.dumps(schema, ensure_ascii=False, sort_keys=True)
    )
    if semantic:
        public_criteria = [{"fields": metric["semantic_fields"], "rubric": metric["rubric"],
                            "criteria": metric["criteria"]} for metric in semantic]
        public["semantic_answer_criteria"] = copy.deepcopy(public_criteria)
        public["initial_user_message"] += (
            "\n\nFor the following explanation fields, faithful paraphrases are accepted. "
            "Their meaning is evaluated against these public criteria, not exact wording:\n"
            + json.dumps(public_criteria, ensure_ascii=False, sort_keys=True))
    return copy.deepcopy(contract)
