"""Lower reward facts through the authored query chain instead of fixture IDs."""
from __future__ import annotations

import copy
import json
import re


def validate_reward_observability(source: dict) -> None:
    """A private answer fact must be exposed by a tool in the reference flow.

    Run before lowering from_tool: compiler-owned internal resolver lookups are
    not facts the policy must compute itself. Raw authored lookups do not get
    that exemption. This is a necessary field-visibility check; it does not
    prove that a particular row is reachable or that business prose is sound.
    """
    if source["environment_plan"]["mode"] != "reference_data":
        return
    called = {step["tool_name"] for scenario in source["scenarios"]
              if scenario.get("kind") == "goal_success" for step in scenario["steps"]
              if step.get("operation") == "tool_call"}
    columns = {table["table_name"]: {item["name"] for item in table["columns"]}
               for table in source["tables"]}
    visible = {}
    counted = set()
    for impl in source["tool_implementations"]:
        if impl["tool_name"] not in called:
            continue
        if impl.get("operation") == "aggregate_count":
            counted.add(impl["table"])
        if impl.get("operation") != "select":
            continue
        table = impl["table"]
        # Projection aliases add named output fields; they do not conceal the
        # underlying field that is actually returned to the policy.
        fields = set(impl.get("projection", []))
        if not fields and not impl.get("projection_aliases"):
            fields = columns.get(table, set()).copy()
        fields.update(impl.get("projection_aliases", {}).values())
        visible.setdefault(table, set()).update(fields)

    def walk(expression):
        if isinstance(expression, list):
            for child in expression:
                walk(child)
        if not isinstance(expression, dict) or set(expression) == {"literal"}:
            return
        query = expression.get("lookup")
        if isinstance(query, dict) and query.get("field") not in visible.get(query.get("table"), set()):
            raise ValueError(f"REWARD_FACT_UNOBSERVABLE: {query.get('table')}.{query.get('field')} "
                "is read by the reward but not returned by any reference tool. Expose the required "
                "business fact through the actual tool interface and bind it with from_tool; "
                "the private reference answer is not evidence available to the policy.")
        aggregate = expression.get("aggregate")
        if isinstance(aggregate, dict):
            table = aggregate.get("table")
            fields = aggregate.get("fields", [aggregate.get("field")])
            exposed = (table in visible or table in counted) if aggregate.get("op") == "count" else (
                isinstance(fields, list) and set(fields) <= visible.get(table, set()))
            if not exposed:
                raise ValueError(f"REWARD_FACT_UNOBSERVABLE: aggregate over {table} uses private facts "
                                 "not returned by reference tools")
        for child in expression.values():
            walk(child)

    for rule in source["metric_implementations"]:
        if rule.get("operator") == "state_predicates":
            rule["expected"] = lower(rule["expected"])
        if rule.get("operator") in {"value_targets", "numeric_targets"}:
            walk(rule["expected"])


def bind_reward_queries(source: dict) -> dict:
    implementations = {item["tool_name"]: item for item in source["tool_implementations"]}
    successes = [item for item in source["scenarios"] if item.get("kind") == "goal_success"]
    bindings, captures = {}, {}

    def lower_argument_expression(expression, depth=0):
        # Match the reference interpreter grammar; retain live data lookups.
        if depth > 12:
            raise ValueError("REWARD_QUERY_BINDING: capture expression is too deep")
        if not isinstance(expression, dict):
            return {"literal": copy.deepcopy(expression)}
        if set(expression) == {"$ref"}:
            reference = str(expression["$ref"])
            if reference not in captures:
                raise ValueError("REWARD_QUERY_BINDING: unavailable capture in expression: " + reference)
            return copy.deepcopy(captures[reference])
        if set(expression) == {"literal"}:
            return copy.deepcopy(expression)
        if set(expression) == {"op", "args"} and isinstance(expression["args"], list):
            return {"op": expression["op"], "args": [lower_argument_expression(arg, depth + 1)
                for arg in expression["args"]]}
        if set(expression) == {"if"} and isinstance(expression["if"], dict):
            return {"if": {key: lower_argument_expression(arg, depth + 1)
                for key, arg in expression["if"].items()}}
        raise ValueError("REWARD_QUERY_BINDING: capture expressions require captures, literals and finite operators")
    if len(successes) == 1:
        for step in successes[0]["steps"]:
            if step.get("operation") != "tool_call":
                continue
            name = step["tool_name"]
            if name in bindings:
                # A tool name is not an unambiguous reference to repeated calls.
                bindings[name] = None
                for alias in step.get("capture", {}):
                    captures.pop(alias, None)
                continue
            impl = implementations[name]
            if impl["operation"] != "select":
                continue
            where = {}
            unbound = False
            for rule in impl.get("filters", []):
                argument = rule["argument"]
                if argument not in step.get("arguments", {}):
                    continue
                value = copy.deepcopy(step["arguments"][argument])
                if isinstance(value, dict) and set(value) == {"$ref"}:
                    if value["$ref"] not in captures:
                        unbound = True
                        break
                    value = copy.deepcopy(captures[value["$ref"]])
                elif isinstance(value, dict) and set(value) == {"$expr"}:
                    value = lower_argument_expression(value["$expr"])
                if rule.get("resolve"):
                    resolver = rule["resolve"]
                    value = {"lookup": {"table": resolver["table"], "field": resolver["value_column"],
                                        "where": {resolver["match_column"]: value}}}
                predicates = where.setdefault(rule["column"], {})
                operator = rule["operator"]
                if operator in predicates and predicates[operator] != value:
                    raise ValueError("REWARD_QUERY_BINDING: conflicting predicates on the same query field")
                predicates[operator] = value
            if unbound:
                bindings[name] = None
                continue
            bindings[name] = {"table": impl["table"], "where": where,
                              "projection": impl.get("projection", []),
                              "aliases": impl.get("projection_aliases", {})}
            result_field = re.escape(impl.get("result_field", "records"))
            for alias, path in step.get("capture", {}).items():
                match = re.fullmatch(r"(?:\$\.)?" + result_field + r"\[0\]\.([A-Za-z_][A-Za-z_0-9]*)", path)
                if match:
                    field = bindings[name]["aliases"].get(match[1], match[1])
                    captures[alias] = {"lookup": {"table": impl["table"], "field": field,
                                                "where": copy.deepcopy(where)}}

    def lower(value):
        if isinstance(value, list):
            return [lower(item) for item in value]
        if not isinstance(value, dict) or set(value) == {"literal"}:
            return copy.deepcopy(value)
        if set(value) == {"from_tool"}:
            ref = value["from_tool"]
            if (not isinstance(ref, dict) or set(ref) - {"name", "field", "where"}
                    or not isinstance(ref.get("name"), str) or not isinstance(ref.get("field"), str)):
                raise ValueError("REWARD_QUERY_BINDING: from_tool requires name, field, and optional where")
            binding = bindings.get(ref["name"])
            if binding is None:
                raise ValueError("REWARD_QUERY_BINDING: from_tool requires one unambiguous select call: " + ref["name"])
            projection = set(binding["projection"])
            aliases = binding["aliases"]
            field = aliases.get(ref["field"], ref["field"])
            if projection and field not in projection and ref["field"] not in aliases:
                raise ValueError("REWARD_QUERY_BINDING: reward field was not returned by the tool: " + ref["field"])
            where = copy.deepcopy(binding["where"])
            extra = ref.get("where", {})
            if not isinstance(extra, dict):
                raise ValueError("REWARD_QUERY_BINDING: where must contain business selection predicates")
            for key, operand in extra.items():
                column = aliases.get(key, key)
                if projection and column not in projection and key not in aliases:
                    raise ValueError("REWARD_QUERY_BINDING: selection field was not returned by the tool: " + key)
                operand = lower(operand)
                predicates = operand if isinstance(operand, dict) and operand and set(operand) <= {
                    "eq", "in", "contains", "gte", "lte"} else {"eq": operand}
                existing = where.setdefault(column, {})
                for operator, expected in predicates.items():
                    if operator in existing and existing[operator] != expected:
                        raise ValueError("REWARD_QUERY_BINDING: additional selection contradicts the tool query")
                    existing[operator] = expected
            return {"lookup": {"table": binding["table"], "field": field, "where": where}}
        return {key: lower(child) for key, child in value.items()}

    for rule in source["metric_implementations"]:
        if rule.get("operator") == "state_predicates":
            rule["expected"] = lower(rule["expected"])
        if rule.get("operator") in {"value_targets", "numeric_targets"}:
            for target in rule["expected"]["targets"]:
                target["expression"] = lower(target["expression"])
    for predicate in source.get("semantic_goal", {}).get("row_predicates", []):
        if "value_expressions" in predicate:
            predicate["value_expressions"] = {field: lower(expression)
                for field, expression in predicate["value_expressions"].items()}
    return {name: binding for name, binding in bindings.items() if binding is not None}


def reject_private_identifier_constants(source: dict, *, validated_scripts=(), public_input=None) -> None:
    """Private relational identifiers must come from a live goal-linked lookup."""
    if source["environment_plan"]["mode"] != "reference_data":
        return
    public = source["description"]["public_input"] if public_input is None else public_input
    prose = source["description"]["task"] + " " + public["initial_user_message"] + " " + json.dumps(
        public.get("materials", []), ensure_ascii=False)
    identifiers = {
        table["table_name"]: set(table["primary_key"]) | {
            fk["column"] for fk in table.get("foreign_keys", [])}
        for table in source["tables"]
    }
    # Only compiler-validated, mandatory disclosures can establish this origin.
    # Scope it to the exact queried table/column/value, across every variant.
    implementations = {item["tool_name"]: item for item in source.get("tool_implementations", [])}
    per_script = []
    for script in validated_scripts:
        origins = set()
        for stage in script["interaction_protocol"]["stages"]:
            binding = stage.get("bind_to", {})
            impl = implementations.get(binding.get("tool"), {})
            if impl.get("operation") != "select" or "private_fact" not in stage:
                continue
            for rule in impl.get("filters", []):
                if (rule.get("argument") == binding.get("argument")
                        and rule.get("operator") == "eq" and not rule.get("resolve")):
                    origins.add((impl["table"], rule["column"], json.dumps(stage["private_fact"], ensure_ascii=False)))
        per_script.append(origins)
    user_origins = set.intersection(*per_script) if per_script else set()

    def disclosed(value, field, table):
        if (table, field, json.dumps(value, ensure_ascii=False)) in user_origins:
            return True
        def material_contains(item):
            if isinstance(item, dict):
                return (field in item and type(item[field]) is type(value) and item[field] == value
                        or any(material_contains(child) for child in item.values()))
            return isinstance(item, list) and any(material_contains(child) for child in item)
        if material_contains(public.get("materials", [])):
            return True
        if not isinstance(value, str) or value.isdecimal():
            # A requested quantity of 1 does not disclose product_id=1.
            return bool(re.search(r"(?:\b(?:id|identifier|code)\b|编号|编码|号码|#|" + re.escape(field)
                + r")\s*[:=：#-]?\s*" + re.escape(str(value)) + r"(?!\w)", prose, re.IGNORECASE))
        if isinstance(value, str) and not value.isascii():
            return value in prose
        token = str(value)
        return bool(re.search(r"(?<!\w)" + re.escape(token) + r"(?!\w)", prose))

    def walk(expression):
        if isinstance(expression, list):
            for child in expression:
                walk(child)
        if not isinstance(expression, dict) or set(expression) == {"literal"}:
            return
        for kind in ("lookup", "aggregate"):
            query = expression.get(kind)
            if isinstance(query, dict):
                for key, selector in query.get("where", {}).items():
                    if key not in identifiers.get(query.get("table"), set()):
                        continue
                    values = selector.values() if isinstance(selector, dict) and set(selector) <= {
                        "eq", "in", "contains", "gte", "lte"} else [selector]
                    for value in values:
                        if isinstance(value, dict) and set(value) == {"literal"}:
                            value = value["literal"]
                        candidates = value if isinstance(value, list) else [value]
                        for candidate in candidates:
                            if not isinstance(candidate, dict) and not disclosed(candidate, key, query.get("table")):
                                raise ValueError(f"PRIVATE_REWARD_IDENTIFIER: {query['table']}.{key} pins a private "
                                    "identifier; use from_tool or a nested lookup from the public named target")
        for child in expression.values():
            walk(child)
    for rule in source["metric_implementations"]:
        if rule.get("operator") in {"numeric_targets", "value_targets"}:
            walk(rule["expected"])
