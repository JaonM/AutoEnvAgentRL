"""Lower reward facts through the authored query chain instead of fixture IDs."""
from __future__ import annotations

import copy
import json
import re

from env_factory.contracts.trajectory_probes import idempotent_call_groups


def reward_temporal_diagnostics(source: dict) -> list[dict]:
    """Flag live reward queries over mutable tables, without rejecting valid goals.

    Run after binding: inherited filters and upstream capture lookups matter as
    much as the returned field. These are review obligations, not proof of a bug.
    """
    mutable = {}
    for impl in source.get("tool_implementations", []):
        if impl.get("operation") in {"insert", "update", "delete"}:
            mutable.setdefault(impl["table"], []).append(impl["tool_name"])
    for event in source.get("business_lifecycle", {}).get("dynamics", {}).get("events", []):
        for change in event.get("changes", []):
            mutable.setdefault(change["table"], []).append("event:" + event["id"])
    findings = []

    def walk(value, path, initial=False):
        if isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]", initial)
        elif isinstance(value, dict) and set(value) != {"literal"}:
            for kind in ("lookup", "aggregate"):
                query = value.get(kind)
                if isinstance(query, dict) and query.get("table") in mutable and not initial:
                    findings.append({
                        "code": "REWARD_CURRENT_MUTABLE_QUERY", "path": path,
                        "table": query["table"], "query": copy.deepcopy(query),
                        "mutators": sorted(set(mutable[query["table"]])),
                        "message": "This reward query reads current state, including inherited filters; "
                            "it is not a saved tool response. Check whether writes invalidate selection "
                            "or change a decision condition. Use initial only for episode-start facts; "
                            "current-state outcome checks should remain current. External events may "
                            "make an observed pre-write fact differ from the initial snapshot.",
                    })
            for key, child in value.items():
                walk(child, f"{path}.{key}", initial or key in {"initial", "at_tool"})

    walk(source.get("metric_implementations", []), "metric_implementations")
    walk(source.get("semantic_goal", {}), "semantic_goal")
    return findings


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
            # This pass audits visibility; query lowering belongs to
            # bind_reward_queries, where the reference bindings are available.
            for predicate in rule["expected"]:
                walk(predicate.get("value_expressions", {}))
        if rule.get("operator") in {"value_targets", "numeric_targets"}:
            walk(rule["expected"])


def bind_reward_queries(source: dict, *, reference_details: list | None = None) -> dict:
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
    reference_bindings = []
    for scenario in successes:
        # Captures belong to one execution. A later reference must never inherit
        # a value or an ambiguity from an earlier reference.
        bindings, captures = {}, {}
        steps = scenario["steps"]
        group_for_step = {position: group for group, positions in enumerate(
            idempotent_call_groups(steps)) for position in positions}
        from ..contracts.trajectory_probes import optional_observation_positions
        optional = optional_observation_positions(steps, source['tool_implementations'])
        binding_groups = {}
        for position, step in enumerate(steps):
            if position in optional:
                continue
            if step.get("operation") != "tool_call":
                continue
            name = step["tool_name"]
            if name in bindings and (bindings[name] is None or
                    binding_groups[name] != group_for_step[position]):
                # Separate logical requests remain ambiguous; cached retries do not.
                bindings[name] = None
                for alias in step.get("capture", {}):
                    captures.pop(alias, None)
                continue
            impl = implementations[name]
            if impl["operation"] != "select":
                continue
            binding_groups[name] = group_for_step[position]
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
            from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, SandboxError
            for alias, path in step.get("capture", {}).items():
                captures.pop(alias, None)
                try:
                    tokens = DeclarativeMetricEvaluator.path_tokens(path)
                except SandboxError:
                    continue
                if (len(tokens) != 3 or tokens[0] != ('field', impl.get('result_field', 'records'))
                        or tokens[2][0] != 'field'
                        or not (tokens[1] == ('index', 0) or tokens[1][0] == 'filter')):
                    continue
                aliases = bindings[name]['aliases']
                projection = set(bindings[name]['projection'])
                def visible(field):
                    return not projection or field in projection or field in aliases
                if not visible(tokens[2][1]):
                    continue
                field = aliases.get(tokens[2][1], tokens[2][1])
                selected_where = copy.deepcopy(where)
                if tokens[1][0] == 'filter':
                    selector, expected = tokens[1][1]
                    if not visible(selector):
                        continue
                    column = aliases.get(selector, selector)
                    predicates = selected_where.setdefault(column, {})
                    if ('eq' in predicates and json.dumps(predicates['eq'], sort_keys=True) !=
                            json.dumps(expected, sort_keys=True)):
                        continue
                    predicates['eq'] = copy.deepcopy(expected)
                # Lookup retains the producer's entire query scope and resolves
                # only one matching row, just like the runtime equality capture.
                captures[alias] = {'lookup': {'table': impl['table'], 'field': field,
                                              'where': selected_where}}
        reference_bindings.append(bindings)
        if reference_details is not None:
            reference_details.append({'scenario_id': scenario.get('scenario_id'),
                'called_tools': sorted({step['tool_name'] for step in steps if step.get('operation') == 'tool_call'}),
                'bindings': copy.deepcopy(bindings)})

    # from_tool describes a shared query, without a reference selector. It can
    # therefore be lowered only when every applicable successful execution
    # agrees on the entire query and its transitive capture chain. A declared
    # user-refusal route may omit an inapplicable query, but a missing required
    # call or an unresolved capture is still an error. Route validity and actual
    # refusal evidence are checked by the compiler and execution preflight.
    routes = {route['id']: route for route in source.get('execution_routes', [])}

    def inactive_refusal_query(scenario, name):
        route = routes.get(scenario.get('execution_route'), {})
        return (set(route.get('when', {})) == {'user_rejection'}
                and name not in route.get('required_tools', []))

    bindings = {}
    if reference_bindings:
        for name in sorted({name for reference in reference_bindings for name in reference}):
            applicable = [reference.get(name) for scenario, reference in zip(successes, reference_bindings)
                          if name in reference or not inactive_refusal_query(scenario, name)]
            binding = applicable[0]
            if binding is None:
                continue
            encoded = json.dumps(binding, sort_keys=True, allow_nan=False)
            if all(json.dumps(other, sort_keys=True, allow_nan=False) == encoded
                   for other in applicable[1:]):
                bindings[name] = binding

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
    for route in source.get("execution_routes", []):
        route["when"] = lower(route["when"])
    for implementation in source.get('tool_implementations', []):
        if 'derived_changes' in implementation:
            implementation['derived_changes'] = lower(implementation['derived_changes'])
    for policy in source.get('business_lifecycle', {}).get('tools', {}).values():
        for guard in policy.get('guards', []):
            for operand in list(guard.get('selector', {}).values()) + [
                    value for predicates in guard.get('conditions', {}).values() for value in predicates.values()]:
                if isinstance(operand, dict) and set(operand) == {'expression'}:
                    operand['expression'] = lower(operand['expression'])
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
            from .interaction_contract import stage_bindings
            for binding in stage_bindings(stage):
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
