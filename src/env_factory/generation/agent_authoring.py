"""Compile agent-authored business contracts; no catalog of task prototypes.

The agent owns the source JSON. This module owns materialization and executable
preflight, and is rerun by the parent outside the agent's editable workspace.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path
import tempfile
import subprocess
import sys

from env_factory.generation.contract_compiler import ContractCompiler as P
from env_factory.tasks.task_routing import training_contract
from env_factory.tasks.task_spec import compile_task_spec
from env_factory.generation.answer_contract import bind_answer_contract, validate_direct_text_targets
from env_factory.generation.reward_bindings import (bind_reward_queries, reject_private_identifier_constants,
    validate_reward_observability, reward_temporal_diagnostics)
from env_factory.generation.input_grounding import validate_query_inputs, validate_stateful_value_origins


def normalize_expression(value):
    """Compact syntax permits scalar literals; preserve all business meaning."""
    if isinstance(value, (str, bool, int, float)):
        return {"literal": value}
    if not isinstance(value, dict):
        return value
    result = copy.deepcopy(value)
    if set(result) == {"op", "args"} and isinstance(result["args"], list):
        result["args"] = [normalize_expression(arg) for arg in result["args"]]
    elif set(result) == {"initial"}:
        result["initial"] = normalize_expression(result["initial"])
    elif set(result) == {"at_tool"} and isinstance(result["at_tool"], dict):
        result["at_tool"]["expression"] = normalize_expression(result["at_tool"].get("expression"))
    elif set(result) == {"array"} and isinstance(result["array"], list):
        result["array"] = [normalize_expression(arg) for arg in result["array"]]
    elif set(result) == {"if"} and isinstance(result["if"], dict):
        result["if"] = {key: normalize_expression(child) for key, child in result["if"].items()}
    elif set(result) in ({"lookup"}, {"aggregate"}):
        operation = next(iter(result.values()))
        if isinstance(operation, dict) and isinstance(operation.get("where"), dict):
            operation["where"] = {key: normalize_expression(child) if isinstance(child, dict) else child
                                  for key, child in operation["where"].items()}
    return result


def expand_capture_fields(calls: list[dict]) -> None:
    """Lower object-field references into existing scalar capture contracts.

    This changes syntax only: paths still originate in an earlier actual tool
    result, and all dependency/process checks see the explicit scalar binding.
    """
    occupied = {name for call in calls for name in call.get("capture", {})}
    producers, generated = {}, {}

    def lower(value):
        if isinstance(value, list):
            return [lower(item) for item in value]
        if not isinstance(value, dict):
            return value
        if set(value) != {"$ref"}:
            return {key: lower(item) for key, item in value.items()}
        reference = value["$ref"]
        if not isinstance(reference, str) or reference in producers:
            return value
        candidates = [name for name in producers if reference.startswith(name + ".") or reference.startswith(name + "[")]
        if not candidates:
            return value  # existing validation diagnoses an unavailable capture
        base = max(candidates, key=len)
        suffix = reference[len(base):]
        if not re.fullmatch(r"(?:\.[A-Za-z_][A-Za-z_0-9]*|\[[0-9]+\])+", suffix):
            raise ValueError("CAPTURE_FIELD_INVALID: use object fields or nonnegative array indices")
        producer, path = producers[base]
        identity = (id(producer), reference)
        if identity not in generated:
            index = len(generated)
            alias = f"compiled_capture_field_{index}"
            while alias in occupied:
                index += 1
                alias = f"compiled_capture_field_{index}"
            occupied.add(alias)
            producer.setdefault("capture", {})[alias] = path + suffix
            generated[identity] = alias
        return {"$ref": generated[identity]}

    for call in calls:
        call["arguments"] = lower(call.get("arguments", {}))
        for alias, path in call.get("capture", {}).items():
            producers[alias] = (call, path)


def validate_capture_paths(scenarios: list[dict], tool_contracts: list[dict]) -> None:
    """Reject paths contradicted by a declared result shape; unknown shapes stay executable."""
    schemas = {tool["name"]: tool.get("output_contract", {}).get("schema", {})
               for tool in tool_contracts}
    for scenario in scenarios:
        for step in scenario.get("steps", []):
            if step.get("operation") != "tool_call":
                continue
            for alias, path in step.get("capture", {}).items():
                if not isinstance(path, str):
                    continue
                tail = path[1:] if path.startswith("$") else "." + path
                tokens = re.findall(r"\.([A-Za-z_][A-Za-z_0-9]*)|\[([0-9]+)\]", tail)
                if "".join("." + field if field else "[" + index + "]" for field, index in tokens) != tail:
                    continue
                schema = schemas.get(step.get("tool_name"), {})
                for field, index in tokens:
                    kind = schema.get("type")
                    mismatch = ((bool(field) and kind == "array") or
                                (bool(index) and kind == "object"))
                    if mismatch:
                        raise ValueError(f"CAPTURE_PATH_INVALID: tool={step.get('tool_name')} capture={alias} path={path}; cannot access {field or '[' + index + ']'} on {kind}")
                    if field:
                        properties = schema.get("properties", {})
                        if field not in properties:
                            if (schema.get("additionalProperties") is False
                                    and not schema.get("patternProperties")
                                    and not any(key in schema for key in ("allOf", "anyOf", "oneOf", "$ref"))):
                                raise ValueError(
                                    f"CAPTURE_PATH_INVALID: tool={step.get('tool_name')} capture={alias} path={path}; "
                                    f"field {field} is absent from closed output schema; available fields={sorted(properties)}")
                            break
                        schema = properties[field]
                    else:
                        schema = schema.get("items", {})
                    if not isinstance(schema, dict):
                        break


def expand_source(source: dict, request: dict) -> dict:
    """Expand invariant wiring from a compact, agent-designed business contract.

    No task, table, answer or business predicate is supplied by this scaffold.
    Capture/ref dependencies are derived from actual authored reference calls.
    """
    from .business_contract import expand_business_contract
    from .decision_table import expand_decision_table
    from .data_recipes import expand_data_recipes
    source = expand_data_recipes(copy.deepcopy(source), seed=0 if request.get('seed') is None else request['seed'])
    source = expand_business_contract(expand_decision_table(source))
    from .answer_fields import expand_answer_fields
    source = expand_answer_fields(source, required=request.get('require_answer_fields', False))
    if "business_tools" not in source:
        return source
    if any(key in source for key in ("tools", "tool_implementations", "metrics", "scenarios")):
        raise ValueError("choose compact business_tools/outcomes/reference or expanded fields, not both")
    tools, implementations, actions, bindings, capabilities, key_steps = [], [], [], [], [], []
    calls = source["reference"]["calls"]
    expand_capture_fields(calls)
    producers, dependencies = {}, {}
    def refs(value):
        if isinstance(value, dict):
            if set(value) == {"$ref"}:
                yield value["$ref"]
            else:
                for child in value.values():
                    yield from refs(child)
        elif isinstance(value, list):
            for child in value:
                yield from refs(child)
    committed_losses = set()
    after_commit_tools = {fault['tool'] for fault in source.get('business_lifecycle', {}).get('dynamics', {}).get('faults', [])
                          if fault.get('phase') == 'after_commit'}
    for call in calls:
        name = call["tool_name"]
        key = call.get('idempotency_key')
        request_identity = (name, key) if isinstance(key, str) and key else None
        recovery_replay = request_identity in committed_losses
        if request_identity and name in after_commit_tools and call.get('expected_status', 200) >= 500:
            committed_losses.add(request_identity)
        deps = dependencies.setdefault(name, [])
        for reference in refs(call.get("arguments", {})):
            if reference not in producers:
                raise ValueError(f"reference call uses unknown capture {reference}")
            if not recovery_replay and producers[reference] != name and producers[reference] not in deps:
                deps.append(producers[reference])
        for capture in call.get("capture", {}):
            producers[capture] = name
    metrics, metric_impl = [], []
    business = source.pop("business_tools")
    for item in business:
        name, description = item["name"], item["description"]
        tools.append({"type": "function", "function": {"name": name, "description": description,
            "parameters": item["parameters"]}})
        implementations.append({**item["implementation"], "tool_name": name})
        if implementations[-1].get("operation") == "select":
            implementations[-1].setdefault("result_field", "records")
        actions.append({"name": name, "description": description,
            "atomicity_rationale": item.get("atomicity_rationale", "one business operation"),
            "inputs": [], "outputs": [], "preconditions": item.get("preconditions", []),
            "effects": item.get("effects", [])})
        bindings.append({"tool_name": name, "action_name": name})
        deps = dependencies.get(name, [])
        capabilities.append({"action_name": name, "kind": "environment_operation", "requires_tool": True,
            "dependencies": deps, "reason": description})
        key_steps.append({"step_id": name, "action_name": name, "required_for_goal": True,
            "dependencies": deps, "rationale": description})
        metrics.append({"id": "process_" + name, "category": "process", "type": "rule-based",
            "scope": "step", "weight": .2 / len(business), "score_range": [0, 1],
            "rubric": description, "target_action": name,
            "evaluator": {"kind": "trajectory_rule", "source": "runtime_rule", "assertion": description,
                          "score_mapping": {"pass": 1, "fail": 0}}})
    outcomes = source.pop("outcomes")
    if not outcomes:
        raise ValueError("at least one business outcome is required")
    for item in outcomes:
        if "semantic" in item:
            from env_factory.generation.semantic_reward import compile_semantic_outcome
            metric, cases = compile_semantic_outcome(item,
                schema=source.get("answer_contract", {}).get("schema", {}),
                weight=(.8 if business else 1) / len(outcomes),
                reference_answers={"reference_success": source['reference'].get('answer'),
                    **{alternate['id']: alternate.get('answer')
                       for alternate in source.get('reference_variants', [])
                       if isinstance(alternate, dict) and isinstance(alternate.get('id'), str)}})
            metrics.append(metric)
            source.setdefault("semantic_calibration", {})[item["id"]] = cases
            continue
        rule = {**item["rule"], "metric_id": item["id"], "score_mapping": {"pass": 1, "fail": 0}}
        if rule["operator"] in {"numeric_targets", "value_targets"}:
            for target in rule["expected"]["targets"]:
                target["expression"] = normalize_expression(target["expression"])
        metric_impl.append(rule)
        metrics.append({"id": item["id"], "category": "outcome", "type": "rule-based", "scope": "terminal",
            "weight": (.8 if business else 1) / len(outcomes), "score_range": [0, 1], "rubric": item["rubric"],
            "evaluation_inputs": ["final_agent_response", "tool_results", "business_data"],
            "criteria": [item["rubric"], "逐项核对最终回答中的业务事实与实际工具结果及当前业务记录一致。"],
            "evaluator": {"kind": "business_state_rule" if rule["source"] == "business_state" else "document_rule",
                "source": "runtime_rule", "assertion": item["rubric"], "score_mapping": {"pass": 1, "fail": 0}}})
    reset = {"operation": "reset", "body": {"episode_id": "reference", "seed": request.get("seed") or 0}}
    reference = source.pop('reference')
    reference_route = reference.get("route_id")
    success_steps = [copy.deepcopy(reset), *[{**call, "operation": "tool_call", "expected_status": call.get("expected_status", 200)} for call in calls],
        {"operation": "agent_response", "content": reference["answer"]},
        {"operation": "reward", "step_id": "reward"}]
    scenarios = [{"scenario_id": "reference_success", "kind": "goal_success", "steps": success_steps,
        **({"execution_route": reference_route} if reference_route else {}),
        **({'user_script_indices': reference['user_script_indices']} if 'user_script_indices' in reference else {}),
        "assertions": [{"source": "step:reward", "path": "$.reward", "operator": "gte", "expected": 1}]},
        {"scenario_id": "empty_failure", "kind": "goal_failure", "steps": [copy.deepcopy(reset),
         {"operation": "reward", "step_id": "reward"}], "assertions": [
            {"source": "step:reward", "path": "$.reward", "operator": "lte", "expected": 0}]}]
    alternatives = source.pop('reference_variants', [])
    if not isinstance(alternatives, list) or len(alternatives) > 8:
        raise ValueError('INTERACTION_REFERENCE_SCOPE: at most eight alternate references are supported')
    reference_ids = {'reference_success', 'empty_failure'}
    for alternate in alternatives:
        if (not isinstance(alternate, dict) or not isinstance(alternate.get('id'), str)
                or not alternate['id'].strip() or alternate['id'] in reference_ids
                or not isinstance(alternate.get('calls'), list)
                or any(not isinstance(call, dict) for call in alternate['calls'])
                or not isinstance(alternate.get('answer'), str) or not alternate['answer'].strip()
                or 'user_script_indices' not in alternate):
            raise ValueError('INTERACTION_REFERENCE_SCOPE: alternate reference requires unique id, calls, answer and user_script_indices')
        reference_ids.add(alternate['id'])
        if not alternate['calls'] and not any(
                route['id'] == alternate.get('route_id') and not route['required_tools']
                and set(route['when']) == {'user_rejection'} for route in source.get('execution_routes', [])):
            raise ValueError('INTERACTION_REFERENCE_SCOPE: zero-call reference requires a declared user-refusal route')
        expand_capture_fields(alternate['calls'])
        scenarios.append({'scenario_id': alternate['id'], 'kind': 'goal_success',
            'user_script_indices': alternate['user_script_indices'],
            **({'execution_route': alternate['route_id']} if alternate.get('route_id') else {}),
            'steps': [copy.deepcopy(reset), *[{**call, 'operation': 'tool_call',
                'expected_status': call.get('expected_status', 200)} for call in alternate['calls']],
                {'operation': 'agent_response', 'content': alternate['answer']},
                {'operation': 'reward', 'step_id': 'reward'}],
            'assertions': [{'source': 'step:reward', 'path': '$.reward', 'operator': 'gte', 'expected': 1}]})
    from .negative_scenarios import derive_negatives
    for negative in source.pop("negative_scenarios", []):
        if negative.get('from_reference') is True:
            base_id = negative.get('reference_id', 'reference_success')
            bases = [case for case in scenarios if case['scenario_id'] == base_id and case['kind'] == 'goal_success']
            if len(bases) != 1:
                raise ValueError('INTERACTION_REFERENCE_SCOPE: negative reference_id must select a successful reference')
            derived = derive_negatives(bases, [{key: value for key, value in negative.items() if key != 'reference_id'}])[1:]
            for case in derived:
                case['derive_interactions'] = True
            scenarios.extend(derived)
        else:
            scenarios.append(negative)
    source.update(tools=tools, tool_implementations=implementations, actions=actions,
        tool_bindings=bindings, capability_plan=capabilities, reward_key_steps=key_steps,
        metrics=metrics, metric_implementations=metric_impl, scenarios=scenarios)
    return source


def compile_source(source: dict, *, root: Path, request: dict, script_count: int = 3,
                   structural_preview: bool = False, defer_privacy_review: bool = False, blueprint: dict | None = None,
                   judgment_cache=None) -> dict:
    """Compile a complete business design against runner-owned graph/route input."""
    source = expand_source(source, request)
    from .scenario_assertions import validate_scenario_fixture_scope
    validate_scenario_fixture_scope(source['scenarios'])
    if blueprint is not None:
        from .business_blueprint import validate_implementation
        validate_implementation(blueprint, source, request)
    from .requirement_coverage import validate_requirement_coverage
    requirement_coverage = validate_requirement_coverage(
        source, required=request.get('require_business_requirements', False))
    from .predicate_selectors import validate_predicate_selectors
    validate_predicate_selectors(source)
    from .realism_requirements import validate_realism_request
    realism_requirements = validate_realism_request(source, request)
    from .authorization_contract import validate_authorizations
    validate_authorizations(source, request)
    from .reference_requests import validate_reference_requests
    validate_reference_requests(source)
    from .business_conditions import compile_business_conditions, validate_condition_bindings, probe_business_conditions
    business_conditions = compile_business_conditions(source)
    from .query_dependencies import validate_query_dependencies
    validate_query_dependencies(source)
    if request.get('require_interaction_contract') and not source.get('interaction_contract'):
        raise ValueError('multi-step authoring requires task-specific interaction_contract')
    for scenario in source["scenarios"]:
        expand_capture_fields([step for step in scenario["steps"] if step.get("operation") == "tool_call"])
    # Return envelope names are platform plumbing, not business decisions.
    result_fields = {"select": "records", "aggregate_count": "count",
                     "insert": "record", "update": "updated_count", "delete": "deleted_count"}
    for implementation in source["tool_implementations"]:
        if implementation.get("operation") in result_fields:
            implementation.setdefault("result_field", result_fields[implementation["operation"]])
        if 'derived_changes' in implementation:
            implementation['derived_changes'] = {field: normalize_expression(expression)
                for field, expression in implementation['derived_changes'].items()}
    if not source.get('interaction_contract'):
        validate_stateful_value_origins(source)
    validate_direct_text_targets(source)
    state_reward_predicates = [predicate for rule in source["metric_implementations"]
        if rule.get("operator") == "state_predicates" for predicate in rule.get("expected", [])]
    for predicate in source.get("semantic_goal", {}).get("row_predicates", []) + state_reward_predicates:
        predicate.setdefault("values", {})
        if "value_expressions" in predicate:
            predicate["value_expressions"] = {field: normalize_expression(expression)
                for field, expression in predicate["value_expressions"].items()}
    for rule in source["metric_implementations"]:
        if (rule.get("source") == "business_state" and rule.get("operator") in {"eq", "ne", "gte", "lte"}
                and isinstance(rule.get("expected"), dict)
                and set(rule["expected"]) & {"from_tool", "lookup", "op", "initial", "if", "at_tool"}):
            raise ValueError("STATE_REWARD_EXPRESSION: ordinary comparisons use literal expected values; "
                "use operator state_predicates, source business_state, path $, and expected row predicates "
                "with value_expressions for dynamic state targets")
    for route in source.get("execution_routes", []):
        route["when"] = normalize_expression(route["when"])
    validate_reward_observability(source)
    reference_query_bindings = []
    query_bindings = bind_reward_queries(source, reference_details=reference_query_bindings)
    from .reward_proofs import validate_reward_goal_consistency
    validate_reward_goal_consistency(source)
    from env_factory.sandbox_runtime import RewardStateSnapshots
    RewardStateSnapshots.validate_tools(source["tool_implementations"],
        [source["metric_implementations"], source.get("semantic_goal", {}), source.get("execution_routes", []),
         source['tool_implementations'], source.get('business_lifecycle', {})])
    temporal_findings = reward_temporal_diagnostics(source)
    validate_condition_bindings(business_conditions, query_bindings, source,
                                reference_details=reference_query_bindings)
    condition_probes = probe_business_conditions(business_conditions, query_bindings, source,
                                                reference_details=reference_query_bindings)
    from env_factory.tasks.task_spec import validate_goal_contract
    for rule in source["metric_implementations"]:
        if rule.get("operator") == "state_predicates":
            validate_goal_contract({"row_predicates": rule["expected"]}, source["tables"], require_change=False)
    identifier_public_input = copy.deepcopy(source["description"]["public_input"])
    # The delivery gate requires this platform-owned evidence rule for every
    # stateful semantic outcome. Publish it before calibration/public binding so
    # authors need not discover an exact required sentence after live replay.
    if source.get('environment_plan', {}).get('mode') == 'stateful':
        P._ground_reference_outcome_metrics(source['metrics'], environment_mode='stateful')
    bind_answer_contract(source)
    category = request["training_category"]
    contract = training_contract(category)
    if source.get("version") != "1.0":
        raise ValueError("authoring source version must be 1.0")
    description = source["description"]
    graph = request["graph_context"]
    P._validate_graph_keyword_alignment(description, graph.get("keywords", []), graph)
    if description["task_intent"] not in contract["allowed_intents"]:
        raise ValueError("task intent is incompatible with requested route")
    if request.get("task_intent") and description["task_intent"] != request["task_intent"]:
        raise ValueError("task intent differs from requested intent")
    tables, tools = source["tables"], source["tools"]
    implementations = source["tool_implementations"]
    plan = source["environment_plan"]
    if plan["mode"] == "stateful" and not source.get("semantic_goal", {}).get("row_predicates"):
        raise ValueError("stateful authoring requires typed row_predicates in semantic_goal")
    if plan["mode"] == "stateful" and (plan.get("requires_persistence") is not True
            or not any(item.get("operation") in {"insert", "update", "delete"} for item in implementations)):
        raise ValueError("stateful authoring requires persistence and an actual mutating tool")
    if plan["mode"] not in contract["allowed_environment_modes"]:
        raise ValueError("environment mode differs from requested route")
    if plan["mode"] not in request["available_environment_modes"]:
        raise ValueError("environment mode is unavailable")
    bounds = contract["business_tools"]
    if len(tools) < bounds["min"] or (bounds["max"] is not None and len(tools) > bounds["max"]):
        raise ValueError("business tool count differs from requested route")
    if tables:
        if plan["mode"] == "stateless":
            raise ValueError("DIRECT_PRIVATE_DATA: stateless tasks must place their evidence in public_input.materials, not private tables")
        P._validate_relational_data(tables)
    elif plan["mode"] != "stateless":
        raise ValueError("business environment requires data tables")
    P._validate_tools(tools)
    P._validate_tool_implementations(implementations, tools=tools, tables=tables)
    for implementation in implementations:
        if implementation.get('pagination'):
            paging = implementation['pagination']
            tool = next(t['function'] for t in tools if t['function']['name'] == implementation['tool_name'])
            tool['description'] += (f"\nPagination: at most {paging['page_size']} rows per page. "
                f"Omit {paging['cursor_argument']} for the first page; copy next_cursor into "
                f"{paging['cursor_argument']} for the next page and keep all other arguments unchanged. "
                "count is this page's row count; total_count counts all matches. An empty next_cursor "
                "and has_more=false indicate completion. Ordering is stable and includes the primary key. "
                "A cursor is bound to this query and episode. PAGINATION_CURSOR_STALE (409) means "
                "query data changed: restart without a cursor; do not combine old and new pages.")
    query_inputs = validate_query_inputs(source)
    if {x["tool_name"] for x in implementations} != {x["function"]["name"] for x in tools}:
        raise ValueError("every business tool requires an implementation")
    actions, bindings = source["actions"], source["tool_bindings"]
    capabilities, key_steps = source["capability_plan"], source["reward_key_steps"]
    metrics, scenarios = source["metrics"], source["scenarios"]
    from .semantic_reward import validate_explanation_rewards
    validate_explanation_rewards(source)
    outcome_ids = {metric["id"] for metric in metrics if metric["category"] == "outcome"}
    for rule in source["metric_implementations"]:
        if (rule["metric_id"] in outcome_ids and rule.get("source") == "final_agent_response"
                and rule.get("operator") not in {"numeric_targets", "value_targets"}):
            raise ValueError("ANSWER_REWARD_UNGROUNDED: use typed value_targets or numeric_targets for "
                             "final-answer outcomes; keyword presence and arbitrary exact text cannot "
                             "establish the requested business result")
    if plan["mode"] == "reference_data":
        for rule in source["metric_implementations"]:
            if rule["metric_id"] in outcome_ids and rule["operator"] not in {"numeric_targets", "value_targets"}:
                raise ValueError("REFERENCE_REWARD_UNGROUNDED: use value_targets for dynamic text/boolean/decision "
                    "answers or numeric_targets for quantities; static contains/eq phrases cannot verify private business facts")
    # Stable operation IDs are plumbing, not a business-design choice. They must
    # match assertion references in the final sandbox, not only the preview.
    for scenario in scenarios:
        counts = {}
        for step in scenario["steps"]:
            operation = step["operation"]
            counts[operation] = counts.get(operation, 0) + 1
            step.setdefault("step_id", operation if counts[operation] == 1 else f"{operation}_{counts[operation]}")
        ids = [step["step_id"] for step in scenario["steps"]]
        if len(ids) != len(set(ids)):
            raise ValueError("scenario step ids must be unique")
    semantic_metrics = [m for m in metrics if m.get("type") != "rule-based"]
    if semantic_metrics:
        from env_factory.generation.semantic_reward import compile_semantic_outcome
        calibration = source.get("semantic_calibration", {})
        if set(calibration) != {m["id"] for m in semantic_metrics}:
            raise ValueError("SEMANTIC_CALIBRATION_COVERAGE: every model metric requires validated cases")
        for metric in semantic_metrics:
            if metric.get("type") != "model-based" or not metric.get("semantic_fields"):
                raise ValueError("SEMANTIC_REWARD_CONTRACT: only field-scoped calibrated model metrics are supported")
            compile_semantic_outcome({"id": metric["id"], "rubric": metric["rubric"], "semantic": {
                "fields": metric["semantic_fields"], "criteria": metric["criteria"],
                "context_fields": metric.get("semantic_context_fields", []),
                "evidence_mode": metric.get("semantic_evidence_mode", "runtime"),
                "cases": calibration[metric["id"]]}},
                schema=source["answer_contract"]["schema"], weight=metric["weight"])
    if not metrics or len({m["id"] for m in metrics}) != len(metrics):
        raise ValueError("reward metric ids must be nonempty and unique")
    weights = {kind: 0.0 for kind in ("process", "outcome", "penalty")}
    for metric in metrics:
        weight = metric["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError("reward weights must be finite and nonnegative")
        weights[metric["category"]] += weight
        if not metric.get("rubric") or metric.get("scope") not in {"step", "state", "terminal", "trajectory"}:
            raise ValueError("reward metrics require rubric and scope")
    if not math.isclose(weights["process"] + weights["outcome"], 1) or weights["outcome"] <= weights["process"]:
        raise ValueError("positive reward weights must sum to one and favor outcome")
    if weights["penalty"] and not math.isclose(weights["penalty"], 1):
        raise ValueError("penalty weights must sum to one when supplied")
    metric_impl = P._compile_process_metric_implementations(
        metrics=metrics, metric_implementations=source["metric_implementations"],
        business_scenarios=scenarios, tool_bindings=bindings)
    P._normalize_compiled_process_metrics(metrics, metric_impl)
    P._validate_metric_implementations(metric_impl, metrics, require_process=True)
    validate_numeric_dependency_coverage(source, metric_impl)
    from .interaction_contract import compile_interactions, interleave_scenarios, bind_captured_interaction_arguments
    privacy_findings = []
    capture_bindings = []
    interaction_scripts = (compile_interactions(source['interaction_contract'], description, tools, script_count,
                           implementations=implementations, scenarios=scenarios,
                           privacy_findings=privacy_findings if defer_privacy_review or structural_preview else None,
                           structured_requests=request.get('require_structured_interactions', False),
                           capture_bindings=capture_bindings, execution_routes=source.get('execution_routes', []))
                           if source.get('interaction_contract') else None)
    bind_captured_interaction_arguments(metric_impl, interaction_scripts or [], capture_bindings)
    reject_private_identifier_constants(source, validated_scripts=interaction_scripts or (),
                                        public_input=identifier_public_input)
    root.mkdir(parents=True, exist_ok=True)
    manifest = P._materialize_business_data(tables, P._render_data_document(tables),
        root / "data/business_data", environment_mode=plan["mode"], manifest_root="data/business_data")
    manifest["data_governance"]["origin"] = "programmatically_generated_synthetic"
    if source.get('_data_recipe_provenance'):
        manifest['row_recipes'] = copy.deepcopy(source['_data_recipe_provenance'])
    if interaction_scripts and request.get('require_structured_interactions'):
        from env_factory.sandbox_runtime import STRUCTURED_INTERACTION_INSTRUCTIONS
        description['public_input'].setdefault('materials', []).append({
            'name': 'user_interaction_format', 'mime_type': 'text/plain',
            'content': STRUCTURED_INTERACTION_INSTRUCTIONS})
        from env_factory.sandbox_runtime import ApprovalTerms
        proposal_tools = {}
        for script in interaction_scripts:
            for stage in script['interaction_protocol']['stages']:
                for effective in [stage, *[{**stage, **branch} for branch in stage.get('branches', [])]]:
                    fields = ApprovalTerms.proposal_fields(effective)
                    if fields:
                        proposal_tools[effective['before_tool']] = sorted(fields)
        if proposal_tools:
            description['public_input']['materials'].append({
                'name': 'approval_proposal_tools', 'mime_type': 'application/json',
                'content': json.dumps(proposal_tools, ensure_ascii=False, sort_keys=True)})
        contexts = [dict(tool=stage['before_tool'], arguments=([] if stage['kind'] == 'execution_confirmation'
                         else [stage['bind_to']['argument']]),
                         context_from=stage['context_from'])
                    for script in interaction_scripts for stage in script['interaction_protocol']['stages']
                    if 'context_from' in stage]
        if contexts:
            contexts = list({json.dumps(item, sort_keys=True): item for item in contexts}.values())
            description['public_input']['materials'].append({
                'name': 'user_interaction_context', 'mime_type': 'application/json',
                'content': json.dumps(contexts, ensure_ascii=False)})
    if interaction_scripts:
        from .interaction_contract import stage_bindings
        scopes = sorted({(binding['tool'], binding['argument']) for script in interaction_scripts
                         for stage in script['interaction_protocol']['stages'] for binding in stage_bindings(stage)})
        description['public_input'].setdefault('materials', []).append({
            'name': 'disclosed_user_input_scope', 'mime_type': 'application/json',
            'content': json.dumps({'rule': 'After the user discloses a value, every call using its '
                'bound argument must preserve that value. A mismatched call is rejected with '
                'HTTP 409 USER_INPUT_MISMATCH before reading or writing business data. '
                'Retrying with the correct value is allowed. Undisclosed/inactive branches '
                'do not authorize guessing, and a terminal refusal still stops execution.',
                'bindings': [{'tool': tool, 'argument': argument} for tool, argument in scopes]}, ensure_ascii=False)})
        for script in interaction_scripts:
            script['user_input'] = copy.deepcopy(description['public_input'])
    scripts = interaction_scripts or P._deterministic_user_scripts(description=description, count=script_count, finalize_on_goal=True)
    for index, script in enumerate(scripts):
        P._validate_user_script_state_machine(script, index)
    users = P._materialize_user_simulation(P._deterministic_user_profiles(script_count), scripts,
        root / "data/user_simulation", manifest_root="data/user_simulation")
    formula = {"type": "separate_sign_weighted_sum", "formula": "R = clip(" +
        " + ".join(f"{m['weight']}*score({m['id']})" for m in metrics) + ", -1, 1)",
        "positive_weight_sum": sum(m["weight"] for m in metrics if m["category"] != "penalty"),
        "negative_weight_sum": sum(m["weight"] for m in metrics if m["category"] == "penalty"),
        "score_range": [-1, 1], "positive_categories": ["process", "outcome"],
        "negative_categories": ["penalty"], "normalization": "positive weights sum to one"}
    acceptance = P._build_acceptance_contract(task_description=description, data_manifest=manifest,
        data_tables=tables, actions=actions, tools=tools, key_steps=key_steps, metrics=metrics,
        reward_formula=formula, tool_implementations=implementations)
    acceptance["executable_scenarios"].extend(scenarios)
    acceptance = P._ground_acceptance_probes_from_success(acceptance, scenarios)
    spec = compile_task_spec(task_description=description, training_category=category,
        environment_plan=plan, data_manifest=manifest, data_tables=tables, tools=tools,
        noise_tools=[], tool_bindings=bindings, tool_implementations=implementations,
        actions=actions, key_steps=key_steps, metrics=metrics, executable_scenarios=scenarios,
        semantic_goal=source.get("semantic_goal"), capability_plan=capabilities)
    from .execution_routes import compile_execution_routes
    compile_execution_routes(source, spec, metrics, metric_impl, interaction_scripts)
    from env_factory.sandbox_runtime import RewardStateSnapshots
    disclosed_bindings = RewardStateSnapshots.validate_user_values(
        [metric_impl, source.get('execution_routes', [])], interaction_scripts or [])
    if disclosed_bindings:
        spec['disclosed_value_bindings'] = disclosed_bindings
    if spec.get('execution_routes'):
        formula['normalization'] = 'exclude inactive route process metrics; normalize active positive weights to one; penalty weights unchanged'
    if interaction_scripts:
        spec['requires_user_interaction'] = True
        spec['interaction_complexity'] = {
            'variants':len(interaction_scripts),
            'required_user_turns':[len(script['interaction_protocol']['stages']) for script in interaction_scripts],
            'kinds':sorted({stage['kind'] for script in interaction_scripts for stage in script['interaction_protocol']['stages']}),
            'runtime':'deterministic_task_protocol'}
        acceptance['executable_scenarios'] = interleave_scenarios(acceptance['executable_scenarios'], interaction_scripts,
            execution_routes=spec.get('execution_routes', []))
        from .interaction_diversity import interaction_diversity
        spec['interaction_complexity']['behavioral_diversity'] = interaction_diversity(
            interaction_scripts, acceptance['executable_scenarios'])
        spec['interaction_complexity']['require_behavioral_diversity'] = bool(request.get('require_behavioral_diversity'))
        if request.get('require_dynamic_choice_coverage'):
            from .dynamic_choice_coverage import requirements as choice_requirements
            spec['interaction_complexity']['dynamic_choice_requirements'] = choice_requirements(interaction_scripts)
        if (request.get('require_behavioral_diversity') and len(interaction_scripts) > 1
                and spec['interaction_complexity']['behavioral_diversity']['wording_only_variants']):
            raise ValueError('INTERACTION_BEHAVIOR_DIVERSITY: multiple user scripts only paraphrase the same behavior; '
                             'author a genuine alternative choice, correction or authorization branch with executable evidence')
        validate_stateful_value_origins(
            {**source, 'scenarios': acceptance['executable_scenarios']},
            validated_scripts=interaction_scripts)
    from .execution_routes import compile_route_probes
    route_probes = compile_route_probes(source, spec, interaction_scripts)
    requirements = {**description["requirements"], "media_truth_mode": "programmatic",
                    "runtime_interface": P._build_runtime_interface(tools, formula)}
    for endpoint in requirements["runtime_interface"]["endpoints"]:
        tool = next((t for t in spec["tool_contracts"] if t["name"] == endpoint.get("name")), None)
        if tool:
            endpoint["response_schema"] = tool["output_contract"]["schema"]
    result = {"task": description["task"], "task_type": request["task_type"],
        "task_intent": description["task_intent"], "complexity": description["complexity"],
        "training_category": category, "training_contract": contract,
        "runtime_capabilities": {"environment_modes": [plan["mode"]]},
        "user_simulation_policy": {"mode": "interactive_goal" if interaction_scripts else "fixed_goal", "required_outcomes": ["goal_satisfied"]},
        "task_spec": spec, "requirements": requirements, "public_input": description["public_input"],
        "business_lifecycle": copy.deepcopy(source.get("business_lifecycle", {})),
        "realism_requirements": realism_requirements,
        "requirement_coverage": requirement_coverage,
        "execution_route_probes": route_probes,
        "environment_plan": plan, "environment": P._environment_records({}, actions),
        "data_manifest": manifest, "user_simulation_manifest": users,
        "tools_manifest": P._materialize_tools(tools, root),
        "media_generation": {"required": False, "language": "python", "code": "", "dependencies": [], "entrypoint": "", "output_dir": ""},
        "actions": actions, "capability_plan": capabilities, "tools": tools, "tool_bindings": bindings,
        "tool_implementations": implementations, "noise_tools": [], "reward_key_steps": key_steps,
        "observation_schema": {"type": "object", "properties": {}}, "metrics": metrics,
        "metric_implementations": metric_impl, "reward_formula": formula, "acceptance_contract": acceptance,
        "graph_context": copy.deepcopy(request["graph_context"])}
    from env_factory.tasks.task_quality import reward_target_qualification_failures
    target_failures = reward_target_qualification_failures(result)
    if target_failures:
        raise ValueError('REWARD_TARGET_QUALIFICATION: ' + '; '.join(target_failures)
            + '. Reward observed private evidence. Query-scope fields may use a direct tool_argument '
            'consumed by a private query filter alongside a separate positive private-evidence outcome; '
            'do not hardcode the user reply or weaken the outcome check.')
    from .requirement_coverage import bind_scenarios
    result['requirement_coverage'] = bind_scenarios(requirement_coverage,
        acceptance['executable_scenarios'], [script['script_id'] for script in scripts])
    if source.get('reward_pair_probes') or request.get('require_business_requirements'):
        from .reward_pairs import derive_reward_pairs
        definitions = []
        if not isinstance(source.get('reward_pair_probes', []), list) or len(source.get('reward_pair_probes', [])) > 8:
            raise ValueError('REWARD_PAIRS: at most eight authored probes are supported')
        known = {case['scenario_id'] for case in acceptance['executable_scenarios']}
        for definition in source.get('reward_pair_probes', []):
            base = definition.get('reference_id')
            matches = [candidate for candidate in [base, *(f"{base}-{script['script_id']}" for script in scripts)]
                       if candidate in known]
            if not matches:
                raise ValueError('REWARD_PAIRS: reference_id does not resolve')
            definitions.extend({**definition, 'reference_id': identity,
                                'id': f"{definition['id']}::{identity}"} for identity in matches)
        result['reward_pair_probes'] = derive_reward_pairs(result, definitions, max_probes=8 * script_count)
    from .scenario_assertions import validate_reward_assertion_paths
    validate_reward_assertion_paths(acceptance['executable_scenarios'], metrics,
                                    routed=bool(spec.get('execution_routes')))
    validate_capture_paths(scenarios, spec["tool_contracts"])
    from env_factory.contracts.tool_chain_contract import tool_chain_issues
    chain_issues = tool_chain_issues(result)
    if chain_issues:
        raise ValueError('TOOL_CHAIN_QUALIFICATION: ' + json.dumps(chain_issues, ensure_ascii=False))
    if source.get('business_lifecycle'):
        from types import SimpleNamespace
        from env_factory.sandbox_runtime import BusinessLifecycle
        BusinessLifecycle(source['business_lifecycle'],
            SimpleNamespace(schemas={table['table_name']: table for table in tables}), None,
            [tool['function']['name'] for tool in tools])
        by_name = {tool['function']['name']: tool['function']['parameters'] for tool in tools}
        for name, policy in source['business_lifecycle'].get('tools', {}).items():
            operands = list(policy.get('transition', {}).get('selector', {}).values())
            for guard in policy.get('guards', []):
                operands.extend(guard['selector'].values())
                operands.extend(value for comparisons in guard['conditions'].values() for value in comparisons.values())
            for value in operands:
                if isinstance(value, dict) and set(value) == {'argument'} and (value['argument'] not in by_name[name].get('required', [])
                        or value['argument'] not in by_name[name].get('properties', {})):
                    raise ValueError('lifecycle selectors and guards must bind required tool arguments')
    # Use the same API-consumption check as final buildability, while the
    # candidate can still be repaired before execution or a model review.
    P._validate_business_tool_semantics(tools=tools, implementations=implementations,
        interaction_scripts=scripts, metrics=metrics, metric_implementations=metric_impl,
        business_lifecycle=source.get('business_lifecycle'))
    if structural_preview:
        # Reuse runtime initial-data checks while the author can still repair
        # cheaply. No episodes, trajectories or model evaluators are run here.
        from env_factory.sandbox_runtime import EpisodeStore, ManifestDataStore
        with tempfile.TemporaryDirectory(prefix="envfactory-preview-data-") as directory:
            ManifestDataStore(manifest, root / manifest["root"],
                              EpisodeStore(Path(directory) / "episodes.sqlite3"))
        # Author-side sandbox previews cannot rely on evaluator network access.
        # These artifacts must never masquerade as an accepted training task.
        from .review_replay import baseline_reward_observations
        reward_observations = baseline_reward_observations({"source": source,
            "compiled_reward_rules": metric_impl})
        # Initial-state targets are not post-write targets. Keep these diagnostic,
        # and only warn about unresolved values when no business tool writes data.
        readonly = all(item.get("operation") in {"select", "aggregate_count"}
                       for item in implementations)
        unresolved = [item for item in reward_observations["targets"]
                      if item.get("value") is None or "error" in item] if readonly else []
        result["task_readiness"] = {"ready": False, "errors": ["EXECUTION_VALIDATION_PENDING"],
            "warnings": (["READONLY_REWARD_TARGET_UNRESOLVED"] if unresolved else [])
                + (["REWARD_CURRENT_MUTABLE_QUERY"] if temporal_findings else [])}
        result["generation_pipeline"] = {"backend": "code_agent", "structural_preview": True,
            "privacy_review_obligations": privacy_findings,
            "business_condition_probes": condition_probes,
            "live_rollout_verified": False, "verification": {
                "passed": False, "structural_checks_passed": True,
                "scope": "structural_preview_only",
                "reward_observations": reward_observations,
                "reward_temporal_diagnostics": temporal_findings,
                "unresolved_readonly_reward_targets": unresolved,
                "pending": [
                    "reference_execution", "policy_ablations", "semantic_calibration",
                    "delivery_preflight", "independent_semantic_review"]}}
        return result
    from .deterministic_preflight import check_all_routes
    check_all_routes(result, root, verify_execution, prepare_route_probe)
    checks = verify_execution(result, root, calibration=source.get("semantic_calibration", {}),
                              condition_probes=condition_probes, judgment_cache=judgment_cache)
    result["task_readiness"] = {"ready": not privacy_findings,
        "errors": ["INTERACTION_PRIVACY_REVIEW_PENDING"] if privacy_findings else [], "warnings": [],
        "training_profile": P._derive_training_profile(business_tool_count=len(bindings),
            noise_tool_count=0, key_step_count=len(key_steps))}
    result["generation_pipeline"] = {"version": "1.0", "backend": "code_agent",
        "privacy_review_obligations": privacy_findings,
        "reward_query_bindings": query_bindings,
        "reward_temporal_diagnostics": temporal_findings,
        "business_condition_probes": condition_probes,
        "query_input_grounding": query_inputs,
        "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "verification": checks, "live_rollout_verified": False}
    return result


def unresolved_lookup_diagnostics(expression: object, state: dict) -> list[dict]:
    """Explain failed unique lookups without changing reward semantics."""
    from env_factory.sandbox_runtime import DeclarativeMetricEvaluator as Evaluator
    findings = []
    def walk(node, path, state):
        if isinstance(node, dict):
            if set(node) == {'if'}:
                branch = node['if']
                walk(branch['condition'], path + '.if.condition', state)
                condition = Evaluator._value_expression(branch['condition'], state)
                if type(condition) is bool:
                    chosen = 'then' if condition else 'else'
                    walk(branch[chosen], path + '.if.' + chosen, state)
                return
            if set(node) == {"initial"}:
                from env_factory.sandbox_runtime import ExpressionBusinessState
                baseline = getattr(state, "initial", {})
                walk(node["initial"], path + ".initial", ExpressionBusinessState(
                    baseline, baseline, getattr(state, "snapshots", {})))
                return
            if set(node) == {"at_tool"}:
                from env_factory.sandbox_runtime import ExpressionBusinessState
                name = node["at_tool"]["name"]
                snapshots = getattr(state, "snapshots", {})
                if name not in snapshots:
                    findings.append({"path": path, "reason": "reward_checkpoint_missing", "tool": name})
                    return
                walk(node["at_tool"]["expression"], path + ".at_tool.expression",
                     ExpressionBusinessState(snapshots[name], getattr(state, "initial", None), snapshots))
                return
            for key, child in node.items():
                walk(child, path + "." + key, state)
            if set(node) == {"lookup"}:
                lookup = node["lookup"]
                if Evaluator._value_expression(node, state) is None:
                    where = Evaluator._numeric_where(lookup["where"], state, depth=0)
                    unresolved = any(value is None for value in where.values())
                    rows = state.get(lookup["table"], [])
                    matches = [] if unresolved else [row for row in rows
                        if isinstance(row, dict) and Evaluator._where_matches(row, where)]
                    findings.append({"path": path, "table": lookup["table"],
                        "field": lookup["field"], "resolved_where": where,
                        "matched_row_count": None if unresolved else len(matches),
                        "reason": "upstream_lookup_unresolved" if unresolved else
                            "lookup_not_unique" if len(matches) != 1 else "field_value_invalid"})
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, path + f"[{index}]", state)
    walk(expression, "$", state)
    return findings


def state_predicate_diagnostic(predicate, state, baseline):
    """Retain the expression cause when a state target cannot resolve an identity."""
    from env_factory.sandbox_runtime import BusinessGoalEvaluator
    actual = [row for row in state.get(predicate['table'], [])
              if all(row.get(key) == value for key, value in predicate.get('where', {}).items())]
    expected = BusinessGoalEvaluator.expected_values(predicate, state, baseline)
    return {'table': predicate['table'], 'where': predicate.get('where', {}),
        'expected_values': expected, 'actual_rows': actual[:5], 'matched_row_count': len(actual),
        'satisfied': BusinessGoalEvaluator.evaluate([predicate], state, baseline),
        'unresolved_lookups': unresolved_lookup_diagnostics(
            predicate.get('value_expressions', {}), state) if expected is None else []}


def validate_numeric_dependency_coverage(source: dict, implementations: list[dict]) -> None:
    """A required private lookup must be able to affect a purely numeric outcome.

    This is a structural necessary condition, not proof of business semantics.
    Full data sensitivity and independent semantic review remain required.
    """
    if source["environment_plan"]["mode"] != "reference_data":
        return
    outcome_ids = {m["id"] for m in source["metrics"] if m["category"] == "outcome"}
    outcomes = [m for m in implementations if m["metric_id"] in outcome_ids]
    if not outcomes or any(m["operator"] not in {"numeric_targets", "value_targets"} for m in outcomes):
        return
    referenced_tables = set()
    tool_tables = {item['tool_name']: item['table'] for item in source['tool_implementations']}
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('tool_result'), dict):
                table = tool_tables.get(value['tool_result'].get('name'))
                if table:
                    referenced_tables.add(table)
            if isinstance(value.get("table"), str):
                referenced_tables.add(value["table"])
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    for outcome in outcomes:
        walk(outcome["expected"])
    required_actions = {step["action_name"] for step in source["reward_key_steps"] if step.get("required_for_goal")}
    required_tools = {b["tool_name"] for b in source["tool_bindings"] if b["action_name"] in required_actions}
    ignored = [impl["tool_name"] for impl in source["tool_implementations"]
        if impl["tool_name"] in required_tools and impl["table"] not in referenced_tables]
    if ignored:
        raise ValueError("NUMERIC_DEPENDENCY_UNUSED: required tools read tables absent from all outcome expressions: "
                         + ", ".join(ignored) + "; encode the actual business dependency in the reward")


def _bind_reference_for_ablations(steps: list[dict], execution: dict) -> list[dict]:
    """Freeze observed arguments and proposals before omitting their producers.

    Replay captures in order: a later reuse of a capture name must not change an
    earlier displayed proposal. The negative still has to satisfy the normal
    tool dependencies, interaction protocol and business outcome checks.
    """
    from env_factory.sandbox_runtime import AcceptanceScenarioRunner, DeclarativeMetricEvaluator
    bound_steps, variables = copy.deepcopy(steps), {}
    for step, observed in zip(bound_steps, execution['history'], strict=True):
        if step.get('operation') == 'tool_call':
            step['arguments'] = AcceptanceScenarioRunner._resolve(step.get('arguments', {}), variables)
            if 'idempotency_key' in step:
                step['idempotency_key'] = AcceptanceScenarioRunner._resolve(step['idempotency_key'], variables)
        elif step.get('operation') == 'dialogue_turn':
            step['content'] = AcceptanceScenarioRunner.resolve_dialogue_content(step.get('content', ''), variables)
        for name, path in step.get('capture', {}).items():
            variables[str(name)] = DeclarativeMetricEvaluator._resolve(observed['body'], str(path))
    return bound_steps


def prepare_route_probe(artifacts, root, probe):
    """Materialize one isolated probe with the exact production overrides."""
    import shutil
    probe_root = root / 'route_probes' / probe['route_id']
    for field in ('data_manifest', 'user_simulation_manifest'):
        relative = artifacts[field]['root']
        shutil.copytree(root / relative, probe_root / relative, dirs_exist_ok=True)
    candidate = copy.deepcopy(artifacts)
    probe_rules = {rule['metric_id']: rule for rule in probe.get('process_rules', [])}
    candidate['metric_implementations'] = [copy.deepcopy(probe_rules.get(rule['metric_id'], rule))
                                           for rule in candidate['metric_implementations']]
    if probe.get('user_variables'):
        from .execution_routes import validate_probe_user_variables
        users = candidate['user_simulation_manifest']
        script_path = probe_root / users['root'] / users['scripts_file']
        probe_scripts = json.loads(script_path.read_text())
        validate_probe_user_variables(probe['user_variables'], probe_scripts)
        for script in probe_scripts:
            script['variables'].update(probe['user_variables'])
        script_path.write_text(json.dumps(probe_scripts, ensure_ascii=False, indent=2) + '\n')
    candidate['execution_route_probes'] = []
    candidate['acceptance_contract']['executable_scenarios'] = probe['scenarios']
    return candidate, probe_root


def verify_execution(artifacts: dict, root: Path, *, calibration=None, condition_probes=None,
                     initial_changes=None, verify_route_probes=True, deterministic_only=False, judgment_cache=None):
    if deterministic_only or not verify_route_probes:
        return _verify_execution(artifacts, root, calibration=calibration,
            condition_probes=condition_probes, initial_changes=initial_changes,
            verify_route_probes=verify_route_probes, deterministic_only=deterministic_only, judgment_cache=judgment_cache)
    from .validation_batch import verify_independent_routes
    return verify_independent_routes(artifacts, root, _verify_execution, prepare_route_probe,
        calibration=calibration, condition_probes=condition_probes, initial_changes=initial_changes, judgment_cache=judgment_cache)


def _verify_execution(artifacts: dict, root: Path, *, calibration: dict | None = None,
                     condition_probes: dict | None = None, initial_changes: list | None = None,
                     verify_route_probes: bool = True, deterministic_only: bool = False,
                     external_routes: bool = False, judgment_cache=None) -> dict:
    """Run actual tools/reward and independent ablations without trusting assertions."""
    from .execution_progress import ExecutionProgress
    progress = ExecutionProgress(root)
    from env_factory.sandbox_runtime import (AcceptanceScenarioRunner, ContractRewardAggregator,
        ContractRewardGate, ContractToolRegistry, DeclarativeMetricEvaluator, DeclarativeToolCompiler,
        EpisodeStore, ManifestDataStore, SandboxError, BusinessGoalEvaluator)
    (root / "preflight_failure.json").unlink(missing_ok=True)
    scenarios = artifacts["acceptance_contract"]["executable_scenarios"]
    successes = [s for s in scenarios if s.get("kind") == "goal_success"]
    failures = [s for s in scenarios if s.get("kind") == "goal_failure"]
    if not successes or not failures:
        raise ValueError("reference success and goal failure scenarios are required")
    with tempfile.TemporaryDirectory(prefix="envfactory-agent-check-") as directory:
        store = EpisodeStore(Path(directory) / "episode.sqlite3")
        manifest = artifacts["data_manifest"]
        data = ManifestDataStore(manifest, root / manifest["root"], store)
        if initial_changes:
            from .execution_routes import apply_initial_changes
            apply_initial_changes(data, initial_changes)
        from env_factory.sandbox_runtime import BusinessLifecycle, RewardStateSnapshots
        lifecycle = BusinessLifecycle(artifacts.get("business_lifecycle"), data, store,
                                      [t['function']['name'] for t in artifacts['tools']])
        registry = ContractToolRegistry(artifacts["tools"],
            DeclarativeToolCompiler(data).compile_all(artifacts["tool_implementations"]),
            event_recorder=store.event, tool_contracts=artifacts["task_spec"]["tool_contracts"], lifecycle=lifecycle,
            reward_snapshots=RewardStateSnapshots(data, artifacts["tool_implementations"],
                [artifacts["metric_implementations"], artifacts["task_spec"].get("goal_contract", {}),
                 artifacts["task_spec"].get("execution_routes", [])]))
        gate = ContractRewardGate(artifacts["task_spec"], artifacts["metrics"])
        evaluator = DeclarativeMetricEvaluator()
        from env_factory.sandbox_runtime import ContractModelMetricEvaluator
        from env_factory.generation.semantic_reward import calibrate_semantic_outcomes, SemanticCalibrationUnavailable
        model_evaluator = ContractModelMetricEvaluator(artifacts, store, reuse_across_resets=True, judgment_cache=judgment_cache)
        last_evaluated_scores = {}
        proof_limit = None
        latest_proof = {}
        class DeferredSemanticEvidence(Exception):
            """The cheap pass cannot decide this trajectory without a judge."""

        def evaluate_scores(context):
            scores = evaluator.evaluate_all(artifacts["metric_implementations"], context)
            latest_proof.clear()
            if deterministic_only:
                from .reward_proofs import reward_upper_bound
                proof = reward_upper_bound(artifacts['metrics'], scores, gate, context)
                if proof is None:
                    raise DeferredSemanticEvidence('unsupported reward bound')
                latest_proof.update(proof)
                last_evaluated_scores.clear()
                last_evaluated_scores.update(scores)
                return {**scores, **dict.fromkeys(proof['assumed_full_credit'], 1.0)}
            if proof_limit is not None:
                from .reward_proofs import reward_upper_bound
                proof = reward_upper_bound(artifacts['metrics'], scores, gate, context)
                if proof is not None and proof['assumed_full_credit']:
                    bound = proof['reward_upper_bound']
                    threshold, strict = proof_limit
                    proven = bound < threshold if strict else bound <= threshold
                    if proven:
                        latest_proof.update(proof)
                        last_evaluated_scores.clear()
                        last_evaluated_scores.update(scores)
                        return {**scores, **dict.fromkeys(proof['assumed_full_credit'], 1.0)}
            before = len(store.replay()["events"])
            with progress.measure("reward_evaluation"):
                scores.update(model_evaluator.evaluate_all(context, scores))
            (root / 'semantic_judgment_reuse.json').write_text(json.dumps(
                model_evaluator.runtime.validation_cache_report(), ensure_ascii=False, indent=2) + '\n')
            if any(event.get("event") == "evaluator_call" and
                   (event.get("payload", {}).get("used_fallback") or
                    event.get("payload", {}).get("mode") == "offline_fixture")
                   for event in store.replay()["events"][before:]):
                raise SemanticCalibrationUnavailable("SEMANTIC_CALIBRATION_UNAVAILABLE: reference judge unavailable or mocked")
            last_evaluated_scores.clear()
            last_evaluated_scores.update(scores)
            return scores
        aggregator = ContractRewardAggregator(artifacts["metrics"])
        from env_factory.sandbox_runtime import ContractUserSimulator
        manifest_users = artifacts['user_simulation_manifest']
        user_root = root / manifest_users['root']
        user = ContractUserSimulator(store,
            profiles=json.loads((user_root / manifest_users['profiles_file']).read_text()),
            scripts=json.loads((user_root / manifest_users['scripts_file']).read_text()),
            execution_routes=gate.routes, route_context=lambda: {
                'business_state': data._all(), 'initial_business_state': data.baseline,
                'reward_state_snapshots': store.get_state(RewardStateSnapshots.KEY, {})})
        def dispatch_call(method, path, body, headers):
            if path == "/v1/reset":
                store.reset(episode_id="authoring-check", seed=(body or {}).get("seed", 17))
                data.reset()
                user.reset()
                last_evaluated_scores.clear()
                result = {}
            elif path.startswith("/v1/tools/"):
                user.guard_tool(path.removeprefix("/v1/tools/"), body)
                result = registry.execute(path.removeprefix("/v1/tools/"), body,
                                          idem_key=headers.get('Idempotency-Key'))
            elif path == "/v1/user_simulator":
                result = user.turn(body["messages"])
            elif path == "/v1/agent_response":
                store.set_state("final_agent_response", body.get("content", ""))
                result = {}
            elif path == "/v1/reward":
                context = {"business_state": {n: data.table(n) for n in data.baseline},
                    "reward_state_snapshots": store.get_state(RewardStateSnapshots.KEY, {}),
                    "initial_business_state": data.baseline, "trajectory": store.replay(),
                    "final_agent_response": store.get_state("final_agent_response", "")}
                result = aggregator.aggregate(gate.apply(evaluate_scores(context), context))
            else:
                raise ValueError(f"unsupported preflight operation: {path}")
            return 200, result, {}
        request_faults = []
        def call(method, path, body, headers):
            # Match the service boundary: business errors are HTTP responses,
            # while interpreter/harness exceptions remain actual failures.
            request_faults.clear()
            event_count = len(store.replay()["events"]) if path.startswith('/v1/tools/') else None
            try:
                return dispatch_call(method, path, body, headers)
            except SandboxError as exc:
                if event_count is not None:
                    request_faults.extend(event['payload'] for event in store.replay()["events"][event_count:]
                                          if event.get('event') == 'injected_fault')
                return exc.status, exc.body("authoring-check"), {}
        user.completion_check = lambda: call("GET", "/v1/reward", None, {})[1]["reward"] >= 1 - 1e-9
        if deterministic_only:
            def deterministic_completion():
                bound = call('GET', '/v1/reward', None, {})[1]['reward']
                if bound < 1 - 1e-9:
                    return False
                if latest_proof.get('assumed_full_credit'):
                    raise DeferredSemanticEvidence('completion requires semantic judgment')
                return True
            def defer_renderer(payload):
                raise DeferredSemanticEvidence('user rendering requires model')
            user.completion_check = deterministic_completion
            user.renderer = defer_renderer
        runner = AcceptanceScenarioRunner(call,
            business_snapshot=lambda: {name: data.table(name) for name in data.baseline},
            injected_fault_check=lambda name, status:
            any(fault.get('tool') == name and fault.get('status') == status for fault in request_faults))
        def diagnostic():
            from env_factory.sandbox_runtime import ExpressionBusinessState
            snapshots = store.get_state(RewardStateSnapshots.KEY, {})
            state = ExpressionBusinessState({name: data.table(name) for name in data.baseline}, data.baseline, snapshots)
            response = store.get_state("final_agent_response", "")
            context = {"business_state": state, "initial_business_state": data.baseline,
                       "reward_state_snapshots": snapshots,
                       "trajectory": store.replay(), "final_agent_response": response}
            scores = evaluator.evaluate_all(artifacts["metric_implementations"], context)
            gate_diagnostics = {}
            result = {"ungated_components": scores,
                      "gated_components": gate.apply(scores, context, diagnostics=gate_diagnostics),
                      "gate_diagnostics": gate_diagnostics,
                      "answer_targets": [], "state_predicates": [], "metric_state_predicates": [],
                      "last_evaluated_components": dict(last_evaluated_scores),
                      "model_evidence": [event for event in store.replay()["events"]
                          if event.get("event") in {"evaluator_call", "evaluator_fact_mismatch"}]}
            try:
                parsed = json.loads(response)
            except (ValueError, TypeError):
                parsed = None
            for rule in artifacts["metric_implementations"]:
                if rule.get("operator") == "state_predicates":
                    for predicate in rule["expected"]:
                        result["metric_state_predicates"].append({
                            "metric_id": rule["metric_id"],
                            **state_predicate_diagnostic(predicate, state, data.baseline)})
                if rule.get("operator") not in {"value_targets", "numeric_targets"}:
                    continue
                for target in rule["expected"]["targets"]:
                    wanted = evaluator._value_expression(target["expression"], state)
                    from decimal import Decimal
                    result["answer_targets"].append({"metric_id": rule["metric_id"],
                        "field": target.get("key", target.get("label")),
                        "expected": wanted,
                        "expected_numeric_exact": str(wanted) if isinstance(wanted, Decimal) else None,
                        "absolute_tolerance": target.get('tolerance', 0),
                        "unresolved_lookups": unresolved_lookup_diagnostics(target["expression"], state),
                        "submitted": parsed.get(target["key"]) if isinstance(parsed, dict) and "key" in target else response})
            goal = artifacts["task_spec"].get("goal_contract", {})
            for predicate in goal.get("row_predicates", []):
                result["state_predicates"].append(state_predicate_diagnostic(predicate, state, data.baseline))
            if goal.get("row_predicates"):
                preservation_baseline = BusinessGoalEvaluator.external_baseline(data.baseline, context["trajectory"]["events"])
                result['preservation_violations'] = []
                result["unrelated_state_preserved"] = BusinessGoalEvaluator.preserves_unrelated(
                    goal, preservation_baseline, state, diagnostics=result['preservation_violations'])
            return result

        execution_outcome = {}
        def reward_impl(steps, *, negative=False, mutation=None):
            call("POST", "/v1/reset", next((step.get("body", {}) for step in steps if step.get("operation") == "reset"), {}), {})
            execution_outcome.clear()
            if mutation:
                rows = data.table(mutation['table'])
                for change in mutation.get('row_mutations', [mutation]):
                    rows[change['row_index']][mutation['column']] = change['mutated_value']
                # Reject invalid fixtures before executing the negative path;
                # schema/FK errors are not proof of a correct reward.
                data.replace_table(mutation['table'], rows)
                data.baseline[mutation['table']] = copy.deepcopy(rows)
            # Reset/reward are runner-owned so scenarios cannot select an earlier reward.
            body = copy.deepcopy([s for s in steps if s.get("operation") not in {"reset", "reward"}])
            if negative:
                # Removing a prerequisite legitimately leaves the user FSM in
                # an earlier stage. A positive-path stage assertion is not a
                # negative execution result.
                for step in body:
                    if step.get("operation") == "dialogue_turn":
                        step.pop("expected_stage", None)
                        step.pop("expected_branch", None)
            try:
                execution = runner.run({"steps": body, "assertions": []}, allow_untriggered_faults=negative)
            except SandboxError as exc:
                status = (exc.details or {}).get("actual_status") if isinstance(exc.details, dict) else None
                if negative and isinstance(status, int) and 400 <= status < 500:
                    execution_outcome.update(kind='client_rejection', status=status, details=exc.details)
                    return 0.0
                raise
            result = call("GET", "/v1/reward", {}, {})[1]
            from .realism_requirements import observe_realism
            execution_outcome['realism'] = observe_realism(store.replay(), body, execution['history'], scripts=user.scripts.values())
            if latest_proof:
                execution_outcome.update(kind='reward_upper_bound', proof=copy.deepcopy(latest_proof))
            else:
                execution_outcome.update(kind='completed', reward=result)
            return result["reward"]
        class ConditionProbeInconclusive(ValueError):
            pass

        def reward(steps, *, negative=False, mutation=None, proof=None):
            nonlocal proof_limit
            previous = proof_limit
            proof_limit = proof
            handler = None
            observed = []
            if mutation and mutation.get('query_where') is not None:
                name = mutation['tool']
                handler = registry.handlers[name]
                def checked_query(arguments):
                    # Dynamics have already run at this point. A scheduled
                    # update may undo a fixture mutation or create alternatives.
                    # Reestablish and verify an isolated query-time counterfactual.
                    # Harness inspection must not become an agent business read.
                    state = data._all()
                    where = evaluator._numeric_where(mutation['query_where'], state, depth=0)
                    eligible = [row for row in state[mutation['table']]
                                if evaluator._where_matches(row, where)]
                    from .report_values import report_value
                    observation = {'eligible_count': len(eligible),
                        'tool': name, 'arguments': copy.deepcopy(arguments),
                        'resolved_where': report_value(where),
                        'external_changes': [event for event in store.replay()['events']
                            if event.get('event') == 'external_business_change']}
                    observed.append(observation)
                    if eligible:
                        from .business_conditions import restore_condition_counterfactual
                        plan = restore_condition_counterfactual(mutation, data, store)
                        observation['query_time_intervention'] = plan
                        if plan['status'] != 'verified':
                            raise ConditionProbeInconclusive('query-time predicate cannot be isolated: ' + plan['reason'])
                        after = data._all()
                        current_where = evaluator._numeric_where(mutation['query_where'], after, depth=0)
                        observation['eligible_count_after'] = sum(
                            evaluator._where_matches(row, current_where) for row in after[mutation['table']])
                        if current_where != where or observation['eligible_count_after']:
                            raise ConditionProbeInconclusive('query-time intervention changed the operand or retained eligible alternatives')
                    return handler(arguments)
                registry.handlers[name] = checked_query
            try:
                value = reward_impl(steps, negative=negative, mutation=mutation)
                if handler is not None and not observed and execution_outcome.get('kind') != 'client_rejection':
                    raise ConditionProbeInconclusive('condition query was not executed')
                return value
            finally:
                if mutation:
                    execution_outcome['condition_observations'] = observed
                if handler is not None:
                    registry.handlers[mutation['tool']] = handler
                proof_limit = previous
        checks = {}
        ablation_failures = []
        condition_runs = []
        realism_observations = []
        reference_observations = []
        dynamic_choice_observations = []
        executed_interaction_behaviors = set()
        validated_runs = {}
        requirement_observations = {}
        if deterministic_only:
            from .deterministic_preflight import run_scenarios
            from .dynamic_choice_coverage import observe as observe_choices
            positive_probes = [{'scenario_id': 'positive-probe:' + pair['id'], 'kind': 'goal_success',
                                'steps': pair['steps'], 'assertions': []}
                               for pair in artifacts.get('reward_pair_probes', [])]
            return run_scenarios(successes + failures + positive_probes, artifacts['metrics'], runner, call,
                                 latest_proof, diagnostic, progress, root, DeferredSemanticEvidence,
                                 observe=lambda: observe_choices(store.replay(), user.scripts.values()))
        reference_failures = []
        for scenario in successes + failures:
            call("POST", "/v1/reset", {}, {})
            try:
                with progress.measure("reference_scenario", scenario_id=scenario.get('scenario_id')):
                    validated_runs[id(scenario)] = runner.run(scenario)
                if artifacts.get('requirement_coverage', {}).get('status') == 'structurally_valid':
                    requirement_observations[scenario['scenario_id']] = {
                        'kind': 'completed', 'reward': call('GET', '/v1/reward', {}, {})[1]}

            except SandboxError as exc:
                detail = {
                    "scenario_id": scenario.get("scenario_id"), "message": str(exc),
                    "code": exc.code, "details": exc.details}
                try:
                    detail["diagnostic"] = diagnostic()
                except (SandboxError, ValueError, TypeError, KeyError) as error:
                    detail["diagnostic_error"] = str(error)
                # Expressions may yield Decimal values; diagnostic serialization
                # must not mask the original scenario failure.
                from decimal import Decimal
                def diagnostic_json(value):
                    if isinstance(value, Decimal):
                        return int(value) if value == value.to_integral_value() else float(value)
                    raise TypeError(f"unsupported diagnostic value: {type(value).__name__}")
                encoded = json.dumps(detail, ensure_ascii=False, indent=2, default=diagnostic_json)
                (root / "preflight_failure.json").write_text(encoded + "\n")
                reference_failures.append(detail)
                progress.feedback('reference_scenario', scenario_id=scenario.get('scenario_id'), code=exc.code)
        coverage = artifacts.get('requirement_coverage', {'status': 'not_declared'})
        paired = None
        if not reference_failures and coverage.get('status') == 'structurally_valid':
            from .requirement_coverage import score_requirement_pairs
            paired = score_requirement_pairs(coverage, artifacts['metrics'], requirement_observations)
        if paired is not None and verify_route_probes:
            (root / 'requirement_reward_validation.json').write_text(
                json.dumps(paired, ensure_ascii=False, indent=2) + '\n')
            coverage['paired_execution_verified'] = paired['passed']
            coverage['reward_validation'] = paired
            if not paired['passed']:
                raise ValueError('REQUIREMENT_REWARD_MISMATCH: ' + json.dumps(
                    [p for p in paired['pairs'] if p['passed'] is not True], ensure_ascii=False))
        if calibration and not reference_failures:
            from .semantic_reward import scope_calibration_references
            calibration, selected_references = scope_calibration_references(calibration, successes)
            contexts, executed_contexts = {}, {}
            for name, scenario in selected_references.items():
                scenario_id = scenario['scenario_id']
                if scenario_id not in executed_contexts:
                    reference_reset = next((step.get("body", {}) for step in scenario["steps"]
                                            if step.get("operation") == "reset"), {})
                    call("POST", "/v1/reset", reference_reset, {})
                    runner.run({"steps": [step for step in scenario["steps"]
                        if step.get("operation") not in {"reset", "reward"}], "assertions": []})
                    executed_contexts[scenario_id] = copy.deepcopy({
                        "business_state": {name: data.table(name) for name in data.baseline},
                        "reward_state_snapshots": store.get_state(RewardStateSnapshots.KEY, {}),
                        "initial_business_state": data.baseline, "trajectory": store.replay()})
                contexts[name] = executed_contexts[scenario_id]
            calibration_store = EpisodeStore(Path(directory) / "semantic-calibration.sqlite3")
            calibration_store.reset(episode_id="semantic-calibration", seed=17)
            try:
                report = calibrate_semantic_outcomes(artifacts, calibration, context=contexts[None],
                                                    reference_contexts=contexts,
                                                    store=calibration_store, progress=progress, judgment_cache=judgment_cache,
                                                    workers=2)
            finally:
                (root / "semantic_calibration_replay.json").write_text(json.dumps(
                    calibration_store.replay(), ensure_ascii=False, indent=2) + "\n")
            report['reference_scenarios'] = {str(name): scenario['scenario_id']
                                             for name, scenario in selected_references.items()}
            (root / "semantic_calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            if report.get('conflicting_judgments'):
                raise SemanticCalibrationUnavailable('SEMANTIC_JUDGMENT_CONFLICT: ' + json.dumps(
                    report['conflicting_judgments'], ensure_ascii=False))
            if not report["passed"]:
                raise ValueError("SEMANTIC_CALIBRATION_FAILED: " + json.dumps(report, ensure_ascii=False))
        if reference_failures:
            from .report_values import report_value
            encoded = json.dumps(report_value(reference_failures), ensure_ascii=False)
            (root / 'preflight_failures.json').write_text(encoded + '\n')
            raise ValueError('AGENT_SCENARIO_FAILED: ' + encoded)
        for i, scenario in enumerate(successes):
            steps = scenario["steps"]
            checks[f"reference_{i}"] = reward(steps) >= 1 - 1e-9
            if checks[f"reference_{i}"]:
                from .dynamic_choice_coverage import observe as observe_choices
                dynamic_choice_observations.extend({**item, 'scenario_id': scenario.get('scenario_id')}
                    for item in observe_choices(store.replay(), user.scripts.values()))
                from .interaction_diversity import executed_behavior_signature
                signature = executed_behavior_signature(store.replay()['events'], user.scripts.values())
                if signature is not None:
                    executed_interaction_behaviors.add(signature)
                realism_observations.append(execution_outcome['realism'])
                from .report_values import report_value
                observed = diagnostic()
                reference_observations.append(report_value({
                    'scenario_id': scenario.get('scenario_id'),
                    'reward': execution_outcome.get('reward'),
                    'state_predicates': observed['state_predicates'],
                    'metric_state_predicates': observed['metric_state_predicates'],
                    'external_changes': [event for event in store.replay()['events']
                                         if event.get('event') == 'external_business_change'],
                    'unrelated_state_preserved': observed.get('unrelated_state_preserved')}))
            if gate.routes.routes:
                actual_route = execution_outcome.get('reward', {}).get('execution_route')
                checks[f"reference_route_{i}"] = actual_route == scenario.get('execution_route') and actual_route is not None
            bound_steps = _bind_reference_for_ablations(steps, validated_runs[id(scenario)])
            selected_route = next((route for route in gate.routes.routes
                                   if route['id'] == scenario.get('execution_route')), None)
            for probe in (condition_probes or {}).get('cases', []):
                if selected_route and probe.get('tool') not in selected_route['required_tools']:
                    continue
                references = probe.get('reference_ids')
                if references is not None and not any(
                        scenario['scenario_id'] == name or scenario['scenario_id'].startswith(name + '-script-')
                        for name in references):
                    continue
                if probe['status'] != 'verified':
                    continue
                trial = copy.deepcopy(bound_steps)
                for step in trial:
                    # The wrong policy reuses its old choices despite changed
                    # results; absent captures must not crash the test harness.
                    step.pop('capture', None)
                original_baseline = copy.deepcopy(data.baseline)
                evidence = {'scenario_id': scenario.get('scenario_id'), 'mutation': probe}
                try:
                    score = reward(trial, negative=True, mutation=probe, proof=(1 - 1e-9, True))
                    key = f"business_condition_{i}_{probe['condition_id']}"
                    checks[key] = score < 1 - 1e-9
                    evidence.update(status='passed' if checks[key] else 'failed',
                                    execution=copy.deepcopy(execution_outcome), replay=store.replay())
                except ConditionProbeInconclusive as exc:
                    checks[f"business_condition_{i}_{probe['condition_id']}"] = False
                    evidence.update(status='needs_authored_case', reason=str(exc),
                                    execution=copy.deepcopy(execution_outcome), replay=store.replay())
                except SandboxError as exc:
                    if exc.code != 'DATA_INVALID':
                        raise
                    checks[f"business_condition_{i}_{probe['condition_id']}"] = False
                    evidence.update(status='needs_authored_case', reason='mutation violates data constraints',
                                    error=str(exc))
                finally:
                    data.baseline = original_baseline
                condition_runs.append(evidence)
                (root / 'business_condition_execution.json').write_text(
                    json.dumps(condition_runs, ensure_ascii=False, indent=2) + '\n')
            empty = [s for s in steps if s.get("operation") not in {"tool_call", "agent_response", "dialogue_turn"}]
            checks[f"empty_{i}"] = reward(empty) <= .2
            for j, step in enumerate(steps):
                if step.get('operation') == 'dialogue_turn':
                    checks[f'skip_interaction_{i}_{j}'] = reward(bound_steps[:j] + bound_steps[j+1:], negative=True, proof=(1 - 1e-9, True)) < 1 - 1e-9
            if artifacts["training_category"] != "direct_response":
                if not (selected_route and selected_route['id'] in gate.routes.dialogue_stop_ids
                        and not selected_route['required_tools']):
                    checks[f"no_tools_{i}"] = reward([s for s in bound_steps if s.get("operation") != "tool_call"], negative=True, proof=(.2, False)) <= .2
                from env_factory.contracts.trajectory_probes import idempotent_call_groups, optional_observation_positions
                optional = optional_observation_positions(steps, artifacts["tool_implementations"])
                for group in idempotent_call_groups(steps):
                    omitted = set(group)
                    is_optional = omitted <= optional
                    ablated_reward = reward(
                        [step for index, step in enumerate(bound_steps) if index not in omitted],
                        negative=not is_optional, proof=None if is_optional else (.2, False))
                    checks[f"skip_{i}_{group[0]}"] = (ablated_reward >= 1 - 1e-9 if is_optional else ablated_reward <= .2)
                    if not checks[f"skip_{i}_{group[0]}"]:
                        ablation_failures.append({
                            'check': f'skip_{i}_{group[0]}', 'scenario_id': scenario.get('scenario_id'),
                            'execution_route': scenario.get('execution_route'),
                            'omitted_step_indexes': sorted(omitted),
                            'omitted_steps': [steps[index] for index in sorted(omitted)],
                            'actual_reward': ablated_reward, 'maximum_allowed_reward': .2,
                            'message': 'Removing this logical request retained reward. Check redundant calls '
                                       'and whether the intended business dependency is actually enforced. '
                                       'Do not pad a recovery route with duplicate reads just to trigger a fault.'})
            from env_factory.contracts.reward_contract import terminal_outcome_weight
            answer_weight = terminal_outcome_weight(artifacts)
            if answer_weight > 0:
                wrong = copy.deepcopy(steps)
                for step in wrong:
                    if step.get("operation") == "agent_response":
                        step["content"] = "无法确定；答案未知。"
                wrong_reward = reward(wrong)
                checks[f"wrong_answer_{i}"] = wrong_reward < 1 - 1e-9 and 1 - wrong_reward >= .5 * answer_weight - 1e-9
        component_failures = []
        for i, scenario in enumerate(failures):
            bound_pairs = [pair for pair in (paired or {}).get('pairs', [])
                           if pair['polarity'] == 'negative' and pair['scenario_id'] == scenario['scenario_id']]
            if bound_pairs:
                # A single incorrect obligation can coexist with correctly scored
                # outcomes. Require fresh component evidence for every binding,
                # and reject a full aggregate reward; do not force all outcomes to zero.
                observed = requirement_observations[scenario['scenario_id']]['reward']['reward']
                checks[f"authored_failure_{i}"] = (
                    all(pair['passed'] is True for pair in bound_pairs)
                    and observed < 1 - 1e-9)
            else:
                from .scenario_assertions import measured_component_failure
                component_failure = measured_component_failure(
                    scenario, validated_runs[id(scenario)], artifacts['metrics'])
                if component_failure is not None:
                    component_failures.append(component_failure)
                    checks[f"authored_failure_{i}"] = component_failure['passed']
                else:
                    checks[f"authored_failure_{i}"] = reward(scenario["steps"], negative=True, proof=(.2, False)) <= .2
        if component_failures:
            (root / 'component_negative_validation.json').write_text(
                json.dumps(component_failures, ensure_ascii=False, indent=2) + '\n')
        if verify_route_probes and 'reward_pair_probes' in artifacts:
            from .reward_pairs import run_reward_pairs
            def execute_pair(steps):
                value = reward(steps)
                return {'reward': value, 'state': copy.deepcopy({name: data.table(name) for name in data.baseline})}
            paired = run_reward_pairs(artifacts['reward_pair_probes'], successes, execute_pair)
            (root / 'reward_pair_validation.json').write_text(
                json.dumps(paired, ensure_ascii=False, indent=2) + '\n')
            artifacts['reward_pair_validation'] = paired
            if paired['status'] != 'not_applicable':
                checks['equivalent_and_recovery_pairs'] = paired['passed']
            if not paired['passed'] and paired['status'] != 'not_applicable':
                raise ValueError('REWARD_PAIR_MISMATCH: ' + json.dumps(
                    [item for item in paired['results'] if not item['passed']], ensure_ascii=False))
        if not all(checks.values()):
            condition_failures = [run for run in condition_runs if run['status'] != 'passed']
            if ablation_failures or condition_failures:
                (root / 'preflight_ablation_failure.json').write_text(
                    json.dumps(ablation_failures, ensure_ascii=False, indent=2) + '\n')
                raise ValueError('AGENT_PREFLIGHT_FAILED: ' + json.dumps({
                    'failed_checks': [k for k, v in checks.items() if not v],
                    'ablations': ablation_failures,
                    'business_conditions': condition_failures}, ensure_ascii=False))
            raise ValueError("AGENT_PREFLIGHT_FAILED: " + ", ".join(k for k, v in checks.items() if not v))
        route_coverage = {'references': sorted({s.get('execution_route') for s in successes if s.get('execution_route')}),
                          'probes': []}
        if verify_route_probes and not external_routes:
            raise RuntimeError('full validation must use verify_execution orchestration')
        return {"passed": True, "checks": checks, "scope": "reference_and_policy_ablations",
                "semantic_judgment_reuse": model_evaluator.runtime.validation_cache_report(),
                "distinct_executed_interaction_behaviors": len(executed_interaction_behaviors),
                "realism_observations": realism_observations,
                "reference_observations": reference_observations,
                "dynamic_choice_observations": dynamic_choice_observations,
                "execution_route_coverage": route_coverage,
                "business_condition_execution": [{key: value for key, value in run.items() if key != 'replay'}
                                                 for run in condition_runs]}


def _archive_delivery(root: Path, category: str) -> None:
    paths = [root / name for name in
             ('task.json', 'compiled_artifacts.json', 'status.json', 'buildability.json')
             if (root / name).exists()]
    if not paths:
        return
    archive_root = root / category
    archive_root.mkdir(parents=True, exist_ok=True)
    archive = Path(tempfile.mkdtemp(prefix='delivery-', dir=archive_root))
    for path in paths:
        path.replace(archive / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--blueprint", type=Path, help="Accepted business blueprint; enforce implementation consistency")
    parser.add_argument("--output", type=Path, required=True)
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument("--structural-preview", action="store_true",
                        help="Check structure without online judges; does not certify or export a training task")
    preview.add_argument("--deterministic-preview", action="store_true",
                        help="Check structure and execute model-free route diagnostics; never certifies a task")
    args = parser.parse_args()
    is_preview = args.structural_preview or args.deterministic_preview
    if is_preview and any((args.output / name).exists()
            for name in ("task.json", "compiled_artifacts.json", "status.json")):
        parser.error("structural preview requires an output directory without delivery artifacts; choose a separate directory")
    if not is_preview:
        # A failed rerun must not leave the previous successful delivery in place.
        # Retain those files for diagnosis, outside the active output paths.
        _archive_delivery(args.output, '.previous_deliveries')
    request = json.loads(args.request.read_text())
    result = compile_source(json.loads(args.source.read_text()), root=args.output,
                            request=request, script_count=request.get('user_script_count', 3), structural_preview=is_preview,
                            blueprint=json.loads(args.blueprint.read_text()) if args.blueprint else None)
    if is_preview:
        report = result["generation_pipeline"]["verification"]
        (args.output / "structural_preview.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if args.deterministic_preview:
            from .deterministic_preflight import check_all_routes
            report = check_all_routes(result, args.output, verify_execution, prepare_route_probe)
        print(json.dumps(report, ensure_ascii=False))
        return
    from env_factory.generation.delivery_preflight import verify_delivery
    verify_delivery(result, args.output)
    from env_factory.generation.artifacts import write_task_artifact
    from env_factory.graph.knowledge_graph import TaskType
    from env_factory.tasks.task import Task
    try:
        write_task_artifact(args.output, Task(desc=result["task"], env=result["environment"],
            metrics=result["metrics"], task_type=TaskType(result["task_type"]),
            task_intent=result["task_intent"], complexity=result["complexity"], artifacts=result),
            result["training_category"])
        (args.output / "compiled_artifacts.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve().parents[3] /
            "scripts/sandbox/assess_task_buildability.py"), "--root", str(args.output),
            "--output", str(args.output / "buildability.json")])
        if completed.returncode:
            raise SystemExit(completed.returncode)
    except BaseException:
        # Keep failure evidence, but never leave a rejected candidate in the
        # active delivery paths consumed by the sandbox builder.
        _archive_delivery(args.output, '.failed_deliveries')
        raise
    print(json.dumps(result["generation_pipeline"]["verification"], ensure_ascii=False))


if __name__ == "__main__":
    main()
