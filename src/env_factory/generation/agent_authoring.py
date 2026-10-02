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

from env_factory.task_pipeline import TaskGenerationPipeline as P
from env_factory.tasks.task_routing import training_contract
from env_factory.tasks.task_spec import compile_task_spec
from env_factory.generation.answer_contract import bind_answer_contract, validate_direct_text_targets
from env_factory.generation.reward_bindings import (bind_reward_queries, reject_private_identifier_constants,
    validate_reward_observability)
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
    source = copy.deepcopy(source)
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
    for call in calls:
        name = call["tool_name"]
        deps = dependencies.setdefault(name, [])
        for reference in refs(call.get("arguments", {})):
            if reference not in producers:
                raise ValueError(f"reference call uses unknown capture {reference}")
            if producers[reference] != name and producers[reference] not in deps:
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
                weight=(.8 if business else 1) / len(outcomes))
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
    success_steps = [copy.deepcopy(reset), *[{**call, "operation": "tool_call", "expected_status": 200} for call in calls],
        {"operation": "agent_response", "content": source.pop("reference")["answer"]},
        {"operation": "reward", "step_id": "reward"}]
    scenarios = [{"scenario_id": "reference_success", "kind": "goal_success", "steps": success_steps,
        "assertions": [{"source": "step:reward", "path": "$.reward", "operator": "gte", "expected": 1}]},
        {"scenario_id": "empty_failure", "kind": "goal_failure", "steps": [copy.deepcopy(reset),
         {"operation": "reward", "step_id": "reward"}], "assertions": [
            {"source": "step:reward", "path": "$.reward", "operator": "lte", "expected": 0}]}]
    scenarios.extend(source.pop("negative_scenarios", []))
    source.update(tools=tools, tool_implementations=implementations, actions=actions,
        tool_bindings=bindings, capability_plan=capabilities, reward_key_steps=key_steps,
        metrics=metrics, metric_implementations=metric_impl, scenarios=scenarios)
    return source


def compile_source(source: dict, *, root: Path, request: dict, script_count: int = 3,
                   structural_preview: bool = False) -> dict:
    """Compile a complete business design against runner-owned graph/route input."""
    source = expand_source(source, request)
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
                and set(rule["expected"]) & {"from_tool", "lookup", "op", "initial", "if"}):
            raise ValueError("STATE_REWARD_EXPRESSION: ordinary comparisons use literal expected values; "
                "use operator state_predicates, source business_state, path $, and expected row predicates "
                "with value_expressions for dynamic state targets")
    validate_reward_observability(source)
    query_bindings = bind_reward_queries(source)
    from env_factory.tasks.task_spec import validate_goal_contract
    for rule in source["metric_implementations"]:
        if rule.get("operator") == "state_predicates":
            validate_goal_contract({"row_predicates": rule["expected"]}, source["tables"], require_change=False)
    identifier_public_input = copy.deepcopy(source["description"]["public_input"])
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
    from .interaction_contract import compile_interactions, interleave_scenarios
    interaction_scripts = (compile_interactions(source['interaction_contract'], description, tools, script_count, implementations=implementations, scenarios=scenarios)
                           if source.get('interaction_contract') else None)
    reject_private_identifier_constants(source, validated_scripts=interaction_scripts or (),
                                        public_input=identifier_public_input)
    root.mkdir(parents=True, exist_ok=True)
    manifest = P._materialize_business_data(tables, P._render_data_document(tables),
        root / "data/business_data", environment_mode=plan["mode"], manifest_root="data/business_data")
    manifest["data_governance"]["origin"] = "programmatically_generated_synthetic"
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
    if interaction_scripts:
        spec['requires_user_interaction'] = True
        spec['interaction_complexity'] = {
            'variants':len(interaction_scripts),
            'required_user_turns':[len(script['interaction_protocol']['stages']) for script in interaction_scripts],
            'kinds':sorted({stage['kind'] for script in interaction_scripts for stage in script['interaction_protocol']['stages']}),
            'runtime':'deterministic_task_protocol'}
        acceptance['executable_scenarios'] = interleave_scenarios(acceptance['executable_scenarios'], interaction_scripts)
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
        "environment_plan": plan, "environment": P._environment_records({}, actions),
        "data_manifest": manifest, "user_simulation_manifest": users,
        "tools_manifest": P._materialize_tools(tools, root),
        "media_generation": {"required": False, "language": "python", "code": "", "dependencies": [], "entrypoint": "", "output_dir": ""},
        "actions": actions, "capability_plan": capabilities, "tools": tools, "tool_bindings": bindings,
        "tool_implementations": implementations, "noise_tools": [], "reward_key_steps": key_steps,
        "observation_schema": {"type": "object", "properties": {}}, "metrics": metrics,
        "metric_implementations": metric_impl, "reward_formula": formula, "acceptance_contract": acceptance,
        "graph_context": copy.deepcopy(request["graph_context"])}
    validate_capture_paths(scenarios, spec["tool_contracts"])
    if structural_preview:
        # Author-side sandbox previews cannot rely on evaluator network access.
        # These artifacts must never masquerade as an accepted training task.
        result["task_readiness"] = {"ready": False, "errors": ["EXECUTION_VALIDATION_PENDING"], "warnings": []}
        result["generation_pipeline"] = {"backend": "code_agent", "structural_preview": True,
            "live_rollout_verified": False, "verification": {
                "passed": False, "structural_checks_passed": True,
                "scope": "structural_preview_only", "pending": [
                    "reference_execution", "policy_ablations", "semantic_calibration",
                    "delivery_preflight", "independent_semantic_review"]}}
        return result
    checks = verify_execution(result, root, calibration=source.get("semantic_calibration", {}))
    result["task_readiness"] = {"ready": True, "errors": [], "warnings": [],
        "training_profile": P._derive_training_profile(business_tool_count=len(bindings),
            noise_tool_count=0, key_step_count=len(key_steps))}
    result["generation_pipeline"] = {"version": "1.0", "backend": "code_agent",
        "reward_query_bindings": query_bindings,
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
            if set(node) == {"initial"}:
                from env_factory.sandbox_runtime import ExpressionBusinessState
                baseline = getattr(state, "initial", {})
                walk(node["initial"], path + ".initial", ExpressionBusinessState(baseline, baseline))
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
    def walk(value):
        if isinstance(value, dict):
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


def verify_execution(artifacts: dict, root: Path, *, calibration: dict | None = None) -> dict:
    """Run actual tools/reward and independent ablations without trusting assertions."""
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
        registry = ContractToolRegistry(artifacts["tools"],
            DeclarativeToolCompiler(data).compile_all(artifacts["tool_implementations"]),
            event_recorder=store.event, tool_contracts=artifacts["task_spec"]["tool_contracts"])
        gate = ContractRewardGate(artifacts["task_spec"], artifacts["metrics"])
        evaluator = DeclarativeMetricEvaluator()
        from env_factory.sandbox_runtime import ContractModelMetricEvaluator
        from env_factory.generation.semantic_reward import calibrate_semantic_outcomes, SemanticCalibrationUnavailable
        model_evaluator = ContractModelMetricEvaluator(artifacts, store)
        last_evaluated_scores = {}
        def evaluate_scores(context):
            scores = evaluator.evaluate_all(artifacts["metric_implementations"], context)
            before = len(store.replay()["events"])
            scores.update(model_evaluator.evaluate_all(context, scores))
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
            scripts=json.loads((user_root / manifest_users['scripts_file']).read_text()))
        def dispatch_call(method, path, body, headers):
            if path == "/v1/reset":
                store.reset(episode_id="authoring-check", seed=(body or {}).get("seed", 17))
                data.reset()
                user.reset()
                last_evaluated_scores.clear()
                result = {}
            elif path.startswith("/v1/tools/"):
                user.guard_tool(path.removeprefix("/v1/tools/"))
                result = registry.execute(path.removeprefix("/v1/tools/"), body)
            elif path == "/v1/user_simulator":
                result = user.turn(body["messages"])
            elif path == "/v1/agent_response":
                store.set_state("final_agent_response", body.get("content", ""))
                result = {}
            elif path == "/v1/reward":
                context = {"business_state": {n: data.table(n) for n in data.baseline},
                    "initial_business_state": data.baseline, "trajectory": store.replay(),
                    "final_agent_response": store.get_state("final_agent_response", "")}
                result = aggregator.aggregate(gate.apply(evaluate_scores(context), context))
            else:
                raise ValueError(f"unsupported preflight operation: {path}")
            return 200, result, {}
        def call(method, path, body, headers):
            # Match the service boundary: business errors are HTTP responses,
            # while interpreter/harness exceptions remain actual failures.
            try:
                return dispatch_call(method, path, body, headers)
            except SandboxError as exc:
                return exc.status, exc.body("authoring-check"), {}
        user.completion_check = lambda: call("GET", "/v1/reward", None, {})[1]["reward"] >= 1 - 1e-9
        runner = AcceptanceScenarioRunner(call)
        def diagnostic():
            from env_factory.sandbox_runtime import ExpressionBusinessState
            state = ExpressionBusinessState({name: data.table(name) for name in data.baseline}, data.baseline)
            response = store.get_state("final_agent_response", "")
            context = {"business_state": state, "initial_business_state": data.baseline,
                       "trajectory": store.replay(), "final_agent_response": response}
            scores = evaluator.evaluate_all(artifacts["metric_implementations"], context)
            result = {"ungated_components": scores, "gated_components": gate.apply(scores, context),
                      "answer_targets": [], "state_predicates": [],
                      "last_evaluated_components": dict(last_evaluated_scores),
                      "model_evidence": [event for event in store.replay()["events"]
                          if event.get("event") in {"evaluator_call", "evaluator_fact_mismatch"}]}
            try:
                parsed = json.loads(response)
            except (ValueError, TypeError):
                parsed = None
            for rule in artifacts["metric_implementations"]:
                if rule.get("operator") not in {"value_targets", "numeric_targets"}:
                    continue
                for target in rule["expected"]["targets"]:
                    result["answer_targets"].append({"metric_id": rule["metric_id"],
                        "field": target.get("key", target.get("label")),
                        "expected": evaluator._value_expression(target["expression"], state),
                        "unresolved_lookups": unresolved_lookup_diagnostics(target["expression"], state),
                        "submitted": parsed.get(target["key"]) if isinstance(parsed, dict) and "key" in target else response})
            goal = artifacts["task_spec"].get("goal_contract", {})
            for predicate in goal.get("row_predicates", []):
                actual = [row for row in state.get(predicate["table"], [])
                          if all(row.get(key) == value for key, value in predicate.get("where", {}).items())]
                result["state_predicates"].append({"table": predicate["table"], "where": predicate["where"],
                    "expected_values": BusinessGoalEvaluator.expected_values(predicate, state, data.baseline),
                    "actual_rows": actual[:5], "matched_row_count": len(actual),
                    "satisfied": BusinessGoalEvaluator.evaluate([predicate], state, data.baseline)})
            if goal.get("row_predicates"):
                result["unrelated_state_preserved"] = BusinessGoalEvaluator.preserves_unrelated(goal, data.baseline, state)
            return result

        def reward(steps, *, negative=False):
            call("POST", "/v1/reset", next((step.get("body", {}) for step in steps if step.get("operation") == "reset"), {}), {})
            # Reset/reward are runner-owned so scenarios cannot select an earlier reward.
            body = copy.deepcopy([s for s in steps if s.get("operation") not in {"reset", "reward"}])
            if negative:
                # Removing a prerequisite legitimately leaves the user FSM in
                # an earlier stage. A positive-path stage assertion is not a
                # negative execution result.
                for step in body:
                    if step.get("operation") == "dialogue_turn":
                        step.pop("expected_stage", None)
            try:
                runner.run({"steps": body, "assertions": []})
            except SandboxError as exc:
                status = (exc.details or {}).get("actual_status") if isinstance(exc.details, dict) else None
                if negative and isinstance(status, int) and 400 <= status < 500:
                    return 0.0
                raise
            return call("GET", "/v1/reward", {}, {})[1]["reward"]
        checks = {}
        validated_runs = {}
        for scenario in successes + failures:
            call("POST", "/v1/reset", {}, {})
            try:
                validated_runs[id(scenario)] = runner.run(scenario)
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
                raise ValueError("AGENT_SCENARIO_FAILED: " + encoded) from exc
        if calibration:
            reference_reset = next((step.get("body", {}) for step in successes[0]["steps"]
                                    if step.get("operation") == "reset"), {})
            call("POST", "/v1/reset", reference_reset, {})
            runner.run({"steps": [step for step in successes[0]["steps"]
                if step.get("operation") not in {"reset", "reward"}], "assertions": []})
            context = {"business_state": {name: data.table(name) for name in data.baseline},
                "initial_business_state": data.baseline, "trajectory": store.replay()}
            calibration_store = EpisodeStore(Path(directory) / "semantic-calibration.sqlite3")
            calibration_store.reset(episode_id="semantic-calibration", seed=17)
            report = calibrate_semantic_outcomes(artifacts, calibration, context=context, store=calibration_store)
            (root / "semantic_calibration.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            (root / "semantic_calibration_replay.json").write_text(json.dumps(calibration_store.replay(), ensure_ascii=False, indent=2) + "\n")
            if not report["passed"]:
                raise ValueError("SEMANTIC_CALIBRATION_FAILED: " + json.dumps(report, ensure_ascii=False))
        for i, scenario in enumerate(successes):
            steps = scenario["steps"]
            checks[f"reference_{i}"] = reward(steps) >= 1 - 1e-9
            bound_steps = copy.deepcopy(steps)
            variables = validated_runs[id(scenario)]["variables"]
            for step in bound_steps:
                if step.get("operation") == "tool_call":
                    step["arguments"] = runner._resolve(step.get("arguments", {}), variables)
            empty = [s for s in steps if s.get("operation") not in {"tool_call", "agent_response", "dialogue_turn"}]
            checks[f"empty_{i}"] = reward(empty) <= .2
            for j, step in enumerate(steps):
                if step.get('operation') == 'dialogue_turn':
                    checks[f'skip_interaction_{i}_{j}'] = reward(bound_steps[:j] + bound_steps[j+1:], negative=True) < 1 - 1e-9
            if artifacts["training_category"] != "direct_response":
                checks[f"no_tools_{i}"] = reward([s for s in bound_steps if s.get("operation") != "tool_call"], negative=True) <= .2
                for j, step in enumerate(steps):
                    if step.get("operation") == "tool_call":
                        checks[f"skip_{i}_{j}"] = reward(bound_steps[:j] + bound_steps[j+1:], negative=True) <= .2
            from env_factory.contracts.reward_contract import terminal_outcome_weight
            answer_weight = terminal_outcome_weight(artifacts)
            if answer_weight > 0:
                wrong = copy.deepcopy(steps)
                for step in wrong:
                    if step.get("operation") == "agent_response":
                        step["content"] = "无法确定；答案未知。"
                wrong_reward = reward(wrong)
                checks[f"wrong_answer_{i}"] = wrong_reward < 1 - 1e-9 and 1 - wrong_reward >= .5 * answer_weight - 1e-9
        for i, scenario in enumerate(failures):
            checks[f"authored_failure_{i}"] = reward(scenario["steps"], negative=True) <= .2
        if not all(checks.values()):
            raise ValueError("AGENT_PREFLIGHT_FAILED: " + ", ".join(k for k, v in checks.items() if not v))
        return {"passed": True, "checks": checks, "scope": "reference_and_policy_ablations"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structural-preview", action="store_true",
                        help="Check structure without online judges; does not certify or export a training task")
    args = parser.parse_args()
    if args.structural_preview and any((args.output / name).exists()
            for name in ("task.json", "compiled_artifacts.json", "status.json")):
        parser.error("structural preview requires an output directory without delivery artifacts; choose a separate directory")
    result = compile_source(json.loads(args.source.read_text()), root=args.output,
                            request=json.loads(args.request.read_text()), structural_preview=args.structural_preview)
    if args.structural_preview:
        report = result["generation_pipeline"]["verification"]
        (args.output / "structural_preview.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False))
        return
    from env_factory.generation.delivery_preflight import verify_delivery
    verify_delivery(result, args.output)
    from env_factory.generation.artifacts import write_task_artifact
    from env_factory.graph.knowledge_graph import TaskType
    from env_factory.tasks.task import Task
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
    print(json.dumps(result["generation_pipeline"]["verification"], ensure_ascii=False))


if __name__ == "__main__":
    main()
