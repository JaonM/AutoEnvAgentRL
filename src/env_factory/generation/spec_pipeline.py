"""Constructive multi-step generation: one executable specification, no repair lottery.

The versioned prototype owns business semantics. Sampled state, tool contracts,
reference execution and outcome predicates are compiled from it. Natural language
is rendered last and never introduces additional facts or goals.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

from .pipeline_errors import PipelineGenerationError

PROTOTYPES = ("lookup_join_sum", "lookup_update", "constraint_create")
SUM_GOAL = {"label": "采购总数量", "unit": "件", "aggregate_table": "entries",
            "field": "quantity", "join_field": "account_id", "lookup_table": "accounts",
            "public_field": "name"}


def instantiate(*, seed: int, subject: str = "办公用品", prototype: str = "lookup_join_sum") -> dict[str, Any]:
    if prototype != "lookup_join_sum":
        from .stateful_spec_pipeline import instantiate as instantiate_stateful
        return instantiate_stateful(seed=seed, prototype=prototype, subject=subject)
    rng = random.Random(seed)
    ids = rng.sample(range(100000, 999999), 3)
    tables = [
        {"table_name": "accounts", "description": "项目名称与内部台账编号的映射", "primary_key": ["account_id"],
         "columns": [{"name": "account_id", "type": "integer", "nullable": False},
                     {"name": "name", "type": "text", "nullable": False}],
         "foreign_keys": [], "indexes": [], "constraints": [],
         "rows": [{"account_id": key, "name": f"{subject}采购项目{index + 1}"} for index, key in enumerate(ids)]},
        {"table_name": "entries", "description": "各项目的内部采购数量明细", "primary_key": ["entry_id"],
         "columns": [{"name": "entry_id", "type": "integer", "nullable": False},
                     {"name": "account_id", "type": "integer", "nullable": False},
                     {"name": "quantity", "type": "integer", "nullable": False}],
         "foreign_keys": [{"column": "account_id", "ref_table": "accounts", "ref_column": "account_id"}],
         "indexes": [], "constraints": [], "rows": []},
    ]
    for key in ids:
        for _ in range(rng.randint(3, 7)):
            tables[1]["rows"].append({"entry_id": len(tables[1]["rows"]) + 1,
                                      "account_id": key, "quantity": rng.randint(11, 199)})
    target = rng.choice(tables[0]["rows"])["name"]
    return {"version": "1.0", "prototype": "lookup_join_sum", "seed": seed,
            "subject": subject, "selector": target, "tables": tables,
            "goal": copy.deepcopy(SUM_GOAL)}


def _tool(name: str, description: str, arguments: dict[str, str]) -> dict:
    descriptions = {
        "name": "用户提供的项目或申请名称", "account_id": "项目查询返回的内部台账编号",
        "id": "查询返回的待更新申请编号", "status": "用户要求的新状态",
        "request_id": "申请查询返回的 id", "supplier_id": "供应商查询返回的 id",
        "quantity": "申请查询返回的采购数量", "min_capacity": "所需的最低可供数量，来自申请 quantity",
        "max_price": "允许的最高单价，来自申请 max_price",
    }
    return {"type": "function", "function": {"name": name, "description": description,
        "parameters": {"type": "object", "properties": {
            key: {"type": kind, "description": descriptions[key]}
            for key, kind in arguments.items()}, "required": list(arguments), "additionalProperties": False}}}


def _metric(identifier: str, category: str, weight: float, assertion: str, action: str = "") -> dict:
    result = {"id": identifier, "category": category, "type": "rule-based",
              "scope": "terminal" if category == "outcome" else "step",
              "weight": weight, "score_range": [0, 1], "rubric": assertion,
              "condition": assertion, "evaluator": {"kind": "document_rule" if category == "outcome" else "trajectory_rule",
              "source": "runtime_rule", "assertion": assertion, "score_mapping": {"pass": 1, "fail": 0}}}
    if action:
        result["target_action"] = action
    return result


def compile_spec(spec: dict, *, artifact_dir: Path, script_count: int = 3) -> dict:
    from env_factory.task_pipeline import TaskGenerationPipeline as P
    from env_factory.sandbox_runtime import DeclarativeMetricEvaluator

    started = time.perf_counter()
    if spec.get("version") != "1.0" or spec.get("prototype") not in PROTOTYPES:
        raise PipelineGenerationError("SPEC_UNSUPPORTED: unsupported specification version or prototype")
    if spec["prototype"] != "lookup_join_sum":
        from .stateful_spec_pipeline import compile_spec as compile_stateful
        return compile_stateful(spec, artifact_dir=artifact_dir, script_count=script_count)
    if spec.get("goal") != SUM_GOAL:
        raise PipelineGenerationError("SPEC_UNSUPPORTED: goal must match the versioned prototype")
    tables = copy.deepcopy(spec["tables"])
    P._validate_data_tables(tables)
    P._validate_relational_data(tables)
    goal = spec["goal"]
    lookup = {"lookup": {"table": goal["lookup_table"], "field": goal["join_field"],
                          "where": {goal["public_field"]: spec["selector"]}}}
    expression = {"aggregate": {"table": goal["aggregate_table"], "field": goal["field"],
                                 "where": {goal["join_field"]: lookup}, "op": "sum"}}
    outcome = {"metric_id": "outcome_quantity_total", "source": "final_agent_response", "path": "$",
               "operator": "numeric_targets", "expected": {"answer_format": "single_labeled_number", "targets": [{"label": goal["label"],
               "unit": goal["unit"], "tolerance": 0, "expression": expression}]},
               "score_mapping": {"pass": 1, "fail": 0}}
    state = {t["table_name"]: t["rows"] for t in tables}
    total = DeclarativeMetricEvaluator._numeric_expression(expression, state)
    if total is None:
        raise PipelineGenerationError("SPEC_UNSOLVABLE: goal does not resolve to a number")
    answer = f"{goal['label']}：{total}{goal['unit']}。"
    # Rendering is deliberately bounded by the typed goal; no model-authored facts.
    task = (f"请核算内部系统中“{spec['selector']}”的采购总数量。我只知道项目名称，"
            "请先查询对应的内部台账编号，再按该编号查询全部采购明细，汇总每条明细的 quantity。"
            "最终只需给出“采购总数量：数字件”。")
    public = {"initial_user_message": task, "materials": []}
    description = {"task": task, "task_intent": "calculate", "goal": "汇总指定项目的全部采购数量",
                   "expected_result": "采购总数量与当前内部明细的 quantity 合计一致", "public_input": public,
                   "requirements": {"input_modalities": ["text"], "output_modalities": ["text"]}, "complexity": "standard"}
    tools = [_tool("lookup_account", "按项目名称查询内部台账编号，返回 records 中的 account_id。", {"name": "string"}),
             _tool("query_entries", "按内部台账编号读取全部采购明细，返回 records 中的 quantity。", {"account_id": "integer"})]
    implementations = [
        {"tool_name": "lookup_account", "operation": "select", "table": "accounts", "result_field": "records",
         "filters": [{"argument": "name", "column": "name", "operator": "eq"}], "projection": ["account_id"]},
        {"tool_name": "query_entries", "operation": "select", "table": "entries", "result_field": "records",
         "filters": [{"argument": "account_id", "column": "account_id", "operator": "eq"}], "projection": ["quantity"]},
    ]
    names = ["查询项目台账编号", "查询采购明细"]
    actions = [{"name": name, "description": tools[i]["function"]["description"], "atomicity_rationale": "一次读取一张私有业务表",
                "inputs": [], "outputs": [], "preconditions": ["对应记录存在"], "effects": ["取得后续任务所需的私有数据"]}
               for i, name in enumerate(names)]
    bindings = [{"tool_name": tool["function"]["name"], "action_name": name} for tool, name in zip(tools, names)]
    capabilities = [{"action_name": name, "kind": "environment_operation", "requires_tool": True,
                     "dependencies": names[:i], "reason": "需要读取私有业务记录"} for i, name in enumerate(names)]
    actions.append({"name": "汇总采购数量", "description": "对已读取的全部 quantity 求和并回答", "atomicity_rationale": "本地数值推理",
                    "inputs": [{"name": "quantity"}], "outputs": [{"name": "采购总数量"}], "preconditions": ["已获得全部明细"], "effects": []})
    capabilities.append({"action_name": "汇总采购数量", "kind": "agent_reasoning", "requires_tool": False,
                         "dependencies": [names[1]], "reason": "对已有工具结果求和，无需新增工具"})
    key_steps = [{"step_id": f"step-{i+1}", "action_name": name, "required_for_goal": True,
                  "dependencies": [] if i == 0 else ["step-1"], "rationale": "提供不可从题面推断的业务事实"}
                 for i, name in enumerate(names)]
    metrics = [_metric("process_lookup_account", "process", .1, "查询内部台账编号", names[0]),
               _metric("process_query_entries", "process", .1, "使用查询结果读取采购明细", names[1]),
               _metric("outcome_quantity_total", "outcome", .8, "数值结论与当前私有业务数据重算结果一致")]
    metrics[-1]["evaluation_inputs"] = ["final_agent_response", "tool_results", "business_data"]
    metrics[-1]["criteria"] = ["最终回答只允许一个采购总数量数值结论；格式、单位及数值均由确定性规则检查，包含其他文本即不合格。", "逐项核对最终回答中的业务事实与实际工具结果及当前业务记录一致。"]
    reset = {"operation": "reset", "body": {"episode_id": "goal-success", "seed": spec["seed"]}, "expected_status": 200}
    calls = [{"step_id": "lookup", "operation": "tool_call", "tool_name": "lookup_account",
              "arguments": {"name": spec["selector"]}, "capture": {"account_id": "$.records[0].account_id"}, "expected_status": 200},
             {"step_id": "read", "operation": "tool_call", "tool_name": "query_entries",
              "arguments": {"account_id": {"$ref": "account_id"}}, "expected_status": 200}]
    scenarios = [{"scenario_id": "goal_success", "kind": "goal_success", "steps": [reset, *calls,
                  {"operation": "agent_response", "content": answer, "expected_status": 200},
                  {"step_id": "reward", "operation": "reward", "expected_status": 200}],
                  "assertions": [{"source": "step:reward", "path": "$.reward", "operator": "gte", "expected": 1}]},
                 {"scenario_id": "goal_failure", "kind": "goal_failure", "steps": [reset,
                  {"step_id": "reward", "operation": "reward", "expected_status": 200}],
                  "assertions": [{"source": "step:reward", "path": "$.reward", "operator": "lte", "expected": 0}]}]
    preview = P._preview_success_tool_results(scenarios=scenarios, data_tables=tables,
                tool_implementations=implementations, environment_mode="reference_data", tools=tools)
    verification = verify_spec(spec, outcome, answer, preview)
    plan = {"mode": "reference_data", "requires_business_data": True, "requires_persistence": False,
            "reason": "查询私有映射和明细，计算目标项目采购数量"}
    return assemble_spec(spec, artifact_dir=artifact_dir, script_count=script_count, started=started,
        tables=tables, description=description, tools=tools, implementations=implementations,
        actions=actions, bindings=bindings, capabilities=capabilities, key_steps=key_steps,
        metrics=metrics, outcome=outcome, scenarios=scenarios, plan=plan, verification=verification)


def assemble_spec(spec, *, artifact_dir, script_count, started, tables, description, tools,
                  implementations, actions, bindings, capabilities, key_steps, metrics,
                  outcome, scenarios, plan, verification, semantic_goal=None):
    """Compile all delivery projections through the same platform adapters."""
    from env_factory.task_pipeline import TaskGenerationPipeline as P
    from env_factory.tasks.task_spec import compile_task_spec
    from env_factory.tasks.task_routing import training_contract
    task, public = description["task"], description["public_input"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest = P._materialize_business_data(tables, P._render_data_document(tables), artifact_dir / "data/business_data",
                                            environment_mode=plan["mode"], manifest_root="data/business_data")
    manifest["data_governance"]["origin"] = "programmatically_generated_synthetic"
    users = P._materialize_user_simulation(P._deterministic_user_profiles(script_count),
                P._deterministic_user_scripts(description=description, count=script_count, finalize_on_goal=True),
                artifact_dir / "data/user_simulation", manifest_root="data/user_simulation")
    formula = {"type": "separate_sign_weighted_sum", "formula": "R = clip(" + " + ".join(f"{m['weight']}*score({m['id']})" for m in metrics) + ", -1, 1)",
               "positive_weight_sum": 1, "negative_weight_sum": 0, "score_range": [-1, 1],
               "positive_categories": ["process", "outcome"], "negative_categories": ["penalty"],
               "normalization": "positive weights sum to one"}
    metric_impl = P._compile_process_metric_implementations(metrics=metrics, metric_implementations=[outcome],
                         business_scenarios=scenarios, tool_bindings=bindings)
    P._normalize_compiled_process_metrics(metrics, metric_impl)
    acceptance = P._build_acceptance_contract(task_description=description, data_manifest=manifest, data_tables=tables,
        actions=actions, tools=tools, key_steps=key_steps, metrics=metrics, reward_formula=formula, tool_implementations=implementations)
    acceptance["executable_scenarios"].extend(scenarios)
    acceptance = P._ground_acceptance_probes_from_success(acceptance, scenarios)
    compiled = compile_task_spec(task_description=description, training_category="multi_step_agentic", environment_plan=plan,
        data_manifest=manifest, data_tables=tables, tools=tools, noise_tools=[], tool_bindings=bindings,
        tool_implementations=implementations, actions=actions, key_steps=key_steps, metrics=metrics,
        executable_scenarios=scenarios, capability_plan=capabilities, semantic_goal=semantic_goal)
    requirements = {**description["requirements"], "media_truth_mode": "programmatic",
                    "runtime_interface": P._build_runtime_interface(tools, formula)}
    for endpoint in requirements["runtime_interface"]["endpoints"]:
        contract = next((c for c in compiled["tool_contracts"] if c["name"] == endpoint.get("name")), None)
        if contract:
            endpoint["response_schema"] = contract["output_contract"]["schema"]
    pipeline = {"version": "2.0", "backend": "spec", "prototype": spec["prototype"],
                "compiler_sha256": hashlib.sha256(b"".join(path.read_bytes() for path in (
                    Path(__file__), Path(__file__).with_name("stateful_spec_pipeline.py"),
                    Path(__file__).with_name("user_simulation_contract.py"),
                    Path(__file__).parents[1] / "task_pipeline.py", Path(__file__).parents[1] / "sandbox_runtime.py",
                    Path(__file__).parents[1] / "tasks/task_spec.py",
                ))).hexdigest(),
                "spec_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                "stages": ["instantiate", "compile", "reference_execution", "counterfactuals", "render", "materialize"],
                "seconds": time.perf_counter() - started, "llm_calls": 0,
                "verification": verification, "live_rollout_verified": False}
    (artifact_dir / "source_spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    result = {"user_simulation_policy": {"mode": "fixed_goal", "required_outcomes": ["goal_satisfied"]}, "task": task, "task_type": "Event", "task_intent": description["task_intent"], "complexity": "standard",
            "training_category": "multi_step_agentic", "training_contract": training_contract("multi_step_agentic"),
            "runtime_capabilities": {"environment_modes": [plan["mode"]]}, "task_spec": compiled,
            "requirements": requirements, "public_input": public, "environment_plan": plan,
            "environment": P._environment_records({}, actions), "data_manifest": manifest,
            "user_simulation_manifest": users, "tools_manifest": P._materialize_tools(tools, artifact_dir),
            "media_generation": {"required": False, "language": "python", "code": "", "dependencies": [], "entrypoint": "", "output_dir": ""},
            "actions": actions, "capability_plan": capabilities, "tools": tools, "tool_bindings": bindings,
            "tool_implementations": implementations, "noise_tools": [], "reward_key_steps": key_steps,
            "observation_schema": {"type": "object", "properties": {}}, "metrics": metrics,
            "metric_implementations": metric_impl, "reward_formula": formula, "acceptance_contract": acceptance,
            "task_readiness": {"ready": True, "training_profile": P._derive_training_profile(business_tool_count=len(bindings), noise_tool_count=0, key_step_count=len(key_steps)), "errors": [], "warnings": []},
            "generation_pipeline": pipeline, "graph_context": {"keywords": [spec["subject"]]}}

    verification["execution_checks"] = verify_execution(result, artifact_dir)
    pipeline["seconds"] = time.perf_counter() - started
    (artifact_dir / "spec_verification.json").write_text(json.dumps(pipeline, ensure_ascii=False, indent=2) + "\n")
    return result


def verify_spec(spec: dict, outcome: dict, answer: str, preview: list) -> dict:
    """Independent summation and metamorphic checks before expensive construction."""
    from env_factory.sandbox_runtime import DeclarativeMetricEvaluator
    evaluator = DeclarativeMetricEvaluator()
    state = {t["table_name"]: copy.deepcopy(t["rows"]) for t in spec["tables"]}
    selected = [r for r in state["accounts"] if r["name"] == spec["selector"]]
    if len(selected) != 1 or len(preview) != 2:
        raise PipelineGenerationError("SPEC_REFERENCE_FAILED: reference must resolve exactly one account")
    key = selected[0]["account_id"]
    expected = sum(r["quantity"] for r in state["entries"] if r["account_id"] == key)
    if (preview[0]["result"]["records"] != [{"account_id": key}]
            or sum(row["quantity"] for row in preview[1]["result"]["records"]) != expected):
        raise PipelineGenerationError("SPEC_REFERENCE_FAILED: actual tool results disagree with business goal")
    if answer != f"采购总数量：{expected}件。":
        raise PipelineGenerationError("SPEC_ORACLE_DISAGREEMENT: independent sum differs")
    def score(data, response=answer):
        return evaluator.evaluate(outcome, {"final_agent_response": response, "business_state": data})
    checks = {"success": score(state) == 1, "wrong_answer": score(state, "采购总数量：0件。") == 0,
              "extra_claim_rejected": score(state, answer + "物料编号：伪造编号。") == 0}
    for index, row in enumerate(state["entries"]):
        if row["account_id"] == key:
            changed = copy.deepcopy(state)
            changed["entries"][index]["quantity"] += 1
            checks[f"quantity_{index}"] = score(changed) == 0
    changed = copy.deepcopy(state)
    changed["entries"].append({"entry_id": 99999, "account_id": key, "quantity": 7})
    checks["new_matching_row"] = score(changed) == 0
    changed = copy.deepcopy(state)
    row = next(r for r in changed["accounts"] if r["name"] == spec["selector"])
    row["account_id"] = -1
    checks["private_join_changes"] = score(changed) == 0
    changed = copy.deepcopy(state)
    next(r for r in changed["entries"] if r["account_id"] != key)["quantity"] += 1
    checks["unrelated_row_invariant"] = score(changed) == 1
    if not all(checks.values()):
        raise PipelineGenerationError("SPEC_COUNTERFACTUAL_FAILED: " + ",".join(k for k,v in checks.items() if not v))
    return {"passed": True, "checks": checks, "scope": "reference_and_numeric_outcome"}


def generate(*, seed: int, artifact_dir: Path | None, subject: str = "办公用品", script_count: int = 3,
             prototype: str = "lookup_join_sum") -> dict:
    root = artifact_dir or Path(tempfile.mkdtemp(prefix="envfactory-spec-"))
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    from env_factory.sandbox_runtime import SandboxError
    try:
        return compile_spec(instantiate(seed=seed, subject=subject, prototype=prototype), artifact_dir=root, script_count=script_count)
    except (PipelineGenerationError, SandboxError, ValueError, TypeError, KeyError) as exc:
        code = str(exc).split(":", 1)[0] if str(exc).startswith("SPEC_") else "SPEC_COMPILE_FAILED"
        defect = {"code": code, "owner": "spec_compiler", "prototype": prototype, "seed": seed,
                  "stage": "prebuild", "evidence": str(exc), "retryable": False,
                  "repair_action": "correct source specification or prototype compiler; recompile dependent artifacts"}
        (root / "generation_defect.json").write_text(json.dumps(defect, ensure_ascii=False, indent=2) + "\n")
        raise PipelineGenerationError(f"{code}: {exc}") from exc


def verify_execution(artifacts: dict, root: Path) -> dict:
    """Execute compiled tools and the actual reward gate, including policy negatives."""
    from env_factory.sandbox_runtime import (
        AcceptanceScenarioRunner, ContractRewardAggregator, ContractRewardGate,
        ContractToolRegistry, DeclarativeMetricEvaluator, DeclarativeToolCompiler,
        EpisodeStore, ManifestDataStore, SandboxError,
    )
    with tempfile.TemporaryDirectory(prefix="envfactory-spec-check-") as temporary:
        store = EpisodeStore(Path(temporary) / "episode.sqlite3")
        manifest = artifacts["data_manifest"]
        data = ManifestDataStore(manifest, root / manifest["root"], store)
        registry = ContractToolRegistry(artifacts["tools"], DeclarativeToolCompiler(data).compile_all(artifacts["tool_implementations"]),
            event_recorder=store.event, tool_contracts=artifacts["task_spec"]["tool_contracts"])
        gate = ContractRewardGate(artifacts["task_spec"], artifacts["metrics"])
        evaluator = DeclarativeMetricEvaluator()
        aggregator = ContractRewardAggregator(artifacts["metrics"])
        def call(method, path, body, headers):
            if path == "/v1/reset":
                store.reset(episode_id="spec-check", seed=17)
                data.reset()
                result = {}
            elif path.startswith("/v1/tools/"):
                result = registry.execute(path.removeprefix("/v1/tools/"), body)
            elif path == "/v1/agent_response":
                store.set_state("final_agent_response", body.get("content", ""))
                result = {}
            elif path == "/v1/reward":
                context = {"business_state": {name: data.table(name) for name in data.baseline},
                           "initial_business_state": data.baseline, "trajectory": store.replay(),
                           "final_agent_response": store.get_state("final_agent_response", "")}
                scores = evaluator.evaluate_all(artifacts["metric_implementations"], context)
                result = aggregator.aggregate(gate.apply(scores, context))
            else:
                raise ValueError(f"unsupported preflight operation: {path}")
            return 200, result, {}
        runner = AcceptanceScenarioRunner(call)
        success = next(s for s in artifacts["acceptance_contract"]["executable_scenarios"] if s.get("kind") == "goal_success")
        def reward(steps):
            try:
                result = runner.run({"steps": steps, "assertions": []})
                return result["history"][-1]["body"]["reward"]
            except SandboxError:
                return None
        steps = success["steps"]
        checks = {"reference_reward": reward(steps) == 1}
        wrong = copy.deepcopy(steps)
        next(s for s in wrong if s.get("operation") == "agent_response")["content"] = "采购总数量：0件。"
        if artifacts["environment_plan"]["mode"] == "reference_data":
            checks["wrong_answer_rejected"] = (reward(wrong) or 0) < .6
        no_tools = [s for s in steps if s.get("operation") != "tool_call"]
        checks["no_tools_rejected"] = (reward(no_tools) or 0) < .6
        for index, step in enumerate(steps):
            if step.get("operation") == "tool_call":
                checks[f"skip_{step['tool_name']}_rejected"] = (reward(steps[:index] + steps[index+1:]) or 0) < .6
        reversed_steps = copy.deepcopy(steps)
        reversed_steps[1], reversed_steps[2] = reversed_steps[2], reversed_steps[1]
        checks["wrong_order_rejected"] = (reward(reversed_steps) or 0) < .6
        extra = copy.deepcopy(steps)
        next(s for s in extra if s.get("operation") == "agent_response")["content"] += "私有编号是伪造编号。"
        if artifacts["environment_plan"]["mode"] == "reference_data":
            checks["extra_claim_rejected"] = (reward(extra) or 0) < .6
        else:
            predicate = artifacts["task_spec"]["goal_contract"]["row_predicates"][0]
            reward(steps)
            field, value = list(predicate["values"].items())[-1]
            data.update(predicate["table"], predicate["where"], {field: value + 1 if isinstance(value, int) else "未完成"})
            checks["wrong_state_rejected"] = call("GET", "/v1/reward", None, {})[1]["reward"] < .6
            reward(steps)
            unrelated = next(row for row in data.table("requests")
                             if row["name"] != artifacts["generation_pipeline"].get("selector", "")
                             and not all(row.get(k) == v for k, v in predicate["where"].items()))
            data.update("requests", {"id": unrelated["id"]}, {"name": "无关记录被修改"})
            checks["unrelated_write_rejected"] = call("GET", "/v1/reward", None, {})[1]["reward"] < .6
            if artifacts["generation_pipeline"]["prototype"] == "constraint_create":
                reward(steps)
                before = data.snapshot_hash()
                arguments = {**predicate["where"], **predicate["values"]}
                try:
                    registry.execute("create_booking", arguments)
                    checks["duplicate_write_rejected_atomically"] = False
                except SandboxError as exc:
                    checks["duplicate_write_rejected_atomically"] = exc.status == 400 and data.snapshot_hash() == before
        if not all(checks.values()):
            raise PipelineGenerationError("SPEC_EXECUTION_FAILED: " + ",".join(k for k,v in checks.items() if not v))
        return checks
