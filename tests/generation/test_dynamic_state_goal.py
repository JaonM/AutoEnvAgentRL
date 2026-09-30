import json
from pathlib import Path
import tempfile
import unittest

from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import (BusinessGoalEvaluator, ContractRewardAggregator,
    ContractRewardGate, ContractToolRegistry, DeclarativeMetricEvaluator,
    DeclarativeToolCompiler, EpisodeStore, ManifestDataStore)
from env_factory.tasks.task_spec import TaskSpecError, validate_goal_contract
from .test_agent_authoring import fixture


def dynamic_source(offset=0):
    source, request = fixture("multi_step_agentic")
    source["environment_plan"].update(mode="stateful", requires_persistence=True)
    source["description"]["task_intent"] = "modify"
    source["description"]["public_input"]["initial_user_message"] = "将库区青松库的温度调整到它的配置温度，并报告调整后的温度。"
    source["tables"][0]["columns"].append({"name": "configured_temperature", "type": "integer", "nullable": False})
    for row in source["tables"][0]["rows"]:
        row["configured_temperature"] = 25
    source["tool_implementations"][1]["projection"] = ["configured_temperature"]
    source["tools"].append({"type": "function", "function": {"name": "update_temperature", "description": "更新库区温度",
        "parameters": {"type": "object", "properties": {
            "id": {"type": "integer", "description": "查询得到的库区编号"},
            "value": {"type": "integer", "description": "读取到的配置温度"}},
            "required": ["id", "value"], "additionalProperties": False}}})
    source["tool_implementations"].append({"tool_name": "update_temperature", "operation": "update",
        "table": "zones", "result_field": "updated_count", "selector": {"id": "id"}, "changes": {"value": "temperature"}})
    source["actions"].append({"name": "update_temperature", "description": "更新库区温度",
        "atomicity_rationale": "单条业务记录更新", "inputs": [], "outputs": [], "preconditions": [], "effects": []})
    source["tool_bindings"].append({"tool_name": "update_temperature", "action_name": "update_temperature"})
    source["capability_plan"].append({"action_name": "update_temperature", "kind": "environment_operation",
        "requires_tool": True, "dependencies": ["read_temperature"], "reason": "以读取到的配置温度进行更新"})
    source["reward_key_steps"].append({"step_id": "update_temperature", "action_name": "update_temperature",
        "required_for_goal": True, "dependencies": ["read_temperature"], "rationale": "持久化更新"})
    steps = source["scenarios"][0]["steps"]
    steps[1]["capture"] = {"target_temperature": "$.records[0].configured_temperature"}
    steps.insert(2, {"operation": "tool_call", "tool_name": "update_temperature",
        "arguments": {"id": {"$ref": "zone_id"}, "value": {"$ref": "target_temperature"}}, "expected_status": 200})
    steps[-1]["content"] = "温度：25摄氏度。"
    source["semantic_goal"] = {"row_predicates": [{"table": "zones", "where": {"name": "青松库"},
        "values": {}, "value_expressions": {"temperature": {
            "from_tool": {"name": "read_temperature", "field": "configured_temperature"}}}, "count": 1}]}
    if offset:
        steps[2]["arguments"]["value"] = {"$expr": {"op": "add", "args": [
            {"$ref": "target_temperature"}, offset]}}
        steps[-1]["content"] = f"温度：{25 + offset}摄氏度。"
        predicate = source["semantic_goal"]["row_predicates"][0]
        predicate["value_expressions"]["temperature"] = {"op": "add", "args": [
            predicate["value_expressions"]["temperature"], {"literal": offset}]}
        source["description"]["public_input"]["initial_user_message"] = f"将库区青松库的温度调整到配置温度加 {offset} 摄氏度，并报告结果。"
    return source, request


class DynamicStateGoalTest(unittest.TestCase):
    def test_copied_private_write_and_goal_are_rejected_before_materialization(self):
        for location in ("write", "goal"):
            source, request = dynamic_source()
            if location == "write":
                source["scenarios"][0]["steps"][2]["arguments"]["value"] = 25
            else:
                predicate = source["semantic_goal"]["row_predicates"][0]
                predicate["values"] = {"temperature": 25}
                predicate.pop("value_expressions")
            with self.subTest(location=location), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "preview"
                with self.assertRaisesRegex(ValueError, "STATE_VALUE_UNGROUNDED"):
                    compile_source(source, root=root, request=request)
                self.assertFalse(root.exists())

    def test_public_numeric_target_is_allowed(self):
        source, request = dynamic_source()
        source["description"]["public_input"]["initial_user_message"] = "将库区青松库温度设为 25 摄氏度并报告。"
        source["scenarios"][0]["steps"][2]["arguments"]["value"] = 25
        predicate = source["semantic_goal"]["row_predicates"][0]
        predicate["values"] = {"temperature": 25}
        predicate.pop("value_expressions")
        from env_factory.generation.input_grounding import validate_stateful_value_origins
        validate_stateful_value_origins(source)

    def test_computed_goal_failure_keeps_serializable_diagnostic(self):
        source, request = dynamic_source(5)
        scenario = source["scenarios"][0]
        scenario["steps"][2]["arguments"]["value"] = {"$ref": "target_temperature"}
        scenario["steps"][-1]["content"] = "温度：25摄氏度。"
        scenario["steps"].append({"operation": "reward", "step_id": "reward"})
        scenario["assertions"] = [{"source": "step:reward", "path": "$.reward", "operator": "gte", "expected": 1}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "AGENT_SCENARIO_FAILED"):
                compile_source(source, root=root, request=request)
            diagnostic = json.loads((root / "preflight_failure.json").read_text())["diagnostic"]
            self.assertEqual(diagnostic["state_predicates"][0]["expected_values"], {"temperature": 30})

    def test_failed_reference_explains_reward_gate_and_actual_state(self):
        source, request = dynamic_source()
        scenario = source["scenarios"][0]
        scenario["steps"][2]["arguments"]["value"] = 24
        scenario["steps"][-1]["content"] = "温度：24摄氏度。"
        scenario["steps"].append({"operation": "reward", "step_id": "reward"})
        scenario["assertions"] = [{"source": "step:reward", "path": "$.reward", "operator": "gte", "expected": 1}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "AGENT_SCENARIO_FAILED"):
                compile_source(source, root=root, request=request)
            report = json.loads((root / "preflight_failure.json").read_text())["diagnostic"]
            self.assertEqual(report["ungated_components"]["temperature_answer"], 1)
            self.assertEqual(report["gated_components"]["temperature_answer"], 0)
            self.assertEqual(report["state_predicates"][0]["expected_values"], {"temperature": 25})
            self.assertEqual(report["state_predicates"][0]["actual_rows"][0]["temperature"], 24)

    def test_stateful_request_rejects_readonly_task_with_modify_label(self):
        source, request = fixture("multi_step_agentic")
        source["description"]["task_intent"] = "modify"
        request["task_intent"] = "modify"
        request["available_environment_modes"] = ["stateful"]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "environment mode is unavailable"):
            compile_source(source, root=Path(directory), request=request)

    def test_current_business_target_controls_actual_write_and_reward(self):
        self.check_dynamic_target(0)

    def test_computed_capture_controls_actual_write_and_dependency_gate(self):
        self.check_dynamic_target(5)

    def test_computed_write_process_metric_uses_observed_capture(self):
        source, request = dynamic_source(5)
        source["metrics"][0]["weight"] = .8
        source["metrics"].append({**source["metrics"][0], "id": "write_process",
            "category": "process", "scope": "trajectory", "weight": .2,
            "target_action": "update_temperature"})
        with tempfile.TemporaryDirectory() as directory:
            artifacts = compile_source(source, root=Path(directory), request=request)
            rule = next(rule for rule in artifacts["metric_implementations"]
                        if rule["metric_id"] == "write_process")
            def event(name, args, result):
                return {"event": "tool_call", "payload": {"tool_name": name, "arguments": args}, "result": result}
            events = [event("lookup_zone", {"name": "青松库"}, {"records": [{"id": 821}]}),
                      event("read_temperature", {"id": 821}, {"records": [{"configured_temperature": 40}]}),
                      event("update_temperature", {"id": 821, "value": 45}, {"updated_count": 1})]
            evaluator = DeclarativeMetricEvaluator()
            self.assertEqual(evaluator.evaluate(rule, {"trajectory": {"events": events}}), 1)
            events[-1]["payload"]["arguments"]["value"] = 30
            self.assertEqual(evaluator.evaluate(rule, {"trajectory": {"events": events}}), 0)

    def check_dynamic_target(self, offset):
        source, request = dynamic_source(offset)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = compile_source(source, root=root, request=request)
            store = EpisodeStore(root / "test.sqlite3")
            data = ManifestDataStore(artifacts["data_manifest"], root / "data/business_data", store)
            registry = ContractToolRegistry(artifacts["tools"], DeclarativeToolCompiler(data).compile_all(
                artifacts["tool_implementations"]), event_recorder=store.event,
                tool_contracts=artifacts["task_spec"]["tool_contracts"])
            def run(target, write=None, unrelated=False):
                for row in data.baseline["zones"]:
                    row["configured_temperature"] = target
                store.reset(episode_id="dynamic", seed=1); data.reset()
                zone = registry.execute("lookup_zone", {"name": "青松库"})["records"][0]["id"]
                observed = registry.execute("read_temperature", {"id": zone})["records"][0]["configured_temperature"]
                registry.execute("update_temperature", {"id": zone, "value": observed + offset if write is None else write})
                if unrelated:
                    data.update("zones", {"id": 547}, {"temperature": 99})
                state = {"zones": data.table("zones")}
                context = {"business_state": state, "initial_business_state": data.baseline,
                    "trajectory": store.replay(), "final_agent_response": f"温度：{observed + offset}摄氏度。"}
                scores = DeclarativeMetricEvaluator().evaluate_all(artifacts["metric_implementations"], context)
                scores = ContractRewardGate(artifacts["task_spec"], artifacts["metrics"]).apply(scores, context)
                return ContractRewardAggregator(artifacts["metrics"]).aggregate(scores)["reward"]
            self.assertEqual(run(25), 1)
            self.assertEqual(run(30), 1)
            self.assertEqual(run(30, write=25), 0)
            self.assertEqual(run(30, unrelated=True), 0)

    def test_unresolved_goal_expression_cannot_pass_zero_count(self):
        predicate = {"table": "zones", "where": {"name": "missing"}, "values": {}, "count": 0,
            "value_expressions": {"temperature": {"lookup": {"table": "settings", "field": "value", "where": {}}}}}
        self.assertFalse(BusinessGoalEvaluator.evaluate([predicate], {"zones": [], "settings": []}))

    def test_static_and_dynamic_goal_fields_must_not_conflict(self):
        source, _ = dynamic_source()
        predicate = source["semantic_goal"]["row_predicates"][0]
        predicate["values"]["temperature"] = 25
        with self.assertRaisesRegex(TaskSpecError, "duplicate columns"):
            validate_goal_contract(source["semantic_goal"], source["tables"])
