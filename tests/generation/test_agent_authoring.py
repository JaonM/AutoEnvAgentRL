import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.task_generator import TaskGenerator, TaskGenerationError
from env_factory.graph.knowledge_graph import SceneNode


def fixture(category):
    """Small independent fixture; production authoring never imports these tasks."""
    task = "库区温度为18摄氏度，回答格式为温度：<数字>摄氏度。" if category == "direct_response" else "查询内部库区青松库的温度，只回答温度数字。"
    source = {"version": "1.0", "description": {"task": task, "task_intent": "extract" if category == "direct_response" else "query",
        "goal": "回答库区温度", "expected_result": "正确的温度数字", "complexity": "standard",
        "requirements": {"input_modalities": ["text"], "output_modalities": ["text"]},
        "public_input": {"initial_user_message": task, "materials": []}},
        "environment_plan": {"mode": "stateless" if category == "direct_response" else "reference_data",
            "requires_business_data": category != "direct_response", "requires_persistence": False, "reason": "读取温度"},
        "tables": [], "tools": [], "tool_implementations": [], "actions": [], "tool_bindings": [],
        "capability_plan": [], "reward_key_steps": [],
        "metrics": [{"id": "temperature_answer", "category": "outcome", "type": "rule-based",
            "scope": "terminal", "weight": 1, "score_range": [0, 1], "rubric": "回答正确温度",
            "evaluator": {"kind": "document_rule", "source": "runtime_rule", "assertion": "温度正确", "score_mapping": {"pass": 1, "fail": 0}}}],
        "metric_implementations": [{"metric_id": "temperature_answer", "source": "final_agent_response",
            "path": "$", "operator": "eq", "expected": "18", "score_mapping": {"pass": 1, "fail": 0}}]}
    calls = []
    if category != "direct_response":
        source["tables"] = [{"table_name": "zones", "description": "库区温度", "primary_key": ["id"],
            "columns": [{"name": "id", "type": "integer", "nullable": False}, {"name": "name", "type": "text", "nullable": False},
                        {"name": "temperature", "type": "integer", "nullable": False}],
            "foreign_keys": [], "indexes": [], "constraints": [],
            "rows": [{"id": 821, "name": "青松库", "temperature": 18}, {"id": 547, "name": "白杨库", "temperature": 23}]}]
        names = ["lookup_zone", "read_temperature"] if category == "multi_step_agentic" else ["read_temperature"]
        for index, name in enumerate(names):
            argument = "id" if index else "name"
            source["tools"].append({"type": "function", "function": {"name": name, "description": "查询库区编号" if name == "lookup_zone" else "读取温度",
                "parameters": {"type": "object", "properties": {argument: {"type": "integer" if index else "string", "description": "上次查询得到的编号" if index else "库区名称"}},
                               "required": [argument], "additionalProperties": False}}})
            source["tool_implementations"].append({"tool_name": name, "operation": "select", "table": "zones", "result_field": "records",
                "filters": [{"argument": argument, "column": argument, "operator": "eq"}], "projection": ["id"] if name == "lookup_zone" else ["temperature"]})
            source["actions"].append({"name": name, "description": "读取库区业务记录", "atomicity_rationale": "一次查询", "inputs": [], "outputs": [], "preconditions": [], "effects": []})
            source["tool_bindings"].append({"tool_name": name, "action_name": name})
            source["capability_plan"].append({"action_name": name, "kind": "environment_operation", "requires_tool": True, "dependencies": names[:index], "reason": "取得内部数据"})
            source["reward_key_steps"].append({"step_id": name, "action_name": name, "required_for_goal": True, "dependencies": names[:index], "rationale": "必要查询"})
            calls.append({"step_id": name, "operation": "tool_call", "tool_name": name,
                "arguments": {argument: {"$ref": "zone_id"} if index else "青松库"}, "expected_status": 200})
            if name == "lookup_zone":
                calls[-1]["capture"] = {"zone_id": "$.records[0].id"}
        source["metric_implementations"][0].update(operator="numeric_targets", expected={"answer_format": "single_labeled_number", "targets": [{
            "label": "温度", "unit": "摄氏度", "tolerance": 0, "expression": {"lookup": {"table": "zones", "field": "temperature", "where": {"name": "青松库"}}}}]})
    if category == "direct_response":
        source["metric_implementations"][0].update(operator="numeric_targets", expected={
            "answer_format": "single_labeled_number", "targets": [{"label": "温度", "unit": "摄氏度",
            "tolerance": 0, "expression": {"literal": 18}}]})
    answer = "温度：18摄氏度。"
    source["scenarios"] = [{"scenario_id": "success", "kind": "goal_success", "steps": calls + [
        {"operation": "agent_response", "content": answer, "expected_status": 200}], "assertions": []},
        {"scenario_id": "failure", "kind": "goal_failure", "steps": [{"operation": "agent_response", "content": "999", "expected_status": 200}], "assertions": []}]
    request = {"training_category": category, "task_type": "Event", "task_intent": None,
        "available_environment_modes": ["stateless", "reference_data", "stateful"],
        "graph_context": {"hops": 2, "nodes": ["仓储", "库区", "温度"], "keywords": ["库区", "温度"]}}
    return source, request


class AgentAuthoringTest(unittest.TestCase):
    def test_graph_relation_metadata_is_rejected_before_data_materialization(self):
        from env_factory.task_pipeline import PipelineGenerationError
        source, request = fixture("direct_response")
        request["graph_context"]["relation"] = "SAME_EVENT_ELEMENT"
        source["description"]["public_input"]["initial_user_message"] += " SAME_EVENT_ELEMENT"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "candidate"
            with self.assertRaisesRegex(PipelineGenerationError, "internal graph relation"):
                compile_source(source, root=root, request=request)
            self.assertFalse(root.exists())

    def test_final_gate_tries_answer_data_after_quoted_selector_breaks_replay(self):
        import json
        import os
        import shutil
        import subprocess
        import sys
        from env_factory.generation.artifacts import write_task_artifact
        from env_factory.tasks.task import Task
        from scripts.sandbox.generate_sandbox_scaffold import generate as scaffold

        source, request = fixture("multi_step_agentic")
        # The name is both a query selector and part of the correct answer.
        # Mutating it invalidates the lookup chain; temperature remains a valid
        # independent counterfactual that must disqualify the stale answer.
        source["scenarios"][0]["steps"][-1]["content"] = "青松库温度：18摄氏度。"
        source["metric_implementations"][0]["expected"]["targets"][0]["label"] = "青松库温度"
        for scenario in source["scenarios"]:
            scenario["steps"].insert(0, {"operation": "reset", "body": {"seed": 17}})
            scenario["steps"].append({"operation": "reward"})
        project = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a = compile_source(source, root=root, request=request)
            path = write_task_artifact(root, Task(a["task"], a["environment"], a["metrics"],
                task_intent=a["task_intent"], artifacts=a), "multi_step_agentic")
            contract = json.loads(path.read_text())
            contract.pop("actions")
            (root / "BUILD_CONTRACT.json").write_text(json.dumps(contract))
            scaffold(root)
            for name in ("sandbox_runtime.py", "runtime_llm.py"):
                shutil.copy2(project / "src/env_factory" / name, root / name)
            report = root / "gate.json"
            result = subprocess.run([sys.executable,
                str(project / "scripts/sandbox/validate_agentic_training_value.py"),
                "--root", str(root), "--output", str(report)], capture_output=True, text=True,
                env={**os.environ, "PYTHONPATH": str(project / "src"),
                     "SANDBOX_TRAINER_API_KEY": "test-selector-key", "SANDBOX_EVALUATOR_MOCK": "1"},
                timeout=30)
            self.assertTrue(report.exists(), result.stderr)
            evidence = json.loads(report.read_text())["evidence"]
            sensitivity = evidence["business_data_reward_sensitivity"]
            self.assertTrue(sensitivity["proved"], sensitivity)
            self.assertLess(sensitivity["changed_reward"], .6)

    def test_compact_literal_normalization_preserves_lookup_and_conditions(self):
        from env_factory.generation.agent_authoring import normalize_expression
        from env_factory.sandbox_runtime import DeclarativeMetricEvaluator
        expression = {"if": {"condition": {"op": "eq", "args": [
            {"lookup": {"table": "checks", "field": "status", "where": {"code": "C-1"}}}, "cleared"]},
            "then": "CLEAR", "else": "HOLD"}}
        normalized = normalize_expression(expression)
        self.assertEqual(normalized["if"]["then"], {"literal": "CLEAR"})
        self.assertEqual(DeclarativeMetricEvaluator._value_expression(normalized,
            {"checks": [{"code": "C-1", "status": "cleared"}]}), "CLEAR")
        self.assertEqual(DeclarativeMetricEvaluator._value_expression(normalized,
            {"checks": [{"code": "C-1", "status": "hold"}]}), "HOLD")

    def test_dynamic_text_reward_reaches_prebuild_gate(self):
        from env_factory.generation.artifacts import write_task_artifact
        from env_factory.tasks.task import Task
        from scripts.sandbox.assess_task_buildability import assess
        full, request = fixture("multi_step_agentic")
        import json
        full = json.loads(json.dumps(full).replace("read_temperature", "get_manager"))
        table = full["tables"][0]
        table["columns"].append({"name": "manager", "type": "text", "nullable": False})
        for row, manager in zip(table["rows"], ["齐工", "严工"]):
            row["manager"] = manager
        full["tool_implementations"][1]["projection"] = ["manager"]
        compact = {key: full[key] for key in ("version", "description", "environment_plan", "tables")}
        text = '请查询内部库区青松库的管理员。只回答 JSON 对象，字段名为 manager。'
        compact["description"].update(task=text, goal="查询库区管理员", expected_result="返回当前管理员姓名")
        compact["description"]["public_input"]["initial_user_message"] = text
        compact["business_tools"] = [{"name": tool["function"]["name"], "description": "按标识查询库区信息",
            "parameters": tool["function"]["parameters"], "implementation": implementation}
            for tool, implementation in zip(full["tools"], full["tool_implementations"])]
        compact["outcomes"] = [{"id": "correct_manager", "rubric": "姓名必须与查询出的当前库区管理员一致",
            "rule": {"source": "final_agent_response", "path": "$", "operator": "value_targets", "expected": {
                "answer_format": "json_object", "targets": [{"key": "manager", "expression": {"lookup": {
                    "table": "zones", "field": "manager", "where": {"id": {"lookup": {
                        "table": "zones", "field": "id", "where": {"name": "青松库"}}}}}}}]}}}]
        compact["reference"] = {"calls": full["scenarios"][0]["steps"][:-1], "answer": '{"manager":"齐工"}'}
        compact["answer_contract"] = {"format": "json_object", "schema": {
            "type": "object", "properties": {"manager": {"type": "string", "description": "库区管理员姓名"}},
            "required": ["manager"], "additionalProperties": False}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = compile_source(compact, root=root, request=request)
            write_task_artifact(root, Task(desc=result["task"], env=result["environment"], metrics=result["metrics"],
                task_intent=result["task_intent"], complexity=result["complexity"], artifacts=result), "multi_step_agentic")
            report = assess(root)
            self.assertTrue(report["buildable"], report)

    def test_stateful_task_executes_a_typed_business_change(self):
        source, request = fixture("multi_step_agentic")
        source["description"]["task_intent"] = "modify"
        source["description"]["task"] = "将青松库的目标温度从18改为19，其他库区保持不变。"
        source["description"]["public_input"]["initial_user_message"] = source["description"]["task"]
        source["environment_plan"].update(mode="stateful", requires_persistence=True)
        source["tools"][1]["function"]["parameters"]["properties"]["temperature"] = {"type": "integer", "description": "新目标温度"}
        source["tools"][1]["function"]["parameters"]["required"].append("temperature")
        source["tool_implementations"][1] = {"tool_name": "read_temperature", "operation": "update", "table": "zones",
            "result_field": "updated_count", "selector": {"id": "id"}, "changes": {"temperature": "temperature"}}
        source["scenarios"][0]["steps"][1]["arguments"]["temperature"] = 19
        source["metric_implementations"][0].update(source="business_state", operator="eq",
            path='$.zones[?(@.name=="青松库")][0].temperature', expected=19)
        source["semantic_goal"] = {"row_predicates": [{"table": "zones", "where": {"name": "青松库"},
            "values": {"temperature": 19}, "count": 1}], "expected_delta": [{"table": "zones", "where": {"name": "青松库"},
            "field": "temperature", "before": 18, "after": 19}]}
        with tempfile.TemporaryDirectory() as directory:
            result = compile_source(source, root=Path(directory), request=request)
            self.assertTrue(result["generation_pipeline"]["verification"]["passed"])

    def test_compact_business_contract_supports_all_routes(self):
        for category in ("direct_response", "simple_agentic", "multi_step_agentic"):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as directory:
                full, request = fixture(category)
                compact = {key: full[key] for key in ("version", "description", "environment_plan", "tables")}
                compact["business_tools"] = [{"name": tool["function"]["name"],
                    "description": tool["function"]["description"], "parameters": tool["function"]["parameters"],
                    "implementation": implementation} for tool, implementation in zip(full["tools"], full["tool_implementations"])]
                compact["outcomes"] = [{"id": full["metrics"][0]["id"], "rubric": full["metrics"][0]["rubric"],
                    "rule": full["metric_implementations"][0]}]
                steps = full["scenarios"][0]["steps"]
                compact["reference"] = {"calls": [s for s in steps if s["operation"] == "tool_call"],
                                        "answer": steps[-1]["content"]}
                result = compile_source(compact, root=Path(directory), request=request)
                self.assertTrue(result["generation_pipeline"]["verification"]["passed"])
                self.assertEqual(len(result["tools"]), len(full["tools"]))
                if category == "multi_step_agentic":
                    self.assertTrue(result["task_spec"]["capability_dag"]["edges"])

    def test_all_three_routes_execute_with_original_runtime(self):
        for category in ("direct_response", "simple_agentic", "multi_step_agentic"):
            with self.subTest(category=category), tempfile.TemporaryDirectory() as directory:
                source, request = fixture(category)
                result = compile_source(source, root=Path(directory), request=request)
                self.assertTrue(result["generation_pipeline"]["verification"]["passed"])
                self.assertEqual(result["training_category"], category)
                self.assertEqual(result["graph_context"], request["graph_context"])

    def test_self_reported_success_cannot_override_actual_reward(self):
        source, request = fixture("direct_response")
        source["scenarios"][0]["steps"][0]["content"] = "999"
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "reference_0"):
            compile_source(source, root=Path(directory), request=request)

    def test_final_scenario_assertions_are_executed_in_preflight(self):
        source, request = fixture("direct_response")
        source["scenarios"][0]["steps"].append({"operation": "reward"})
        source["scenarios"][0]["assertions"] = [{"source": "step:reward", "path": "$.reward", "operator": "lte", "expected": 0}]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, '"actual": 1'):
            compile_source(source, root=Path(directory), request=request)

    def test_missing_reward_id_is_assigned_consistently(self):
        source, request = fixture("direct_response")
        source["scenarios"][0]["steps"].append({"operation": "reward"})
        source["scenarios"][0]["assertions"] = [{"source": "step:reward", "path": "$.reward", "operator": "gte", "expected": 1}]
        with tempfile.TemporaryDirectory() as directory:
            result = compile_source(source, root=Path(directory), request=request)
            scenario = next(s for s in result["acceptance_contract"]["executable_scenarios"] if s.get("kind") == "goal_success")
            self.assertEqual(scenario["steps"][-1]["step_id"], "reward")

    def test_always_true_reward_is_rejected(self):
        source, request = fixture("direct_response")
        source["metric_implementations"][0].update(operator="contains", expected="")
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "ANSWER_REWARD_UNGROUNDED"):
            compile_source(source, root=Path(directory), request=request)

    def test_requested_route_cannot_be_changed(self):
        source, request = fixture("simple_agentic")
        request["training_category"] = "multi_step_agentic"
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "tool count"):
            compile_source(source, root=Path(directory), request=request)

    def test_required_lookup_cannot_be_reward_irrelevant(self):
        source, request = fixture("multi_step_agentic")
        source["tables"].append({**copy.deepcopy(source["tables"][0]), "table_name": "inspections"})
        source["tool_implementations"][1]["table"] = "inspections"
        # Keep the actual reward fact observable, so this test isolates the
        # separate defect: the second required tool is irrelevant to reward.
        source["tool_implementations"][0]["projection"].append("temperature")
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "NUMERIC_DEPENDENCY_UNUSED"):
            compile_source(source, root=Path(directory), request=request)

    def test_real_multihop_is_required_before_agent_invocation(self):
        store = Mock()
        store.random_scene_event_path.return_value = (SceneNode(name="仓储", words=("仓储",)),)
        generator = TaskGenerator(store, None, generation_backend="code_agent")
        with patch("env_factory.generation.code_agent.generate") as agent:
            with self.assertRaisesRegex(TaskGenerationError, "full multi-hop"):
                generator.generate(hops=2)
            agent.assert_not_called()


if __name__ == "__main__":
    unittest.main()
