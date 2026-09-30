import copy
import tempfile
import unittest
from pathlib import Path
from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import BusinessGoalEvaluator, DeclarativeMetricEvaluator
from .test_dynamic_state_goal import dynamic_source

class InitialStateGoalTest(unittest.TestCase):
    def test_initial_target_accepts_correct_increment_and_rejects_stale_or_missing_baseline(self):
        lookup = {"lookup": {"table": "specs", "field": "plies", "where": {"product": "P-1"}}}
        expression = {"op": "add", "args": [{"initial": lookup}, {"literal": 2}]}
        predicate = {"table": "specs", "where": {"product": "P-1"}, "values": {},
                     "value_expressions": {"plies": expression}, "count": 1}
        rule = {"metric_id": "answer", "source": "final_agent_response", "path": "$",
                "operator": "value_targets", "expected": {"answer_format": "json_object",
                "targets": [{"key": "plies", "expression": expression}]}}
        import json
        for before in (4, 6, 8):
            baseline = {"specs": [{"product": "P-1", "plies": before}]}
            state = {"specs": [{"product": "P-1", "plies": before + 2}]}
            self.assertTrue(BusinessGoalEvaluator.evaluate([predicate], state, baseline))
            self.assertFalse(BusinessGoalEvaluator.evaluate([predicate], baseline, baseline))
            self.assertFalse(BusinessGoalEvaluator.evaluate([predicate], state))
            context = {"business_state": state, "initial_business_state": baseline,
                       "final_agent_response": json.dumps({"plies": before + 2})}
            self.assertEqual(DeclarativeMetricEvaluator().evaluate(rule, context), 1)
            context["final_agent_response"] = json.dumps({"plies": before + 4})
            self.assertEqual(DeclarativeMetricEvaluator().evaluate(rule, context), 0)

    def test_authoring_compiles_initial_from_tool_and_executes_increment(self):
        source, request = dynamic_source(5)
        source["tool_implementations"][1]["projection"] = ["temperature"]
        source["tool_implementations"][2].pop("result_field")
        steps = source["scenarios"][0]["steps"]
        steps[1]["capture"] = {"target_temperature": "$.records[0].temperature"}
        steps[-1]["content"] = "温度：23摄氏度。"
        source["semantic_goal"]["row_predicates"][0]["value_expressions"]["temperature"] = {
            "op": "add", "args": [{"initial": {"from_tool": {"name": "read_temperature", "field": "temperature"}}}, 5]}
        for scenario in source["scenarios"]:
            scenario["steps"].insert(0, {"operation": "reset", "body": {"seed": 17}})
            scenario["steps"].append({"operation": "reward", "step_id": "reward"})
        with tempfile.TemporaryDirectory() as directory:
            result = compile_source(source, root=Path(directory), request=request)
            self.assertEqual(result["tool_implementations"][2]["result_field"], "updated_count")
            self.assertIn("initial", str(result["task_spec"]["goal_contract"]))
            from env_factory.generation.delivery_preflight import verify_delivery
            verify_delivery(result, Path(directory))

    def test_typed_state_reward_compiles_dynamic_target_and_ignores_row_order(self):
        source, request = dynamic_source(5)
        source["metric_implementations"] = [{"metric_id": "temperature_answer", "source": "business_state",
            "path": "$", "operator": "state_predicates", "score_mapping": {"pass": 1, "fail": 0},
            "expected": copy.deepcopy(source["semantic_goal"]["row_predicates"])}]
        for scenario in source["scenarios"]:
            scenario["steps"].insert(0, {"operation": "reset", "body": {"seed": 17}})
            scenario["steps"].append({"operation": "reward", "step_id": "reward"})
        with tempfile.TemporaryDirectory() as directory:
            artifacts = compile_source(source, root=Path(directory), request=request)
            from env_factory.generation.delivery_preflight import verify_delivery
            verify_delivery(artifacts, Path(directory))
            rule = artifacts["metric_implementations"][0]
            baseline = {"zones": copy.deepcopy(source["tables"][0]["rows"])}
            state = copy.deepcopy(baseline)
            state["zones"].reverse()
            target = next(row for row in state["zones"] if row["name"] == "青松库")
            target["configured_temperature"] = 40
            target["temperature"] = 45
            context = {"business_state": state, "initial_business_state": baseline}
            self.assertEqual(DeclarativeMetricEvaluator().evaluate(rule, context), 1)
            target["temperature"] = 30
            self.assertEqual(DeclarativeMetricEvaluator().evaluate(rule, context), 0)
            invalid = copy.deepcopy(rule)
            invalid["expected"][0]["value_expressions"]["nonexistent"] = {"literal": 1}
            source["metric_implementations"] = [invalid]
            with self.assertRaisesRegex(ValueError, "invalid or duplicate columns"):
                compile_source(source, root=Path(directory), request=request)
