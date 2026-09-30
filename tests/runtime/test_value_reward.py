import copy
import json
import unittest

from env_factory.sandbox_runtime import DeclarativeMetricEvaluator as Evaluator, SandboxError


def lookup(table, field, where):
    return {"lookup": {"table": table, "field": field, "where": where}}


class ValueRewardTest(unittest.TestCase):
    def test_query_predicates_and_empty_aggregate_preserve_tool_semantics(self):
        state = {"offers": [{"name": "small", "portions": 18, "diet": "vegetarian"},
                            {"name": "large", "portions": 32, "diet": "vegetarian"},
                            {"name": "other", "portions": 40, "diet": "meat"}]}
        where = {"portions": {"gte": 24, "lte": 36}, "diet": {"in": ["vegetarian", "vegan"]}}
        expression = lookup("offers", "name", where)
        self.assertEqual(Evaluator._value_expression(expression, state), "large")
        state["offers"][1]["portions"] = 20
        self.assertIsNone(Evaluator._value_expression(expression, state))
        for op in ("count", "sum"):
            aggregate = {"aggregate": {"table": "offers", "field": "portions", "where": where, "op": op}}
            self.assertEqual(Evaluator._value_expression(aggregate, state), 0)
            self.assertIsNone(Evaluator._value_expression(aggregate, {}))
        singleton = lookup("settings", "ready", {})
        self.assertTrue(Evaluator._value_expression(singleton, {"settings": [{"ready": True}]}))
        self.assertIsNone(Evaluator._value_expression(singleton, {"settings": [{"ready": True}, {"ready": False}]}))

    def setUp(self):
        self.state = {"stalls": [{"name": "青松", "lot": "L-7", "units": 18}],
                      "inspections": [{"lot": "L-7", "status": "cleared", "inspector": "陈甲"},
                                      {"lot": "L-8", "status": "hold", "inspector": "刘乙"}]}
        lot = lookup("stalls", "lot", {"name": "青松"})
        self.status = lookup("inspections", "status", {"lot": lot})
        self.condition = {"op": "eq", "args": [self.status, {"literal": "cleared"}]}
        self.capacity = {"if": {"condition": self.condition,
            "then": lookup("stalls", "units", {"name": "青松"}), "else": {"literal": 0}}}
        self.spec = {"source": "final_agent_response", "path": "$", "operator": "value_targets",
            "expected": {"answer_format": "json_object", "targets": [
                {"key": "status", "expression": self.status},
                {"key": "approved", "expression": self.condition},
                {"key": "capacity", "expression": self.capacity}]}, "score_mapping": {"pass": 1, "fail": 0}}

    def score(self, answer, state=None):
        return Evaluator().evaluate(self.spec, {"final_agent_response": answer if isinstance(answer, str) else json.dumps(answer),
            "business_state": self.state if state is None else state})

    def test_actual_text_boolean_and_conditional_numeric_values(self):
        self.assertEqual(self.score({"status": "cleared", "approved": True, "capacity": 18}), 1)
        changed = copy.deepcopy(self.state)
        changed["inspections"][0]["status"] = "hold"
        self.assertEqual(self.score({"status": "cleared", "approved": True, "capacity": 18}, changed), 0)
        self.assertEqual(self.score({"status": "hold", "approved": False, "capacity": 0}, changed), 1)

    def test_changed_string_foreign_key_changes_answer(self):
        changed = copy.deepcopy(self.state)
        changed["stalls"][0]["lot"] = "L-8"
        self.assertEqual(self.score({"status": "hold", "approved": False, "capacity": 0}, changed), 1)
        self.assertEqual(self.score({"status": "cleared", "approved": True, "capacity": 18}, changed), 0)

    def test_missing_and_ambiguous_private_lookup_fail_closed(self):
        for changed in ({}, {**self.state, "inspections": self.state["inspections"] * 2}):
            self.assertEqual(self.score({"status": "cleared", "approved": True, "capacity": 18}, changed), 0)

    def test_metrics_can_own_separate_fields_without_allowing_unowned_claims(self):
        specs = [{**self.spec, "metric_id": target["key"], "expected": {"answer_format": "json_object", "targets": [target]}}
                 for target in self.spec["expected"]["targets"]]
        context = {"business_state": self.state,
                   "final_agent_response": '{"status":"cleared","approved":true,"capacity":18}'}
        self.assertEqual(Evaluator().evaluate_all(specs, context), {"status": 1, "approved": 1, "capacity": 1})
        context["final_agent_response"] = '{"status":"cleared","approved":true,"capacity":19}'
        self.assertEqual(Evaluator().evaluate_all(specs, context), {"status": 1, "approved": 1, "capacity": 0})
        context["final_agent_response"] = '{"status":"cleared","approved":true,"capacity":18,"invented":"yes"}'
        self.assertEqual(Evaluator().evaluate_all(specs, context), {"status": 0, "approved": 0, "capacity": 0})

    def test_rejects_extra_claim_duplicate_key_wrong_type_and_nonfinite(self):
        for answer in [
            {"status": "cleared", "approved": True, "capacity": 18, "invented": "yes"},
            {"status": "cleared", "approved": 1, "capacity": 18},
            {"status": "cleared", "approved": True, "capacity": "18"},
            {"status": "cleared", "approved": True, "capacity": float("nan")},
            '{"status":"hold","status":"cleared","approved":true,"capacity":18}',
            'Here is the result: {"status":"cleared","approved":true,"capacity":18}',
        ]:
            with self.subTest(answer=answer):
                self.assertEqual(self.score(answer), 0)

    def test_numeric_targets_support_condition_and_text_join(self):
        spec = {**self.spec, "operator": "numeric_targets", "expected": {"answer_format": "single_labeled_number",
            "targets": [{"label": "容量", "unit": "份", "tolerance": 0, "expression": self.capacity}]}}
        self.assertEqual(Evaluator().evaluate(spec, {"final_agent_response": "容量：18份", "business_state": self.state}), 1)
        changed = copy.deepcopy(self.state)
        changed["inspections"][0]["status"] = "hold"
        self.assertEqual(Evaluator().evaluate(spec, {"final_agent_response": "容量：18份", "business_state": changed}), 0)
        self.assertEqual(Evaluator().evaluate(spec, {"final_agent_response": "容量：0份", "business_state": changed}), 1)

    def test_structured_literals_and_dynamic_arrays_preserve_types(self):
        spec = {**self.spec, "expected": {"answer_format": "json_object", "targets": [
            {"key": "parts", "expression": {"literal": ["男中音", "男高音", "男低音"]}},
            {"key": "evidence", "expression": {"array": [self.status, self.condition, self.capacity]}},
            {"key": "metadata", "expression": {"literal": {"ordered": True, "count": 3}}}]}}
        valid = {"parts": ["男中音", "男高音", "男低音"], "evidence": ["cleared", True, 18],
                 "metadata": {"ordered": True, "count": 3}}
        def score(value, state=None):
            return Evaluator().evaluate(spec, {"final_agent_response": json.dumps(value),
                "business_state": self.state if state is None else state})
        self.assertEqual(score(valid), 1)
        for field, value in [("parts", "男中音, 男高音, 男低音"),
                             ("parts", list(reversed(valid["parts"]))),
                             ("evidence", ["cleared", 1, 18]),
                             ("metadata", {"ordered": 1, "count": 3}),
                             ("metadata", {"ordered": True, "count": float("nan")})]:
            self.assertEqual(score({**valid, field: value}), 0)
        changed = copy.deepcopy(self.state)
        changed["inspections"][0]["status"] = "hold"
        self.assertEqual(score(valid, changed), 0)
        self.assertEqual(score({**valid, "evidence": ["hold", False, 0]}, changed), 1)
        self.assertEqual(score(valid, {}), 0)

    def test_structured_literal_limits_and_nested_nonfinite_are_rejected(self):
        nested = "x"
        for _ in range(14):
            nested = [nested]
        for value in [[None], {"value": float("inf")}, [0] * 65, nested]:
            with self.subTest(value=value), self.assertRaises(SandboxError):
                Evaluator._value_expression({"literal": value}, {})

    def test_inactive_branch_and_recursion_are_validated(self):
        bad = copy.deepcopy(self.spec["expected"])
        bad["targets"][-1]["expression"]["if"]["else"] = {"eval": "unsafe"}
        with self.assertRaises(SandboxError):
            Evaluator.validate_value_targets(bad)
        expression = {"literal": True}
        for _ in range(20):
            expression = {"op": "and", "args": [expression, {"literal": True}]}
        with self.assertRaises(SandboxError):
            Evaluator._value_expression(expression, {})


if __name__ == "__main__":
    unittest.main()
