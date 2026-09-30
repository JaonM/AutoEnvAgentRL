import copy
import json
import tempfile
import unittest
from pathlib import Path

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.answer_contract import bind_answer_contract
from .test_agent_authoring import fixture


def answer_source():
    source, request = fixture("direct_response")
    source["description"]["public_input"]["initial_user_message"] = (
        "给定声部顺序为男中音、男高音、男低音。按原顺序列出声部。")
    source["answer_contract"] = {"format": "json_object", "schema": {
        "type": "object", "properties": {"voice_types": {"type": "array", "items": {"type": "string"},
            "description": "按给定顺序列出的声部"}}, "required": ["voice_types"], "additionalProperties": False}}
    source["metric_implementations"][0].update(operator="value_targets", expected={
        "answer_format": "json_object", "targets": [{"key": "voice_types",
            "expression": {"literal": ["男中音", "男高音", "男低音"]}}]})
    source["scenarios"][0]["steps"][0]["content"] = json.dumps({"voice_types": ["男中音", "男高音", "男低音"]})
    return source, request


class AnswerContractTest(unittest.TestCase):
    def test_private_canonical_summary_text_is_rejected_before_schema_publication(self):
        from env_factory.generation.answer_contract import validate_direct_text_targets
        source, _ = answer_source()
        source["answer_contract"]["schema"]["properties"]["voice_types"] = {"type": "string"}
        target = source["metric_implementations"][0]["expected"]["targets"][0]
        target["expression"] = {"literal": "private canonical summary wording"}
        with self.assertRaisesRegex(ValueError, "DIRECT_EXACT_TEXT_UNDISCLOSED"):
            validate_direct_text_targets(source)
        source["description"]["public_input"]["materials"] = [{"text": "private canonical summary wording"}]
        validate_direct_text_targets(source)

    def test_public_categorical_choices_are_allowed_without_publishing_gold(self):
        from env_factory.generation.answer_contract import validate_direct_text_targets
        source, _ = answer_source()
        source["answer_contract"]["schema"]["properties"]["voice_types"] = {
            "type": "string", "enum": ["eligible", "ineligible"]}
        source["metric_implementations"][0]["expected"]["targets"][0]["expression"] = {"literal": "eligible"}
        validate_direct_text_targets(source)

    def test_decision_labels_are_public_for_both_branches_without_gold_leak(self):
        source, _ = answer_source()
        source["answer_contract"]["schema"]["properties"] = {"voice_types": {"type": "string"}}
        expression = {"if": {"condition": {"literal": True},
                             "then": {"literal": "ESCALATE"}, "else": {"literal": "HOLD"}}}
        source["metric_implementations"][0]["expected"]["targets"][0]["expression"] = expression
        source["scenarios"][0]["steps"][0]["content"] = '{"voice_types":"ESCALATE"}'
        bound = bind_answer_contract(source)
        self.assertEqual(bound["schema"]["properties"]["voice_types"]["enum"], ["ESCALATE", "HOLD"])
        self.assertIn('"enum": ["ESCALATE", "HOLD"]', source["description"]["public_input"]["initial_user_message"])

    def test_explicit_enum_cannot_hide_an_inactive_branch(self):
        source, _ = answer_source()
        source["answer_contract"]["schema"]["properties"] = {
            "voice_types": {"type": "string", "enum": ["ESCALATE", "OTHER"]}}
        source["metric_implementations"][0]["expected"]["targets"][0]["expression"] = {
            "if": {"condition": {"literal": True}, "then": {"literal": "ESCALATE"}, "else": {"literal": "HOLD"}}}
        source["scenarios"][0]["steps"][0]["content"] = '{"voice_types":"ESCALATE"}'
        with self.assertRaisesRegex(ValueError, "ANSWER_CONTRACT_ENUM"):
            bind_answer_contract(source)

    def test_delivery_preflight_rebuilds_scaffold_and_rejects_false_reference(self):
        from env_factory.generation.delivery_preflight import verify_delivery
        source, request = answer_source()
        for scenario in source["scenarios"]:
            scenario["steps"].insert(0, {"operation": "reset", "body": {"seed": 7}})
            scenario["steps"].append({"operation": "reward"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = compile_source(source, root=root, request=request)
            (root / "app.py").write_text('raise RuntimeError("do not trust author preview")')
            self.assertTrue(verify_delivery(result, root)["passed"])
            for scenario in result["acceptance_contract"]["executable_scenarios"]:
                if scenario.get("kind") == "goal_success":
                    for step in scenario["steps"]:
                        if step.get("operation") == "agent_response":
                            step["content"] = '{"voice_types":["invented"]}'
            with self.assertRaisesRegex(ValueError, "AGENT_DELIVERY_PREFLIGHT_FAILED"):
                verify_delivery(result, root)

    def test_array_answer_executes_and_schema_is_public_without_reference_leak(self):
        source, request = answer_source()
        original = copy.deepcopy(source)
        with tempfile.TemporaryDirectory() as directory:
            result = compile_source(source, root=Path(directory), request=request)
        self.assertEqual(source, original)
        self.assertTrue(result["generation_pipeline"]["verification"]["passed"])
        public = result["public_input"]
        self.assertEqual(public["answer_contract"], source["answer_contract"])
        self.assertIn(json.dumps(source["answer_contract"]["schema"], ensure_ascii=False, sort_keys=True),
                      public["initial_user_message"])
        self.assertNotIn("literal", public["initial_user_message"])
        self.assertEqual(result["task_spec"]["task_contract"]["public_input"], public)

    def test_hidden_json_format_is_rejected_before_materializing_sandbox_data(self):
        source, request = answer_source()
        source.pop("answer_contract")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "candidate"
            with self.assertRaisesRegex(ValueError, "ANSWER_CONTRACT_REQUIRED"):
                compile_source(source, root=root, request=request)
            self.assertFalse(root.exists())

    def test_reference_string_cannot_satisfy_public_array_requirement(self):
        source, request = answer_source()
        source["scenarios"][0]["steps"][0]["content"] = '{"voice_types":"男中音, 男高音, 男低音"}'
        source["metric_implementations"][0]["expected"]["targets"][0]["expression"]["literal"] = "男中音, 男高音, 男低音"
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "ANSWER_CONTRACT_REFERENCE"):
            compile_source(source, root=Path(directory), request=request)

    def test_ungraded_or_hidden_fields_and_unsupported_schema_are_rejected(self):
        for defect in ("missing_public", "ungraded", "duplicate_owner", "unsupported"):
            source, _ = answer_source()
            schema = source["answer_contract"]["schema"]
            if defect == "missing_public":
                source["metric_implementations"][0]["expected"]["targets"][0]["key"] = "hidden"
            elif defect == "ungraded":
                schema["properties"]["extra"] = {"type": "string"}
                schema["required"].append("extra")
            elif defect == "duplicate_owner":
                source["metric_implementations"].append(copy.deepcopy(source["metric_implementations"][0]))
            else:
                schema["properties"]["voice_types"]["uniqueItems"] = True
            with self.subTest(defect=defect), self.assertRaisesRegex(ValueError, "ANSWER_CONTRACT"):
                bind_answer_contract(source)
