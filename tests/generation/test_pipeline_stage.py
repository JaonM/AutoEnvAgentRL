"""Boundary cases for the shared model-stage response parser."""

import unittest

from env_factory.generation.pipeline_stage import normalize_stage_result, validate_stage_result_shape


class PipelineStageContractTest(unittest.TestCase):
    def test_nested_json_string_is_unwrapped(self):
        normalized = normalize_stage_result(
            {"result": '{"output":{"metrics":[],"observation_schema":{},"reward_formula":{}}}'},
            payload={"output": {
                "metrics": [], "observation_schema": {}, "reward_formula": {},
            }},
        )
        self.assertEqual(set(normalized), {"metrics", "observation_schema", "reward_formula"})

    def test_required_and_optional_arrays_have_distinct_empty_rules(self):
        with self.assertRaisesRegex(ValueError, "non-empty array"):
            validate_stage_result_shape(
                {"actions": []},
                payload={"output": {"actions": [{"name": "string"}]}},
            )
        validate_stage_result_shape(
            {"tool_bindings": []},
            payload={"output": {"tool_bindings": [{"tool_name": "string"}]}},
        )
