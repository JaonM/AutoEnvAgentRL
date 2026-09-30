import unittest

from env_factory.generation.input_grounding import validate_query_inputs


def source():
    return {"description": {"public_input": {"initial_user_message": "Find calm, reflective passages at old harbor."}},
            "tools": [{"function": {"name": "search", "description": "Search passages.",
                "parameters": {"properties": {"mood": {"type": "string"}}}}}],
            "tool_implementations": [{"tool_name": "search", "operation": "select",
                "filters": [{"argument": "mood", "operator": "eq", "column": "mood"}]}],
            "scenarios": [{"kind": "goal_success", "steps": [
                {"operation": "tool_call", "tool_name": "search", "arguments": {"mood": "calm and reflective"}}]}]}


class QueryInputGroundingTest(unittest.TestCase):
    def test_hidden_exact_selector_is_rejected_despite_similar_public_words(self):
        with self.assertRaisesRegex(ValueError, "QUERY_INPUT_UNDISCOVERABLE"):
            validate_query_inputs(source())

    def test_public_domain_enum_or_broad_query_is_discoverable(self):
        s = source()
        s["tools"][0]["function"]["parameters"]["properties"]["mood"]["enum"] = ["calm and reflective", "energetic"]
        self.assertEqual(validate_query_inputs(s)[0]["origin"], "public_input_or_tool_contract")
        s = source()
        s["scenarios"][0]["steps"][0]["arguments"] = {}
        self.assertEqual(validate_query_inputs(s), [])

    def test_capture_requires_a_preceding_tool_result(self):
        s = source()
        step = s["scenarios"][0]["steps"][0]
        step["arguments"]["mood"] = {"$ref": "selected_mood"}
        with self.assertRaisesRegex(ValueError, "unavailable capture"):
            validate_query_inputs(s)
        s["scenarios"][0]["steps"].insert(0, {"operation": "tool_call", "tool_name": "search",
            "arguments": {}, "capture": {"selected_mood": "$.records[0].mood"}})
        self.assertEqual(validate_query_inputs(s)[0]["origin"], "previous_tool_capture")

    def test_identifier_substring_is_not_public_disclosure(self):
        s = source()
        s["description"]["public_input"]["initial_user_message"] = "Lookup A10"
        s["scenarios"][0]["steps"][0]["arguments"]["mood"] = "A1"
        with self.assertRaisesRegex(ValueError, "QUERY_INPUT_UNDISCOVERABLE"):
            validate_query_inputs(s)
