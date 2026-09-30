import tempfile
import unittest
from pathlib import Path
from env_factory.generation.agent_authoring import compile_source, expand_capture_fields
from env_factory.generation.delivery_preflight import verify_delivery
from .test_dynamic_state_goal import dynamic_source

class CaptureFieldsTest(unittest.TestCase):
    def test_object_and_array_fields_compile_to_real_executable_dependencies(self):
        for capture, path, reference in [("zone", "$.records[0]", "zone.id"),
                                         ("zones", "$.records", "zones[0].id")]:
            source, request = dynamic_source(5)
            steps = source["scenarios"][0]["steps"]
            steps[0]["capture"] = {capture: path}
            steps[1]["arguments"]["id"] = {"$ref": reference}
            steps[2]["arguments"]["id"] = {"$ref": reference}
            steps[1]["capture"] = {"configuration": "$.records[0]"}
            steps[2]["arguments"]["value"]["$expr"]["args"][0] = {"$ref": "configuration.configured_temperature"}
            for scenario in source["scenarios"]:
                scenario["steps"].insert(0, {"operation": "reset", "body": {"seed": 17}})
                scenario["steps"].append({"operation": "reward"})
            with self.subTest(reference=reference), tempfile.TemporaryDirectory() as directory:
                artifacts = compile_source(source, root=Path(directory), request=request)
                verify_delivery(artifacts, Path(directory))
                self.assertIn("$.records[0].configured_temperature", str(artifacts["task_spec"]["capability_dag"]))

    def test_compact_source_derives_process_rules_from_object_fields(self):
        source, request = dynamic_source()
        calls = source["scenarios"][0]["steps"][:-1]
        calls[0]["capture"] = {"zone": "$.records[0]"}
        calls[1]["arguments"]["id"] = {"$ref": "zone.id"}
        calls[2]["arguments"]["id"] = {"$ref": "zone.id"}
        compact = {key: source[key] for key in ["version", "description", "environment_plan", "tables", "semantic_goal"]}
        compact["business_tools"] = [{"name": tool["function"]["name"], "description": tool["function"]["description"],
            "parameters": tool["function"]["parameters"], "implementation": impl}
            for tool, impl in zip(source["tools"], source["tool_implementations"])]
        compact["outcomes"] = [{"id": "temperature_answer", "rubric": "报告实际持久化的温度", "rule": source["metric_implementations"][0]}]
        compact["reference"] = {"calls": calls, "answer": source["scenarios"][0]["steps"][-1]["content"]}
        with tempfile.TemporaryDirectory() as directory:
            artifacts = compile_source(compact, root=Path(directory), request=request)
            verify_delivery(artifacts, Path(directory))
            rule = next(rule for rule in artifacts["metric_implementations"] if rule["metric_id"] == "process_read_temperature")
            self.assertEqual(rule["expected"]["captures"][0]["path"], "$.records[0].id")

    def test_explicit_alias_wins_and_generated_alias_avoids_collision(self):
        calls = [{"capture": {"row": "$.records[0]", "compiled_capture_field_0": "$.count", "row.id": "$.explicit_id"}},
                 {"arguments": {"a": {"$ref": "row.id"}, "b": {"$ref": "row.name"}}}]
        expand_capture_fields(calls)
        self.assertEqual(calls[1]["arguments"]["a"], {"$ref": "row.id"})
        alias = calls[1]["arguments"]["b"]["$ref"]
        self.assertNotEqual(alias, "compiled_capture_field_0")
        self.assertEqual(calls[0]["capture"][alias], "$.records[0].name")

    def test_unavailable_capture_is_not_invented(self):
        calls = [{"arguments": {"id": {"$ref": "later.id"}}}, {"capture": {"later": "$.record"}}]
        expand_capture_fields(calls)
        self.assertEqual(calls[0]["arguments"]["id"], {"$ref": "later.id"})
        self.assertEqual(calls[1]["capture"], {"later": "$.record"})
