import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from env_factory.task_pipeline import TaskGenerationPipeline
from env_factory.contracts.runtime_contract import missing_system_endpoints
from env_factory.contracts.reward_contract import reward_contract_issues, terminal_outcome_weight


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "assess_task_buildability", ROOT / "scripts/sandbox/assess_task_buildability.py"
)
buildability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(buildability)


class BuildabilityTest(unittest.TestCase):
    def test_captured_tool_rows_must_fit_downstream_schema(self):
        task = {
            "task_spec": {"tool_contracts": [
                {"name": "lookup", "output_contract": {"schema": {
                    "type": "object", "properties": {"records": {
                        "type": "array", "items": {"type": "object",
                            "properties": {"name": {"type": "string"}, "id": {"type": "string"}},
                            "required": ["name", "id"]},
                    }},
                }}},
                {"name": "update", "input_schema": {"type": "object", "properties": {
                    "records": {"type": "array", "items": {"type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"], "additionalProperties": False}},
                }}},
            ]},
            "acceptance_contract": {"executable_scenarios": [{
                "scenario_id": "goal", "kind": "goal_success", "steps": [
                    {"operation": "tool_call", "tool_name": "lookup",
                     "arguments": {}, "capture": {"found": "$.records"}},
                    {"operation": "tool_call", "tool_name": "update",
                     "arguments": {"records": {"$ref": "found"}}},
                ],
            }]},
        }
        issues = buildability._tool_chain_issues(task)
        self.assertEqual([item["code"] for item in issues], ["TASK_TOOL_CHAIN_SCHEMA"])
        self.assertIn("id", issues[0]["evidence"]["conflict"])
        target = task["task_spec"]["tool_contracts"][1]["input_schema"]["properties"]["records"]["items"]
        target["properties"]["id"] = {"type": "string"}
        self.assertEqual(buildability._tool_chain_issues(task), [])

    def test_generator_runtime_interface_covers_builder_minimum(self):
        interface = TaskGenerationPipeline._build_runtime_interface([], {})
        self.assertEqual(missing_system_endpoints(interface), set())

    def test_reward_contract_rejects_literal_process_and_missing_terminal(self):
        task = {
            "environment_plan": {"mode": "reference_data"},
            "metrics": [
                {"id": "process_parse", "category": "process", "type": "rule-based"},
                {"id": "outcome_answer", "category": "outcome", "type": "rule-based"},
            ],
            "metric_implementations": [{
                "metric_id": "process_parse", "operator": "contains_tool_call",
                "expected": {"tool_name": "parse_csv", "arguments": {
                    "csv_text": "name,quantity\n" + "item,1\n" * 15,
                }},
            }],
        }
        self.assertEqual({item["code"] for item in reward_contract_issues(task)}, {
            "TASK_TERMINAL_REWARD_UNDECLARED", "TASK_LITERAL_PROCESS_REWARD",
        })
        task["metrics"][1].update({
            "type": "model-based", "evaluation_inputs": ["final_agent_response"],
            "weight": 0.6,
        })
        task["metric_implementations"] = []
        self.assertEqual(reward_contract_issues(task), [])
        self.assertEqual(terminal_outcome_weight(task), 0.6)
        task["environment_plan"]["mode"] = "stateful"
        self.assertEqual(terminal_outcome_weight(task), 0.0)

    def write_task(self, root: Path, mode: str = "stateless") -> None:
        (root / "task.json").write_text(json.dumps({
            "task": "整理公开输入",
            "environment_plan": {"mode": mode},
            "task_spec": {},
            "requirements": {"runtime_interface": TaskGenerationPipeline._build_runtime_interface([], {})},
            "tools": [], "noise_tools": [], "tool_implementations": [],
            "metrics": [{"id": "outcome_answer", "category": "outcome",
                         "type": "model-based", "evaluation_inputs": ["final_agent_response"]}],
            "metric_implementations": [], "artifacts": {},
        }), encoding="utf-8")

    def test_external_task_without_provider_is_owned_by_task_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "external_capability")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
                patch.dict(os.environ, {}, clear=True),
            ):
                report = buildability.assess(root)
            self.assertFalse(report["buildable"])
            self.assertEqual(report["failure_owner"], "task_generation")
            self.assertIn("EXTERNAL_CAPABILITY_UNAVAILABLE", {
                item["code"] for item in report["issues"]
            })

    def test_valid_platform_contract_reaches_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root)
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
            ):
                report = buildability.assess(root)
            self.assertTrue(report["buildable"])
            self.assertEqual(report["issues"], [])

    def test_unresolved_business_probe_is_rejected_before_code_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root)
            task = json.loads((root / "task.json").read_text(encoding="utf-8"))
            task["tools"] = [{"function": {"name": "validate_batch", "parameters": {
                "type": "object", "properties": {"batch_id": {"type": "string"}},
                "required": ["batch_id"],
            }}}]
            task["acceptance_contract"] = {"argument_probes": [{
                "probe_id": "validate_batch.valid_shape.argument_sensitivity",
                "tool_name": "validate_batch",
                "arguments": {"batch_id": "任务输入中的batch_id"},
            }]}
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
                patch.object(TaskGenerationPipeline, "_validate_business_tool_semantics"),
            ):
                report = buildability.assess(root)
            self.assertIn("TASK_ACCEPTANCE_FIXTURE", {
                item["code"] for item in report["issues"]
            })

    def test_missing_trainer_state_endpoint_fails_before_code_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root)
            task = json.loads((root / "task.json").read_text(encoding="utf-8"))
            endpoints = task["requirements"]["runtime_interface"]["endpoints"]
            task["requirements"]["runtime_interface"]["endpoints"] = [
                item for item in endpoints if item.get("name") != "state"
            ]
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
            ):
                report = buildability.assess(root)
            self.assertFalse(report["buildable"])
            self.assertEqual(report["failure_owner"], "task_generation")
            issue = next(item for item in report["issues"] if item["code"] == "TASK_RUNTIME_INTERFACE")
            self.assertIn(("state", "GET", "/v1/state"), issue["evidence"]["missing"])

    def test_project_relative_manifest_is_resolved_from_self_contained_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "reference_data")
            schema = root / "schemas" / "items.json"
            schema.parent.mkdir()
            schema.write_text(json.dumps({
                "columns": [{"name": "id", "type": "INTEGER", "nullable": False}],
                "primary_key": ["id"],
            }), encoding="utf-8")
            rows = root / "rows" / "items.jsonl"
            rows.parent.mkdir()
            rows.write_text('{"id":1}\n', encoding="utf-8")
            task = json.loads((root / "task.json").read_text(encoding="utf-8"))
            task["artifacts"] = {"data_manifest": {
                "root": "output/task_artifacts/task-42",
                "tables": [{"table_name": "items", "schema_file": "schemas/items.json",
                            "rows_file": "rows/items.jsonl"}],
            }}
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
            ):
                report = buildability.assess(root)
            self.assertTrue(report["buildable"])
            self.assertEqual(report["issues"], [])

    def test_invalid_business_foreign_key_fails_before_builder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "reference_data")
            (root / "schemas").mkdir()
            (root / "rows").mkdir()
            (root / "schemas/users.json").write_text(json.dumps({
                "columns": [{"name": "id", "type": "INTEGER", "nullable": False}],
                "primary_key": ["id"],
            }), encoding="utf-8")
            (root / "rows/users.jsonl").write_text('{"id":1}\n', encoding="utf-8")
            (root / "schemas/orders.json").write_text(json.dumps({
                "columns": [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "user_id", "type": "INTEGER", "nullable": False},
                ],
                "primary_key": ["id"],
                "foreign_keys": [{"column": "user_id", "ref_table": "users", "ref_column": "id"}],
            }), encoding="utf-8")
            (root / "rows/orders.jsonl").write_text('{"id":1,"user_id":2}\n', encoding="utf-8")
            task = json.loads((root / "task.json").read_text(encoding="utf-8"))
            task["artifacts"] = {"data_manifest": {"root": ".", "tables": [
                {"table_name": "users", "schema_file": "schemas/users.json", "rows_file": "rows/users.jsonl"},
                {"table_name": "orders", "schema_file": "schemas/orders.json", "rows_file": "rows/orders.jsonl"},
            ]}}
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
            ):
                report = buildability.assess(root)
            self.assertFalse(report["buildable"])
            self.assertIn("BUSINESS_DATA_INVALID", {item["code"] for item in report["issues"]})


if __name__ == "__main__":
    unittest.main()
