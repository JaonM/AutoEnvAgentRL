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
from env_factory.contracts.reward_contract import ambiguous_metric_captures, reward_contract_issues, terminal_outcome_weight


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "assess_task_buildability", ROOT / "scripts/sandbox/assess_task_buildability.py"
)
buildability = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(buildability)


class BuildabilityTest(unittest.TestCase):
    def test_multirow_positional_reward_capture_is_rejected_before_build(self):
        task = {"metric_implementations": [{
            "metric_id": "process_parts", "expected": {"captures": [{
                "name": "order_id", "tool_name": "query_orders",
                "path": "$.records[0].order_id",
            }]},
        }]}
        preview = [{"tool_name": "query_orders", "result": {"records": [
            {"order_id": 1}, {"order_id": 3},
        ]}}]
        issues = ambiguous_metric_captures(task, preview)
        self.assertEqual([issue["code"] for issue in issues], ["AMBIGUOUS_METRIC_CAPTURE"])
        self.assertEqual(issues[0]["metric_id"], "process_parts")
        preview[0]["result"]["records"] = [{"order_id": 3}]
        self.assertEqual(ambiguous_metric_captures(task, preview), [])

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
        self.assertEqual(terminal_outcome_weight(task), 0.6)

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

    def test_manifest_root_never_uses_shadow_business_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rows").mkdir()
            (root / "rows/items.jsonl").write_text('{"id": 999}\n', encoding="utf-8")
            manifest = {"root": "data/business_data", "tables": [{
                "table_name": "items", "rows_file": "rows/items.jsonl",
            }]}
            self.assertEqual(
                buildability._manifest_root(root, manifest),
                root / "data/business_data",
            )

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

    def test_conditional_noop_stateful_task_is_rejected_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "stateful")
            task_path = root / "task.json"
            task = json.loads(task_path.read_text(encoding="utf-8"))
            task["task"] = "若状态为待确认则更新；若已确认则保持现状并说明。"
            task["public_input"] = {"initial_user_message":
                                    "如果已经确认就不用动，否则请更新。", "materials": []}
            task_path.write_text(json.dumps(task), encoding="utf-8")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            with patch.object(buildability, "score_file", return_value=quality), \
                 patch.object(buildability, "validate_task_spec"):
                report = buildability.assess(root)
            self.assertIn("STATEFUL_NOOP_BRANCH", {item["code"] for item in report["issues"]})

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

    def test_empty_success_capture_is_rejected_before_code_agent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_task(root, "reference_data")
            task = json.loads((root / "task.json").read_text(encoding="utf-8"))
            task["artifacts"] = {"data_manifest": {"root": ".", "tables": []}}
            task["tools"] = [{"function": {"name": "lookup", "parameters": {
                "type": "object", "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            }}}]
            task["tool_implementations"] = [{
                "tool_name": "lookup", "operation": "select", "table": "items",
                "result_field": "records", "filters": [{
                    "argument": "item_id", "column": "id", "operator": "eq",
                }],
            }]
            task["acceptance_contract"] = {"executable_scenarios": [{
                "kind": "goal_success", "steps": [
                    {"operation": "reset", "body": {}},
                    {"operation": "tool_call", "tool_name": "lookup",
                     "arguments": {"item_id": "missing"},
                     "capture": {"found": "$.records[0].id"}},
                ],
            }]}
            (root / "task.json").write_text(json.dumps(task), encoding="utf-8")
            quality = SimpleNamespace(to_dict=lambda: {"eligible": True, "score": 9.0})
            data = SimpleNamespace(baseline={"items": [{"id": "known"}]}, schemas={})
            with (
                patch.object(buildability, "score_file", return_value=quality),
                patch.object(buildability, "validate_task_spec"),
                patch.object(buildability, "ManifestDataStore", return_value=data),
                patch.object(TaskGenerationPipeline, "_validate_business_tool_semantics"),
            ):
                report = buildability.assess(root)
            self.assertIn("SUCCESS_TOOL_PREVIEW_INVALID", {
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

    def test_manifest_with_missing_declared_root_cannot_borrow_task_root(self):
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
            self.assertFalse(report["buildable"])
            self.assertIn("BUSINESS_DATA_MISSING", {
                item["code"] for item in report["issues"]
            })

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
