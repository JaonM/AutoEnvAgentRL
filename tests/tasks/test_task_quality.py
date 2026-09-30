import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from env_factory.contracts.reward_contract import REFERENCE_FACTUALITY_CRITERION
from env_factory.tasks.task_quality import (
    discover_task_files, error_report, has_private_tool_dependency, score_file, score_task,
    unsourced_real_world_exemplars,
    unobservable_tool_preconditions,
)
from env_factory.task_pipeline import TaskGenerationPipeline


def valid_task():
    return {
        "training_category": "simple_agentic",
        "task": "查询公开库存记录，比较候选商品并给出有依据的采购建议。",
        "task_intent": "recommend",
        "complexity": "standard",
        "requirements": {"output_format": "markdown", "runtime_interface": TaskGenerationPipeline._build_runtime_interface([], {})},
        "environment_plan": {"mode": "reference_data"},
        "actions": [
            {"name": "读取库存"}, {"name": "筛选候选"},
            {"name": "比较候选"}, {"name": "形成建议"},
        ],
        "tools": [
            {"function": {"name": "read_inventory"}},
            {"function": {"name": "get_weather"}},
        ],
        "noise_tools": [{"name": "get_weather", "category": "unrelated"}],
        "metrics": [
            {"id": "process_read", "category": "process", "type": "hybrid", "evaluator": {}},
            {"id": "outcome", "category": "outcome", "type": "model-based",
             "evaluator": {"kind": "external_llm_judge"},
             "evaluation_inputs": ["final_agent_response", "tool_results", "business_data"],
             "criteria": [REFERENCE_FACTUALITY_CRITERION]},
            {"id": "noise", "category": "penalty", "type": "rule-based", "evaluator": {}},
        ],
        "metric_implementations": [
            {"metric_id": "process_read"},
            {"metric_id": "noise"},
        ],
        "reward_formula": {"score_range": [-1, 1]},
        "acceptance_contract": {
            "executable_scenarios": [
                {
                    "kind": "goal_success",
                    "steps": [{
                        "operation": "tool_call",
                        "tool_name": "read_inventory",
                        "arguments": {"category": "办公设备"},
                    }],
                },
                {"kind": "goal_failure"},
                {"kind": "noise_selection"},
            ],
            "mutation_tests": [{"id": "constant_reward"}],
        },
        "task_readiness": {"ready": True, "warnings": []},
        "task_spec": {
            "version": "1.0",
            "task_contract": {},
            "training_contract": {"category": "simple_agentic", "environment_archetype": "single_read"},
            "environment_contract": {"mode": "reference_data", "archetype": "single_read"},
            "tool_contracts": [{"name": "read_inventory", "role": "business"}],
            "capability_dag": {"nodes": ["read_inventory"], "edges": []},
            "goal_contract": {"expected_delta": []},
        },
    }


class TaskQualityTest(unittest.TestCase):
    def test_declared_training_category_is_required_and_matches_task_spec(self):
        task = valid_task()
        self.assertTrue(score_task(task).eligible)
        for category in (None, "nonsense"):
            task["training_category"] = category
            report = score_task(task)
            self.assertFalse(report.eligible)
            self.assertIn("training_category 缺失或无效", report.eligibility_failures)
        task["training_category"] = "simple_agentic"
        task["task_spec"]["training_contract"]["category"] = "multi_step_agentic"
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertIn("training_category 与 task_spec 类别不一致", report.eligibility_failures)

    def test_report_preserves_diagnostic_score_after_hard_gate(self):
        task = valid_task()
        task["training_category"] = "invalid"
        report = score_task(task)
        self.assertEqual(report.score, 0.0)
        self.assertGreater(report.to_dict()["raw_score"], 0.0)

    def test_task_spec_cannot_disagree_with_runtime_contract(self):
        task = valid_task()
        task["task_spec"]["task_contract"]["task"] = "查询不同的业务目标"
        task["task_spec"]["environment_contract"]["mode"] = "stateful"
        task["task_spec"]["tool_contracts"][0]["name"] = "other_tool"
        task["task_spec"]["reward_contract"] = {"metric_ids": ["different_reward"]}
        report = score_task(task)
        for reason in (
            "任务描述与 task_spec 不一致",
            "环境模式与 task_spec 不一致",
            "工具清单与 task_spec 不一致",
            "奖励指标清单与 task_spec 不一致",
        ):
            self.assertIn(reason, report.eligibility_failures)

    def test_file_score_rejects_data_task_without_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task.json"
            path.write_text(json.dumps(valid_task(), ensure_ascii=False), encoding="utf-8")
            report = score_file(path)
        self.assertEqual(report.score, 0.0)
        self.assertFalse(report.eligible)
        self.assertIn("数据环境缺少可加载的业务表 manifest", report.eligibility_failures)

    def test_reference_outcome_requires_factual_evidence_inputs(self):
        task = valid_task()
        outcome = next(item for item in task["metrics"] if item["category"] == "outcome")
        outcome["evaluator"] = {"kind": "external_llm_judge"}
        outcome["evaluation_inputs"] = ["final_agent_response", "business_data"]
        outcome["criteria"] = ["是否整理出库存清单"]
        report = score_task(task)
        self.assertIn("只读数据结果奖励缺少答案与业务证据核对：outcome",
                      report.eligibility_failures)
        TaskGenerationPipeline._ground_reference_outcome_metrics(
            task["metrics"], environment_mode="reference_data",
        )
        report = score_task(task)
        self.assertNotIn("只读数据结果奖励缺少答案与业务证据核对：outcome",
                         report.eligibility_failures)
        outcome["criteria"].append("无需核对工具结果与业务记录。")
        self.assertIn("只读数据结果奖励缺少答案与业务证据核对：outcome",
                      score_task(task).eligibility_failures)
        outcome["criteria"].pop()
        outcome["rubric"] = "不需要检查最终回答与业务数据一致。"
        self.assertIn("只读数据结果奖励缺少答案与业务证据核对：outcome",
                      score_task(task).eligibility_failures)

    def test_reference_outcome_cannot_use_literal_response_rule(self):
        task = valid_task()
        outcome = next(item for item in task["metrics"] if item["category"] == "outcome")
        outcome.update({
            "type": "rule-based",
            "evaluator": {"kind": "document_rule", "assertion": "回答包含建议"},
        })
        task["metric_implementations"].append({
            "metric_id": "outcome", "source": "final_agent_response", "path": "$",
            "operator": "contains", "expected": "建议",
            "score_mapping": {"pass": 1, "fail": 0},
        })
        report = score_task(task)
        self.assertEqual(report.score, 0.0)
        self.assertFalse(report.eligible)
        self.assertIn("只读数据结果奖励缺少答案与业务证据核对：outcome",
                      report.eligibility_failures)

    def test_reference_outcome_judge_cannot_be_overridden_by_literal_rule(self):
        task = valid_task()
        task["metric_implementations"].append({
            "metric_id": "outcome", "source": "final_agent_response", "path": "$",
            "operator": "contains", "expected": "建议",
            "score_mapping": {"pass": 1, "fail": 0},
        })
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertIn("语义奖励被声明式实现覆盖：outcome",
                      report.eligibility_failures)
        task["metrics"][1]["type"] = "rule-based"
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertIn("只读数据结果奖励缺少答案与业务证据核对：outcome",
                      report.eligibility_failures)

    def test_stateful_semantic_reward_cannot_be_overridden_by_rule(self):
        task = valid_task()
        task["environment_plan"]["mode"] = "stateful"
        task["task_spec"]["environment_contract"].update(
            mode="stateful", archetype="single_mutation",
        )
        task["tool_implementations"] = [{
            "tool_name": "read_inventory", "operation": "update",
            "table": "inventory", "changes": {"quantity": "quantity"},
        }]
        task["task_spec"]["goal_contract"] = {"row_predicates": [{
            "table": "inventory", "where": {"id": "i1"},
            "values": {"quantity": 3}, "count": 1,
        }]}
        TaskGenerationPipeline._ground_reference_outcome_metrics(
            task["metrics"], environment_mode="stateful",
        )
        self.assertTrue(score_task(task).eligible)
        task["metric_implementations"].append({
            "metric_id": "outcome", "source": "final_agent_response", "path": "$",
            "operator": "contains", "expected": "建议",
        })
        report = score_task(task)
        self.assertEqual(report.score, 0.0)
        self.assertFalse(report.eligible)
        self.assertIn("语义奖励被声明式实现覆盖：outcome",
                      report.eligibility_failures)

    def test_public_selector_echo_is_not_a_private_multistep_dependency(self):
        echo = [{"kind": "goal_success", "steps": [
            {"operation": "tool_call", "tool_name": "reviews",
             "arguments": {"dish_name": "牛肉河粉"},
             "capture": {"dish": "$.records[0].dish_name"}},
            {"operation": "tool_call", "tool_name": "menu",
             "arguments": {"dish_name": {"$ref": "dish"}}},
        ]}]
        self.assertFalse(has_private_tool_dependency(echo))
        private = [{"kind": "goal_success", "steps": [
            {"operation": "tool_call", "tool_name": "reviews",
             "arguments": {"dish_name": "牛肉河粉"},
             "capture": {"canteen": "$.records[0].canteen_id"}},
            {"operation": "tool_call", "tool_name": "menu",
             "arguments": {"canteen_id": {"$ref": "canteen"}}},
        ]}]
        self.assertTrue(has_private_tool_dependency(private))
        task = valid_task()
        task["training_category"] = "multi_step_agentic"
        task["tools"].insert(1, {"function": {"name": "menu"}})
        task["acceptance_contract"]["executable_scenarios"][0]["steps"] = echo[0]["steps"]
        report = score_task(task)
        self.assertIn("多步工具依赖只回传前序查询的公开选择器", report.eligibility_failures)

    def _write_manifest_task(self, root: Path, *, child_parent_id: int) -> Path:
        task = valid_task()
        task["artifacts"] = {"data_manifest": {
            "version": "1.0", "environment_mode": "reference_data", "root": ".",
            "tables": [
                {"table_name": "parent", "schema_file": "schemas/parent.json",
                 "rows_file": "rows/parent.jsonl"},
                {"table_name": "child", "schema_file": "schemas/child.json",
                 "rows_file": "rows/child.jsonl"},
            ],
        }}
        (root / "schemas").mkdir(parents=True)
        (root / "rows").mkdir()
        (root / "schemas" / "parent.json").write_text(json.dumps({
            "table_name": "parent",
            "columns": [{"name": "id", "type": "INTEGER", "nullable": False}],
            "primary_key": ["id"], "foreign_keys": [], "indexes": [], "constraints": [],
        }), encoding="utf-8")
        (root / "schemas" / "child.json").write_text(json.dumps({
            "table_name": "child",
            "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "parent_id", "type": "INTEGER", "nullable": False},
            ],
            "primary_key": ["id"],
            "foreign_keys": [{
                "column": "parent_id", "references_table": "parent", "references_column": "id",
            }],
            "indexes": [], "constraints": ["parent_id > 0"],
        }), encoding="utf-8")
        (root / "rows" / "parent.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
        (root / "rows" / "child.jsonl").write_text(
            json.dumps({"id": 10, "parent_id": child_parent_id}) + "\n", encoding="utf-8"
        )
        task_path = root / "task.json"
        task_path.write_text(json.dumps(task), encoding="utf-8")
        return task_path

    def test_score_file_accepts_data_valid_under_shared_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            report = score_file(self._write_manifest_task(Path(directory), child_parent_id=1))
        self.assertTrue(report.eligible)

    def test_file_score_rejects_insert_missing_non_generated_primary_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            schema_path = path.parent / "schemas" / "child.json"
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            schema["columns"][0]["type"] = "VARCHAR(32)"
            schema_path.write_text(json.dumps(schema), encoding="utf-8")
            (path.parent / "rows" / "child.jsonl").write_text(
                '{"id":"C-10","parent_id":1}\n', encoding="utf-8",
            )
            task = json.loads(path.read_text(encoding="utf-8"))
            task["tool_implementations"] = [{
                "tool_name": "read_inventory", "operation": "insert", "table": "child",
                "values": {"item_id": "parent_id"}, "result_field": "records",
            }]
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
        self.assertFalse(report.eligible)
        self.assertTrue(any("声明式插入缺少存储列" in reason and "id" in reason
                            for reason in report.eligibility_failures))

    def test_file_score_rejects_optional_declarative_mutation_argument(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            task["tools"][0]["function"]["parameters"] = {
                "properties": {"id": {"type": "integer"}, "parent_id": {"type": "integer"}},
                "required": ["parent_id"],
            }
            task["tool_implementations"] = [{
                "tool_name": "read_inventory", "operation": "insert", "table": "child",
                "values": {"id": "id", "parent_id": "parent_id"}, "result_field": "records",
            }]
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
        self.assertFalse(report.eligible)
        self.assertTrue(any("声明式写入参数未设为必填" in reason and "id" in reason
                            for reason in report.eligibility_failures))

    def test_file_score_rejects_ambiguous_reward_capture_before_build(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            task["metric_implementations"][0]["expected"] = {"captures": [{
                "name": "chosen_id", "tool_name": "read_inventory",
                "path": "$.records[0].id",
            }]}
            path.write_text(json.dumps(task), encoding="utf-8")
            preview = [{"tool_name": "read_inventory", "result": {"records": [
                {"id": 1}, {"id": 3},
            ]}}]
            with patch.object(TaskGenerationPipeline, "_preview_success_tool_results", return_value=preview):
                report = score_file(path)
            self.assertFalse(report.eligible)
            self.assertEqual(report.score, 0)
            self.assertIn("奖励指标从多条业务结果中按位置取值", report.eligibility_failures[-1])

    def test_file_score_rejects_task_spec_fixture_that_differs_from_runtime_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            fixture = task["task_spec"]["environment_contract"]["initial_fixture"] = {
                "manifest": {**task["artifacts"]["data_manifest"], "root": "old-data"},
                "table_names": ["parent", "child"],
            }
            path.write_text(json.dumps(task), encoding="utf-8")
            self.assertIn(
                "TaskSpec 初始数据 manifest 与实际业务数据不一致",
                score_file(path).eligibility_failures,
            )
            fixture["manifest"] = task["artifacts"]["data_manifest"]
            fixture["table_names"] = ["parent"]
            path.write_text(json.dumps(task), encoding="utf-8")
            self.assertIn(
                "TaskSpec 初始业务表与实际业务数据不一致",
                score_file(path).eligibility_failures,
            )

    def test_stateful_outcome_must_be_false_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_manifest_task(root, child_parent_id=1)
            with (root / "rows/parent.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('{"id": 2}\n')
            task = json.loads(path.read_text(encoding="utf-8"))
            task["task_intent"] = "modify"
            task["environment_plan"]["mode"] = "stateful"
            task["artifacts"]["data_manifest"]["environment_mode"] = "stateful"
            task["task_spec"]["environment_contract"].update(
                mode="stateful", archetype="single_mutation",
            )
            task["task_spec"]["goal_contract"] = {"row_predicates": [{
                "table": "child", "where": {"id": 10},
                "values": {"parent_id": 2}, "count": 1,
            }]}
            task["tool_implementations"] = [{
                "tool_name": "read_inventory", "operation": "update", "table": "child",
                "changes": {"parent_id": "parent_id"},
            }]
            task["tools"][0]["function"]["parameters"] = {
                "type": "object", "properties": {"parent_id": {"type": "integer"}},
                "required": ["parent_id"],
            }
            task["acceptance_contract"]["executable_scenarios"][0]["steps"][0]["arguments"] = {
                "parent_id": 2,
            }
            outcome = task["metrics"][1]
            outcome.update({"type": "rule-based", "evaluator": {"kind": "business_state_rule"}})
            task["metric_implementations"].append({
                "metric_id": "outcome", "source": "business_state",
                "path": "$.child", "operator": "count_eq", "expected": 1,
                "score_mapping": {"pass": 1, "fail": 0},
            })
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
            self.assertFalse(report.eligible)
            self.assertIn("状态结果奖励在变更前已满足：outcome",
                          report.eligibility_failures)
            task["metric_implementations"][-1].update(
                operator="contains", expected={"id": 10, "parent_id": 2},
            )
            path.write_text(json.dumps(task), encoding="utf-8")
            self.assertTrue(score_file(path).eligible)

    def test_read_only_data_task_requires_a_baseline_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_manifest_task(root, child_parent_id=1)
            (root / "rows/parent.jsonl").write_text("", encoding="utf-8")
            (root / "rows/child.jsonl").write_text("", encoding="utf-8")
            report = score_file(path)
        self.assertFalse(report.eligible)
        self.assertIn("只读数据任务缺少基线业务记录", report.eligibility_failures)

    def test_file_score_does_not_borrow_business_data_from_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self._write_manifest_task(base / "shared", child_parent_id=1)
            isolated = base / "isolated"
            isolated.mkdir()
            task = json.loads(source.read_text(encoding="utf-8"))
            task["artifacts"]["data_manifest"]["root"] = "shared"
            path = isolated / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(base)
                report = score_file(path)
            finally:
                os.chdir(previous)
        self.assertFalse(report.eligible)
        self.assertTrue(any("共享持久化契约" in item for item in report.eligibility_failures))

    def test_file_score_uses_declared_root_when_shadow_files_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_manifest_task(root, child_parent_id=1)
            declared = root / "data/business_data"
            shutil.copytree(root / "schemas", declared / "schemas")
            shutil.copytree(root / "rows", declared / "rows")
            task = json.loads(path.read_text(encoding="utf-8"))
            task["artifacts"]["data_manifest"]["root"] = "data/business_data"
            path.write_text(json.dumps(task), encoding="utf-8")
            (declared / "rows/child.jsonl").write_text(
                '{"id": 10, "parent_id": 99}\n', encoding="utf-8",
            )
            report = score_file(path)
            self.assertFalse(report.eligible)
            self.assertTrue(any("共享持久化契约" in item
                                for item in report.eligibility_failures))
            (declared / "rows/child.jsonl").write_text(
                '{"id": 10, "parent_id": 1}\n', encoding="utf-8",
            )
            (root / "rows/child.jsonl").write_text(
                '{"id": 10, "parent_id": 99}\n', encoding="utf-8",
            )
            self.assertTrue(score_file(path).eligible)

    def test_file_score_rejects_manifest_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self._write_manifest_task(root, child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            task["artifacts"]["data_manifest"]["root"] = "../outside"
            path.write_text(json.dumps(task), encoding="utf-8")
            self.assertIn("业务数据 manifest 必须位于任务目录内",
                          score_file(path).eligibility_failures)
            task["artifacts"]["data_manifest"]["root"] = "."
            task["artifacts"]["data_manifest"]["tables"][0]["rows_file"] = "../rows.jsonl"
            path.write_text(json.dumps(task), encoding="utf-8")
            self.assertIn("业务数据 manifest 文件路径必须位于任务目录内",
                          score_file(path).eligibility_failures)

    def test_file_score_rejects_symlinked_business_data_outside_task(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            source = self._write_manifest_task(base / "outside", child_parent_id=1)
            task = json.loads(source.read_text(encoding="utf-8"))
            task["artifacts"]["data_manifest"]["root"] = "data"
            isolated = base / "isolated"
            isolated.mkdir()
            (isolated / "data").symlink_to(source.parent, target_is_directory=True)
            path = isolated / "task.json"
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
        self.assertFalse(report.eligible)
        self.assertIn("业务数据 manifest 指向任务目录外的文件",
                      report.eligibility_failures)

    def test_score_file_rejects_data_invalid_under_shared_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            report = score_file(self._write_manifest_task(Path(directory), child_parent_id=99))
        self.assertFalse(report.eligible)
        self.assertEqual(report.tier, "rejected")
        self.assertTrue(any("共享持久化契约" in item for item in report.eligibility_failures))

    def test_score_file_rejects_stale_declared_data_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            task["acceptance_contract"]["fixtures"] = {"initial_data_hash": "stale"}
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
        self.assertFalse(report.eligible)
        self.assertTrue(any("initial_data_hash mismatch" in item
                            for item in report.eligibility_failures))

    def test_score_file_rejects_stale_manifest_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            task["artifacts"]["data_manifest"]["tables"][0]["row_count"] = 999
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
        self.assertFalse(report.eligible)
        self.assertTrue(any("row_count differs" in item for item in report.eligibility_failures))

    def test_score_file_rejects_malformed_manifest_without_crashing_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            task["artifacts"]["data_manifest"]["tables"].append(None)
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
        self.assertFalse(report.eligible)
        self.assertTrue(any("table manifest entry is not an object" in item
                            for item in report.eligibility_failures))

    def test_score_file_rejects_private_rows_copied_from_public_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            task["public_input"] = {"materials": [{
                "name": "候选记录", "content": '{"candidates":[{"id":1}]}',
            }]}
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
        self.assertFalse(report.eligible)
        self.assertTrue(any("私有业务表完整复制了公开材料" in item
                            for item in report.eligibility_failures))

    def test_score_file_verifies_numeric_oracle_against_rows_and_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            task["task_intent"] = "calculate"
            task["metrics"][1]["type"] = "rule-based"
            task["metrics"][1]["evaluator"] = {"kind": "document_rule"}
            task["metric_implementations"].append({
                "metric_id": "outcome", "source": "final_agent_response",
                "path": "$", "operator": "numeric_targets",
                "expected": {"targets": [{
                    "label": "总数", "unit": "", "tolerance": 0,
                    "expression": {"lookup": {"table": "child", "field": "parent_id",
                                              "where": {"id": 10}}},
                }]},
                "score_mapping": {"pass": 1, "fail": 0},
            })
            task["acceptance_contract"]["executable_scenarios"][0]["steps"].append({
                "operation": "agent_response", "content": "总数1",
            })
            path.write_text(json.dumps(task), encoding="utf-8")
            self.assertTrue(score_file(path).eligible)
            task["acceptance_contract"]["executable_scenarios"][0]["steps"][-1]["content"] = "总数2"
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
            self.assertFalse(report.eligible)
            self.assertTrue(any("成功答案" in item for item in report.eligibility_failures))

    def test_score_file_rejects_fixed_row_enumeration_for_open_total(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            with (Path(directory) / "rows" / "child.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('{"id": 11, "parent_id": 1}\n')
            task = json.loads(path.read_text(encoding="utf-8"))
            task["task_intent"] = "calculate"
            task["task_spec"]["task_contract"]["expected_result"] = "汇总每条记录的总数"
            task["metrics"][1]["type"] = "rule-based"
            task["metrics"][1]["evaluator"] = {"kind": "document_rule"}
            task["metric_implementations"].append({
                "metric_id": "outcome", "source": "final_agent_response",
                "path": "$", "operator": "numeric_targets",
                "expected": {"targets": [{
                    "label": "总数", "unit": "", "tolerance": 0,
                    "expression": {"op": "add", "args": [
                        {"lookup": {"table": "child", "field": "parent_id", "where": {"id": 10}}},
                        {"lookup": {"table": "child", "field": "parent_id", "where": {"id": 11}}},
                    ]},
                }]},
                "score_mapping": {"pass": 1, "fail": 0},
            })
            task["acceptance_contract"]["executable_scenarios"][0]["steps"].append({
                "operation": "agent_response", "content": "总数2",
            })
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
        self.assertFalse(report.eligible)
        self.assertTrue(any("开放记录集合" in item for item in report.eligibility_failures))

    def test_score_file_requires_new_matching_row_to_change_open_total(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_manifest_task(Path(directory), child_parent_id=1)
            task = json.loads(path.read_text(encoding="utf-8"))
            task["task_intent"] = "calculate"
            task["task_spec"]["task_contract"]["expected_result"] = "统计每条记录的总数"
            task["metrics"][1]["type"] = "rule-based"
            task["metrics"][1]["evaluator"] = {"kind": "document_rule"}
            task["metric_implementations"].append({
                "metric_id": "outcome", "source": "final_agent_response",
                "path": "$", "operator": "numeric_targets",
                "expected": {"targets": [{
                    "label": "总数", "unit": "", "tolerance": 0,
                    "expression": {"aggregate": {
                        "table": "child", "field": "id", "where": {"id": 10}, "op": "count",
                    }},
                }]},
                "score_mapping": {"pass": 1, "fail": 0},
            })
            task["acceptance_contract"]["executable_scenarios"][0]["steps"].append({
                "operation": "agent_response", "content": "总数1",
            })
            path.write_text(json.dumps(task), encoding="utf-8")
            report = score_file(path)
            self.assertFalse(report.eligible)
            self.assertTrue(any("新增匹配记录不敏感" in item
                                for item in report.eligibility_failures))
            task["metric_implementations"][-1]["expected"]["targets"][0]["expression"]["aggregate"]["where"] = {"parent_id": 1}
            path.write_text(json.dumps(task), encoding="utf-8")
            self.assertTrue(score_file(path).eligible)

    def test_tool_capability_cannot_disclaim_its_own_need(self):
        task = valid_task()
        task["capability_plan"] = [{
            "action_name": "generate_table", "requires_tool": True,
            "reason": "属于分析、计算和生成自然语言回答，不需要额外工具。",
        }]
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertTrue(any("Agent 自行完成" in item
                            for item in report.eligibility_failures))
        task["capability_plan"][0]["reason"] = (
            "基于已读取字段分析、总结并生成最终自然语言回答，不需要调用工具。"
        )
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertTrue(any("Agent 自行完成" in item
                            for item in report.eligibility_failures))

    def test_simple_postprocessing_of_tool_output_is_not_a_business_step(self):
        task = valid_task()
        task["capability_plan"] = [{
            "action_name": "sum_pit_counts", "requires_tool": True,
            "reason": "该动作对前序工具返回的数量进行求和，属于确定性计算能力。",
        }]
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertTrue(any("简单加工" in item for item in report.eligibility_failures))

    def test_dependent_simulation_of_prior_records_is_not_a_second_business_step(self):
        task = valid_task()
        task["capability_plan"] = [
            {"action_name": "get_inventory", "requires_tool": True,
             "dependencies": [], "reason": "读取私有库存记录"},
            {"action_name": "simulate_demand", "requires_tool": True,
             "dependencies": ["get_inventory"],
             "reason": "基于前序环境操作获取的库存数据执行确定性模拟计算"},
        ]
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertTrue(any("未声明新的环境访问" in item
                            for item in report.eligibility_failures))

    def test_dependent_query_of_new_private_records_remains_valid(self):
        task = valid_task()
        task["capability_plan"] = [
            {"action_name": "get_inventory", "requires_tool": True,
             "dependencies": [], "reason": "读取私有库存记录"},
            {"action_name": "get_supplier", "requires_tool": True,
             "dependencies": ["get_inventory"],
             "reason": "根据前序结果中的供应商编号进一步查询私有供应商记录"},
        ]
        self.assertTrue(score_task(task).eligible)

    def test_graph_task_must_retain_sampled_scene_theme(self):
        task = valid_task()
        task["artifacts"] = {"graph_context": {
            "nodes": ["赫拉特", "中亚"], "keywords": ["赫拉特", "中亚"],
        }}
        self.assertFalse(score_task(task).eligible)
        task["task"] = "查询中亚商品库存并给出采购建议。"
        self.assertTrue(score_task(task).eligible)
        task["task"] = "查询沙箱内的中亚商品库存并给出采购建议。"
        self.assertFalse(score_task(task).eligible)

    def test_stateful_mutation_without_goal_assertion_is_ineligible(self):
        task = valid_task()
        task["task_intent"] = "modify"
        task["environment_plan"] = {"mode": "stateful"}
        task["tool_implementations"] = [{
            "tool_name": "update_export", "operation": "update",
            "table": "exports", "changes": {"code": "code"},
        }]
        task["task_spec"]["goal_contract"] = {"row_predicates": []}
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertTrue(any("状态目标与变更工具不闭合" in item
                            for item in report.eligibility_failures))

    def test_stateful_empty_goal_and_no_mutation_cannot_pass_vacuously(self):
        task = valid_task()
        task["environment_plan"] = {"mode": "stateful"}
        task["task_spec"]["environment_contract"].update(
            mode="stateful", archetype="single_mutation",
        )
        report = score_task(task)
        self.assertGreaterEqual(sum(item.score for item in report.dimensions.values()), 8.0)
        self.assertEqual(report.score, 0.0)
        self.assertFalse(report.eligible)
        self.assertTrue(any("状态任务缺少最终行状态断言" in item
                            for item in report.eligibility_failures))

    def test_stateful_goal_requires_actual_mutation_in_success_trace(self):
        task = valid_task()
        task["environment_plan"] = {"mode": "stateful"}
        task["task_spec"]["environment_contract"].update(
            mode="stateful", archetype="single_mutation",
        )
        task["task_spec"]["goal_contract"] = {"row_predicates": [{
            "table": "inventory", "where": {"id": "i1"},
            "values": {"quantity": 3}, "count": 1,
        }]}
        task["tool_implementations"] = [{
            "tool_name": "update_inventory", "operation": "update",
            "table": "inventory", "changes": {"quantity": "quantity"},
        }]
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertTrue(any("成功轨迹未调用目标表 inventory 的变更工具" in item
                            for item in report.eligibility_failures))

    def test_stateful_semantic_outcome_requires_business_goal_evidence(self):
        task = valid_task()
        task["environment_plan"] = {"mode": "stateful"}
        task["task_spec"]["environment_contract"].update(
            mode="stateful", archetype="single_mutation",
        )
        task["task_spec"]["goal_contract"] = {"row_predicates": [{
            "table": "inventory", "where": {"id": "i1"},
            "values": {"quantity": 3}, "count": 1,
        }]}
        task["tool_implementations"] = [{
            "tool_name": "read_inventory", "operation": "update",
            "table": "inventory", "changes": {"quantity": "quantity"},
        }]
        outcome = task["metrics"][1]
        outcome["evaluation_inputs"] = ["final_agent_response"]
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertIn("状态结果奖励缺少最终业务状态核对：outcome",
                      report.eligibility_failures)
        TaskGenerationPipeline._ground_reference_outcome_metrics(
            task["metrics"], environment_mode="stateful",
        )
        self.assertTrue(score_task(task).eligible)
        outcome["criteria"].append("无需核对业务数据")
        self.assertIn("状态结果奖励缺少最终业务状态核对：outcome",
                      score_task(task).eligibility_failures)

    def test_object_array_filter_without_matching_column_is_ineligible(self):
        task = valid_task()
        task["tools"][0]["function"]["parameters"] = {"properties": {
            "inventory_records": {"type": "array", "items": {"type": "object",
                "properties": {"latest_batch_no": {"type": "string"}}}},
        }}
        task["tool_implementations"] = [{
            "tool_name": "read_inventory", "filters": [{
                "argument": "inventory_records", "column": "sample_code", "operator": "in",
            }],
        }]
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertTrue(any("对象数组工具筛选无法映射列" in issue
                            for issue in report.eligibility_failures))

    def test_stateful_mutation_with_goal_assertion_is_closed(self):
        task = valid_task()
        task["task_intent"] = "modify"
        task["environment_plan"] = {"mode": "stateful"}
        task["tool_implementations"] = [{
            "tool_name": "read_inventory", "operation": "update",
            "table": "inventory", "changes": {"quantity": "quantity"},
        }]
        task["task_spec"]["goal_contract"] = {"row_predicates": [{
            "table": "inventory", "where": {"id": "i1"},
            "values": {"quantity": 3}, "count": 1,
        }]}
        report = score_task(task)
        self.assertFalse(any("状态目标与变更工具不闭合" in item
                             for item in report.eligibility_failures))

    def test_mutation_with_deferred_business_truth_is_ineligible(self):
        task = valid_task()
        task["task_intent"] = "modify"
        task["task"] = "修改记录，但新商品编码暂时没想好，请先问我"
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertIn("关键业务真值依赖未定义的后续用户回复",
                      report.eligibility_failures)

    def test_complete_agentic_task_passes(self):
        report = score_task(valid_task(), min_score=8)
        self.assertTrue(report.passed)
        self.assertEqual(report.score, 10)
        self.assertEqual(report.tier, "high_value")

    def test_calculation_with_model_only_outcome_is_not_high_value(self):
        task = valid_task()
        task["task_intent"] = "calculate"
        report = score_task(task)
        self.assertTrue(report.eligible)
        self.assertTrue(report.passed)
        self.assertEqual(report.tier, "usable")
        self.assertTrue(any("尚无确定性结果指标" in item for item in report.findings))
        task["metrics"][1]["type"] = "rule-based"
        task["metrics"][1]["evaluator"] = {"kind": "document_rule"}
        task["metric_implementations"].append({"metric_id": "outcome"})
        self.assertEqual(score_task(task).tier, "rejected")
        task["metric_implementations"][-1].update({
            "source": "final_agent_response", "path": "$", "operator": "numeric_targets",
            "expected": {"targets": [{
                "label": "总金额", "unit": "元", "tolerance": 0,
                "expression": {"lookup": {"table": "inventory", "field": "amount",
                                          "where": {"id": 1}}},
            }]},
        })
        self.assertEqual(score_task(task).tier, "high_value")
        task["metric_implementations"].append(dict(task["metric_implementations"][-1]))
        self.assertFalse(score_task(task).eligible)

    def test_missing_trainer_state_endpoint_is_ineligible(self):
        task = valid_task()
        endpoints = task["requirements"]["runtime_interface"]["endpoints"]
        task["requirements"]["runtime_interface"]["endpoints"] = [
            item for item in endpoints if item.get("name") != "state"
        ]
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertIn("运行接口缺少必需端点：state", report.eligibility_failures)

    def test_agentic_process_reward_requires_deterministic_implementation(self):
        task = valid_task()
        task["metric_implementations"] = [{"metric_id": "noise"}]
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertTrue(any("过程奖励缺少可复现实现" in item
                            for item in report.eligibility_failures))

    def test_public_quote_arithmetic_cannot_pass_agentic_task_score(self):
        task = valid_task()
        task["task_intent"] = "calculate"
        task["task"] = "根据供应商报价单计算总费用"
        task["public_input"] = {
            "initial_user_message": "我有 80 人，每人两餐，报价单给你了，算总费用。",
            "materials": [{"name": "quote.txt", "content": "每人每餐 350 元；服务费 10%"}],
        }
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertIn("公开输入已足以完成计算或估算，业务工具缺少必要的私有数据依赖",
                      report.eligibility_failures)

    def test_public_ratio_estimate_cannot_pass_agentic_task_score(self):
        task = valid_task()
        task["task_intent"] = "estimate"
        task["task"] = "估算活动物料数量"
        task["public_input"] = {
            "initial_user_message": "报名 86 人，预计 80% 到场，手册按 110%、茶歇按 90% 准备。请估算份数。",
            "materials": [],
        }
        report = score_task(task)
        self.assertFalse(report.eligible)

    def test_unverifiable_reward_contract_cannot_pass_on_score_alone(self):
        cases = {
            "missing_outcome": lambda task: task["metrics"].pop(1),
            "missing_rule_implementation": lambda task: task["metric_implementations"].pop(),
            "missing_failure_scenario": lambda task: task["acceptance_contract"]["executable_scenarios"].pop(1),
            "invalid_reward_range": lambda task: task["reward_formula"].update(score_range=[0, 1]),
        }
        for name, damage in cases.items():
            with self.subTest(name=name):
                task = valid_task()
                damage(task)
                report = score_task(task, min_score=8)
                self.assertGreaterEqual(sum(item.score for item in report.dimensions.values()), 8)
                self.assertEqual(report.score, 0.0)
                self.assertFalse(report.eligible)
                self.assertFalse(report.passed)

    def test_simple_no_tool_task_is_rejected(self):
        task = valid_task()
        task.update({"task": "提取时间", "complexity": "simple", "actions": [{"name": "提取"}], "tools": [], "noise_tools": []})
        report = score_task(task, min_score=8)
        self.assertFalse(report.passed)
        self.assertEqual(report.tier, "rejected")
        self.assertIn("simple_agentic 缺少必要业务工具调用", report.findings)

    def test_simple_task_with_only_noise_is_quality_capped(self):
        task = valid_task()
        task.update({"complexity": "simple", "tools": [{"function": {"name": "get_weather"}}]})
        report = score_task(task, min_score=8)
        self.assertLessEqual(report.score, 7.8)
        self.assertFalse(report.passed)

    def test_direct_response_can_be_high_value_without_business_tools(self):
        task = valid_task()
        task.update({
            "training_category": "direct_response",
            "complexity": "simple",
            "environment_plan": {"mode": "stateless"},
            "tools": [{"function": {"name": "get_weather"}}],
            "noise_tools": [{"name": "get_weather", "category": "unrelated"}],
            "actions": [{"name": "回答用户"}],
        })
        task["task_spec"] = {
            **task["task_spec"],
            "training_contract": {"category": "direct_response", "environment_archetype": "text_only"},
            "environment_contract": {"mode": "stateless", "archetype": "text_only"},
            "tool_contracts": [{"name": "get_weather", "role": "noise"}],
            "capability_dag": {"nodes": [], "edges": []},
        }
        task["acceptance_contract"]["executable_scenarios"] = [
            {"kind": "goal_success", "steps": [{"operation": "agent_response"}]},
            {"kind": "goal_failure"},
            {"kind": "noise_selection"},
        ]
        report = score_task(task)
        self.assertTrue(report.passed)
        self.assertEqual(report.training_category, "direct_response")
        self.assertEqual(report.tool_policy_target, "do_not_call")

    def test_direct_response_rejects_unsourced_real_world_examples(self):
        task = valid_task()
        task.update({
            "training_category": "direct_response",
            "task": "解释粤剧唱腔并列出各自代表剧目",
            "public_input": {"initial_user_message": "请介绍平喉和薛腔的代表剧目", "materials": [
                {"name": "背景", "content": "平喉和薛腔是粤剧唱腔。"},
            ]},
        })
        self.assertTrue(unsourced_real_world_exemplars(task, training_category="direct_response"))
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertIn("直接回答要求现实领域代表例证，但公开材料缺少可追溯来源", report.eligibility_failures)
        task["public_input"]["materials"][0]["source_url"] = "https://example.org/source"
        self.assertFalse(unsourced_real_world_exemplars(task, training_category="direct_response"))
        task["public_input"] = "invalid"
        self.assertTrue(unsourced_real_world_exemplars(task, training_category="direct_response"))

    def test_active_record_precondition_must_be_observable(self):
        task = valid_task()
        task["task_spec"]["tool_contracts"][0]["preconditions"] = [
            "库存记录中存在 status 为 active 的商品。"
        ]
        task["tool_implementations"] = [{
            "tool_name": "read_inventory", "operation": "select", "table": "inventory",
            "filters": [{"argument": "category", "column": "category", "operator": "eq"}],
            "projection": ["name", "quantity"], "result_field": "records",
        }]
        self.assertEqual(unobservable_tool_preconditions(task), ["read_inventory"])
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertTrue(any("既不筛选也不返回 status" in reason for reason in report.eligibility_failures))
        task["tool_implementations"][0]["projection"].append("status")
        self.assertEqual(unobservable_tool_preconditions(task), [])

    def test_multi_step_route_rejects_single_business_call(self):
        task = valid_task()
        task["training_category"] = "multi_step_agentic"
        report = score_task(task)
        self.assertFalse(report.passed)
        self.assertFalse(report.eligible)
        self.assertGreater(sum(item.score for item in report.dimensions.values()), 7.8)
        self.assertEqual(report.score, 0.0)

    def test_overdesigned_user_input_environment_is_penalized(self):
        task = valid_task()
        task["task"] = "从用户明确提供的活动选项中推荐一个结果"
        task["requirements"] = {"constraints": "仅依赖用户提供的列表"}
        report = score_task(task)
        self.assertIn("仅依赖用户输入的任务被过度设计为数据环境", report.findings)

    def test_user_supplied_complete_specs_do_not_justify_reference_data(self):
        task = valid_task()
        task["task"] = "比较两款电脑，基于用户提供的规格数据给出结论"
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertEqual(report.tier, "rejected")

    def test_referenced_public_material_must_be_delivered(self):
        task = valid_task()
        task["task"] = "请根据用户提供的产品说明查询库存并给出建议"
        task["public_input"] = {"initial_user_message": task["task"], "materials": []}
        report = score_task(task)
        self.assertFalse(report.eligible)
        self.assertIn("public_input 缺少题面引用的实际输入材料", report.eligibility_failures)

    def test_system_provided_private_records_need_no_public_attachment(self):
        task = valid_task()
        task["task"] = "查询系统提供的库存记录，筛选可用物料并给出建议"
        task["public_input"] = {"initial_user_message": task["task"], "materials": []}
        report = score_task(task)
        self.assertNotIn("public_input 缺少题面引用的实际输入材料",
                         report.eligibility_failures)

    def test_intent_label_does_not_override_agentic_evidence(self):
        task = valid_task()
        task["task_intent"] = "explain"
        report = score_task(task)
        self.assertTrue(report.passed)
        self.assertEqual(report.tier, "high_value")

    def test_cross_domain_vocabulary_does_not_change_structural_score(self):
        task = valid_task()
        baseline = score_task(task)
        task["task"] = "根据语言服务需求和城市资料选择学习交流目的地"
        task["actions"] = [
            {"name": "读取参考数据"},
            {"name": "比较候选城市气候"},
            {"name": "选择最优城市"},
        ]
        task["metrics"][1]["id"] = "outcome_city_selection"
        report = score_task(task)
        self.assertEqual(report.score, baseline.score)
        self.assertEqual(report.eligible, baseline.eligible)

    def test_unknown_task_domain_does_not_trigger_false_cross_domain_cap(self):
        task = valid_task()
        task["task"] = "比较皮革与毛皮在历史贸易中的角色差异"
        task["actions"] = [
            {"name": "读取参考数据", "description": "读取商品维度记录"},
            {"name": "提取两类材料事实"},
            {"name": "形成历史对比"},
        ]
        report = score_task(task)
        self.assertNotIn("训练资格失败：任务、动作、工具或奖励发生跨领域语义漂移", report.findings)

    def test_wildlife_trip_requirements_are_not_a_domain_conflict(self):
        task = valid_task()
        baseline = score_task(task)
        task["task"] = "从藏羚羊和雪豹中推荐一个自然观察目标动物"
        task["requirements"] = {
            **task["requirements"],
            "rule": "按照候选地点的车程和门票筛选目的地",
        }
        report = score_task(task)
        self.assertEqual(report.score, baseline.score)
        self.assertEqual(report.eligible, baseline.eligible)

    def test_discover_tasks_is_incremental_layout_aware(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "task-2").mkdir()
            (root / "task-2" / "task.json").write_text("{}", encoding="utf-8")
            self.assertEqual(discover_task_files(root), [root / "task-2" / "task.json"])

    def test_invalid_artifact_becomes_zero_score_report(self):
        report = error_report(Path("task-7/task.json"), ValueError("bad json"))
        self.assertEqual(report.score, 0)
        self.assertFalse(report.passed)
        self.assertFalse(report.eligible)
        self.assertEqual(report.training_category, "unknown")
        self.assertEqual(report.tier, "rejected")
