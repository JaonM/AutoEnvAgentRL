import json
import tempfile
import unittest
from pathlib import Path

from env_factory.tasks.task_quality import discover_task_files, error_report, score_file, score_task
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
            {"id": "outcome", "category": "outcome", "type": "model-based", "evaluator": {}},
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

    def test_multi_step_route_rejects_single_business_call(self):
        task = valid_task()
        task["training_category"] = "multi_step_agentic"
        report = score_task(task)
        self.assertFalse(report.passed)
        self.assertFalse(report.eligible)
        self.assertGreater(report.score, 7.8)

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
        self.assertEqual(report.tier, "rejected")
