import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from env_factory.sandbox_runtime import sha256_json
from env_factory.task_pipeline import PipelineGenerationError, TaskGenerationPipeline


class StageRetryTest(unittest.TestCase):
    def test_table_rows_repair_is_local_and_constraint_aware(self):
        table = {
            "table_name": "order", "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "prep_duration_minutes", "type": "INTEGER", "nullable": False},
            ],
            "primary_key": ["id"], "constraints": ["prep_duration_minutes >= 0"],
        }
        pipeline = TaskGenerationPipeline.__new__(TaskGenerationPipeline)
        pipeline.retries = 3
        responses = [
            {"rows": [{"id": 1, "prep_duration_minutes": -2}]},
            {"rows": [{"id": 1, "prep_duration_minutes": 12}]},
        ]
        with patch.object(pipeline, "_call", side_effect=responses) as call:
            generated = pipeline._generate_table_rows(
                table, task_description={"task": "排程"}, entities=[],
            )
        self.assertEqual(generated["rows"][0]["prep_duration_minutes"], 12)
        self.assertEqual(call.call_count, 2)
        self.assertIn("CHECK constraint", call.call_args_list[1].args[2]["validation_error"])
        self.assertEqual(call.call_args_list[1].args[2]["previous_rows"], responses[0]["rows"])

    def test_table_rows_reject_empty_and_duplicate_keys(self):
        table = {
            "table_name": "item", "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
            ], "primary_key": ["id"],
        }
        with self.assertRaisesRegex(PipelineGenerationError, "non-empty rows"):
            TaskGenerationPipeline._validate_generated_table_rows(table, [])
        with self.assertRaisesRegex(PipelineGenerationError, "duplicate primary keys"):
            TaskGenerationPipeline._validate_generated_table_rows(
                table, [{"id": 1}, {"id": 1}],
            )

    def test_consistency_output_can_reach_grounding_repair_for_check_violation(self):
        table = {
            "table_name": "schedule", "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "duration", "type": "INTEGER", "nullable": False},
            ], "primary_key": ["id"], "constraints": ["duration >= 0"],
            "rows": [{"id": 1, "duration": -1}],
        }
        TaskGenerationPipeline._validate_data_tables([table])
        with self.assertRaisesRegex(PipelineGenerationError, "CHECK constraint"):
            TaskGenerationPipeline._validate_relational_data([table])

    def test_transient_llm_failure_uses_backoff_before_retry(self):
        class FakeLLM:
            def __init__(self):
                self.calls = 0

            def complete(self, prompt, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("LLM returned HTTP 503: Service is too busy")
                return SimpleNamespace(content=json.dumps({"value": True}))

        llm = FakeLLM()
        pipeline = TaskGenerationPipeline(llm, retries=2)
        with patch("env_factory.task_pipeline.time.sleep") as sleep:
            result = pipeline._call("some_stage", "prompt", {"output": {}})
        self.assertTrue(result["value"])
        sleep.assert_called_once()

    def test_task_description_shape_error_is_retried_with_feedback(self):
        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                if len(self.calls) == 1:
                    return SimpleNamespace(content=json.dumps({"output": "错误包装", "type": "text"}))
                return SimpleNamespace(content=json.dumps({
                    "task": "查询指定对象的信息",
                    "task_intent": "query",
                    "goal": "完成查询",
                    "context": [],
                    "expected_result": "结构化结果",
                    "complexity": "simple",
                    "requirements": {"input_modalities": ["text"]},
                }))

        llm = FakeLLM()
        pipeline = TaskGenerationPipeline(llm, retries=2)
        result = pipeline._call(
            "task_description",
            "生成任务描述",
            {"output": {"task": "string", "complexity": "simple"}},
        )
        self.assertEqual(result["task"], "查询指定对象的信息")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("previous_validation_error", llm.calls[1][0])

    def test_table_data_json_error_is_retried_with_compact_rows_contract(self):
        class FakeLLM:
            def __init__(self):
                self.calls = []

            def complete(self, prompt, **kwargs):
                self.calls.append((prompt, kwargs))
                if len(self.calls) == 1:
                    return SimpleNamespace(content='{"rows":[{"name":"未闭合}')
                return SimpleNamespace(content=json.dumps({"rows": [{"name": "工坊"}]}))

        llm = FakeLLM()
        pipeline = TaskGenerationPipeline(llm, retries=2)
        result = pipeline._call(
            "environment_table_data.workshop",
            "生成表数据",
            {"table": {"columns": [{"name": "name"}]}, "output": {"rows": []}},
        )
        self.assertEqual(result["rows"][0]["name"], "工坊")
        self.assertEqual(len(llm.calls), 2)
        self.assertIn("rows 数组", llm.calls[1][1]["system_prompt"])


class PublicInputContractTest(unittest.TestCase):
    def test_rejects_referenced_but_missing_material(self):
        description = {
            "task": "请总结用户提供的产品说明",
            "context": ["用户已提供产品说明"],
            "requirements": {"input_modalities": ["text"]},
        }
        public_input = TaskGenerationPipeline._normalize_public_input(description)
        with self.assertRaisesRegex(PipelineGenerationError, "missing the concrete material"):
            TaskGenerationPipeline._validate_public_input(
                task_description=description,
                public_input=public_input,
                environment_mode="stateless",
            )

    def test_accepts_concrete_material(self):
        description = {
            "task": "请总结以下产品说明",
            "public_input": {"initial_user_message": "请总结材料", "materials": [{
                "name": "说明", "mime_type": "text/plain",
                "content": "鞋底采用天然橡胶，适合室内木地板使用。",
            }]},
        }
        public_input = TaskGenerationPipeline._normalize_public_input(description)
        TaskGenerationPipeline._validate_public_input(
            task_description=description,
            public_input=public_input,
            environment_mode="stateless",
        )
        self.assertEqual(public_input["materials"][0]["name"], "说明")


class BusinessDataArtifactsTest(unittest.TestCase):
    def test_acceptance_probe_uses_correlated_business_rows_for_object_arrays(self):
        tables = [
            {"table_name": "compliance_rule", "rows": [
                {"id": 1, "rule_id": "R1", "rule_description": "品牌须获授权"},
                {"id": 2, "rule_id": "R2", "rule_description": "成分须准确"},
            ]},
            {"table_name": "authorized_brand", "rows": [
                {"id": 1, "brand_name": "BrandA"},
            ]},
        ]
        parameters = {"type": "object", "properties": {
            "compliance_rules": {"type": "array", "items": {
                "type": "object", "properties": {
                    "rule_id": {"type": "string"},
                    "rule_description": {"type": "string"},
                }, "required": ["rule_id", "rule_description"],
                "additionalProperties": False,
            }},
            "authorized_brands": {"type": "array", "items": {
                "type": "object", "properties": {"brand_name": {"type": "string"}},
                "required": ["brand_name"], "additionalProperties": False,
            }},
        }, "required": ["compliance_rules", "authorized_brands"]}
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description={"task": "验证批次"}, data_manifest={}, data_tables=tables,
            actions=[], tools=[{"function": {"name": "validate_batch", "parameters": parameters}}],
            key_steps=[], metrics=[], reward_formula={},
        )
        arguments = contract["argument_probes"][0]["arguments"]
        self.assertEqual(arguments["compliance_rules"], [
            {"rule_id": "R1", "rule_description": "品牌须获授权"},
            {"rule_id": "R2", "rule_description": "成分须准确"},
        ])
        self.assertEqual(arguments["authorized_brands"], [{"brand_name": "BrandA"}])

    def test_acceptance_probe_uses_public_json_and_derived_daily_sales(self):
        parameters = {"type": "object", "properties": {
            "sales_data": {"type": "array", "items": {"type": "object", "properties": {
                "date": {"type": "string"}, "item": {"type": "string"},
                "quantity": {"type": "number"},
            }, "required": ["date", "item", "quantity"]}},
            "average_daily_sales": {"type": "array", "items": {"type": "object", "properties": {
                "item": {"type": "string"}, "average_daily_sales": {"type": "number"},
            }, "required": ["item", "average_daily_sales"]}},
        }, "required": ["sales_data", "average_daily_sales"]}
        task = {"task": "模拟库存", "public_input": {"materials": [{
            "mime_type": "application/json", "content": json.dumps({"sales": [
                {"date": "2025-03-01", "item": "舞鞋", "quantity": 4},
                {"date": "2025-03-02", "item": "舞鞋", "quantity": 6},
            ]}, ensure_ascii=False),
        }]}}
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description=task, data_manifest={}, data_tables=[], actions=[],
            tools=[{"function": {"name": "simulate", "parameters": parameters}}],
            key_steps=[], metrics=[], reward_formula={},
        )
        arguments = contract["argument_probes"][0]["arguments"]
        self.assertEqual(arguments["sales_data"][0], {
            "date": "2025-03-01", "item": "舞鞋", "quantity": 4,
        })
        self.assertEqual(arguments["average_daily_sales"], [
            {"item": "舞鞋", "average_daily_sales": 5.0},
        ])

    def test_acceptance_probe_binds_tool_argument_to_source_column(self):
        parameters = {"type": "object", "properties": {
            "product_id": {"type": "string"},
        }, "required": ["product_id"]}
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description={"task": "查询价格"}, data_manifest={},
            data_tables=[{"table_name": "products", "rows": [{"field_abc": "P00001"}]}],
            actions=[], tools=[{"function": {"name": "lookup_product", "parameters": parameters}}],
            key_steps=[], metrics=[], reward_formula={},
            tool_implementations=[{"tool_name": "lookup_product", "operation": "select",
                                   "table": "products", "selector": {"product_id": "field_abc"}}],
        )
        self.assertEqual(contract["argument_probes"][0]["arguments"], {"product_id": "P00001"})

    def test_acceptance_probe_uses_top_level_public_json_array(self):
        rows = [
            {"款式": "运动T恤", "尺寸": "M", "材质": "涤纶"},
            {"款式": "羽绒服", "尺寸": "L", "材质": "尼龙"},
        ]
        record_schema = {"type": "object", "properties": {
            name: {"type": "string"} for name in rows[0]
        }, "required": list(rows[0])}
        parameters = {"type": "object", "properties": {
            "clothing_records": {"type": "array", "items": record_schema},
        }, "required": ["clothing_records"]}
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description={"task": "筛选服装", "public_input": {"materials": [{
                "name": "服装数据表", "mime_type": "application/json",
                "content": json.dumps(rows, ensure_ascii=False),
            }]}}, data_manifest={}, data_tables=[], actions=[],
            tools=[{"function": {"name": "filter_by_size", "parameters": parameters}}],
            key_steps=[], metrics=[], reward_formula={},
        )
        self.assertEqual(contract["argument_probes"][0]["arguments"]["clothing_records"], rows)

    def test_success_scenario_grounds_probe_and_preserves_ref_fallback(self):
        contract = {"tool_cases": [{
            "tool_name": "calculate", "kind": "schema_and_business_smoke",
            "arguments_template": {"order": "任务输入中的order", "stock": 1},
        }], "argument_probes": [{
            "tool_name": "calculate",
            "arguments": {"order": "任务输入中的order", "stock": 1},
        }]}
        scenarios = [{"kind": "goal_success", "steps": [{
            "operation": "tool_call", "tool_name": "calculate",
            "arguments": {"order": "B2024-001", "stock": {"$ref": "available"}},
        }]}]
        grounded = TaskGenerationPipeline._ground_acceptance_probes_from_success(
            contract, scenarios
        )
        expected = {"order": "B2024-001", "stock": 1}
        self.assertEqual(grounded["tool_cases"][0]["arguments_template"], expected)
        self.assertEqual(grounded["argument_probes"][0]["arguments"], expected)
        self.assertNotEqual(contract["argument_probes"][0]["arguments"], expected)

    def test_business_fixture_values_alias_localized_count_fields(self):
        values = TaskGenerationPipeline._business_fixture_values([{
            "table_name": "corpus_statistics",
            "rows": [{"table_a_撮口呼_count": 48, "table_a_开口呼_count": 312}],
        }])
        schema = {
            "type": "object",
            "properties": {
                "撮口呼样本计数": {"type": "integer"},
                "开口呼样本计数": {"type": "integer"},
            },
            "required": ["撮口呼样本计数", "开口呼样本计数"],
        }
        self.assertEqual(
            TaskGenerationPipeline._schema_fixture(schema, fixture_values=values),
            {"撮口呼样本计数": 48, "开口呼样本计数": 312},
        )

    def test_stateless_manifest_has_no_business_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = TaskGenerationPipeline._materialize_business_data(
                [], "# 无业务数据", Path(directory), environment_mode="stateless"
            )
            self.assertEqual(manifest["environment_mode"], "stateless")
            self.assertEqual(manifest["tables"], [])
            self.assertEqual(manifest["data_governance"], {
                "origin": "model_generated_synthetic",
                "contains_real_user_data": False,
                "intended_use": "agentic_rl_training_material",
            })

    def test_materialized_source_rows_keep_public_dataset_origin(self):
        governance = {
            "origin": "public_dataset", "provider": "heywhale",
            "source_url": "https://www.heywhale.com/home/dataset/example",
            "source_sha256": "a" * 64, "license": None,
            "contains_real_user_data": "undetermined",
            "intended_use": "agentic_rl_training_material",
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = TaskGenerationPipeline._materialize_business_data(
                [], "# source", Path(directory), data_governance=governance,
            )
            self.assertEqual(manifest["data_governance"], governance)

    def test_materialize_business_data_writes_schema_rows_and_document(self):
        tables = [{
            "table_name": "clothing_items",
            "description": "衣物",
            "columns": [{"name": "item_id", "type": "string", "description": "标识"}],
            "primary_key": ["item_id"],
            "foreign_keys": [],
            "rows": [{"item_id": "item-1"}],
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = TaskGenerationPipeline._materialize_business_data(tables, "# 数据说明", root)

            self.assertEqual(manifest["tables"][0]["schema_file"], "schemas/clothing_items.json")
            self.assertEqual(manifest["tables"][0]["rows_file"], "rows/clothing_items.jsonl")
            self.assertEqual(json.loads((root / "schemas/clothing_items.json").read_text())["table_name"], "clothing_items")
            self.assertEqual((root / "rows/clothing_items.jsonl").read_text().strip(), '{"item_id":"item-1"}')
            self.assertEqual((root / "data_document.md").read_text(), "# 数据说明\n")

    def test_materialized_business_data_can_publish_a_portable_root(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = TaskGenerationPipeline._materialize_business_data(
                [],
                "# 无业务数据",
                Path(directory) / "data" / "business_data",
                environment_mode="stateless",
                manifest_root="data/business_data",
            )
            self.assertEqual(manifest["root"], "data/business_data")


class AgentActionContractTest(unittest.TestCase):
    def test_stateful_task_rejects_deferred_business_truth(self):
        with self.assertRaisesRegex(PipelineGenerationError, "future user reply"):
            TaskGenerationPipeline._validate_no_deferred_business_truth({
                "task": "更新商品记录，新商品编码暂时没想好，请先问我",
                "requirements": {},
            }, "modify")

    def test_stateful_task_accepts_authoritative_replacement_value(self):
        TaskGenerationPipeline._validate_no_deferred_business_truth({
            "task": "把商品编码更新为 SKU-2026-09",
            "requirements": {"new_product_code": "SKU-2026-09"},
        }, "modify")

    def test_stateful_task_ignores_optional_clarification_outside_goal(self):
        TaskGenerationPipeline._validate_no_deferred_business_truth({
            "task": "把商品编码更新为 SKU-2026-09",
            "goal": "记录保存 SKU-2026-09",
            "requirements": {"presentation": "格式不清时需要向用户询问"},
        }, "modify")

    def test_goal_tool_coverage_rejects_unasserted_mutation_table(self):
        with self.assertRaisesRegex(PipelineGenerationError, "no declarative mutation|does not assert"):
            TaskGenerationPipeline._validate_goal_tool_coverage(
                semantic_goal={"row_predicates": [{
                    "table": "notes", "where": {"id": "n1"},
                    "values": {"name": "correct"}, "count": 1,
                }]},
                tool_implementations=[{
                    "tool_name": "update_export", "operation": "update",
                    "table": "exports", "changes": {"code": "code"},
                }],
            )

    def test_goal_tool_coverage_accepts_asserted_mutation_table(self):
        TaskGenerationPipeline._validate_goal_tool_coverage(
            semantic_goal={"row_predicates": [{
                "table": "exports", "where": {"id": "e1"},
                "values": {"code": "SKU-2"}, "count": 1,
            }]},
            tool_implementations=[{
                "tool_name": "update_export", "operation": "update",
                "table": "exports", "changes": {"code": "code"},
            }],
        )

    def test_goal_tool_coverage_rejects_goal_without_mutation(self):
        with self.assertRaisesRegex(PipelineGenerationError, "no declarative mutation"):
            TaskGenerationPipeline._validate_goal_tool_coverage(
                semantic_goal={"row_predicates": [{
                    "table": "inventory", "where": {"id": "i1"},
                    "values": {"quantity": 3}, "count": 1,
                }]},
                tool_implementations=[{
                    "tool_name": "read_inventory", "operation": "select",
                    "table": "inventory", "result_field": "records",
                }],
            )

    def test_deterministic_stateful_update_compilation(self):
        tools = [{"function": {
            "name": "update_inventory", "description": "更新库存数量",
            "parameters": {"properties": {
                "item_id": {"type": "string"}, "quantity": {"type": "number"},
            }},
        }}]
        tables = [{"table_name": "inventory", "columns": [
            {"name": "item_id", "type": "VARCHAR(32)"},
            {"name": "quantity", "type": "DECIMAL(10,2)"},
        ]}]
        result = TaskGenerationPipeline._complete_stateful_tool_implementations(
            implementations=[], tools=tools, tables=tables,
            semantic_goal={"row_predicates": [{
                "table": "inventory", "where": {"item_id": "i1"},
                "values": {"quantity": 3}, "count": 1,
            }]},
        )
        self.assertEqual(result, [{
            "tool_name": "update_inventory", "operation": "update",
            "table": "inventory", "selector": {"item_id": "item_id"},
            "changes": {"quantity": "quantity"}, "result_field": "records",
        }])

    def test_deterministic_stateful_update_compiles_unique_parameter_aliases(self):
        tools = [{"function": {
            "name": "update_product", "description": "更新商品状态",
            "parameters": {"properties": {
                "product_name": {"type": "string"},
                "new_status": {"type": "string"},
            }},
        }}]
        tables = [{"table_name": "product", "columns": [
            {"name": "name", "type": "VARCHAR(64)"},
            {"name": "status", "type": "VARCHAR(32)"},
        ]}]
        result = TaskGenerationPipeline._complete_stateful_tool_implementations(
            implementations=[], tools=tools, tables=tables,
            semantic_goal={"row_predicates": [{
                "table": "product", "where": {"name": "A"},
                "values": {"status": "active"}, "count": 1,
            }]},
        )
        self.assertEqual(result[0]["selector"], {"product_name": "name"})
        self.assertEqual(result[0]["changes"], {"new_status": "status"})

    def test_deterministic_stateful_update_replaces_non_mutating_model_spec(self):
        tools = [{"function": {
            "name": "update_product", "description": "更新商品状态",
            "parameters": {"properties": {
                "product_name": {"type": "string"},
                "new_status": {"type": "string"},
            }},
        }}]
        tables = [{"table_name": "product", "columns": [
            {"name": "name", "type": "VARCHAR(64)"},
            {"name": "status", "type": "VARCHAR(32)"},
        ]}]
        result = TaskGenerationPipeline._complete_stateful_tool_implementations(
            implementations=[{
                "tool_name": "update_product", "operation": "select",
                "table": "product", "filters": [], "projection": ["name"],
                "order_by": [], "result_field": "records",
            }],
            tools=tools, tables=tables,
            semantic_goal={"row_predicates": [{
                "table": "product", "where": {"name": "A"},
                "values": {"status": "active"}, "count": 1,
            }]},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["operation"], "update")
        self.assertEqual(result[0]["changes"], {"new_status": "status"})

    def test_file_deliverable_detection(self):
        self.assertTrue(TaskGenerationPipeline._requires_file_deliverable({
            "task": "生成可打印的工作坊流程表",
            "requirements": {"output_format": "PDF"},
        }))
        self.assertFalse(TaskGenerationPipeline._requires_file_deliverable({
            "task": "用 Markdown 编写工作坊流程表",
            "requirements": {"output_format": "Markdown"},
        }))

    def test_authoritative_reference_schema_requires_provenance(self):
        tables = [{"columns": [{"name": "content"}]}]
        with self.assertRaisesRegex(PipelineGenerationError, "provenance columns"):
            TaskGenerationPipeline._validate_authoritative_source_schema(
                task_description={"task": "验证澳门法律条文"},
                table_definitions=tables,
            )

    def test_authoritative_reference_schema_accepts_traceable_sources(self):
        tables = [{"columns": [
            {"name": "content"}, {"name": "source_url"},
            {"name": "retrieved_at"}, {"name": "content_hash"},
        ]}]
        TaskGenerationPipeline._validate_authoritative_source_schema(
            task_description={"task": "验证澳门法律条文"},
            table_definitions=tables,
        )

    def test_authoritative_reference_task_requires_supplied_source_urls(self):
        with self.assertRaisesRegex(PipelineGenerationError, "source URLs"):
            TaskGenerationPipeline._validate_authoritative_task_input(
                task_description={"task": "验证澳门法律条文"},
                environment_mode="reference_data",
                graph_context={"nodes": []},
            )

    def test_authoritative_reference_task_accepts_supplied_source_urls(self):
        TaskGenerationPipeline._validate_authoritative_task_input(
            task_description={"task": "验证澳门法律条文"},
            environment_mode="reference_data",
            graph_context={"sources": ["https://bo.io.gov.mo/example"]},
        )

    def test_media_generation_rejects_fabricated_runtime_input(self):
        with self.assertRaisesRegex(PipelineGenerationError, "fabricates runtime user input"):
            TaskGenerationPipeline._validate_media_generation(
                {
                    "code": "# 模拟数据\nintervals = [1, 2]\nopen('output/result.png', 'wb')",
                    "output_dir": "output",
                },
                input_media_required=True,
                deliverable_required=True,
                task_description={"requirements": {"output_format": "PNG"}},
            )

    def test_media_generation_requires_declared_file_format(self):
        with self.assertRaisesRegex(PipelineGenerationError, "declared file format"):
            TaskGenerationPipeline._validate_media_generation(
                {"code": "open('output/result.txt', 'w')", "output_dir": "output"},
                input_media_required=False,
                deliverable_required=True,
                task_description={"requirements": {"output_format": "PDF"}},
            )

    def test_extract_without_persistence_is_forced_to_stateless(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateful", "requires_business_data": True, "requires_persistence": True, "reason": "模型过度设计"},
            task_description={"task": "从对话中提取植物信息", "goal": "返回结构化结果", "expected_result": "JSON"},
            task_intent="extract",
        )
        self.assertEqual(plan["mode"], "stateless")
        self.assertFalse(plan["requires_business_data"])

    def test_explicit_persistence_keeps_extract_task_stateful(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateful", "requires_business_data": True, "requires_persistence": True, "reason": "任务明确要求保存"},
            task_description={"task": "提取信息并写入数据库", "goal": "保存记录", "expected_result": "新记录"},
            task_intent="extract",
        )
        self.assertEqual(plan["mode"], "stateful")

    def test_environment_plan_normalizes_unambiguous_alias(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "text-only", "reason": "纯文本处理"},
            task_description={"task": "整理会议日程"},
            task_intent="schedule",
        )
        self.assertEqual(plan["mode"], "stateless")

    def test_environment_plan_uses_deterministic_fallback_for_unknown_mode(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "planning_workspace", "reason": "模型返回了未知标签"},
            task_description={"task": "生成一份半天工作坊时间表"},
            task_intent="schedule",
        )
        self.assertEqual(plan["mode"], "stateless")

    def test_environment_plan_fallback_preserves_explicit_persistence(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "database_mode"},
            task_description={"task": "将预约写入数据库并更新记录"},
            task_intent="execute",
        )
        self.assertEqual(plan["mode"], "stateful")

    def test_environment_plan_upgrades_public_latest_lookup(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateless", "reason": "无需数据"},
            task_description={
                "task": "查询青岛观光森林公园并整理列表",
                "requirements": {"factual_constraint": "仅列出公开可查的信息"},
            },
            task_intent="query",
        )
        self.assertEqual(plan["mode"], "external_capability")

    def test_environment_plan_upgrades_current_market_lookup(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateless", "reason": "使用常识"},
            task_description={"task": "列出当前市场上的知名运动鞋品牌"},
            task_intent="query",
        )
        self.assertEqual(plan["mode"], "external_capability")

    def test_environment_plan_upgrades_administrative_area_fact(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateless", "reason": "模型常识"},
            task_description={"task": "把三个地名按所属省级行政区归类"},
            task_intent="classify",
        )
        self.assertEqual(plan["mode"], "external_capability")

    def test_environment_plan_upgrades_price_estimate_without_prices(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateless", "reason": "直接计算"},
            task_description={"task": "根据菜品名称计算三人组合并预估总价"},
            task_intent="calculate",
        )
        self.assertEqual(plan["mode"], "external_capability")

    def test_environment_plan_uses_runtime_supplied_comparison_data(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "reference_data", "reason": "便于比较"},
            task_description={"task": "基于用户提供的规格数据比较两款笔记本"},
            task_intent="compare",
        )
        self.assertEqual(plan["mode"], "stateless")

    def test_environment_plan_uses_user_supplied_recommendation_options(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "reference_data", "reason": "便于推荐"},
            task_description={
                "task": "从用户明确提供的周末活动选项中筛选并推荐一个选项",
                "requirements": {"constraints": "仅依赖用户提供的列表和属性"},
            },
            task_intent="recommend",
        )
        self.assertEqual(plan["mode"], "stateless")

    def test_environment_plan_keeps_parameter_only_recommendation_data_backed(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateless", "reason": "用户提供参数"},
            task_description={"task": "根据用户提供的碳纤维参数推荐适合的布料选项"},
            task_intent="recommend",
        )
        self.assertEqual(plan["mode"], "reference_data")

    def test_environment_plan_gives_recommendations_reference_evidence(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateless", "reason": "可直接回答"},
            task_description={"task": "比较两类材料并给出推荐"},
            task_intent="recommend",
        )
        self.assertEqual(plan["mode"], "reference_data")

    def test_environment_plan_gives_explanations_reference_evidence(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateless", "reason": "使用模型常识"},
            task_description={"task": "解释一道地方菜的历史和烹饪技法"},
            task_intent="explain",
        )
        self.assertEqual(plan["mode"], "reference_data")

    def test_environment_plan_keeps_common_knowledge_explanation_stateless(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "reference_data", "reason": "模型过度设计资料库"},
            task_description={"task": "解释枕头的常见用途和基本构造"},
            task_intent="explain",
        )
        self.assertEqual(plan["mode"], "stateless")
        self.assertFalse(plan["requires_business_data"])

    def test_agentic_route_aligns_stateless_query_to_reference_data(self):
        plan = TaskGenerationPipeline._align_environment_plan_with_route(
            {"mode": "stateless", "requires_business_data": False,
             "requires_persistence": False, "reason": "model noise"},
            route_plan={"environment_operations": [
                {"action_name": "lookup_record", "purpose": "read hidden fact", "dependencies": []},
            ]},
            task_intent="query",
            supported_modes={"stateless", "reference_data", "stateful"},
        )
        self.assertEqual(plan["mode"], "reference_data")
        self.assertTrue(plan["requires_business_data"])
        self.assertFalse(plan["requires_persistence"])

    def test_agentic_route_aligns_mutation_to_stateful(self):
        plan = TaskGenerationPipeline._align_environment_plan_with_route(
            {"mode": "stateless", "requires_business_data": False,
             "requires_persistence": False, "reason": "model noise"},
            route_plan={"environment_operations": [
                {"action_name": "update_record", "purpose": "persist change", "dependencies": []},
            ]},
            task_intent="modify",
            supported_modes={"stateless", "reference_data", "stateful"},
        )
        self.assertEqual(plan["mode"], "stateful")
        self.assertTrue(plan["requires_persistence"])

    def test_agentic_route_rejects_missing_required_environment_mode(self):
        with self.assertRaisesRegex(PipelineGenerationError, "route requires environment mode"):
            TaskGenerationPipeline._align_environment_plan_with_route(
                {"mode": "stateless", "requires_business_data": False,
                 "requires_persistence": False, "reason": "model noise"},
                route_plan={"environment_operations": [
                    {"action_name": "update_record", "purpose": "persist change", "dependencies": []},
                ]},
                task_intent="execute",
                supported_modes={"stateless", "reference_data"},
            )

    def test_metric_runtime_semantics_accepts_structured_expected_value(self):
        TaskGenerationPipeline._validate_metric_runtime_semantics(
            metrics=[{"id": "outcome", "category": "outcome"}],
            metric_implementations=[{
                "metric_id": "outcome", "source": "business_state",
                "operator": "equals", "expected": {"status": "complete"},
                "score_mapping": {"pass": 1, "fail": 0},
            }],
            noise_tools=[],
        )

    def test_environment_plan_gives_training_plans_reference_evidence(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateless", "reason": "直接规划"},
            task_description={"task": "制定四周足部稳定性训练计划"},
            task_intent="plan",
        )
        self.assertEqual(plan["mode"], "reference_data")

    def test_environment_plan_gives_decisions_reference_evidence(self):
        plan = TaskGenerationPipeline._resolve_environment_plan(
            {"mode": "stateless", "reason": "直接判断"},
            task_description={"task": "在三类装备中决定最优先购买哪一类"},
            task_intent="decide",
        )
        self.assertEqual(plan["mode"], "reference_data")

    def test_task_description_repair_envelope_is_unwrapped(self):
        repaired = {"grounding_issues": [], "output": {"task": "低风险任务", "complexity": "simple"}}
        self.assertEqual(
            TaskGenerationPipeline._unwrap_task_description(repaired)["task"],
            "低风险任务",
        )

    def test_task_scope_rejects_chain_of_thought(self):
        with self.assertRaisesRegex(PipelineGenerationError, "chain-of-thought"):
            TaskGenerationPipeline._validate_task_generation_scope(
                {"task": "输出你的思考过程和建议"}, graph_context={}
            )

    def test_task_scope_rejects_unsourced_high_stakes_advice(self):
        with self.assertRaisesRegex(PipelineGenerationError, "authoritative source"):
            TaskGenerationPipeline._validate_task_generation_scope(
                {"task": "根据腹痛症状判断是否需要用药"}, graph_context={}
            )

    def test_task_scope_rejects_unsourced_regulation_task_early(self):
        with self.assertRaisesRegex(PipelineGenerationError, "authoritative source"):
            TaskGenerationPipeline._validate_task_generation_scope(
                {"task": "根据现行法规完成合规审查"}, graph_context={}
            )

    def test_task_scope_rejects_medical_factual_explanation_without_sources(self):
        with self.assertRaisesRegex(PipelineGenerationError, "authoritative source"):
            TaskGenerationPipeline._validate_task_generation_scope(
                {"task": "解释癌症与冠状病毒感染的已知关联"}, graph_context={}
            )

    def test_task_scope_does_not_treat_ordinary_chemical_fact_as_high_stakes(self):
        TaskGenerationPipeline._validate_task_generation_scope(
            {"task": "把乙醇的密度记录整理成表格"}, graph_context={}
        )

    def test_task_scope_treats_chemical_safety_decision_as_high_stakes(self):
        with self.assertRaisesRegex(PipelineGenerationError, "authoritative source"):
            TaskGenerationPipeline._validate_task_generation_scope(
                {"task": "判断乙醇操作是否安全并给出防护建议"}, graph_context={}
            )

    def test_data_environment_gets_a_read_action_fallback(self):
        actions = [{"name": "比较候选", "description": "比较选项"}]
        result = TaskGenerationPipeline._ensure_data_access_action(
            actions, environment_mode="reference_data", table_names=["products"]
        )
        self.assertEqual(result[0]["name"], "读取任务参考数据")
        self.assertEqual(result[1:], actions)

    def test_data_access_capability_is_normalized_to_environment_operation(self):
        result = TaskGenerationPipeline._normalize_data_access_capabilities(
            [{"action_name": "读取资料", "kind": "agent_reasoning", "requires_tool": False, "reason": "读取"}],
            actions=[{"name": "读取资料", "description": "查询参考表"}],
            environment_mode="reference_data",
        )
        self.assertEqual(result[0]["kind"], "environment_operation")
        self.assertTrue(result[0]["requires_tool"])

    def test_user_script_state_machine_accepts_reachable_terminal_graph(self):
        TaskGenerationPipeline._validate_user_script_state_machine({
            "initial_state": "request",
            "variables": {"budget": None},
            "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。", "handled_outcomes": ["agent_off_topic", "agent_premature_completion", "unrecognized"]},
            "states": [
                {"state_id": "request", "user_behavior": "提出请求", "terminal": False},
                {"state_id": "clarify", "user_behavior": "补充预算", "terminal": False},
                {"state_id": "done", "user_behavior": "确认结束", "terminal": True},
            ],
            "transitions": [
                {"transition_id": "ask", "outcome_category": "information_required", "from_state": "request", "to_state": "clarify", "condition": "需要预算", "should_end": False, "updates": {"budget": 100}},
                {"transition_id": "correct", "outcome_category": "user_correction", "from_state": "request", "to_state": "clarify", "condition": "纠正要求", "should_end": False, "updates": {}},
                {"transition_id": "reject", "outcome_category": "user_rejection", "from_state": "request", "to_state": "clarify", "condition": "拒绝方案", "should_end": False, "updates": {}},
                {"transition_id": "satisfy", "outcome_category": "goal_satisfied", "from_state": "request", "to_state": "clarify", "condition": "当前目标完成", "should_end": False, "updates": {}},
                {"transition_id": "finish", "outcome_category": "user_acceptance", "from_state": "clarify", "to_state": "done", "condition": "用户接受", "should_end": True, "updates": {}},
            ],
        }, 0)

    def test_deterministic_user_scripts_are_valid_and_have_requested_count(self):
        scripts = TaskGenerationPipeline._deterministic_user_scripts(
            description={"description": "比较两个候选并给出建议"}, count=2
        )
        self.assertEqual([item["script_id"] for item in scripts], ["script-1", "script-2"])
        self.assertTrue(all(len(item["transitions"]) >= 5 for item in scripts))
        self.assertTrue(all(
            {transition["outcome_category"] for transition in item["transitions"]}
            == {"goal_satisfied", "information_required", "user_correction", "user_rejection", "user_acceptance"}
            for item in scripts
        ))
        for index, script in enumerate(scripts):
            TaskGenerationPipeline._validate_user_script_state_machine(script, index)

    def test_user_simulation_manifest_contains_only_runtime_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = TaskGenerationPipeline._materialize_user_simulation(
                [{"profile_id": "profile-1"}],
                [{"script_id": "script-1"}],
                root,
            )
            self.assertEqual(manifest["version"], "3.0")
            self.assertNotIn("sessions", manifest)
            self.assertTrue((root / manifest["profiles_file"]).is_file())
            self.assertTrue((root / manifest["scripts_file"]).is_file())

    def test_user_simulation_manifest_can_publish_a_portable_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "data" / "user_simulation"
            manifest = TaskGenerationPipeline._materialize_user_simulation(
                [{"profile_id": "profile-1"}],
                [{"script_id": "script-1"}],
                root,
                manifest_root="data/user_simulation",
            )
            self.assertEqual(manifest["root"], "data/user_simulation")

    def test_stateful_baseline_must_preserve_explicit_old_value(self):
        description = {"task": "请把商品场景从“景观建筑”更正为“服装”。"}
        with self.assertRaisesRegex(PipelineGenerationError, "precondition"):
            TaskGenerationPipeline._validate_stateful_preconditions(
                [{"table_name": "product", "rows": [{"scene": "服装"}]}],
                task_description=description,
                environment_mode="stateful",
            )
        TaskGenerationPipeline._validate_stateful_preconditions(
            [{"table_name": "product", "rows": [{"scene": "景观建筑"}]}],
            task_description=description,
            environment_mode="stateful",
        )

    def test_user_script_state_machine_rejects_unreachable_state(self):
        with self.assertRaisesRegex(PipelineGenerationError, "unreachable"):
            TaskGenerationPipeline._validate_user_script_state_machine({
                "initial_state": "request",
                "variables": {},
                "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。", "handled_outcomes": ["agent_off_topic", "agent_premature_completion", "unrecognized"]},
                "states": [
                    {"state_id": "request", "user_behavior": "提出请求", "terminal": False},
                    {"state_id": "orphan", "user_behavior": "不可达", "terminal": False},
                    {"state_id": "done", "user_behavior": "结束", "terminal": True},
                ],
                "transitions": [
                    {"transition_id": "finish", "outcome_category": "user_acceptance", "from_state": "request", "to_state": "done", "condition": "完成", "should_end": True, "updates": {}},
                    {"transition_id": "orphan-finish", "outcome_category": "user_acceptance", "from_state": "orphan", "to_state": "done", "condition": "完成", "should_end": True, "updates": {}},
                ],
            }, 0)

    def test_user_script_state_machine_rejects_cycles(self):
        with self.assertRaisesRegex(PipelineGenerationError, "acyclic"):
            TaskGenerationPipeline._validate_user_script_state_machine({
                "initial_state": "request", "variables": {},
                "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。", "handled_outcomes": ["agent_off_topic", "agent_premature_completion", "unrecognized"]},
                "states": [
                    {"state_id": "request", "user_behavior": "请求", "terminal": False},
                    {"state_id": "clarify", "user_behavior": "澄清", "terminal": False},
                    {"state_id": "done", "user_behavior": "结束", "terminal": True},
                ],
                "transitions": [
                    {"transition_id": "forward", "outcome_category": "information_required", "from_state": "request", "to_state": "clarify", "condition": "追问", "should_end": False, "updates": {}},
                    {"transition_id": "back", "outcome_category": "user_correction", "from_state": "clarify", "to_state": "request", "condition": "重试", "should_end": False, "updates": {}},
                    {"transition_id": "finish", "outcome_category": "user_acceptance", "from_state": "clarify", "to_state": "done", "condition": "完成", "should_end": True, "updates": {}},
                ],
            }, 0)

    def test_metric_constraints_must_come_from_task(self):
        metric = {
            "rubric": "最终回答约 200 字",
            "criteria": ["至少包含 2 个类别"],
            "condition": "回答完整",
            "evaluator": {},
        }
        with self.assertRaisesRegex(PipelineGenerationError, "numeric constraints"):
            TaskGenerationPipeline._validate_metric_constraint_grounding(
                [metric], task_description={"task": "撰写简洁的宣传文案"}
            )
        TaskGenerationPipeline._validate_metric_constraint_grounding(
            [metric],
            task_description={"task": "撰写约 200 字的宣传文案，至少包含 2 个类别"},
        )

    def test_success_fixture_enforces_declared_character_range(self):
        description = {"requirements": ["回答限制在 100-200 字"]}
        with self.assertRaisesRegex(PipelineGenerationError, "violates required"):
            TaskGenerationPipeline._validate_success_fixture_constraints(
                "太短", task_description=description
            )
        TaskGenerationPipeline._validate_success_fixture_constraints(
            "字" * 120, task_description=description
        )

    def test_success_fixture_audit_rejects_ignored_latest_instruction(self):
        with self.assertRaisesRegex(PipelineGenerationError, "latest_instructions_satisfied"):
            TaskGenerationPipeline._validate_success_fixture_audit({
                "latest_instructions_satisfied": False,
                "grounded": True,
                "goal_completed": True,
                "issues": ["未将IO口改成I/O口"],
            })

    def test_capability_plan_separates_tools_from_reasoning_and_response(self):
        actions = [
            {"name": "查询库存"},
            {"name": "比较候选项"},
            {"name": "生成最终回答"},
        ]
        capabilities = [
            {"action_name": "查询库存", "kind": "environment_operation", "requires_tool": True, "reason": "读取私有库存"},
            {"action_name": "比较候选项", "kind": "agent_reasoning", "requires_tool": False, "reason": "根据查询结果推理"},
            {"action_name": "生成最终回答", "kind": "agent_response", "requires_tool": False, "reason": "由 Agent 回复"},
        ]
        TaskGenerationPipeline._validate_capability_plan(capabilities, actions)
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_capability_plan(capabilities[:2], actions)

    def test_capability_dependency_aliases_are_canonicalized(self):
        actions = [
            {"name": "查询订单", "action_id": "lookup"},
            {"name": "更新订单"},
        ]
        capabilities = [
            {"action_name": "查询订单", "kind": "environment_operation",
             "requires_tool": True, "dependencies": [], "reason": "读取状态"},
            {"action_name": "更新订单", "kind": "environment_operation",
             "requires_tool": True, "dependencies": ["step_1", "lookup"],
             "reason": "使用查询结果更新状态"},
        ]
        normalized = TaskGenerationPipeline._normalize_capability_dependencies(
            capabilities, actions=actions
        )
        self.assertEqual(normalized[1]["dependencies"], ["查询订单"])
        TaskGenerationPipeline._validate_capability_plan(
            normalized,
            actions,
            environment_mode="stateful",
            has_business_data=True,
            training_category="multi_step_agentic",
        )

    def test_route_plan_defines_multi_step_environment_chain(self):
        route_plan = {"environment_operations": [
            {"action_name": "查询订单", "purpose": "读取订单", "dependencies": []},
            {"action_name": "更新订单", "purpose": "根据查询结果更新订单",
             "dependencies": ["查询订单"]},
        ]}
        TaskGenerationPipeline._validate_route_plan(
            route_plan, "multi_step_agentic"
        )
        with self.assertRaisesRegex(PipelineGenerationError, "dependent"):
            TaskGenerationPipeline._validate_route_plan(
                {"environment_operations": [route_plan["environment_operations"][0]]},
                "multi_step_agentic",
            )
        with self.assertRaisesRegex(PipelineGenerationError, "earlier actions"):
            TaskGenerationPipeline._validate_route_plan({
                "environment_operations": [{
                    "action_name": "更新订单", "purpose": "更新",
                    "dependencies": ["查询订单"],
                }]
            }, "simple_agentic")

    def test_route_plan_is_projected_onto_capabilities(self):
        route_plan = {"environment_operations": [
            {"action_name": "查询订单", "purpose": "读取订单", "dependencies": []},
            {"action_name": "更新订单", "purpose": "更新订单",
             "dependencies": ["查询订单"]},
        ]}
        capabilities = [
            {"action_name": "查询订单", "kind": "agent_reasoning",
             "requires_tool": False, "dependencies": [], "reason": "错误分类"},
            {"action_name": "更新订单", "kind": "agent_reasoning",
             "requires_tool": False, "dependencies": [], "reason": "错误分类"},
        ]
        normalized = TaskGenerationPipeline._apply_route_capability_contract(
            capabilities, route_plan=route_plan
        )
        self.assertTrue(all(item["requires_tool"] for item in normalized))
        self.assertEqual(normalized[1]["dependencies"], ["查询订单"])

    def test_agentic_route_rejects_task_fully_solved_by_public_input(self):
        with self.assertRaisesRegex(PipelineGenerationError, "fully solvable from public user input"):
            TaskGenerationPipeline._validate_route_input_boundary({
                "task": "基于用户提供的销售数据计算三种价格下的利润",
                "public_input": {
                    "initial_user_message": "这是完整销售数据，请计算利润。",
                    "materials": [{"name": "sales.json", "content": '{"sales": 10}'}],
                },
            }, "multi_step_agentic")

    def test_agentic_route_allows_public_identifier_for_private_lookup(self):
        TaskGenerationPipeline._validate_route_input_boundary({
            "task": "根据用户给出的订单号查询内部订单记录并更新状态",
            "public_input": {"initial_user_message": "请处理订单 ORD-1", "materials": []},
        }, "multi_step_agentic")

    def test_agentic_route_allows_public_material_with_private_catalog_dependency(self):
        TaskGenerationPipeline._validate_route_input_boundary({
            "task": "基于用户提供的采购清单匹配内部库存目录",
            "public_input": {"initial_user_message": "请基于以下提供的清单完成匹配"},
            "route_plan": {"environment_operations": [{
                "action_name": "query_internal_inventory",
                "purpose": "查询内部库存数据库",
                "dependencies": [],
            }]},
        }, "multi_step_agentic")

    def test_route_actions_must_be_preserved_by_decomposition(self):
        with self.assertRaisesRegex(PipelineGenerationError, "更新订单"):
            TaskGenerationPipeline._validate_route_action_coverage(
                [{"name": "查询订单"}],
                route_plan={"environment_operations": [
                    {"action_name": "查询订单"}, {"action_name": "更新订单"},
                ]},
            )

    def test_unknown_capability_dependency_is_not_silently_discarded(self):
        actions = [{"name": "查询订单"}]
        capabilities = [{
            "action_name": "查询订单", "kind": "environment_operation",
            "requires_tool": True, "dependencies": ["invented-step"],
            "reason": "读取状态",
        }]
        normalized = TaskGenerationPipeline._normalize_capability_dependencies(
            capabilities, actions=actions
        )
        with self.assertRaisesRegex(PipelineGenerationError, "known action names"):
            TaskGenerationPipeline._validate_capability_plan(normalized, actions)

    def test_data_backed_capability_plan_requires_data_access_tool(self):
        actions = [{"name": "读取面粉属性"}, {"name": "生成回答"}]
        capabilities = [
            {"action_name": "读取面粉属性", "kind": "agent_reasoning", "requires_tool": False, "reason": "读取"},
            {"action_name": "生成回答", "kind": "agent_response", "requires_tool": False, "reason": "回答"},
        ]
        with self.assertRaisesRegex(PipelineGenerationError, "data access operation"):
            TaskGenerationPipeline._validate_capability_plan(
                capabilities, actions,
                environment_mode="reference_data", has_business_data=True,
            )

    def test_data_grounding_audit_rejects_ambiguous_decision(self):
        with self.assertRaisesRegex(PipelineGenerationError, "decision_determinate"):
            TaskGenerationPipeline._validate_data_grounding_audit({
                "task_supported": True,
                "decision_determinate": False,
                "facts_consistent": True,
                "issues": ["两个候选项具有相同优先级，无法唯一推荐"],
            })

    def test_task_grounding_audit_rejects_missing_factual_inputs(self):
        with self.assertRaisesRegex(PipelineGenerationError, "no_unprovided_facts"):
            TaskGenerationPipeline._validate_task_grounding_audit({
                "self_contained": True,
                "no_unprovided_facts": False,
                "expected_result_derivable": False,
                "internally_consistent": True,
                "issues": ["要求替换时价，但没有提供权威价格"],
            })

    def test_recipe_shopping_list_is_not_a_domain_conflict(self):
        TaskGenerationPipeline._validate_task_description_consistency({
            "task": "按食谱生成采购购物清单",
            "requirements": {"categories": ["食材"], "output_format": "购物清单"},
        })

    def test_task_description_consistency_accepts_matching_list_contract(self):
        TaskGenerationPipeline._validate_task_description_consistency({
            "task": "把食材整理成分类购物清单并排序",
            "requirements": {
                "categories": ["蔬菜", "主食"],
                "sort_within_category": "按拼音排序",
            },
        })

    def test_coding_event_management_is_allowed(self):
        TaskGenerationPipeline._validate_task_description_consistency({
            "task": "编写视频会议日程管理脚本",
            "requirements": {"input": "活动方案和人员分工", "output": "日程"},
        })

    def test_action_alignment_audit_rejects_cross_task_drift(self):
        with self.assertRaisesRegex(PipelineGenerationError, "not task-aligned"):
            TaskGenerationPipeline._validate_action_alignment_audit({
                "aligned": False,
                "complete": True,
                "atomic": True,
                "issues": ["黑熊估算动作错误引用了小区借用次数"],
            })

    def test_action_grounding_rejects_cross_task_keywords(self):
        with self.assertRaisesRegex(PipelineGenerationError, "task-specific keyword"):
            TaskGenerationPipeline._validate_action_grounding(
                [{"name": "设计两天团队建设", "description": "安排北京团建活动"}],
                task_description={"task": "制定美国东岸连锁超市考察行程"},
                keywords=["美国东岸"],
            )

    def test_action_grounding_rejects_invented_numeric_constraints(self):
        with self.assertRaisesRegex(PipelineGenerationError, "invent numeric constraints"):
            TaskGenerationPipeline._validate_action_grounding(
                [{"name": "分配预算", "description": "为美国东岸行程按20人、每人500元分配"}],
                task_description={"task": "制定5天美国东岸考察行程"},
                keywords=["美国东岸"],
            )

    def test_related_action_concepts_need_not_be_literal_in_task(self):
        TaskGenerationPipeline._validate_action_grounding(
            [{"name": "设计活动场地布置", "description": "为读书活动安排场地布置和宣传方式"}],
            task_description={"task": "组织读书活动"}, keywords=["读书活动"],
        )

    def test_action_grounding_does_not_accept_keyword_echo_in_inputs(self):
        with self.assertRaisesRegex(PipelineGenerationError, "task-specific keyword"):
            TaskGenerationPipeline._validate_action_grounding(
                [{
                    "name": "统计咖啡店收入",
                    "description": "汇总每日营业额",
                    "inputs": [{"name": "customary_law", "description": "习惯法模拟"}],
                    "outputs": [{"name": "revenue_total"}],
                }],
                task_description={"task": "模拟习惯法争议解决过程"},
                keywords=["习惯法"],
            )

    def test_action_grounding_rejects_predominantly_generic_steps(self):
        with self.assertRaisesRegex(PipelineGenerationError, "generic placeholders"):
            TaskGenerationPipeline._validate_action_grounding(
                [
                    {"name": "接收并解析用户输入", "description": "提取历史教育材料"},
                    {"name": "检查信息完整性", "description": "检查中国东北地区资料"},
                    {"name": "执行诊断处理", "description": "处理教学难点"},
                    {"name": "格式化输出结果", "description": "输出排查路径"},
                ],
                task_description={"task": "诊断中国东北地区历史教育的教学难点"},
                keywords=["中国东北地区", "历史教育"],
            )

    def test_action_grounding_rejects_generic_action_paraphrases(self):
        with self.assertRaisesRegex(PipelineGenerationError, "generic placeholders"):
            TaskGenerationPipeline._validate_action_grounding(
                [
                    {"name": "接收并规范化用户输入", "description": "处理襟边故障描述"},
                    {"name": "检查输入完整性与明确要求", "description": "检查输入"},
                    {"name": "生成补充信息请求", "description": "请求信息"},
                    {"name": "格式化并输出最终结果", "description": "输出襟边诊断"},
                ],
                task_description={"task": "诊断襟边开裂原因"},
                keywords=["襟边"],
            )

    def test_cross_domain_requirements_are_left_to_semantic_audit(self):
        TaskGenerationPipeline._validate_task_description_consistency({
            "task": "制作食品冷藏设备监控程序",
            "requirements": {"input_format": "食材保鲜条件与设备日志", "output_format": "告警"},
        })

    def test_grounding_evidence_is_required_independently_of_domain(self):
        for issue in ("分类规则缺失", "价格来源缺失", "状态变更目标缺失"):
            with self.assertRaisesRegex(PipelineGenerationError, "not grounded"):
                TaskGenerationPipeline._validate_task_grounding_audit({
                    "self_contained": False, "no_unprovided_facts": False,
                    "expected_result_derivable": False, "internally_consistent": True,
                    "issues": [issue],
                })

    def test_cross_domain_comparison_is_not_rejected_by_keywords(self):
        TaskGenerationPipeline._validate_task_description_consistency({
            "task_intent": "compare",
            "task": "比较地方食材包装和城市建筑中的相同纹样",
            "requirements": {"input": "用户提供的纹样描述", "criteria": "按相同分类规则比较"},
        })

    def test_task_description_structure_remains_validated(self):
        for value in (None, {"task": ""}, {"task": "选择候选项", "requirements": []}):
            with self.assertRaises(PipelineGenerationError):
                TaskGenerationPipeline._validate_task_description_consistency(value)

    def test_reasoning_request_is_normalized_to_concise_rationale(self):
        normalized = TaskGenerationPipeline._normalize_reasoning_request({
            "task": "给出逐步思考和推理过程",
            "requirements": {"constraint": "show your reasoning"},
        })
        text = json.dumps(normalized, ensure_ascii=False).lower()
        self.assertNotIn("思考过程", text)
        self.assertNotIn("推理过程", text)
        self.assertNotIn("show your reasoning", text)

    def test_no_fixed_topic_fallback_is_exposed(self):
        self.assertFalse(hasattr(TaskGenerationPipeline, "_deterministic_low_risk_task"))

    def test_executable_scenario_accepts_null_capture_as_empty(self):
        TaskGenerationPipeline._validate_executable_scenarios(
            [{
                "scenario_id": "success-1",
                "kind": "goal_success",
                "steps": [
                    {"operation": "reset", "capture": None},
                    {"operation": "agent_response", "content": "done", "capture": None},
                    {"operation": "reward", "capture": None},
                ],
                "assertions": [{"path": "$.reward", "operator": "gte", "expected": 0.5}],
            }, {
                "scenario_id": "failure-1",
                "kind": "goal_failure",
                "steps": [
                    {"operation": "reset", "capture": None},
                    {"operation": "agent_response", "content": "incomplete", "capture": None},
                    {"operation": "reward", "capture": None},
                ],
                "assertions": [{"path": "$.reward", "operator": "lte", "expected": 0.2}],
            }],
            tools=[],
            noise_tools=[],
        )

    def test_action_numeric_examples_are_replaced_when_not_grounded(self):
        actions = TaskGenerationPipeline._normalize_action_numeric_examples(
            [{
                "name": "估算费用",
                "description": "例如按1000元和6个月估算",
                "inputs": [], "outputs": [],
                "preconditions": ["已有1000元示例"], "effects": [],
            }],
            task_description={"task": "根据用户提供金额和期限估算费用"},
        )
        self.assertNotIn("1000", json.dumps(actions, ensure_ascii=False))
        self.assertNotIn("6个月", json.dumps(actions, ensure_ascii=False))

    def test_capability_plan_rejects_response_composition_tool(self):
        with self.assertRaisesRegex(PipelineGenerationError, "response composition"):
            TaskGenerationPipeline._validate_capability_plan(
                [{
                    "action_name": "生成Markdown对比表格",
                    "kind": "environment_operation",
                    "requires_tool": True,
                    "reason": "调用格式化工具",
                }],
                [{"name": "生成Markdown对比表格"}],
            )

    def test_data_keyword_alignment_rejects_cross_task_fixture(self):
        with self.assertRaisesRegex(PipelineGenerationError, "environment data omits"):
            TaskGenerationPipeline._validate_data_keyword_alignment(
                [{"table_name": "candidate_city", "rows": [{"name": "A市"}]}],
                task_description={"task": "判断日语和阿依努语哪一种符合多式综合语"},
                keywords=["多式综合语", "日语", "阿依努语"],
            )

    def test_data_keyword_alignment_accepts_matching_fixture(self):
        TaskGenerationPipeline._validate_data_keyword_alignment(
            [{"table_name": "language", "rows": [{"name": "阿依努语"}]}],
            task_description={"task": "判断日语和阿依努语哪一种符合多式综合语"},
            keywords=["多式综合语", "日语", "阿依努语"],
        )

    def test_success_fixture_length_normalization_trims_to_explicit_limit(self):
        content = "甲" * 125
        normalized = TaskGenerationPipeline._normalize_success_fixture_length(
            content,
            task_description={"requirements": {"length": "80-120字"}},
        )
        self.assertEqual(len(normalized), 120)
        TaskGenerationPipeline._validate_success_fixture_constraints(
            normalized,
            task_description={"requirements": {"length": "80-120字"}},
        )

    def test_structured_output_unwraps_common_model_envelopes(self):
        audit = {
            "output": {
                "self_contained": True,
                "no_unprovided_facts": True,
                "expected_result_derivable": True,
                "internally_consistent": True,
                "issues": [],
            },
            "type": "object",
        }
        normalized = TaskGenerationPipeline._unwrap_structured_output(
            audit,
            expected_fields={
                "self_contained", "no_unprovided_facts",
                "expected_result_derivable", "internally_consistent", "issues",
            },
        )
        TaskGenerationPipeline._validate_task_grounding_audit(normalized)

    def test_metric_constraint_grounding_ignores_boolean_control_values(self):
        metrics = [{
            "rubric": "无无效循环",
            "condition": "invalid_loop_count == 0 and completed == 1",
            "evaluator": {},
        }]
        TaskGenerationPipeline._validate_metric_constraint_grounding(
            metrics, task_description={"task": "完成任务且避免无效循环"}
        )

    def test_metric_grounding_accepts_equivalent_numeric_format(self):
        metric = [{"rubric": "The exact cost difference is 58200."}]
        TaskGenerationPipeline._validate_metric_constraint_grounding(
            metric, task_description={"verified_reward_facts": {"difference": 58200.0}}
        )
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_metric_constraint_grounding(
                metric, task_description={"verified_reward_facts": {"difference": 58201.0}}
            )

    def test_ungrounded_numeric_metric_is_dropped_without_losing_valid_metric(self):
        metrics = [
            {"id": "invented", "rubric": "回答至少 50 字", "evaluator": {}},
            {"id": "valid", "rubric": "回答忠实于输入", "evaluator": {}},
        ]
        retained = TaskGenerationPipeline._drop_ungrounded_numeric_metrics(
            metrics, task_description={"task": "根据输入生成回答"}
        )
        self.assertEqual([item["id"] for item in retained], ["valid"])

    def test_exhausted_grounding_is_not_replaced_with_a_template(self):
        pipeline = TaskGenerationPipeline(SimpleNamespace(), retries=1)
        description = {"task": "按未提供的规则分类输入记录", "task_intent": "classify",
                       "complexity": "simple", "requirements": {},
                       "route_plan": {"environment_operations": []}}
        audit = {"self_contained": False, "no_unprovided_facts": False,
                 "expected_result_derivable": False, "internally_consistent": True,
                 "issues": ["分类规则缺失"]}
        with patch.object(pipeline, "_call", side_effect=[description, audit]) as call:
            with self.assertRaisesRegex(PipelineGenerationError, "not grounded"):
                pipeline.generate(keywords=["记录"], task_type="QA", style="standard",
                                  task_intent="classify", graph_context={}, training_category="direct_response")
        self.assertEqual([args.args[0] for args in call.call_args_list],
                         ["task_description", "task_description_grounding_audit"])

    def test_reference_data_outcome_is_normalized_to_response_judge(self):
        metrics = [{
            "id": "outcome_recommendation_persisted",
            "category": "outcome",
            "type": "rule-based",
            "scope": "state",
            "condition": "recommendation persisted",
            "rubric": "推荐结果正确且有依据",
            "evaluator": {
                "kind": "business_state_rule", "source": "runtime_rule",
                "assertion": "recommendation_persisted == true",
                "score_mapping": {"true": 1, "false": 0},
            },
        }]
        normalized = TaskGenerationPipeline._normalize_metrics_for_environment(
            metrics, environment_mode="reference_data"
        )
        self.assertEqual(normalized[0]["type"], "model-based")
        self.assertEqual(normalized[0]["evaluator"]["kind"], "external_llm_judge")
        self.assertEqual(
            normalized[0]["evaluation_inputs"],
            ["final_agent_response", "business_data"],
        )

    def test_actions_use_semantic_atomic_fields(self):
        action = {
            "name": "record_clothing_items",
            "description": "记录用户提供的衣物信息",
            "atomicity_rationale": "该动作只有一个记录边界，不能继续拆成独立的 Agent 决策",
            "inputs": [{"name": "items", "description": "用户提供的衣物列表"}],
            "outputs": [{"name": "recorded_items", "description": "已保存的衣物记录"}],
            "preconditions": ["items 非空"],
            "effects": ["持久化衣物业务记录"],
        }
        TaskGenerationPipeline._validate_actions([action])

        invalid = dict(action)
        invalid["inputs"] = [{"name": "items"}]
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_actions([invalid])

    def test_deterministic_actions_preserve_task_keyword(self):
        actions = TaskGenerationPipeline._deterministic_actions(
            task_description={"task": "分析表音文字的构词特点"},
            keywords=["表音文字"],
            environment_mode="stateless",
        )
        TaskGenerationPipeline._validate_actions(actions)
        self.assertTrue(any("表音文字" in action["name"] for action in actions))

    def test_deterministic_actions_include_reference_data_read(self):
        actions = TaskGenerationPipeline._deterministic_actions(
            task_description={"task": "根据库存记录给出补货建议"},
            keywords=["库存"],
            environment_mode="reference_data",
        )
        self.assertTrue(any("读取" in action["name"] for action in actions))

    def test_deterministic_actions_include_external_lookup(self):
        actions = TaskGenerationPipeline._deterministic_actions(
            task_description={"task": "查询天气并规划行程"},
            keywords=["天气"],
            environment_mode="external_capability",
        )
        self.assertTrue(any("查询" in action["name"] for action in actions))


class OpenAIToolArtifactTest(unittest.TestCase):
    def test_success_scenario_uses_fixture_when_model_omits_agent_response(self):
        scenarios = TaskGenerationPipeline._normalize_executable_scenarios(
            [{
                "scenario_id": "success",
                "kind": "goal_success",
                "steps": [{"operation": "agent_response", "content": ""}],
                "assertions": [{"path": "$.reward", "operator": "gte", "expected": 0.6}],
            }],
            success_content="完整且经过验证的成功回答",
        )
        self.assertEqual(
            scenarios[0]["steps"][0]["content"],
            "完整且经过验证的成功回答",
        )

    def test_noise_audit_identifies_materially_useful_tool(self):
        unsafe = TaskGenerationPipeline._validate_noise_tool_audit(
            [{"name": "route_lookup", "is_safe_noise": False, "reason": "可提供排查证据"}],
            noise_names={"route_lookup"},
        )
        self.assertEqual(unsafe, {"route_lookup"})

    def test_noise_audit_accepts_safe_noise_tool(self):
        unsafe = TaskGenerationPipeline._validate_noise_tool_audit(
            [{"name": "roll_virtual_die", "is_safe_noise": True, "reason": "与任务无关"}],
            noise_names={"roll_virtual_die"},
        )
        self.assertEqual(unsafe, set())

    def test_noise_tool_cannot_claim_external_side_effect(self):
        tool = {
            "type": "function",
            "function": {
                "name": "book_restaurant_table",
                "description": "预订餐厅座位",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        with self.assertRaisesRegex(PipelineGenerationError, "side effect"):
            TaskGenerationPipeline._validate_noise_tool_safety([tool])

    def test_read_only_noise_tool_may_describe_booking_information(self):
        tool = {
            "type": "function",
            "function": {
                "name": "search_academic_workshop_venue",
                "description": "搜索学术研讨会场地及其预订状态，只返回公开信息。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        TaskGenerationPipeline._validate_noise_tool_safety([tool])

    def test_read_only_estimator_may_reference_order_lifecycle(self):
        tool = {
            "type": "function",
            "function": {
                "name": "estimate_shipping_time",
                "description": "用于下单后估算运输时间，只返回预计天数，不创建或修改订单。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        TaskGenerationPipeline._validate_noise_tool_safety([tool])

    def test_schema_derived_noise_fixture_maps_every_parameter(self):
        rows, mapping = TaskGenerationPipeline._build_noise_fixture({
            "name": "lookup_unrelated_catalog",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询词"},
                    "limit": {"type": "integer", "description": "数量"},
                },
                "required": ["query"],
            },
        })
        self.assertEqual(mapping, {"query": "query", "limit": "limit"})
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(set(mapping.values()) <= set(row) for row in rows))
        self.assertNotEqual(rows[0]["query"], rows[1]["query"])

    def test_noise_tool_description_cannot_hide_external_side_effect(self):
        tool = {
            "type": "function",
            "function": {
                "name": "manage_academic_workshop_venue",
                "description": "用于预订研讨会场地并提交预约请求。",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
        with self.assertRaisesRegex(PipelineGenerationError, "side effect"):
            TaskGenerationPipeline._validate_noise_tool_safety([tool])

    def test_noise_has_no_fixed_dice_fallback(self):
        self.assertFalse(hasattr(TaskGenerationPipeline, "_fallback_noise_tool"))

    def test_placeholder_tool_name_is_rejected(self):
        tool = {
            "type": "function",
            "function": {
                "name": "noise_tool_1",
                "description": "占位工具",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        with self.assertRaisesRegex(PipelineGenerationError, "placeholder name"):
            TaskGenerationPipeline._validate_tools([tool])

    def test_missing_nested_descriptions_are_filled_deterministically(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "query_matching_treatments",
                "description": "查询匹配治疗方案",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "patient_profile": {
                            "type": "object",
                            "properties": {
                                "genomic_alterations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["patient_profile"],
                },
            },
        }]
        TaskGenerationPipeline._fill_tool_schema_descriptions(tools)
        TaskGenerationPipeline._validate_tools(tools)
        items = tools[0]["function"]["parameters"]["properties"]["patient_profile"]["properties"]["genomic_alterations"]["items"]
        self.assertTrue(items["description"])

    def test_declarative_tool_implementation_validation(self):
        tools = [{"function": {"name": "lookup", "parameters": {
            "properties": {"kind": {"type": "string"}}
        }}}]
        tables = [{"table_name": "items", "columns": [
            {"name": "id"}, {"name": "kind"}
        ]}]
        specs = [{
            "tool_name": "lookup", "operation": "select", "table": "items",
            "filters": [{"argument": "kind", "column": "kind", "operator": "eq"}],
            "projection": ["id", "kind"], "order_by": ["id"], "result_field": "records",
        }]
        TaskGenerationPipeline._validate_tool_implementations(specs, tools=tools, tables=tables)
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_tool_implementations(
                [dict(specs[0], table="missing")], tools=tools, tables=tables
            )
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_tool_implementations(
                [dict(specs[0], projection=[{"column": "id"}])],
                tools=tools, tables=tables,
            )

    def test_foreign_key_filter_resolver_is_completed_and_validated(self):
        tools = [{"function": {"name": "count_products", "parameters": {
            "properties": {"brand_name": {"type": "string"}}
        }}}]
        tables = [
            {"table_name": "brand", "columns": [
                {"name": "id"}, {"name": "name"},
            ], "foreign_keys": []},
            {"table_name": "product", "columns": [
                {"name": "id"}, {"name": "brand_id"},
            ], "foreign_keys": [{
                "column": "brand_id", "ref_table": "brand", "ref_column": "id",
            }]},
        ]
        specs = [{
            "tool_name": "count_products", "operation": "aggregate_count",
            "table": "product", "result_field": "count",
            "filters": [{
                "argument": "brand_name", "column": "brand_id", "operator": "eq",
            }],
        }]
        completed = TaskGenerationPipeline._complete_filter_resolvers(
            implementations=specs, tables=tables,
        )
        self.assertEqual(completed[0]["filters"][0]["resolve"], {
            "table": "brand", "match_column": "name", "value_column": "id",
        })
        TaskGenerationPipeline._validate_tool_implementations(
            completed, tools=tools, tables=tables,
        )
        broken = copy.deepcopy(completed)
        broken[0]["filters"][0]["resolve"]["value_column"] = "name"
        with self.assertRaisesRegex(PipelineGenerationError, "foreign key"):
            TaskGenerationPipeline._validate_tool_implementations(
                broken, tools=tools, tables=tables,
            )

    def test_descriptive_unique_non_null_constraint_is_canonicalized(self):
        tables = [{
            "table_name": "ingredient",
            "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "name", "type": "TEXT", "nullable": True},
            ],
            "primary_key": ["id"], "foreign_keys": [], "indexes": [],
            "constraints": ["name 唯一且非空"],
        }]
        normalized = TaskGenerationPipeline._normalize_structural_constraints(tables)
        self.assertEqual(normalized[0]["constraints"], [])
        self.assertFalse(normalized[0]["columns"][1]["nullable"])
        self.assertEqual(normalized[0]["indexes"], [{
            "name": "uq_ingredient_name", "columns": ["name"], "unique": True,
        }])
        TaskGenerationPipeline._validate_table_definitions(normalized)

    def test_materialize_tools_writes_only_standard_tool_array(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "record_clothing_items",
                "description": "记录衣物信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "items": {"type": "array", "description": "衣物列表"},
                    },
                    "required": ["items"],
                },
            },
        }]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = TaskGenerationPipeline._materialize_tools(tools, root)
            self.assertEqual(manifest["file"], "tools.json")
            self.assertEqual(json.loads((root / "tools.json").read_text()), tools)

    def test_runtime_interface_lists_tools_and_reward_function(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "query_recipe",
                "description": "查询食谱",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "dish": {"type": "string", "description": "菜名"},
                    },
                    "required": ["dish"],
                },
            },
        }]
        formula = {"type": "separate_sign_weighted_sum", "score_range": [-1, 1]}
        interface = TaskGenerationPipeline._build_runtime_interface(tools, formula)
        TaskGenerationPipeline._validate_runtime_interface(interface, tools, formula)
        tool_endpoint = next(item for item in interface["endpoints"] if item["name"] == "query_recipe")
        self.assertEqual(tool_endpoint["method"], "POST")
        self.assertEqual(tool_endpoint["path"], "/v1/tools/{tool_name}")
        self.assertEqual(tool_endpoint["request_schema"], tools[0]["function"]["parameters"])
        self.assertIn("不包含 observation 或 reward", tool_endpoint["response_schema"]["description"])
        user_endpoint = next(item for item in interface["endpoints"] if item["name"] == "user_simulator")
        self.assertEqual(user_endpoint["path"], "/v1/user_simulator")
        self.assertEqual(user_endpoint["access"], "rl_trainer_only")
        self.assertEqual(
            user_endpoint["response_schema"]["required"],
            [
                "user_query", "should_end", "match_status",
                "outcome_category", "reason_code", "attachments",
                "fsm_script_id", "fsm_transition_id",
                "fsm_state_before", "fsm_state_after",
                "fsm_transition_applied", "fsm_recovery_count",
            ],
        )
        self.assertEqual(user_endpoint["request_schema"]["required"], ["messages"])
        self.assertIn("Trainer", user_endpoint["request_schema"]["description"])
        response_endpoint = next(item for item in interface["endpoints"] if item["name"] == "agent_response")
        self.assertEqual(response_endpoint["path"], "/v1/agent_response")
        self.assertEqual(response_endpoint["request_schema"]["required"], ["content"])
        self.assertEqual(interface["reward_functions"], [{"name": "reward", "endpoint": "/v1/reward"}])
        reward_endpoint = next(item for item in interface["endpoints"] if item["name"] == "reward")
        self.assertEqual(reward_endpoint["access"], "rl_trainer_only")
        replay_endpoint = next(item for item in interface["endpoints"] if item["name"] == "replay")
        self.assertEqual(replay_endpoint["access"], "rl_trainer_only")
        self.assertEqual(interface["security"]["trainer"]["environment_variable"], "SANDBOX_TRAINER_API_KEY")
        self.assertTrue(interface["episode"]["deterministic_replay"])
        self.assertEqual(interface["llm_runtime"]["api_key_environment_variable"], "SANDBOX_LLM_API_KEY")
        self.assertEqual(interface["evaluator_runtime"]["mock_mode_environment_variable"], "SANDBOX_EVALUATOR_MOCK")
        self.assertIn("ContractToolRegistry", interface["shared_runtime"]["required_components"])
        self.assertIn("ContractRewardAggregator", interface["shared_runtime"]["required_components"])


class RewardContractTest(unittest.TestCase):
    def test_reference_data_outcome_cannot_score_preloaded_state_alone(self):
        with self.assertRaisesRegex(PipelineGenerationError, "cannot by itself prove"):
            TaskGenerationPipeline._validate_task_readiness(
                capability_plan=[{
                    "action_name": "查询法规", "requires_tool": True,
                }],
                tool_bindings=[{"action_name": "查询法规", "tool_name": "search_law"}],
                noise_tools=[],
                metrics=[{
                    "id": "outcome", "category": "outcome", "type": "rule-based",
                    "evaluator": {"kind": "business_state_rule"},
                }],
                metric_implementations=[{"metric_id": "outcome"}],
                business_scenarios=[
                    {"kind": "goal_success"}, {"kind": "goal_failure"},
                ],
                environment_mode="reference_data",
                has_business_data=True,
                require_noise=False,
            )

    def test_noise_penalty_is_compiled_as_deterministic_rule(self):
        metrics = [{
            "id": "penalty_noise_old", "category": "penalty", "type": "model-based",
            "condition": "noise", "weight": 1,
        }]
        TaskGenerationPipeline._ensure_deterministic_noise_penalty(
            metrics, noise_names=["weather"]
        )
        self.assertEqual(len(metrics), 1)
        self.assertEqual(metrics[0]["id"], "penalty_noise_tool_usage")
        self.assertEqual(metrics[0]["type"], "rule-based")

    def test_stateless_metrics_reject_process_and_business_state(self):
        process = [{"category": "process", "evaluator": {"kind": "hybrid_tool_call"}}]
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_metrics_for_environment(
                process, environment_mode="stateless"
            )
        business = [{
            "category": "outcome",
            "evaluation_inputs": ["business_data"],
            "evaluator": {"kind": "external_llm_judge"},
        }]
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_metrics_for_environment(
                business, environment_mode="stateless"
            )

    def test_uncompiled_rule_metric_is_promoted_to_external_judge(self):
        metrics = [{
            "id": "grounded", "category": "outcome", "type": "rule-based",
            "scope": "terminal", "rubric": "回答与业务数据一致", "condition": "grounded",
            "evaluator": {"kind": "document_rule", "source": "runtime_rule", "assertion": "grounded", "score_mapping": {"true": 1, "false": 0}},
            "score_range": [0, 1], "weight": 1,
        }]
        TaskGenerationPipeline._promote_unimplemented_rule_metrics(metrics, [])
        self.assertEqual(metrics[0]["type"], "model-based")
        self.assertEqual(metrics[0]["evaluator"]["kind"], "external_llm_judge")
        self.assertTrue(metrics[0]["evaluation_inputs"])
        self.assertTrue(metrics[0]["criteria"])

    def test_task_readiness_requires_noise_and_business_trajectories(self):
        kwargs = {
            "capability_plan": [{"action_name": "查询", "requires_tool": True}],
            "tool_bindings": [{"tool_name": "query", "action_name": "查询"}],
            "noise_tools": [{"name": "weather"}],
            "metrics": [{
                "id": "penalty_noise_tool_usage", "category": "penalty",
                "type": "rule-based",
            }],
            "metric_implementations": [{
                "metric_id": "penalty_noise_tool_usage", "source": "trajectory",
                "path": "$.events", "operator": "none_tool_calls",
                "expected": ["weather"], "score_mapping": {"pass": 0, "fail": -1},
            }],
            "business_scenarios": [
                {"kind": "goal_success"}, {"kind": "goal_failure"}, {"kind": "noise_selection"},
            ],
        }
        TaskGenerationPipeline._validate_task_readiness(**kwargs)
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_task_readiness(**(kwargs | {"noise_tools": []}))

    def test_complexity_is_derived_from_executable_structure(self):
        self.assertEqual(TaskGenerationPipeline._derive_complexity(business_tool_count=1, metric_count=4, key_step_count=1), "simple")
        self.assertEqual(TaskGenerationPipeline._derive_complexity(business_tool_count=2, metric_count=6, key_step_count=2), "standard")
        self.assertEqual(TaskGenerationPipeline._derive_complexity(business_tool_count=5, metric_count=9, key_step_count=5), "complex")

    def test_training_profile_distinguishes_abstention_from_tool_use(self):
        self.assertEqual(
            TaskGenerationPipeline._derive_training_profile(
                business_tool_count=0, noise_tool_count=1, key_step_count=0
            ),
            "tool_abstention",
        )
        self.assertEqual(
            TaskGenerationPipeline._derive_training_profile(
                business_tool_count=2, noise_tool_count=1, key_step_count=2
            ),
            "multi_step_tool_use",
        )

    def test_executable_scenarios_require_success_failure_and_noise(self):
        tools = [{"function": {"name": "lookup", "parameters": {"properties": {}}}},
                 {"function": {"name": "weather", "parameters": {"properties": {}}}}]
        scenarios = [
            {"scenario_id": "ok", "kind": "goal_success", "steps": [{"operation": "tool_call", "tool_name": "lookup", "arguments": {}}], "assertions": [{"path": "$.reward", "operator": "gte", "expected": 0.8}]},
            {"scenario_id": "bad", "kind": "goal_failure", "steps": [{"operation": "reward"}], "assertions": [{"path": "$.reward", "operator": "lte", "expected": 0.2}]},
            {"scenario_id": "noise", "kind": "noise_selection", "steps": [{"operation": "tool_call", "tool_name": "weather", "arguments": {}}], "assertions": [{"path": "$.reward", "operator": "lte", "expected": 0}]},
        ]
        TaskGenerationPipeline._validate_executable_scenarios(
            scenarios, tools=tools, noise_tools=[{"name": "weather"}]
        )
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_executable_scenarios(
                scenarios[:2], tools=tools, noise_tools=[{"name": "weather"}]
            )

    def test_multi_step_route_requires_capture_ref_dependency(self):
        tools = [
            {"function": {"name": "search", "parameters": {"type": "object", "properties": {}}}},
            {"function": {"name": "inspect", "parameters": {
                "type": "object", "properties": {"item_id": {"type": "string"}},
                "required": ["item_id"],
            }}},
        ]
        def scenarios(arguments, capture=None):
            first = {"operation": "tool_call", "tool_name": "search", "arguments": {}}
            if capture:
                first["capture"] = capture
            return [
                {"scenario_id": "ok", "kind": "goal_success", "steps": [
                    first,
                    {"operation": "tool_call", "tool_name": "inspect", "arguments": arguments},
                ], "assertions": [{"path": "$.reward", "operator": "gte", "expected": 0.8}]},
                {"scenario_id": "bad", "kind": "goal_failure", "steps": [
                    {"operation": "reward"},
                ], "assertions": [{"path": "$.reward", "operator": "lte", "expected": 0.2}]},
            ]
        with self.assertRaisesRegex(PipelineGenerationError, "capture/\\$ref dependency"):
            TaskGenerationPipeline._validate_executable_scenarios(
                scenarios({"item_id": "item-1"}), tools=tools, noise_tools=[],
                training_category="multi_step_agentic",
            )
        TaskGenerationPipeline._validate_executable_scenarios(
            scenarios({"item_id": {"$ref": "selected_id"}}, {"selected_id": "$.items[0].id"}),
            tools=tools, noise_tools=[], training_category="multi_step_agentic",
        )

    def test_success_scenario_must_cover_each_state_goal_predicate(self):
        goal = {"row_predicates": [
            {"table": "notes", "where": {"id": "n1"}, "values": {"label": "a"}, "count": 1},
            {"table": "notes", "where": {"id": "n2"}, "values": {"label": "b"}, "count": 1},
        ]}
        implementations = [{
            "tool_name": "update_note", "operation": "update", "table": "notes",
            "selector": {"id": "id"}, "changes": {"label": "label"},
        }]
        scenario = {"steps": [
            {"operation": "tool_call", "tool_name": "update_note",
             "arguments": {"id": "n1", "label": "a"}},
        ]}
        with self.assertRaisesRegex(PipelineGenerationError, r"row_predicates\[1\]"):
            TaskGenerationPipeline._validate_success_scenario_goal_coverage(
                scenario, semantic_goal=goal, tool_implementations=implementations,
            )
        scenario["steps"].append({
            "operation": "tool_call", "tool_name": "update_note",
            "arguments": {"id": "n2", "label": "b"},
        })
        TaskGenerationPipeline._validate_success_scenario_goal_coverage(
            scenario, semantic_goal=goal, tool_implementations=implementations,
        )

    def test_declarative_business_tool_must_consume_every_public_parameter(self):
        tool = {"type": "function", "function": {
            "name": "query_notes", "description": "查询笔记", "parameters": {
                "type": "object", "properties": {
                    "category": {"type": "string"},
                    "columns": {"type": "array", "items": {"type": "string"}},
                }, "required": ["category", "columns"],
            },
        }}
        spec = {"tool_name": "query_notes", "operation": "select", "table": "notes",
                "filters": [{"argument": "category", "column": "category", "operator": "eq"}],
                "projection": ["id"], "result_field": "records"}
        with self.assertRaisesRegex(PipelineGenerationError, "ignored"):
            TaskGenerationPipeline._validate_business_tool_semantics(
                tools=[tool], implementations=[spec],
            )
        tool["function"]["parameters"]["properties"].pop("columns")
        tool["function"]["parameters"]["required"].remove("columns")
        TaskGenerationPipeline._validate_business_tool_semantics(
            tools=[tool], implementations=[spec],
        )

    def test_plain_select_cannot_claim_calculation_semantics(self):
        tool = {"type": "function", "function": {
            "name": "calculate_average", "description": "计算平均值", "parameters": {
                "type": "object", "properties": {}, "required": [],
            },
        }}
        spec = {"tool_name": "calculate_average", "operation": "select", "table": "values",
                "filters": [], "projection": ["value"], "result_field": "records"}
        with self.assertRaisesRegex(PipelineGenerationError, "overclaims"):
            TaskGenerationPipeline._validate_business_tool_semantics(
                tools=[tool], implementations=[spec],
            )

    def test_executable_scenario_identity_is_normalized(self):
        scenarios = TaskGenerationPipeline._normalize_executable_scenarios([
            {"scenario_id": "runtime_reset_replay", "steps": []},
            {"scenario_id": "happy_success", "kind": "success", "steps": [], "assertions": []},
            {"scenario_id": "bad_failure", "kind": "failure", "steps": [], "assertions": []},
            {"scenario_id": "noise_case", "kind": "noise", "steps": [], "assertions": []},
            {"scenario_id": "noise_case", "kind": "noise-selection", "steps": [], "assertions": []},
        ])
        self.assertEqual(
            [item["kind"] for item in scenarios],
            ["goal_success", "goal_failure", "noise_selection", "noise_selection"],
        )
        self.assertEqual(len({item["scenario_id"] for item in scenarios}), 4)

    def test_deterministic_business_scenario_baseline_is_complete(self):
        tools = [
            {"type": "function", "function": {"name": "query", "description": "查询", "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "weather", "description": "天气", "parameters": {"type": "object", "properties": {}, "required": []}}},
        ]
        scenarios = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "给出结论和依据"},
            tools=tools,
            noise_tools=[{"name": "weather", "category": "unrelated"}],
        )
        TaskGenerationPipeline._validate_executable_scenarios(
            scenarios, tools=tools, noise_tools=[{"name": "weather"}]
        )
        success = next(item for item in scenarios if item["kind"] == "goal_success")
        self.assertIn("agent_response", [step["operation"] for step in success["steps"]])

    def test_baseline_binds_business_argument_through_declared_selector(self):
        tools = [{"type": "function", "function": {
            "name": "lookup_product", "parameters": {"type": "object", "properties": {
                "product_id": {"type": "string"},
            }, "required": ["product_id"]},
        }}]
        scenarios = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "查到价格"}, tools=tools, noise_tools=[],
            data_tables=[{"table_name": "products", "rows": [{"field_abc": "P00001"}]}],
            tool_implementations=[{"tool_name": "lookup_product", "operation": "select",
                                   "table": "products", "selector": {"product_id": "field_abc"}}],
            training_category="simple_agentic",
        )
        success = next(item for item in scenarios if item["kind"] == "goal_success")
        call = next(step for step in success["steps"] if step["operation"] == "tool_call")
        self.assertEqual(call["arguments"], {"product_id": "P00001"})

    def test_deterministic_baseline_expands_mutations_for_each_goal_row(self):
        tools = [
            {"type": "function", "function": {
                "name": "query_notes", "description": "查询", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                },
            }},
            {"type": "function", "function": {
                "name": "update_note", "description": "更新", "parameters": {
                    "type": "object", "properties": {
                        "id": {"type": "string"}, "label": {"type": "string"},
                    }, "required": ["id", "label"],
                },
            }},
        ]
        implementations = [
            {"tool_name": "query_notes", "operation": "select", "table": "notes",
             "projection": ["id", "label"], "result_field": "records"},
            {"tool_name": "update_note", "operation": "update", "table": "notes",
             "selector": {"id": "id"}, "changes": {"label": "label"},
             "result_field": "records"},
        ]
        goal = {"row_predicates": [
            {"table": "notes", "where": {"id": "n1"}, "values": {"label": "a"}, "count": 1},
            {"table": "notes", "where": {"id": "n2"}, "values": {"label": "b"}, "count": 1},
        ]}
        scenarios = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "更新完成"}, tools=tools,
            noise_tools=[], tool_implementations=implementations,
            semantic_goal=goal, training_category="multi_step_agentic",
        )
        success = next(item for item in scenarios if item["kind"] == "goal_success")
        updates = [step for step in success["steps"] if step.get("tool_name") == "update_note"]
        self.assertEqual(len(updates), 2)
        self.assertEqual(updates[1]["arguments"], {"id": "n2", "label": "b"})
        TaskGenerationPipeline._validate_executable_scenarios(
            scenarios, tools=tools, noise_tools=[],
            training_category="multi_step_agentic",
            tool_implementations=implementations, semantic_goal=goal,
        )

    def test_business_scenario_fixture_uses_real_rows_and_nonempty_arrays(self):
        schema = {
            "type": "object",
            "properties": {
                "material_categories": {"type": "array", "items": {"type": "string"}},
                "reference_data": {"type": "string"},
            },
            "required": ["material_categories", "reference_data"],
        }
        tools = [{"type": "function", "function": {
            "name": "lookup", "description": "查询", "parameters": schema,
        }}]
        scenarios = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "完成比较"}, tools=tools,
            noise_tools=[], data_tables=[{
                "table_name": "material_category",
                "rows": [{"material_category": "玻璃材料"}, {"material_category": "透光材料"}],
            }],
        )
        success = next(item for item in scenarios if item["kind"] == "goal_success")
        arguments = next(
            step["arguments"] for step in success["steps"]
            if step["operation"] == "tool_call"
        )
        self.assertEqual(arguments["material_categories"], ["玻璃材料", "透光材料"])
        self.assertEqual(arguments["reference_data"], "material_category")
        TaskGenerationPipeline._validate_executable_scenarios(
            scenarios, tools=tools, noise_tools=[]
        )

    def test_business_scenario_fixture_maps_dishes_and_named_tables(self):
        tools = [{"type": "function", "function": {
            "name": "query_menu_status", "description": "查菜品", "parameters": {
                "type": "object", "properties": {
                    "target_dishes": {"type": "array", "items": {"type": "string"}},
                    "size_chart_table": {"type": "string"},
                }, "required": ["target_dishes", "size_chart_table"],
            },
        }}]
        scenarios = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "完成"}, tools=tools, noise_tools=[],
            data_tables=[
                {"table_name": "menu", "rows": [{"dish_name": "担担面"}]},
                {"table_name": "size_chart", "rows": [{"size": "M"}]},
            ],
        )
        arguments = next(
            step["arguments"] for step in scenarios[0]["steps"]
            if step.get("operation") == "tool_call"
        )
        self.assertEqual(arguments["target_dishes"], ["担担面"])
        self.assertEqual(arguments["size_chart_table"], "size_chart")

    def test_baseline_dependency_requires_projected_array_item_fields(self):
        properties = {"items": {"type": "array", "items": {
            "type": "object", "properties": {
                "item_name": {"type": "string"}, "quantity": {"type": "number"},
            }, "required": ["item_name", "quantity"],
        }}}
        self.assertIsNone(TaskGenerationPipeline._match_baseline_dependency(
            required=["items"], properties=properties,
            prior_outputs=[("records", ["item_name"])],
        ))
        self.assertEqual(TaskGenerationPipeline._match_baseline_dependency(
            required=["items"], properties=properties,
            prior_outputs=[("records", ["item_name", "quantity"])],
        ), ("items", "upstream_records", "$.records"))

    def test_relational_data_accepts_nullable_in_check(self):
        table = {
            "table_name": "purchase_request_item",
            "columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "is_matched", "type": "INTEGER", "nullable": True},
            ],
            "primary_key": ["id"], "foreign_keys": [], "indexes": [],
            "constraints": [{"type": "CHECK", "expression":
                "is_matched IN (0, 1) OR is_matched IS NULL"}],
            "rows": [{"id": 1, "is_matched": None}, {"id": 2, "is_matched": 1}],
        }
        TaskGenerationPipeline._validate_data_tables([table])
        TaskGenerationPipeline._validate_relational_data([table])

    def test_business_scenario_argument_repair_uses_validated_fixture_values(self):
        tools = [{"type": "function", "function": {
            "name": "lookup", "description": "查询", "parameters": {
                "type": "object", "properties": {
                    "summary_id": {"type": "string"},
                    "topics": {"type": "array", "items": {"type": "string"}},
                }, "required": ["summary_id", "topics"],
                "additionalProperties": False,
            },
        }}]
        baseline = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "完成查询"}, tools=tools,
            noise_tools=[], data_tables=[{
                "table_name": "summary",
                "rows": [{"summary_id": "summary-1", "topic": "出口统计"}],
            }],
        )
        repaired = TaskGenerationPipeline._repair_business_scenario_arguments(
            [{"kind": "goal_success", "steps": [{
                "operation": "tool_call", "tool_name": "lookup",
                "arguments": {"summary_id": "", "topics": []},
            }]}],
            baseline=baseline, tools=tools,
        )
        arguments = repaired[0]["steps"][0]["arguments"]
        self.assertEqual(arguments["summary_id"], "summary-1")
        self.assertTrue(arguments["topics"])

    def test_business_scenario_uses_public_materials_and_repairs_placeholders(self):
        tools = [{"type": "function", "function": {
            "name": "inspect_material", "description": "读取素材", "parameters": {
                "type": "object", "properties": {
                    "material_names": {"type": "array", "items": {"type": "string"}},
                    "corpus_text": {"type": "string"},
                    "selected_supplier": {"type": "string"},
                }, "required": ["material_names", "corpus_text", "selected_supplier"],
            },
        }}]
        task = {"expected_result": "完成", "public_input": {"materials": [{
            "name": "规格说明.txt", "mime_type": "text/plain", "content": "供应商甲符合规格。",
        }]}}
        baseline = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description=task, tools=tools, noise_tools=[],
            data_tables=[{"table_name": "suppliers", "rows": [{"supplier": "供应商甲"}]}],
        )
        arguments = next(
            step["arguments"] for step in baseline[0]["steps"]
            if step.get("operation") == "tool_call"
        )
        self.assertEqual(arguments, {
            "material_names": ["规格说明.txt"],
            "corpus_text": "供应商甲符合规格。",
            "selected_supplier": "供应商甲",
        })
        repaired = TaskGenerationPipeline._repair_business_scenario_arguments(
            [{"kind": "goal_success", "steps": [{"operation": "tool_call",
                "tool_name": "inspect_material", "arguments": {
                    "material_names": ["任务输入中的material_name"],
                    "corpus_text": "placeholder", "selected_supplier": "供应商甲",
                }}]}], baseline=baseline, tools=tools,
        )
        self.assertEqual(repaired[0]["steps"][0]["arguments"], arguments)

    def test_business_scenario_uses_complete_public_material_objects(self):
        materials = [{
            "name": "规格.txt", "mime_type": "text/plain", "content": "尺寸 M 可用。",
        }]
        tools = [{"type": "function", "function": {
            "name": "get_reference_data", "description": "读取资料", "parameters": {
                "type": "object", "properties": {"materials": {
                    "type": "array", "items": {"type": "object", "properties": {
                        "name": {"type": "string"}, "content": {"type": "string"},
                    }, "required": ["name", "content"]},
                }}, "required": ["materials"],
            },
        }}]
        scenarios = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"public_input": {"materials": materials}},
            tools=tools, noise_tools=[],
        )
        arguments = next(
            step["arguments"] for step in scenarios[0]["steps"]
            if step.get("operation") == "tool_call"
        )
        self.assertEqual(arguments["materials"], [{
            "name": "规格.txt", "content": "尺寸 M 可用。",
        }])

    def test_business_scenario_argument_repair_handles_nullable_numeric_schema(self):
        tools = [{"type": "function", "function": {
            "name": "score", "description": "评分", "parameters": {
                "type": "object", "properties": {
                    "value": {"type": ["number", "null"]},
                }, "required": ["value"],
            },
        }}]
        baseline = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "完成"}, tools=tools,
            noise_tools=[], data_tables=[{"table_name": "scores", "rows": [{"value": 3.5}]}],
        )
        repaired = TaskGenerationPipeline._repair_business_scenario_arguments(
            [{"kind": "goal_success", "steps": [{"operation": "tool_call",
                "tool_name": "score", "arguments": {"value": "invalid"}}]}],
            baseline=baseline, tools=tools,
        )
        self.assertEqual(repaired[0]["steps"][0]["arguments"]["value"], 3.5)

    def test_business_scenario_argument_repair_preserves_capture_reference(self):
        tools = [{"type": "function", "function": {
            "name": "update", "description": "更新", "parameters": {
                "type": "object", "properties": {
                    "record": {"type": "object", "properties": {
                        "id": {"type": "string"},
                    }, "required": ["id"]},
                }, "required": ["record"],
            },
        }}]
        baseline = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "完成更新"}, tools=tools,
            noise_tools=[],
        )
        repaired = TaskGenerationPipeline._repair_business_scenario_arguments(
            [{"kind": "goal_success", "steps": [{
                "operation": "tool_call", "tool_name": "update",
                "arguments": {"record": {"$ref": "selected_record"}},
            }]}],
            baseline=baseline, tools=tools,
        )
        self.assertEqual(
            repaired[0]["steps"][0]["arguments"]["record"],
            {"$ref": "selected_record"},
        )

    def test_multi_step_baseline_compiles_fixture_backed_dependency(self):
        tools = [
            {"type": "function", "function": {
                "name": "query_summary", "description": "查询摘要", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                },
            }},
            {"type": "function", "function": {
                "name": "update_summary", "description": "更新摘要", "parameters": {
                    "type": "object", "properties": {
                        "summary_id": {"type": "string", "description": "摘要标识"},
                    }, "required": ["summary_id"],
                },
            }},
        ]
        scenarios = TaskGenerationPipeline._build_business_scenario_baseline(
            task_description={"expected_result": "完成更新"}, tools=tools,
            noise_tools=[], training_category="multi_step_agentic",
            tool_implementations=[{
                "tool_name": "query_summary", "operation": "select",
                "result_field": "records", "projection": ["summary_id"],
            }],
        )
        success = next(item for item in scenarios if item["kind"] == "goal_success")
        calls = [step for step in success["steps"] if step["operation"] == "tool_call"]
        self.assertEqual(calls[0]["capture"], {"upstream_summary_id": "$.records[0].summary_id"})
        self.assertEqual(
            calls[1]["arguments"]["summary_id"], {"$ref": "upstream_summary_id"}
        )
        TaskGenerationPipeline._validate_executable_scenarios(
            scenarios, tools=tools, noise_tools=[],
            training_category="multi_step_agentic",
        )

    def test_goal_success_rejects_empty_required_array_and_placeholder(self):
        tools = [{"function": {"name": "lookup", "parameters": {
            "type": "object", "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "source": {"type": "string"},
            }, "required": ["items", "source"],
        }}}]
        def scenarios(arguments):
            return [
                {"scenario_id": "ok", "kind": "goal_success", "steps": [
                    {"operation": "tool_call", "tool_name": "lookup", "arguments": arguments},
                ], "assertions": [{"path": "$.reward", "operator": "gte", "expected": 0.8}]},
                {"scenario_id": "bad", "kind": "goal_failure", "steps": [
                    {"operation": "reward"},
                ], "assertions": [{"path": "$.reward", "operator": "lte", "expected": 0.2}]},
            ]
        with self.assertRaisesRegex(PipelineGenerationError, "non-empty array"):
            TaskGenerationPipeline._validate_executable_scenarios(
                scenarios({"items": [], "source": "catalog"}), tools=tools, noise_tools=[]
            )
        with self.assertRaisesRegex(PipelineGenerationError, "placeholder"):
            TaskGenerationPipeline._validate_executable_scenarios(
                scenarios({"items": ["fixture-value"], "source": "catalog"}),
                tools=tools, noise_tools=[],
            )

    def test_declarative_metric_implementation_validation(self):
        metrics = [{"id": "done", "type": "rule-based", "score_range": [0, 1]}]
        specs = [{
            "metric_id": "done", "source": "business_state", "path": "$.done",
            "operator": "eq", "expected": True,
            "score_mapping": {"pass": 1, "fail": 0},
        }]
        TaskGenerationPipeline._validate_metric_implementations(specs, metrics)
        with self.assertRaises(PipelineGenerationError):
            TaskGenerationPipeline._validate_metric_implementations(
                [dict(specs[0], operator="python_eval")], metrics
            )

    def test_process_metric_compiles_from_success_scenario(self):
        metrics = [{
            "id": "write", "category": "process", "type": "hybrid",
            "score_range": [0, 1], "target_action": "save_item",
            "evaluator": {"kind": "hybrid_tool_call"},
        }]
        scenarios = [{
            "kind": "goal_success", "steps": [
                {"operation": "tool_call", "tool_name": "find_item", "arguments": {},
                 "capture": {"item_id": "$.records[0].id"}},
                {"operation": "tool_call", "tool_name": "update_item",
                 "arguments": {"id": {"$ref": "item_id"}, "quantity": 5}},
            ],
        }]
        specs = TaskGenerationPipeline._compile_process_metric_implementations(
            metrics=metrics,
            metric_implementations=[],
            business_scenarios=scenarios,
            tool_bindings=[{"action_name": "save_item", "tool_name": "update_item"}],
        )
        self.assertEqual(specs[0]["operator"], "contains_tool_call")
        self.assertEqual(specs[0]["expected"]["arguments"]["id"], {"$ref": "item_id"})
        self.assertEqual(specs[0]["expected"]["captures"], [{
            "name": "item_id", "tool_name": "find_item", "path": "$.records[0].id",
        }])
        TaskGenerationPipeline._validate_metric_implementations(
            specs, metrics, require_process=True
        )

    def test_process_metric_requires_success_call(self):
        metrics = [{
            "id": "write", "category": "process", "type": "hybrid",
            "score_range": [0, 1], "target_action": "save_item",
            "evaluator": {"kind": "hybrid_tool_call"},
        }]
        with self.assertRaisesRegex(PipelineGenerationError, "no matching success tool call"):
            TaskGenerationPipeline._compile_process_metric_implementations(
                metrics=metrics, metric_implementations=[],
                business_scenarios=[{"kind": "goal_success", "steps": []}],
                tool_bindings=[{"action_name": "save_item", "tool_name": "update_item"}],
            )

    def test_compiled_process_metric_declares_runtime_rule(self):
        metrics = [{
            "id": "write", "category": "process", "type": "hybrid",
            "target_action": "save_item", "evaluation_inputs": ["tool_call"],
            "criteria": ["exact"],
            "evaluator": {"kind": "hybrid_tool_call", "source": "external_llm"},
        }]
        specs = [{
            "metric_id": "write", "source": "trajectory", "path": "$.events",
            "operator": "contains_tool_call",
            "expected": {"tool_name": "update_item", "arguments": {"id": 1}, "captures": []},
            "score_mapping": {"pass": 1, "fail": 0},
        }]
        TaskGenerationPipeline._normalize_compiled_process_metrics(metrics, specs)
        self.assertEqual(metrics[0]["type"], "rule-based")
        self.assertEqual(metrics[0]["evaluator"]["kind"], "trajectory_rule")
        self.assertEqual(metrics[0]["evaluator"]["source"], "runtime_rule")
        self.assertNotIn("evaluation_inputs", metrics[0])

    def test_raw_final_agent_response_cannot_use_synthetic_object_path(self):
        metrics = [{"id": "count", "type": "rule-based", "score_range": [0, 1]}]
        specs = [{
            "metric_id": "count", "source": "final_agent_response",
            "path": "$.tea_varieties", "operator": "count_gte", "expected": 5,
            "score_mapping": {"pass": 1, "fail": 0},
        }]
        with self.assertRaisesRegex(PipelineGenerationError, "raw final_agent_response"):
            TaskGenerationPipeline._validate_metric_implementations(specs, metrics)

    def test_business_state_cannot_contain_final_agent_response(self):
        metrics = [{"id": "consistent", "type": "rule-based", "score_range": [0, 1]}]
        specs = [{
            "metric_id": "consistent", "source": "business_state",
            "path": "$.final_agent_response.choice", "operator": "eq", "expected": "A",
            "score_mapping": {"pass": 1, "fail": 0},
        }]
        with self.assertRaisesRegex(PipelineGenerationError, "through business_state"):
            TaskGenerationPipeline._validate_metric_implementations(specs, metrics)

    def test_document_rules_are_promoted_before_dsl_compilation(self):
        metrics = [{
            "id": "count", "category": "outcome", "type": "rule-based",
            "scope": "terminal", "condition": "至少五种茶", "rubric": "至少五种茶",
            "evaluator": {
                "kind": "document_rule", "source": "runtime_rule",
                "assertion": "count_tea_varieties(final_agent_response) >= 5",
                "score_mapping": {"true": 1, "false": 0},
            },
            "score_range": [0, 1], "weight": 1,
        }]
        TaskGenerationPipeline._promote_unstructured_document_rules(metrics)
        self.assertEqual(metrics[0]["type"], "model-based")
        self.assertEqual(metrics[0]["evaluator"]["kind"], "external_llm_judge")
        self.assertNotIn("condition", metrics[0])

    def test_response_business_join_is_promoted_to_model_judge(self):
        metrics = [{
            "id": "consistent", "category": "outcome", "type": "rule-based",
            "condition": "回答与数据一致", "rubric": "回答与数据一致",
            "evaluator": {
                "kind": "business_state_rule", "source": "runtime_rule",
                "source_fields": ["final_agent_response", "business_data.items"],
                "assertion": "final_agent_response 中的选择存在于业务数据",
                "score_mapping": {"true": 1, "false": 0},
            },
        }]
        TaskGenerationPipeline._promote_mixed_response_business_rules(metrics)
        self.assertEqual(metrics[0]["type"], "model-based")
        self.assertEqual(
            metrics[0]["evaluation_inputs"],
            ["business_data", "final_agent_response"],
        )

    def test_observation_schema_is_canonical_even_when_model_returns_empty(self):
        schema = TaskGenerationPipeline._canonical_observation_schema({})
        self.assertEqual(schema["type"], "object")
        self.assertEqual(schema["properties"]["final_agent_response"]["type"], "string")
        self.assertEqual(schema["properties"]["public_observation"]["type"], "object")

    def test_noise_penalty_removes_duplicate_nested_noise_reference(self):
        metrics = [{
            "id": "penalty_irrelevant_tool_use", "category": "penalty",
            "condition": "avoid irrelevant tools", "rubric": "avoid tools",
            "evaluator": {
                "kind": "trajectory_rule",
                "assertion": "tool_call.name != 'weather'",
            },
        }]
        TaskGenerationPipeline._ensure_deterministic_noise_penalty(
            metrics, noise_names=["weather"]
        )
        self.assertEqual([item["id"] for item in metrics], ["penalty_noise_tool_usage"])

    def test_acceptance_penalty_cases_use_zero_for_success(self):
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description={"task": "test"},
            data_manifest={}, data_tables=[], actions=[], tools=[], key_steps=[],
            metrics=[{
                "id": "penalty", "category": "penalty",
                "evaluator": {"score_mapping": {"pass": 0, "fail": -1}},
            }],
            reward_formula={},
        )
        cases = {item["case_id"]: item for item in contract["reward_cases"]}
        self.assertEqual(cases["penalty.pass"]["expected_score"], 0)
        self.assertEqual(cases["penalty.fail"]["expected_score"], -1)

    def test_acceptance_initial_data_hash_matches_runtime_manifest_hash(self):
        tables = [{
            "table_name": "items",
            "columns": [{"name": "id", "type": "INTEGER"}],
            "rows": [{"id": 1}],
        }]
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description={"task": "test"}, data_manifest={}, data_tables=tables,
            actions=[], tools=[], key_steps=[], metrics=[], reward_formula={},
        )
        self.assertEqual(
            contract["fixtures"]["initial_data_hash"],
            sha256_json({"items": [{"id": 1}]}),
        )

    def test_acceptance_empty_data_hash_matches_stateless_runtime(self):
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description={"task": "test"}, data_manifest={}, data_tables=[],
            actions=[], tools=[], key_steps=[], metrics=[], reward_formula={},
        )
        self.assertEqual(contract["fixtures"]["initial_data_hash"], sha256_json({}))

    def test_executable_dependency_rejects_incompatible_projection(self):
        tools = [
            {"type": "function", "function": {
                "name": "read_rows", "parameters": {
                    "type": "object", "properties": {}, "required": [],
                },
            }},
            {"type": "function", "function": {
                "name": "consume_rows", "parameters": {
                    "type": "object",
                    "properties": {"rows": {
                        "type": "array", "items": {
                            "type": "object",
                            "properties": {"date": {"type": "string"}},
                            "required": ["date"],
                        },
                    }},
                    "required": ["rows"],
                },
            }},
        ]
        scenarios = [{
            "scenario_id": "success", "kind": "goal_success",
            "steps": [
                {"operation": "tool_call", "tool_name": "read_rows", "arguments": {},
                 "capture": {"rows": "$.records"}},
                {"operation": "tool_call", "tool_name": "consume_rows",
                 "arguments": {"rows": {"$ref": "rows"}}},
                {"operation": "agent_response", "content": "done"},
                {"operation": "reward"},
            ],
            "assertions": [{"path": "$.reward", "operator": "gte", "expected": 0.6}],
        }]
        implementations = [{
            "tool_name": "read_rows", "operation": "select", "table": "records",
            "result_field": "records", "projection": ["observation_date"],
        }]
        with self.assertRaisesRegex(PipelineGenerationError, "misses projected fields.*date"):
            TaskGenerationPipeline._validate_executable_scenarios(
                scenarios, tools=tools, noise_tools=[],
                training_category="multi_step_agentic",
                tool_implementations=implementations,
            )

    def test_acceptance_rejects_reversed_penalty_case(self):
        metrics = [{
            "id": "penalty", "category": "penalty",
            "evaluator": {"score_mapping": {"pass": 0, "fail": -1}},
        }]
        contract = TaskGenerationPipeline._build_acceptance_contract(
            task_description={"task": "test"}, data_manifest={}, data_tables=[],
            actions=[], tools=[], key_steps=[], metrics=metrics, reward_formula={},
        )
        contract["reward_cases"][0]["expected_score"] = -1
        with self.assertRaisesRegex(PipelineGenerationError, "reversed expected score"):
            TaskGenerationPipeline._validate_acceptance_contract(
                contract, actions=[], tools=[], metrics=metrics
            )

    def test_acceptance_contract_merge_preserves_required_scenario_shape(self):
        baseline = {
            "authority": "env_factory_outer_workflow",
            "scenarios": [{"scenario_id": "s1", "kind": "lifecycle", "steps": ["reset"]}],
            "tool_cases": [], "argument_probes": [], "executable_scenarios": [],
            "reward_cases": [], "fixtures": {}, "invariants": ["x"],
            "mutations": [], "mutation_tests": ["x"],
        }
        candidate = {"scenarios": [{"scenario_id": "s1", "kind": "", "steps": None}]}
        merged = TaskGenerationPipeline._merge_acceptance_contract(candidate, baseline)
        self.assertEqual(merged["scenarios"], baseline["scenarios"])

    def test_acceptance_merge_keeps_fixture_backed_tool_request(self):
        baseline = {"tool_cases": [{
            "case_id": "check.valid_shape", "tool_name": "check",
            "kind": "schema_and_business_smoke",
            "arguments_template": {"brand_name": "BrandA"},
            "expected": {"status_class": [200]},
        }], "argument_probes": [{
            "probe_id": "check.valid_shape.argument_sensitivity",
            "tool_name": "check", "arguments": {"brand_name": "BrandA"},
        }]}
        candidate = {"tool_cases": [{
            "case_id": "check.valid_shape",
            "arguments_template": {"brand_name": "任务输入中的brand_name"},
            "expected": {"status_class": [500]},
            "description": "业务校验",
        }], "argument_probes": [{"arguments": {"brand_name": "占位符"}}]}
        merged = TaskGenerationPipeline._merge_acceptance_contract(candidate, baseline)
        self.assertEqual(merged["tool_cases"][0]["arguments_template"], {"brand_name": "BrandA"})
        self.assertEqual(merged["tool_cases"][0]["expected"], {"status_class": [200]})
        self.assertEqual(merged["tool_cases"][0]["description"], "业务校验")
        self.assertEqual(merged["argument_probes"], baseline["argument_probes"])

    def test_metric_type_is_recovered_from_executable_evaluator(self):
        metrics = [{
            "id": "outcome",
            "type": "business-rule",
            "evaluator": {"kind": "business_state_rule"},
        }]
        normalized = TaskGenerationPipeline._normalize_metric_types(metrics)
        self.assertEqual(normalized[0]["type"], "rule-based")

    def test_relational_data_rejects_missing_foreign_rows(self):
        tables = [
            {"table_name": "users", "columns": [{"name": "id"}],
             "primary_key": ["id"], "rows": [{"id": 1}]},
            {"table_name": "orders", "columns": [{"name": "id"}, {"name": "user_id"}],
             "primary_key": ["id"], "foreign_keys": [{
                 "column": "user_id", "references_table": "users", "references_column": "id",
             }], "rows": [{"id": 1, "user_id": 2}]},
        ]
        with self.assertRaisesRegex(PipelineGenerationError, "missing foreign values"):
            TaskGenerationPipeline._validate_relational_data(tables)

    def test_relational_data_accepts_valid_foreign_rows(self):
        tables = [
            {"table_name": "users", "columns": [{"name": "id"}],
             "primary_key": ["id"], "rows": [{"id": 1}]},
            {"table_name": "orders", "columns": [{"name": "id"}, {"name": "user_id"}],
             "primary_key": ["id"], "foreign_keys": [{
                 "column": "user_id", "ref_table": "users", "ref_column": "id",
             }], "rows": [{"id": 1, "user_id": 1}]},
        ]
        TaskGenerationPipeline._validate_relational_data(tables)

    def test_relational_data_rejects_natural_language_constraint(self):
        tables = [{
            "table_name": "items",
            "columns": [{"name": "id"}],
            "primary_key": ["id"],
            "constraints": ["id 非空且唯一"],
            "rows": [{"id": 1}],
        }]
        with self.assertRaisesRegex(PipelineGenerationError, "unsupported CHECK"):
            TaskGenerationPipeline._validate_relational_data(tables)

    def test_table_definition_rejects_uncompilable_constraint_before_rows(self):
        tables = [{
            "table_name": "items",
            "columns": [{"name": "id"}, {"name": "name"}],
            "primary_key": ["id"],
            "constraints": ["name 唯一且非空"],
        }]
        with self.assertRaisesRegex(PipelineGenerationError, "unsupported CHECK"):
            TaskGenerationPipeline._validate_table_definitions(tables)

    def test_table_definition_requires_runtime_foreign_key_shape(self):
        tables = [
            {"table_name": "users", "columns": [{"name": "id"}], "primary_key": ["id"]},
            {
                "table_name": "orders",
                "columns": [{"name": "id"}, {"name": "user_id"}],
                "primary_key": ["id"],
                "foreign_keys": [{
                    "columns": ["user_id"], "references_table": "users",
                    "references_columns": ["id"],
                }],
            },
        ]
        with self.assertRaisesRegex(PipelineGenerationError, "invalid foreign key"):
            TaskGenerationPipeline._validate_table_definitions(tables)

    def test_relational_data_checks_constraint_against_rows(self):
        tables = [{
            "table_name": "items",
            "columns": [{"name": "quantity"}],
            "primary_key": ["quantity"],
            "constraints": ["quantity >= 0"],
            "rows": [{"quantity": -1}],
        }]
        with self.assertRaisesRegex(PipelineGenerationError, "violates CHECK"):
            TaskGenerationPipeline._validate_relational_data(tables)

    def test_rule_assertion_is_recovered_from_metric_condition(self):
        metrics = [{
            "id": "business_complete",
            "type": "rule-based",
            "condition": "completed_items >= requested_items",
            "evaluator": {
                "kind": "business_state_rule",
                "source": "runtime_rule",
                "score_mapping": {"true": 1, "false": 0},
            },
        }]
        normalized = TaskGenerationPipeline._normalize_metric_evaluators(metrics)
        self.assertEqual(
            normalized[0]["evaluator"]["assertion"],
            "completed_items >= requested_items",
        )

    def test_rule_condition_is_recovered_from_evaluator_assertion(self):
        metrics = [{
            "id": "business_complete",
            "type": "rule-based",
            "evaluator": {
                "kind": "business_state_rule",
                "source": "runtime_rule",
                "assertion": "completed == true",
                "score_mapping": {"true": 1, "false": 0},
            },
        }]
        normalized = TaskGenerationPipeline._normalize_metric_evaluators(metrics)
        self.assertEqual(normalized[0]["condition"], "completed == true")

    def test_prose_hybrid_outcome_becomes_an_explicit_semantic_judge(self):
        metrics = [{
            "id": "outcome", "type": "hybrid", "condition": "combined",
            "criteria": ["answer is clear"],
            "evaluator": {
                "kind": "hybrid_outcome", "source": "external_llm",
                "rule": {"kind": "business_state_rule", "assertion": "state satisfies target"},
                "external_llm": {"judge_criteria": "answer is grounded"},
                "score_mapping": {"rule_pass_and_llm_pass": 1, "rule_fail": 0},
            },
        }]
        metric = TaskGenerationPipeline._normalize_metric_evaluators(metrics)[0]
        self.assertEqual(metric["type"], "model-based")
        self.assertEqual(metric["evaluator"]["kind"], "external_llm_judge")
        self.assertNotIn("condition", metric)
        self.assertEqual(
            metric["criteria"],
            ["answer is clear", "state satisfies target", "answer is grounded"],
        )

    def test_observation_schema_includes_platform_envelope(self):
        schema = TaskGenerationPipeline._canonical_observation_schema({
            "type": "object",
            "properties": {
                "conversation": {"type": "object"},
                "tool_results": {"type": "object"},
                "custom_hint": {"type": "string"},
            },
            "required": ["conversation"],
            "additionalProperties": False,
        })
        self.assertIn("episode_id", schema["properties"])
        self.assertIn("episode_id", schema["required"])
        self.assertIn("final_agent_response", schema["required"])
        self.assertEqual(schema["properties"]["conversation"]["type"], "array")
        self.assertEqual(schema["properties"]["tool_results"]["type"], "array")
        self.assertEqual(schema["properties"]["custom_hint"]["type"], "string")
        self.assertFalse(schema["additionalProperties"])

    def test_dependency_projection_includes_nested_downstream_fields(self):
        implementations = [{
            "tool_name": "list_products", "operation": "select", "table": "products",
            "projection": ["id"], "filters": [], "order_by": [], "result_field": "records",
        }]
        tools = [{"type": "function", "function": {
            "name": "summarize_products", "parameters": {
                "type": "object", "properties": {"items": {
                    "type": "array", "items": {"type": "object", "properties": {
                        "product_name": {"type": "string"}, "origin": {"type": "string"},
                    }},
                }},
            },
        }}]
        tables = [{"table_name": "products", "columns": [
            {"name": "id"}, {"name": "product_name"}, {"name": "origin"}, {"name": "cost"},
        ]}]
        completed = TaskGenerationPipeline._complete_dependency_projections(
            implementations=implementations, tools=tools, tables=tables
        )
        self.assertEqual(completed[0]["projection"], ["id", "product_name", "origin"])

    def test_route_projection_audit_finds_only_uncovered_record_dependencies(self):
        route = {"environment_operations": [
            {"action_name": "read_inventory", "dependencies": []},
            {"action_name": "compare_items", "dependencies": ["read_inventory"]},
        ]}
        bindings = [
            {"action_name": "read_inventory", "tool_name": "read_inventory"},
            {"action_name": "compare_items", "tool_name": "compare_items"},
        ]
        tools = [{"function": {"name": "compare_items", "parameters": {
            "type": "object", "properties": {"items": {
                "type": "array", "items": {"type": "object", "properties": {
                    "item_name": {"type": "string"}, "quantity": {"type": "number"},
                }, "required": ["item_name", "quantity"]},
            }}, "required": ["items"],
        }}}]
        implementation = [{
            "tool_name": "read_inventory", "operation": "select", "table": "inventory",
            "projection": ["item_name"], "result_field": "records",
        }]
        gaps = TaskGenerationPipeline._tool_chain_projection_gaps(
            route_plan=route, tool_bindings=bindings,
            tools=tools, implementations=implementation,
        )
        self.assertEqual(gaps[0]["missing_fields"], ["quantity"])
        implementation[0]["projection_aliases"] = {"quantity": "on_hand_count"}
        self.assertEqual(TaskGenerationPipeline._tool_chain_projection_gaps(
            route_plan=route, tool_bindings=bindings,
            tools=tools, implementations=implementation,
        ), [])
        tools[0]["function"]["parameters"]["properties"]["record_id"] = {"type": "string"}
        tools[0]["function"]["parameters"]["required"].append("record_id")
        implementation[0]["projection_aliases"] = {}
        implementation[0]["projection"].append("record_id")
        self.assertEqual(TaskGenerationPipeline._tool_chain_projection_gaps(
            route_plan=route, tool_bindings=bindings,
            tools=tools, implementations=implementation,
        ), [])

    def test_compiled_chain_gate_rejects_strict_downstream_shape(self):
        spec = {"tool_contracts": [
            {"name": "read_rows", "output_contract": {"schema": {
                "type": "object", "properties": {"records": {"type": "array", "items": {
                    "type": "object", "properties": {"id": {}, "extra": {}},
                    "required": ["id", "extra"],
                }}},
            }}},
            {"name": "consume_rows", "input_schema": {
                "type": "object", "properties": {"rows": {"type": "array", "items": {
                    "type": "object", "properties": {"id": {}}, "required": ["id"],
                    "additionalProperties": False,
                }}}, "required": ["rows"],
            }},
        ]}
        scenarios = [{"kind": "goal_success", "scenario_id": "success", "steps": [
            {"operation": "tool_call", "tool_name": "read_rows",
             "capture": {"rows": "$.records"}},
            {"operation": "tool_call", "tool_name": "consume_rows",
             "arguments": {"rows": {"$ref": "rows"}}},
        ]}]
        with self.assertRaisesRegex(PipelineGenerationError, "required upstream fields"):
            TaskGenerationPipeline._validate_compiled_tool_chain(spec, scenarios)
        spec["tool_contracts"][1]["input_schema"]["properties"]["rows"]["items"]["properties"]["extra"] = {}
        TaskGenerationPipeline._validate_compiled_tool_chain(spec, scenarios)

    def test_compiled_chain_gate_rejects_missing_capture_path(self):
        spec = {"tool_contracts": [
            {"name": "read_budget", "output_contract": {"schema": {
                "type": "object", "properties": {"records": {"type": "array", "items": {
                    "type": "object", "properties": {"material_limit": {}},
                    "required": ["material_limit"],
                }}},
            }}},
            {"name": "plan_schedule", "input_schema": {
                "type": "object", "properties": {"max_capacity": {"type": "integer"}},
                "required": ["max_capacity"],
            }},
        ]}
        scenarios = [{"kind": "goal_success", "scenario_id": "success", "steps": [
            {"operation": "tool_call", "tool_name": "read_budget",
             "capture": {"capacity": "$.records[0].max_capacity"}},
            {"operation": "tool_call", "tool_name": "plan_schedule",
             "arguments": {"max_capacity": {"$ref": "capacity"}}},
        ]}]
        with self.assertRaisesRegex(PipelineGenerationError, "TASK_TOOL_CAPTURE_PATH"):
            TaskGenerationPipeline._validate_compiled_tool_chain(spec, scenarios)

    def test_reference_task_outcome_reward_requires_terminal_answer(self):
        metrics = [
            {"id": "outcome_records", "category": "outcome", "type": "rule-based",
             "scope": "trajectory", "weight": 0.8, "rubric": "找到履约原因",
             "evaluator": {"kind": "trajectory_rule", "source": "runtime_rule",
                           "assertion": "tool result exists", "score_mapping": {"pass": 1, "fail": 0}}},
            {"id": "process_lookup", "category": "process", "type": "rule-based", "weight": 0.2},
        ]
        specs = [
            {"metric_id": "outcome_records", "source": "trajectory", "path": "$.tool_results"},
            {"metric_id": "process_lookup", "source": "trajectory", "path": "$.events"},
        ]
        TaskGenerationPipeline._ensure_terminal_outcome_metrics(
            metrics, specs, environment_mode="reference_data",
        )
        self.assertEqual(metrics[0]["evaluator"]["kind"], "external_llm_judge")
        self.assertIn("final_agent_response", metrics[0]["evaluation_inputs"])
        self.assertEqual([spec["metric_id"] for spec in specs], ["process_lookup"])

    def test_existing_terminal_outcome_reward_is_preserved(self):
        metrics = [{"id": "answer", "category": "outcome", "type": "model-based",
                    "evaluation_inputs": ["final_agent_response"],
                    "evaluator": {"kind": "external_llm_judge"}}]
        specs = []
        TaskGenerationPipeline._ensure_terminal_outcome_metrics(
            metrics, specs, environment_mode="reference_data",
        )
        self.assertEqual(metrics[0]["evaluation_inputs"], ["final_agent_response"])

    def test_literal_csv_process_metric_requires_causal_downstream_capture(self):
        metrics = [
            {"id": "parse", "category": "process", "weight": 0.2},
            {"id": "calculate", "category": "process", "weight": 0.2},
            {"id": "answer", "category": "outcome", "weight": 0.6},
        ]
        raw = "item,count\n" + "\n".join(f"item-{n},{n}" for n in range(20))
        specs = [
            {"metric_id": "parse", "operator": "contains_tool_call", "expected": {
                "tool_name": "parse_csv", "arguments": {"raw_text": raw}, "captures": [],
            }},
            {"metric_id": "calculate", "operator": "contains_tool_call", "expected": {
                "tool_name": "calculate", "arguments": {"items": {"$ref": "parsed"}},
                "captures": [{"name": "parsed", "tool_name": "parse_csv", "path": "$.items"}],
            }},
        ]
        with self.assertRaisesRegex(PipelineGenerationError, "without downstream capture"):
            TaskGenerationPipeline._prune_literal_payload_process_metrics(
                copy.deepcopy(metrics), specs[:1],
            )
        TaskGenerationPipeline._prune_literal_payload_process_metrics(metrics, specs)
        self.assertEqual([item["id"] for item in metrics], ["calculate", "answer"])
        self.assertEqual([item["metric_id"] for item in specs], ["calculate"])
        self.assertAlmostEqual(sum(item["weight"] for item in metrics), 1.0)

    def test_select_projection_includes_label_for_identified_entity(self):
        implementations = [{
            "tool_name": "query_dish_status", "operation": "select", "table": "dish",
            "projection": ["dish_id", "is_available", "current_price", "store_id"],
            "result_field": "records",
        }]
        tables = [{"table_name": "dish", "primary_key": ["dish_id"], "columns": [
            {"name": "dish_id"}, {"name": "dish_name"},
            {"name": "is_available"}, {"name": "current_price"},
            {"name": "store_id"}, {"name": "internal_note"},
        ]}]
        completed = TaskGenerationPipeline._complete_dependency_projections(
            implementations=implementations, tools=[], tables=tables,
        )
        self.assertEqual(completed[0]["projection"], [
            "dish_id", "is_available", "current_price", "store_id", "dish_name",
        ])

    def test_dependency_projection_aliases_table_qualified_public_field(self):
        implementations = [{
            "tool_name": "read_products", "operation": "select", "table": "product",
            "projection": ["id"], "filters": [], "order_by": [], "result_field": "records",
        }]
        tools = [
            {"function": {"name": "read_products", "parameters": {
                "type": "object", "properties": {}, "required": [],
            }}},
            {"function": {"name": "consume_products", "parameters": {
                "type": "object", "properties": {"rows": {
                    "type": "array", "items": {"type": "object", "properties": {
                        "product_name": {"type": "string", "description": "name"},
                    }, "required": ["product_name"]},
                }}, "required": ["rows"],
            }}},
        ]
        tables = [{"table_name": "product", "columns": [
            {"name": "id", "type": "INTEGER"}, {"name": "name", "type": "TEXT"},
        ]}]
        completed = TaskGenerationPipeline._complete_dependency_projections(
            implementations=implementations, tools=tools, tables=tables,
        )
        self.assertEqual(completed[0]["projection_aliases"], {"product_name": "name"})
        TaskGenerationPipeline._validate_tool_implementations(
            completed, tools=tools, tables=tables,
        )

    def test_stateful_tool_surface_requires_goal_fields(self):
        goal = {"row_predicates": [{
            "table": "items", "where": {"id": "I-1"},
            "values": {"status": "approved"}, "count": 1,
        }]}
        incomplete = [{"type": "function", "function": {
            "name": "update_item", "description": "更新记录",
            "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
        }}]
        with self.assertRaisesRegex(PipelineGenerationError, "compilable mutation parameters"):
            TaskGenerationPipeline._validate_stateful_tool_surface(
                incomplete, semantic_goal=goal
            )
        complete = copy.deepcopy(incomplete)
        complete[0]["function"]["parameters"]["properties"]["status"] = {"type": "string"}
        TaskGenerationPipeline._validate_stateful_tool_surface(
            complete, semantic_goal=goal
        )

    def test_unsourced_governance_normalization_does_not_rewrite_medical_task(self):
        source = {"task": "检查服装出口合规性并参考法规要求", "context": "医疗诊断不可用"}
        normalized = TaskGenerationPipeline._normalize_unsourced_governance_task(source)
        self.assertIn("内部规则一致性", normalized["task"])
        self.assertIn("内部规则要求", normalized["task"])
        self.assertEqual(normalized["context"], "医疗诊断不可用")

    def test_action_numeric_grounding_ignores_identifier_suffixes(self):
        TaskGenerationPipeline._validate_action_grounding(
            [{
                "name": "lookup_stage_1", "description": "查询服装目录",
                "inputs": [], "outputs": [], "preconditions": [], "effects": [],
            }, {
                "name": "filter_stage_2", "description": "筛选服装记录",
                "inputs": [], "outputs": [], "preconditions": [], "effects": [],
            }],
            task_description={"task": "查询并筛选服装目录"},
            keywords=["服装"],
        )

    def test_metric_weights_are_normalized_by_sign_group(self):
        metrics = [
            {"category": "process", "weight": 2.0},
            {"category": "outcome", "weight": 3.0},
            {"category": "penalty", "weight": 4.0},
        ]
        normalized = TaskGenerationPipeline._normalize_metric_weights(metrics)
        self.assertAlmostEqual(normalized[0]["weight"], 0.4)
        self.assertAlmostEqual(normalized[1]["weight"], 0.6)
        self.assertAlmostEqual(normalized[2]["weight"], 1.0)

    def test_process_outcome_penalty_and_normalized_formula(self):
        metrics = [
            {
                "id": "process",
                "category": "process",
                "type": "hybrid",
                "scope": "step",
                "target_action": "query_recipe",
                "evaluation_inputs": ["recent_conversation", "tool_call", "tool_arguments"],
                "criteria": ["动作和参数符合上下文"],
                "condition": "llm_expected_tool_call_exact_match",
                "evaluator": {"kind": "hybrid_tool_call", "source": "external_llm", "comparison": "exact_tool_name_and_canonical_arguments", "score_mapping": {"match": 1, "mismatch": 0}},
                "score_range": [0, 1],
                "weight": 0.25,
                "rubric": "过程动作正确",
            },
            {
                "id": "outcome",
                "category": "outcome",
                "type": "rule-based",
                "scope": "terminal",
                "condition": "task_completed == true",
                "evaluator": {"kind": "business_state_rule", "source": "runtime_rule", "assertion": "task_completed == true", "score_mapping": {"true": 1, "false": 0}},
                "score_range": [0, 1],
                "weight": 0.75,
                "rubric": "任务完成",
            },
            {
                "id": "penalty",
                "category": "penalty",
                "type": "hybrid",
                "scope": "step",
                "condition": "irrelevant_output == true",
                "evaluation_inputs": ["recent_conversation", "agent_message"],
                "criteria": ["是否偏离用户诉求"],
                "evaluator": {"kind": "hybrid_outcome", "source": "external_llm", "rule": {"kind": "trajectory_rule", "assertion": "irrelevant_output == true"}, "external_llm": {"criteria": ["是否偏离用户诉求"]}, "score_mapping": {"violation": -1, "clear": 0}},
                "score_range": [-1, 0],
                "weight": 1.0,
                "rubric": "偏离用户诉求惩罚",
            },
        ]
        TaskGenerationPipeline._validate_metrics(metrics)
        TaskGenerationPipeline._validate_reward_formula(
            {"type": "separate_sign_weighted_sum", "formula": "R = clip(pos + neg, -1, 1)", "positive_weight_sum": 1, "negative_weight_sum": 1, "score_range": [-1, 1]},
            metrics,
        )


if __name__ == "__main__":
    unittest.main()
