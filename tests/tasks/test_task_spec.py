import unittest

from env_factory.tasks.task_spec import TaskSpecError, compile_task_spec


def tool(name, *, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": name,
            "parameters": {
                "type": "object",
                "properties": {item: {"type": "string"} for item in required},
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


class TaskSpecCompilerTest(unittest.TestCase):
    def test_select_output_schema_declares_projected_fields_and_aliases(self):
        spec = compile_task_spec(
            task_description={"task": "查询菜品状态。"},
            training_category="simple_agentic",
            environment_plan={"mode": "reference_data", "requires_persistence": False},
            data_manifest={"tables": []},
            data_tables=[{"table_name": "dish", "rows": [{"dish_id": "1", "dish_name": "面"}]}],
            tools=[tool("query_dish")], noise_tools=[],
            tool_bindings=[{"tool_name": "query_dish", "action_name": "查询"}],
            tool_implementations=[{
                "tool_name": "query_dish", "operation": "select", "table": "dish",
                "result_field": "dishes", "projection": ["dish_id", "dish_name"],
                "projection_aliases": {"name": "dish_name"},
            }],
            actions=[{"name": "查询"}], key_steps=[], metrics=[{"id": "goal"}],
            executable_scenarios=[],
        )
        items = spec["tool_contracts"][0]["output_contract"]["schema"]["properties"]["dishes"]["items"]
        self.assertEqual(items["required"], ["dish_id", "dish_name", "name"])

    def test_compiles_stateful_delta_and_single_mutation_archetype(self):
        spec = compile_task_spec(
            task_description={
                "task": "把商品场景从“景观建筑”更正为“服装”。",
                "task_intent": "modify",
                "goal": "更正场景",
                "expected_result": "场景为服装",
                "requirements": {},
            },
            training_category="simple_agentic",
            environment_plan={"mode": "stateful", "requires_persistence": True},
            data_manifest={"tables": []},
            data_tables=[{"table_name": "product", "rows": [{"scene": "景观建筑"}]}],
            tools=[tool("update_product", required=("scene",))],
            noise_tools=[],
            tool_bindings=[{"tool_name": "update_product", "action_name": "更新场景"}],
            tool_implementations=[{"tool_name": "update_product", "operation": "update", "table": "product"}],
            actions=[{"name": "更新场景", "preconditions": ["旧值存在"], "expected_effects": ["场景更新"]}],
            key_steps=[],
            metrics=[{"id": "goal"}],
            executable_scenarios=[{"scenario_id": "success", "kind": "goal_success", "steps": []}],
        )
        self.assertEqual(spec["environment_contract"]["archetype"], "single_mutation")
        self.assertEqual(
            spec["goal_contract"]["expected_delta"],
            [{"before": "景观建筑", "after": "服装"}],
        )
        self.assertEqual(spec["tool_contracts"][0]["execution"]["operation"], "update")
        self.assertEqual(
            spec["tool_contracts"][0]["output_contract"]["schema"]["required"],
            ["updated_count"],
        )

    def test_rejects_stateful_fixture_that_already_dropped_old_value(self):
        with self.assertRaisesRegex(TaskSpecError, "precondition"):
            compile_task_spec(
                task_description={"task": "把商品场景从“景观建筑”更正为“服装”。"},
                training_category="simple_agentic",
                environment_plan={"mode": "stateful", "requires_persistence": True},
                data_manifest={"tables": []},
                data_tables=[{"table_name": "product", "rows": [{"scene": "服装"}]}],
                tools=[tool("update_product")], noise_tools=[],
                tool_bindings=[{"tool_name": "update_product", "action_name": "更新"}],
                tool_implementations=[], actions=[{"name": "更新"}], metrics=[{"id": "goal"}],
                key_steps=[],
                executable_scenarios=[],
            )

    def test_multi_step_requires_captured_dependency_edge(self):
        common = dict(
            task_description={"task": "查询记录后更新记录。"},
            training_category="multi_step_agentic",
            environment_plan={"mode": "stateful", "requires_persistence": True},
            data_manifest={"tables": []},
            data_tables=[{"table_name": "records", "rows": [{"id": "1"}]}],
            tools=[tool("lookup"), tool("update", required=("id",))],
            noise_tools=[],
            tool_bindings=[
                {"tool_name": "lookup", "action_name": "查询"},
                {"tool_name": "update", "action_name": "更新"},
            ],
            tool_implementations=[],
            actions=[{"name": "查询"}, {"name": "更新"}], metrics=[{"id": "goal"}],
            key_steps=[],
        )
        with self.assertRaisesRegex(TaskSpecError, "dependency edge"):
            compile_task_spec(**common, executable_scenarios=[])
        spec = compile_task_spec(**common, executable_scenarios=[{
            "scenario_id": "success", "kind": "goal_success", "steps": [
                {"operation": "tool_call", "tool_name": "lookup", "arguments": {}, "capture": {"record_id": "$.records[0].id"}},
                {"operation": "tool_call", "tool_name": "update", "arguments": {"id": {"$ref": "record_id"}}},
            ],
        }])
        self.assertEqual(spec["capability_dag"]["edges"][0]["via"], "record_id")

    def test_compiles_capability_edge_from_key_step_dependencies(self):
        spec = compile_task_spec(
            task_description={"task": "查询后更新记录。"},
            training_category="multi_step_agentic",
            environment_plan={"mode": "stateful", "requires_persistence": True},
            data_manifest={"tables": []},
            data_tables=[{"table_name": "records", "rows": [{"id": "1"}]}],
            tools=[tool("lookup"), tool("update")], noise_tools=[],
            tool_bindings=[
                {"tool_name": "lookup", "action_name": "查询"},
                {"tool_name": "update", "action_name": "更新"},
            ],
            tool_implementations=[], actions=[{"name": "查询"}, {"name": "更新"}],
            key_steps=[
                {"step_id": "read", "action_name": "查询", "dependencies": []},
                {"step_id": "write", "action_name": "更新", "dependencies": ["read"]},
            ],
            metrics=[{"id": "goal"}], executable_scenarios=[],
        )
        self.assertEqual(spec["capability_dag"]["edges"], [{
            "from_tool": "lookup", "to_tool": "update",
            "via": "step_dependency:read",
        }])


if __name__ == "__main__":
    unittest.main()
