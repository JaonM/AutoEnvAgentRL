import copy
import json
from pathlib import Path
import tempfile
import unittest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.reward_bindings import validate_reward_observability
from env_factory.sandbox_runtime import (DeclarativeMetricEvaluator, DeclarativeToolCompiler,
    EpisodeStore, ManifestDataStore)
from .test_agent_authoring import fixture


class RewardBindingsTest(unittest.TestCase):
    def test_raw_reward_lookup_cannot_read_a_fact_hidden_from_tools(self):
        source, request = fixture("simple_agentic")
        source["tool_implementations"][0]["projection"] = ["id", "name"]
        # The reference has a private answer and would otherwise execute with
        # full reward even though no policy can obtain the temperature.
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "REWARD_FACT_UNOBSERVABLE"):
            compile_source(source, root=Path(directory), request=request)
        source["tool_implementations"][0]["projection"].append("temperature")
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(compile_source(source, root=Path(directory), request=request)["task_readiness"]["ready"])

    def test_alias_only_projection_does_not_expose_all_private_columns(self):
        source, _ = fixture("simple_agentic")
        impl = source["tool_implementations"][0]
        impl.update(projection=[], projection_aliases={"zone": "name"})
        with self.assertRaisesRegex(ValueError, "REWARD_FACT_UNOBSERVABLE"):
            validate_reward_observability(source)
        impl["projection_aliases"]["degrees"] = "temperature"
        validate_reward_observability(source)

    def test_private_aggregate_is_not_an_escape_from_fact_visibility(self):
        source, _ = fixture("simple_agentic")
        target = source["metric_implementations"][0]["expected"]["targets"][0]
        target["expression"] = {"aggregate": {"table": "unexposed", "field": "capacity", "where": {}, "op": "sum"}}
        with self.assertRaisesRegex(ValueError, "REWARD_FACT_UNOBSERVABLE"):
            validate_reward_observability(source)

    def test_same_public_goal_remains_correct_after_identifier_rebinding(self):
        source, request = fixture("multi_step_agentic")
        source["metric_implementations"][0]["expected"]["targets"][0]["expression"] = {
            "from_tool": {"name": "read_temperature", "field": "temperature"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = compile_source(source, root=root, request=request)
            store = EpisodeStore(root / "test.sqlite3")
            store.reset(episode_id="binding", seed=1)
            data = ManifestDataStore(result["data_manifest"], root / "data/business_data", store)
            data.reset()
            tools = DeclarativeToolCompiler(data).compile_all(result["tool_implementations"])
            rule = result["metric_implementations"][0]
            def run():
                identifier = tools["lookup_zone"]({"name": "青松库"})["records"][0]["id"]
                temperature = tools["read_temperature"]({"id": identifier})["records"][0]["temperature"]
                state = {"zones": data.table("zones")}
                reward = DeclarativeMetricEvaluator().evaluate(rule, {
                    "business_state": state, "final_agent_response": f"温度：{temperature}摄氏度。"})
                return identifier, temperature, reward
            self.assertEqual(run(), (821, 18, 1))
            rows = data.table("zones")
            rows[0]["name"], rows[1]["name"] = "青松库旧址", "青松库"
            data.replace_table("zones", rows)
            self.assertEqual(run(), (547, 23, 1))
            self.assertEqual(DeclarativeMetricEvaluator().evaluate(rule, {
                "business_state": {"zones": data.table("zones")},
                "final_agent_response": "温度：18摄氏度。"}), 0)
            self.assertIn("lookup_zone", result["generation_pipeline"]["reward_query_bindings"])

    def test_unrooted_capture_paths_preserve_dynamic_reward_binding(self):
        for path in ("records[0].id", "$.records[0].id"):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                source, request = fixture("multi_step_agentic")
                source["metric_implementations"][0]["expected"]["targets"][0]["expression"] = {
                    "from_tool": {"name": "read_temperature", "field": "temperature"}}
                for scenario in source["scenarios"]:
                    for step in scenario["steps"]:
                        if step.get("tool_name") == "lookup_zone":
                            step["capture"] = {"zone_id": path}
                result = compile_source(source, root=Path(directory), request=request)
                self.assertTrue(result["task_readiness"]["ready"])
                rule = result["metric_implementations"][0]
                # The query must follow current business identity, not the fixture ID.
                state = {"zones": [{"id": 999, "name": "青松库", "temperature": 31}]}
                evaluator = DeclarativeMetricEvaluator()
                for value, expected in [(31, 1), (18, 0)]:
                    self.assertEqual(evaluator.evaluate(rule, {
                        "business_state": state,
                        "final_agent_response": f"温度：{value}摄氏度。"}), expected)

    def test_private_fixture_id_is_rejected_but_public_named_lookup_is_valid(self):
        source, request = fixture("multi_step_agentic")
        source["metric_implementations"][0]["expected"]["targets"][0]["expression"] = {
            "lookup": {"table": "zones", "field": "temperature", "where": {"id": 821}}}
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "PRIVATE_REWARD_IDENTIFIER"):
            compile_source(source, root=Path(directory), request=request)
        source["description"]["public_input"]["initial_user_message"] += " 本批次有821件货物。"
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "PRIVATE_REWARD_IDENTIFIER"):
            compile_source(source, root=Path(directory), request=request)

    def test_range_filtered_tool_and_reward_select_the_same_current_record(self):
        source, request = fixture("simple_agentic")
        tool = source["tools"][0]["function"]
        tool["parameters"] = {"type": "object", "properties": {
            "minimum": {"type": "integer", "description": "最低温度"}},
            "required": ["minimum"], "additionalProperties": False}
        source["tool_implementations"][0]["filters"] = [{"argument": "minimum", "column": "temperature", "operator": "gte"}]
        steps = source["scenarios"][0]["steps"]
        steps[0]["arguments"] = {"minimum": 20}
        steps[-1]["content"] = "温度：23摄氏度。"
        source["metric_implementations"][0]["expected"]["targets"][0]["expression"] = {
            "from_tool": {"name": "read_temperature", "field": "temperature"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = compile_source(source, root=root, request=request)
            store = EpisodeStore(root / "test.sqlite3"); store.reset(episode_id="range", seed=1)
            data = ManifestDataStore(result["data_manifest"], root / "data/business_data", store); data.reset()
            tool = DeclarativeToolCompiler(data).compile_all(result["tool_implementations"])["read_temperature"]
            rows = data.table("zones")
            rows[0]["temperature"], rows[1]["temperature"] = 24, 19
            data.replace_table("zones", rows)
            self.assertEqual(tool({"minimum": 20})["records"], [{"temperature": 24}])
            rule = result["metric_implementations"][0]
            self.assertEqual(DeclarativeMetricEvaluator().evaluate(rule, {
                "business_state": {"zones": data.table("zones")}, "final_agent_response": "温度：24摄氏度。"}), 1)

    def test_tool_fact_cannot_use_an_unreturned_field(self):
        source, request = fixture("multi_step_agentic")
        source["metric_implementations"][0]["expected"]["targets"][0]["expression"] = {
            "from_tool": {"name": "lookup_zone", "field": "temperature"}}
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "not returned"):
            compile_source(source, root=Path(directory), request=request)

    def test_stateless_private_tables_cannot_hide_answer_evidence(self):
        source, request = fixture("direct_response")
        source["tables"] = fixture("simple_agentic")[0]["tables"]
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(ValueError, "DIRECT_PRIVATE_DATA"):
            compile_source(source, root=Path(directory), request=request)
