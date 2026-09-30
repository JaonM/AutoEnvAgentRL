"""Behavioral regressions for state, trace and reward consistency."""
import json
import tempfile
import unittest
import runpy
import types
import copy
import io
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from env_factory.sandbox_runtime import (
    EpisodeStore, ManifestDataStore, DeclarativeToolCompiler,
    ContractToolRegistry, ContractEvaluatorRuntime, ContractModelMetricEvaluator, ContractRewardGate, SandboxError,
)


class IntegrityTest(unittest.TestCase):
    def test_model_metric_offline_fixture_is_not_semantic_verification(self):
        contract = {"metrics": [{"id": "quality", "score_range": [0, 1], "evaluator": {
            "kind": "external_llm_judge", "score_mapping": {"pass": 1, "fail": 0}}}],
            "acceptance_contract": {"executable_scenarios": [{"kind": "goal_success", "steps": [
                {"operation": "agent_response", "content": "reference answer"}]}]}}
        evaluator = ContractModelMetricEvaluator(contract, self.store)
        with patch.dict("os.environ", {"SANDBOX_EVALUATOR_MOCK": "1"}):
            self.assertEqual(evaluator.evaluate_all({"final_agent_response": "reference answer"}, {}), {"quality": 1})
            self.assertEqual(evaluator.evaluate_all({"final_agent_response": "I completed everything"}, {}), {"quality": 0})
            self.assertEqual(evaluator.evaluate_all({}, {"quality": 0.5}), {})
        event = self.store.replay()["events"][-1]
        self.assertFalse(event["payload"]["semantic_verification"])

    def test_model_metric_live_uses_constrained_labels_and_cached_result(self):
        contract = {"metrics": [{"id": "quality", "score_range": [0, 1], "evaluator": {
            "kind": "external_llm_judge", "score_mapping": {"pass": 1, "fail": 0}}}]}
        evaluator = ContractModelMetricEvaluator(contract, self.store)
        with patch.dict("os.environ", {"SANDBOX_EVALUATOR_MOCK": "0"}), patch(
            "env_factory.sandbox_runtime.RuntimeLLMClient"
        ) as client:
            call = client.return_value.json_chat
            call.return_value = {"label": "pass"}
            for _ in range(2):
                self.assertEqual(evaluator.evaluate_all({"final_agent_response": "answer"}, {}), {"quality": 1})
            self.assertEqual(call.call_count, 1)
            self.assertEqual(call.call_args.kwargs["response_schema"]["properties"]["label"]["enum"], ["pass", "fail"])

    def test_empty_terminal_answer_is_scored_without_external_judge(self):
        contract = {"metrics": [{
            "id": "answer_quality", "category": "outcome", "scope": "terminal",
            "score_range": [0, 1],
            "evaluation_inputs": ["final_agent_response", "tool_results"],
            "evaluator": {"kind": "external_llm_judge",
                          "score_mapping": {"full": 1, "partial": 0.5, "missing": 0}},
        }]}
        evaluator = ContractModelMetricEvaluator(contract, self.store)
        with patch.dict("os.environ", {"SANDBOX_EVALUATOR_MOCK": "0"}), patch(
            "env_factory.sandbox_runtime.RuntimeLLMClient"
        ) as client:
            self.assertEqual(evaluator.evaluate_all({"final_agent_response": ""}, {}),
                             {"answer_quality": 0.0})
            self.assertEqual(evaluator.evaluate_all({"final_agent_response": "  "}, {}),
                             {"answer_quality": 0.0})
            client.assert_not_called()

    def test_model_metric_maps_declared_business_data_to_runtime_state(self):
        contract = {"metrics": [{
            "id": "quality", "score_range": [0, 1],
            "evaluation_inputs": ["business_data"],
            "evaluator": {
                "kind": "external_llm_judge",
                "score_mapping": {"pass": 1, "fail": 0},
            },
        }]}
        evaluator = ContractModelMetricEvaluator(contract, self.store)
        captured = {}

        def fake_chat(client, messages, **kwargs):
            captured["evidence"] = json.loads(messages[1]["content"])["runtime_evidence"]
            return {"label": "pass"}

        with patch.dict("os.environ", {"SANDBOX_EVALUATOR_MOCK": "0"}), patch(
            "env_factory.sandbox_runtime.RuntimeLLMClient.json_chat", new=fake_chat,
        ):
            self.assertEqual(evaluator.evaluate_all({
                "business_state": {"orders": [{"id": 1}]},
            }, {}), {"quality": 1})
        self.assertEqual(captured["evidence"], {
            "business_data": {"orders": [{"id": 1}]},
        })

    def test_reference_answer_with_stale_private_fact_loses_outcome_credit(self):
        contract = {
            "task_spec": {"environment_contract": {"mode": "reference_data", "initial_fixture": {
                "manifest": {"tables": [{"table_name": "archive", "primary_key": ["id"]}]}}}},
            "metrics": [{"id": "answer", "category": "outcome", "scope": "terminal",
                         "score_range": [0, 1],
                         "evaluation_inputs": ["final_agent_response", "tool_results", "business_data"],
                         "evaluator": {"kind": "external_llm_judge", "score_mapping": {"pass": 1, "fail": 0}}}],
        }
        evaluator = ContractModelMetricEvaluator(contract, self.store)
        original = {"archive": [{"id": 1, "updated_at": "2024-06-15"}]}
        changed = {"archive": [{"id": 1, "updated_at": "2025-07-20"}]}
        context = {
            "initial_business_state": original, "business_state": changed,
            "final_agent_response": "档案更新日期为 2024-06-15。",
            "trajectory": {"events": [{"event": "tool_call", "result": {
                "archive": {"id": 1, "updated_at": "2025-07-20"}}}]},
        }
        with patch.dict("os.environ", {"SANDBOX_EVALUATOR_MOCK": "0"}), patch(
            "env_factory.sandbox_runtime.RuntimeLLMClient.json_chat", return_value={"label": "pass"},
        ):
            self.assertEqual(evaluator.evaluate_all(context, {}), {"answer": 0})
            context["final_agent_response"] = "档案更新日期为 2025-07-20。"
            self.assertEqual(evaluator.evaluate_all(context, {}), {"answer": 1})
        self.assertEqual([event["event"] for event in self.store.replay()["events"]
                          if event["event"] == "evaluator_fact_mismatch"],
                         ["evaluator_fact_mismatch"])

    def test_model_metric_reads_declared_inputs_from_observation_envelope(self):
        contract = {"metrics": [{
            "id": "quality", "score_range": [0, 1],
            "evaluation_inputs": ["recent_conversation", "public_observation"],
            "evaluator": {
                "kind": "external_llm_judge",
                "score_mapping": {"pass": 1, "fail": 0},
            },
        }]}
        evaluator = ContractModelMetricEvaluator(contract, self.store)
        captured = {}

        def fake_chat(client, messages, **kwargs):
            captured["evidence"] = json.loads(messages[1]["content"])["runtime_evidence"]
            return {"label": "pass"}

        with patch.dict("os.environ", {"SANDBOX_EVALUATOR_MOCK": "0"}), patch(
            "env_factory.sandbox_runtime.RuntimeLLMClient.json_chat", new=fake_chat,
        ):
            evaluator.evaluate_all({"observation": {
                "conversation": [{"role": "user", "content": "hello"}],
                "public_observation": {"task_input": {"id": 1}},
            }}, {})
        self.assertEqual(captured["evidence"], {
            "recent_conversation": [{"role": "user", "content": "hello"}],
            "public_observation": {"task_input": {"id": 1}},
        })

    def test_hybrid_outcome_is_a_platform_model_metric(self):
        contract = {"metrics": [{"id": "quality", "score_range": [0, 1], "evaluator": {
            "kind": "hybrid_outcome",
            "source": "external_llm",
            "rule": {"kind": "business_state_rule", "assertion": "state matches"},
            "external_llm": {"judge_criteria": "answer is grounded"},
            "score_mapping": {
                "rule_pass_and_llm_pass": 1,
                "rule_pass_llm_fail": 0.5,
                "rule_fail": 0,
            },
        }}], "acceptance_contract": {"executable_scenarios": [{
            "kind": "goal_success", "steps": [
                {"operation": "agent_response", "content": "grounded answer"},
            ],
        }]}}
        evaluator = ContractModelMetricEvaluator(contract, self.store)
        with patch.dict("os.environ", {"SANDBOX_EVALUATOR_MOCK": "1"}):
            self.assertEqual(
                evaluator.evaluate_all({"final_agent_response": "grounded answer"}, {}),
                {"quality": 1},
            )
            self.assertEqual(
                evaluator.evaluate_all({"final_agent_response": "wrong"}, {}),
                {"quality": 0},
            )

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        schema = {"columns": [{"name": "id", "type": "integer"},
                              {"name": "quantity", "type": "integer"}], "primary_key": ["id"]}
        (root / "schema.json").write_text(json.dumps(schema))
        (root / "rows.jsonl").write_text('\n'.join(json.dumps(row) for row in [
            {"id": 1, "quantity": 2}, {"id": 2, "quantity": 10}]))
        self.store = EpisodeStore(root / "state.sqlite3")
        self.store.reset(episode_id="A", seed=1)
        self.data = ManifestDataStore({"tables": [{"table_name": "items", "schema_file": "schema.json",
                                                   "rows_file": "rows.jsonl"}]}, root, self.store)
        self.data.reset()

    def test_failed_tool_rolls_back_state_trace_and_idempotency(self):
        def handler(arguments):
            self.data.update("items", {"id": 1}, {"quantity": 99})
            raise ValueError("interrupted")
        registry = ContractToolRegistry([{"type": "function", "function": {
            "name": "write", "parameters": {"type": "object"}}}], {"write": handler},
            event_recorder=self.store.event)
        for key in (None, "request-1"):
            with self.assertRaises(ValueError):
                registry.execute("write", {}, idem_key=key)
            self.assertEqual(self.data.table("items")[0]["quantity"], 2)
            self.assertEqual(self.store.replay()["events"], [])
        registry.handlers["write"] = lambda args: {"ok": True}
        self.assertEqual(registry.execute("write", {}, idem_key="request-1"), {"ok": True})

    def test_schema_survives_empty_table_and_rejects_bad_writes(self):
        for row in ({"id": 1, "quantity": 3}, {"id": 3, "quantity": "bad"}):
            with self.assertRaises(SandboxError):
                self.data.insert("items", row)
        self.data.delete("items", {})
        self.data.insert("items", {"id": 3, "quantity": 4})
        self.assertEqual(self.data.table("items"), [{"id": 3, "quantity": 4}])

    def test_numeric_sort(self):
        handler = DeclarativeToolCompiler(self.data).compile({
            "operation": "select", "table": "items", "result_field": "records", "order_by": ["quantity"]})
        self.assertEqual([row["quantity"] for row in handler({})["records"]], [2, 10])

    def test_tool_trace_records_private_business_reads(self):
        tools = [{"type": "function", "function": {
            "name": "lookup", "parameters": {"type": "object"},
        }}]
        registry = ContractToolRegistry(
            tools, {"lookup": lambda args: {"records": self.data.select("items", id=1)}},
            event_recorder=self.store.event,
        )
        registry.execute("lookup", {})
        event = self.store.replay()["events"][-1]
        self.assertEqual(event["payload"]["business_data_reads"], ["items"])

        registry.handlers["lookup"] = lambda args: {"records": [{"id": 1}]}
        registry.execute("lookup", {})
        event = self.store.replay()["events"][-1]
        self.assertEqual(event["payload"]["business_data_reads"], [])

    def test_explicit_episode_context_isolates_interleaved_work(self):
        self.store.reset(episode_id="B", seed=2)
        self.data.reset()
        with self.store.episode_context("A"):
            self.data.update("items", {"id": 1}, {"quantity": 99})
        with self.store.episode_context("B"):
            self.assertEqual(self.data.table("items")[0]["quantity"], 2)
        with self.store.episode_context("A"):
            self.assertEqual(self.data.table("items")[0]["quantity"], 99)

    def test_reset_selects_an_older_episode_as_current(self):
        self.store.reset(episode_id="B", seed=2)
        self.assertEqual(self.store.current().episode_id, "B")
        self.store.reset(episode_id="A", seed=3)
        self.assertEqual(self.store.current().episode_id, "A")

    def test_separate_store_instances_do_not_lose_updates(self):
        def increment(_):
            store = EpisodeStore(Path(self.tmp.name) / "state.sqlite3")
            data = copy.copy(self.data)
            data.episode_store = store
            with store.episode_context("A"), store.transaction():
                quantity = data.table("items")[0]["quantity"]
                data.update("items", {"id": 1}, {"quantity": quantity + 1})
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(increment, range(12)))
        self.assertEqual(self.data.table("items")[0]["quantity"], 14)

    def test_foreign_keys_are_checked_on_mutations(self):
        self.data.schemas["child"] = {"columns": [{"name": "id", "type": "integer"}, {"name": "parent", "type": "integer"}],
            "primary_key": ["id"], "foreign_keys": [{"column": "parent", "ref_table": "items", "ref_column": "id"}]}
        self.data.baseline["child"] = [{"id": 10, "parent": 1}]
        self.data.reset()
        with self.assertRaises(SandboxError):
            self.data.insert("child", {"id": 11, "parent": 999})
        with self.assertRaises(SandboxError):
            self.data.delete("items", {"id": 1})
        self.assertEqual(len(self.data.table("items")), 2)

    def test_runtime_llm_schema_failure_gets_feedback_retry(self):
        from env_factory.runtime_llm import RuntimeLLMClient, RuntimeLLMConfig
        replies = [io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(value)}}]}).encode())
                   for value in ({"ok": "yes"}, {"ok": True})]
        with patch("env_factory.runtime_llm.urlopen", side_effect=replies) as request:
            client = RuntimeLLMClient(RuntimeLLMConfig("test", "http://unused", "test", max_retries=1))
            result = client.json_chat([{"role": "user", "content": "test"}], response_schema={
                "type": "object", "required": ["ok"], "properties": {"ok": {"type": "boolean"}}})
            self.assertEqual(result, {"ok": True})
            corrected_request = json.loads(request.call_args_list[1].args[0].data)
            self.assertIn("failed validation", corrected_request["messages"][-1]["content"])

    def test_runtime_llm_trace_records_actual_provider_model_without_raw_id(self):
        from env_factory.runtime_llm import (
            RuntimeLLMClient, RuntimeLLMConfig, capture_runtime_llm_trace,
            summarize_runtime_llm_trace,
        )
        payload = {
            "id": "provider-secret-response-id",
            "model": "actual-runtime-snapshot",
            "usage": {"total_tokens": 7},
            "choices": [{"message": {"content": '{"ok":true}'}}],
        }
        with patch(
            "env_factory.runtime_llm.urlopen",
            return_value=io.BytesIO(json.dumps(payload).encode()),
        ):
            client = RuntimeLLMClient(
                RuntimeLLMConfig("test", "http://unused", "configured-alias")
            )
            with capture_runtime_llm_trace() as trace:
                client.json_chat(
                    [{"role": "user", "content": "test"}],
                    response_schema={
                        "type": "object", "required": ["ok"],
                        "properties": {"ok": {"type": "boolean"}},
                    },
                )
        summary = summarize_runtime_llm_trace(trace)
        self.assertEqual(summary["models"], {"actual-runtime-snapshot": 1})
        self.assertEqual(summary["usage"], {"total_tokens": 7})
        self.assertNotIn("provider-secret-response-id", str(summary))

    def test_runtime_llm_semantic_failure_gets_feedback_retry(self):
        from env_factory.runtime_llm import RuntimeLLMClient, RuntimeLLMConfig, RuntimeLLMError
        replies = [io.BytesIO(json.dumps({"choices": [{"message": {"content": json.dumps(value)}}]}).encode())
                   for value in ({"outcome": "wrong"}, {"outcome": "accepted"})]

        def semantic(value):
            if value["outcome"] != "accepted":
                raise RuntimeLLMError("outcome does not match transition")

        with patch("env_factory.runtime_llm.urlopen", side_effect=replies) as request:
            client = RuntimeLLMClient(RuntimeLLMConfig("test", "http://unused", "test", max_retries=1))
            result = client.json_chat(
                [{"role": "user", "content": "test"}],
                response_schema={
                    "type": "object", "required": ["outcome"],
                    "properties": {"outcome": {"type": "string"}},
                },
                semantic_validator=semantic,
            )
            self.assertEqual(result, {"outcome": "accepted"})
            corrected_request = json.loads(request.call_args_list[1].args[0].data)
            self.assertIn("outcome does not match transition", corrected_request["messages"][-1]["content"])

    def test_acceptance_compiler_preserves_calls_but_owns_assertions(self):
        from env_factory.task_pipeline import TaskGenerationPipeline
        baseline = [{"kind": "goal_success", "steps": [{"operation": "reset"},
                    {"operation": "agent_response", "content": "verified answer"}, {"operation": "reward"}],
                    "assertions": [{"path": "$.reward", "operator": "gte", "expected": 0.6}]},
                    {"kind": "goal_failure", "steps": [{"operation": "reset"}, {"operation": "reward"}], "assertions": []}]
        proposal = [{"kind": "goal_success", "steps": [{"operation": "mutate_business_state", "mutation": {}},
                    {"operation": "tool_call", "tool_name": "read", "arguments": {}, "capture": {"id": "$.records[0].id"}},
                    {"operation": "tool_call", "tool_name": "write", "arguments": {"id": {"$ref": "id"}}}],
                    "assertions": [{"operator": "equals", "expected": "anything"}]}]
        result = TaskGenerationPipeline._compile_business_scenario_structure(proposal, baseline)
        self.assertEqual(result[0]["assertions"], baseline[0]["assertions"])
        self.assertEqual(result[0]["steps"][2]["arguments"], {"id": {"$ref": "id"}})
        self.assertNotIn("mutate_business_state", [step["operation"] for step in result[0]["steps"]])
        self.assertEqual(result[0]["steps"][-2]["content"], "verified answer")

    def test_evaluator_cache_uses_real_replay_shape(self):
        evaluator = ContractEvaluatorRuntime(self.store)
        with patch("env_factory.sandbox_runtime.RuntimeLLMClient.json_chat",
                   return_value={"score": 1}) as judge:
            for _ in range(2):
                evaluator.json_judge({"id": "goal"}, {"trajectory": self.store.replay()}, fallback={"score": 0})
        self.assertEqual(judge.call_count, 1)
        self.assertEqual(len(self.store.replay()["events"]), 1)

    def test_evaluator_cache_invalidates_when_rubric_or_criteria_change(self):
        evaluator = ContractEvaluatorRuntime(self.store)
        metric = {"id": "goal", "rubric": "Explain the distinction", "criteria": ["Mention different roles"]}
        context = {"final_agent_response": "These have different roles."}
        with patch("env_factory.sandbox_runtime.RuntimeLLMClient.json_chat",
                   side_effect=[{"label": "pass"}, {"label": "fail"}, {"label": "pass"}]) as judge:
            first = evaluator.json_judge(metric, context, fallback={"label": "fail"})
            self.assertEqual(first["label"], "pass")
            self.assertEqual(evaluator.json_judge(metric, context, fallback={"label": "fail"}), first)
            metric["criteria"] = ["Also identify the preparation method"]
            self.assertEqual(evaluator.json_judge(metric, context, fallback={"label": "fail"})["label"], "fail")
            metric["rubric"] = "Assess only the stated role distinction"
            self.assertEqual(evaluator.json_judge(metric, context, fallback={"label": "fail"})["label"], "pass")
            self.assertEqual(judge.call_count, 3)

    def test_later_recovery_can_establish_dependency(self):
        gate = ContractRewardGate({"training_contract": {"category": "multi_step_agentic"},
            "capability_dag": {"nodes": ["read", "write"], "edges": [{"from_tool": "read", "to_tool": "write"}]}},
            [{"id": "goal", "category": "outcome", "score_range": [0, 1]}])
        events = [{"event": "tool_call", "payload": {"tool_name": name}} for name in ["write", "read", "write"]]
        self.assertEqual(gate.apply({"goal": 1}, {"trajectory": {"events": events}})["goal"], 1)

    def test_dependency_requires_actual_output_consumption(self):
        gate = ContractRewardGate({"training_contract": {"category": "multi_step_agentic"},
            "capability_dag": {"nodes": ["read", "write"], "edges": [{"from_tool": "read", "to_tool": "write",
                              "result_path": "$.records[0].id", "argument_path": "$.id"}]}},
            [{"id": "goal", "category": "outcome", "score_range": [0, 1]}])
        events = [{"event": "tool_call", "payload": {"tool_name": "read"}, "result": {"records": [{"id": 1}]}},
                  {"event": "tool_call", "payload": {"tool_name": "write", "arguments": {"id": 2}}}]
        self.assertEqual(gate.apply({"goal": 1}, {"trajectory": {"events": events}})["goal"], 0)
        events[-1]["payload"]["arguments"]["id"] = 1
        self.assertEqual(gate.apply({"goal": 1}, {"trajectory": {"events": events}})["goal"], 1)

    def test_nested_dependency_requires_output_consumption_and_complete_paths(self):
        task_spec = {"training_contract": {"category": "multi_step_agentic"},
            "capability_dag": {"nodes": ["read", "write"], "edges": [{
                "from_tool": "read", "to_tool": "write",
                "result_path": "$.records[0].id", "argument_path": "$.changes[0].id",
            }]}}
        gate = ContractRewardGate(task_spec, [
            {"id": "goal", "category": "outcome", "score_range": [0, 1]},
        ])
        events = [
            {"event": "tool_call", "payload": {"tool_name": "read"},
             "result": {"records": [{"id": 1}]}},
            {"event": "tool_call", "payload": {"tool_name": "write",
             "arguments": {"changes": [{"id": 2}]}}},
        ]
        context = {"trajectory": {"events": events}}
        self.assertEqual(gate.apply({"goal": 1}, context)["goal"], 0)
        events[1]["payload"]["arguments"]["changes"][0]["id"] = 1
        self.assertEqual(gate.apply({"goal": 1}, context)["goal"], 1)
        del task_spec["capability_dag"]["edges"][0]["argument_path"]
        self.assertEqual(gate.apply({"goal": 1}, context)["goal"], 0)

    def test_state_goal_cannot_be_satisfied_by_call_coverage(self):
        gate = ContractRewardGate({
            "training_contract": {"category": "simple_agentic"},
            "capability_dag": {"nodes": ["write"]},
            "goal_contract": {"requires_state_change": True, "row_predicates": [
                {"table": "items", "where": {"id": 1}, "values": {"quantity": 5}, "count": 1}]}},
            [{"id": "goal", "category": "outcome", "score_range": [0, 1]}])
        context = {"initial_business_state": {"items": [{"id": 1, "quantity": 2}]},
                   "business_state": {"items": [{"id": 1, "quantity": 2}]},
                   "trajectory": {"events": [{"event": "tool_call", "payload": {"tool_name": "write"}}]}}
        self.assertEqual(gate.apply({"goal": 1}, context)["goal"], 0)
        context["business_state"]["items"][0]["quantity"] = 5
        self.assertEqual(gate.apply({"goal": 1}, context)["goal"], 1)
        context["initial_business_state"] = context["business_state"]
        self.assertEqual(gate.apply({"goal": 1}, context)["goal"], 0)

    def test_output_contract_failure_rolls_back_write(self):
        def bad(args):
            self.data.update("items", {"id": 1}, {"quantity": 99})
            return {"updated_count": "incorrect"}
        registry = ContractToolRegistry([{"type": "function", "function": {
            "name": "write", "parameters": {"type": "object"}}}], {"write": bad},
            event_recorder=self.store.event, tool_contracts=[{"name": "write", "output_contract": {"schema": {
                "type": "object", "properties": {"updated_count": {"type": "integer"}}, "required": ["updated_count"]}}}])
        with self.assertRaises(SandboxError):
            registry.execute("write", {})
        self.assertEqual(self.data.table("items")[0]["quantity"], 2)

    def test_noise_returns_filtered_business_shaped_data(self):
        registry = ContractToolRegistry([{"type": "function", "function": {
            "name": "lookup_venue", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}], {},
            noise_tools=[{"name": "lookup_venue", "category": "unrelated", "parameter_columns": {"city": "city"},
                          "records": [{"city": "A", "capacity": 10}, {"city": "B", "capacity": 20}]}])
        self.assertEqual(registry.execute("lookup_venue", {"city": "B"}), {"records": [{"city": "B", "capacity": 20}], "count": 1})

    def test_noise_behavior_is_driven_by_fixture_not_function_name(self):
        for name in ("lookup_dice_history", "roll_virtual_die", "roll_virtual_die_2"):
            registry = ContractToolRegistry([{"type": "function", "function": {
                "name": name, "parameters": {"type": "object", "properties": {"id": {"type": "integer"}}}}}], {},
                noise_tools=[{"name": name, "parameter_columns": {"id": "id"},
                              "records": [{"id": 1, "value": 4}, {"id": 2, "value": 5}]}])
            self.assertEqual(registry.execute(name, {"id": 2}), {"records": [{"id": 2, "value": 5}], "count": 1})

    def test_generated_scaffold_executes_persistent_stateful_episode(self):
        import env_factory.sandbox_runtime as runtime
        root = Path(self.tmp.name)
        scaffold = runpy.run_path(str(Path(__file__).parents[2] / "scripts/sandbox/generate_sandbox_scaffold.py"))
        tool = {"type": "function", "function": {"name": "update_item", "parameters": {
            "type": "object", "properties": {"id": {"type": "integer"}, "quantity": {"type": "integer"}},
            "required": ["id", "quantity"], "additionalProperties": False}}}
        contract = {
            "tools": [tool], "training_category": "simple_agentic",
            "training_contract": {"category": "simple_agentic", "sandbox_profile": "single_tool"},
            "artifacts": {"data_manifest": {"root": ".", "tables": [{"table_name": "items", "schema_file": "schema.json", "rows_file": "rows.jsonl"}]},
                          "user_simulation_manifest": {"root": ".", "profiles_file": "profiles.json", "scripts_file": "scripts.json"}},
            "tool_implementations": [{"tool_name": "update_item", "operation": "update", "table": "items", "selector": {"id": "id"}, "changes": {"quantity": "quantity"}}],
            "metrics": [{"id": "goal", "category": "outcome", "weight": 1, "score_range": [0, 1]}],
            "metric_implementations": [{"metric_id": "goal", "source": "business_state", "path": "$.items[0].quantity", "operator": "eq", "expected": 5, "score_mapping": {"pass": 1, "fail": 0}}],
            "task_spec": {"training_contract": {"category": "simple_agentic"}, "capability_dag": {"nodes": ["update_item"]},
                          "goal_contract": {"requires_state_change": True, "row_predicates": [{"table": "items", "where": {"id": 1}, "values": {"quantity": 5}, "count": 1}]}},
        }
        (root / "BUILD_CONTRACT.json").write_text(json.dumps(contract))
        (root / "profiles.json").write_text('[{"profile_id":"p"}]')
        (root / "scripts.json").write_text('[{"script_id":"s","initial_state":"start","states":[{"state_id":"start"}],"transitions":[]}]')
        scaffold["generate"](root)
        hooks = types.ModuleType("task_impl")
        with patch.dict("sys.modules", {"sandbox_runtime": runtime, "task_impl": hooks}), patch.dict("os.environ", {"SANDBOX_TRAINER_API_KEY": "test-key"}):
            exec(scaffold["TASK_IMPL_SOURCE"], hooks.__dict__)
            namespace = {"__name__": "compiled_test", "__file__": str(root / "app.py")}
            exec(scaffold["APP_SOURCE"], namespace)
            app = namespace["create_app"](db_path=root / "integration.sqlite3")
            auth = {"Authorization": "Bearer test-key", "X-Episode-ID": "integration"}
            self.assertEqual(app.handle("POST", "/v1/reset", {"episode_id": "integration", "seed": 1}, auth)[0], 200)
            self.assertEqual(app.handle("GET", "/v1/reward", headers=auth)[1]["reward"], 0)
            self.assertEqual(app.handle("POST", "/v1/tools/update_item", {"id": 1, "quantity": 5}, auth)[0], 200)
            self.assertEqual(app.handle("GET", "/v1/reward", headers=auth)[1]["reward"], 1)
            reopened = namespace["create_app"](db_path=root / "integration.sqlite3")
            self.assertEqual(reopened.handle("GET", "/v1/reward", headers=auth)[1]["reward"], 1)
            reopened.handle("POST", "/v1/reset", {"episode_id": "integration", "seed": 1}, auth)
            self.assertEqual(reopened.handle("GET", "/v1/reward", headers=auth)[1]["reward"], 0)
