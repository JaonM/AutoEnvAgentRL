import copy
import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from env_factory.sandbox_runtime import (
    ContractRewardAggregator,
    ContractRewardGate,
    ContractEvaluatorRuntime,
    ContractModelMetricEvaluator,
    ContractToolRegistry,
    ContractUserSimulator,
    DeclarativeToolCompiler,
    DeclarativeMetricEvaluator,
    AcceptanceScenarioRunner,
    EpisodeStore,
    ExternalCapabilityClient,
    DataManifestValidator,
    SandboxError,
    SandboxApplication,
    ManifestDataStore,
    require_trainer,
    validate_json_schema,
)


class SandboxRuntimeTest(unittest.TestCase):
    @staticmethod
    def _wsgi_request(app, path, raw=b"{broken", authorization=None):
        environ = {
            "REQUEST_METHOD": "POST", "PATH_INFO": path,
            "CONTENT_LENGTH": str(len(raw)), "wsgi.input": BytesIO(raw),
        }
        if authorization:
            environ["HTTP_AUTHORIZATION"] = authorization
        captured = {}
        def start_response(status, headers):
            captured.update(status=status, headers=dict(headers))
        body = b"".join(app(environ, start_response))
        captured["body"] = json.loads(body)
        return captured

    def test_metric_paths_support_literal_row_filters(self):
        rows = {"entries": [{"name": "固有词", "definition": "new"},
                            {"name": "other", "definition": "old"}]}
        resolve = DeclarativeMetricEvaluator._resolve
        self.assertEqual(resolve(rows, "$.entries[?(@.name=='固有词')].definition"), "new")
        self.assertEqual(resolve(rows, "$.entries[0].definition"), "new")
        self.assertEqual(resolve(rows, "$.entries[?(@.name=='missing')]"), [])
        rows["entries"].append({"name": "固有词", "definition": "conflict"})
        self.assertEqual(resolve(rows, "$.entries[?(@.name=='固有词')].definition"), ["new", "conflict"])

    def test_user_simulator_semantics_are_inside_feedback_retry_boundary(self):
        payload = {
            "profile": {}, "messages": [], "transitions": [{
                "transition_id": "accept", "outcome_category": "user_acceptance",
            }],
        }
        with patch("env_factory.sandbox_runtime.RuntimeLLMClient") as client:
            client.return_value.json_chat.return_value = {
                "user_query": "收到", "match_status": "matched",
                "outcome_category": "user_acceptance", "reason_code": "accepted",
                "transition_id": "accept",
            }
            ContractUserSimulator._llm_render(payload)
        kwargs = client.return_value.json_chat.call_args.kwargs
        self.assertIn("semantic_validator", kwargs)
        self.assertIn("not hidden tool traces", client.return_value.json_chat.call_args.args[0][0]["content"])
        with self.assertRaisesRegex(Exception, "same category"):
            kwargs["semantic_validator"]({
                "user_query": "收到", "match_status": "matched",
                "outcome_category": "goal_satisfied", "reason_code": "wrong",
                "transition_id": "accept",
            })

    def test_metric_paths_reject_unsupported_expressions(self):
        for path in ("$.entries[?(@.x>1)]", "$.entries[-1]"):
            with self.subTest(path=path), self.assertRaises(SandboxError):
                DeclarativeMetricEvaluator.path_tokens(path)

    def test_metric_paths_project_array_wildcards(self):
        rows = {"entries": [{"id": "a"}, {"id": "b"}]}
        self.assertEqual(
            DeclarativeMetricEvaluator._resolve(rows, "$.entries[*].id"),
            ["a", "b"],
        )

    def test_external_capability_client_uses_training_fixture_or_explicit_gap(self):
        previous = os.environ.get("SANDBOX_EXTERNAL_FIXTURES")
        try:
            os.environ.pop("SANDBOX_EXTERNAL_FIXTURES", None)
            self.assertEqual(ExternalCapabilityClient().query("prices", {})["status"], "data_unavailable")
            with tempfile.TemporaryDirectory() as directory:
                fixture = Path(directory) / "external.json"
                fixture.write_text(json.dumps({"prices": {"status": "ok", "items": [1]}}), encoding="utf-8")
                os.environ["SANDBOX_EXTERNAL_FIXTURES"] = str(fixture)
                self.assertEqual(ExternalCapabilityClient().query("prices", {})["items"], [1])
        finally:
            if previous is None:
                os.environ.pop("SANDBOX_EXTERNAL_FIXTURES", None)
            else:
                os.environ["SANDBOX_EXTERNAL_FIXTURES"] = previous

    def test_contract_evaluator_runtime_retries_failed_external_call(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SANDBOX_EVALUATOR_MOCK": "1"}):
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            store.reset(episode_id="eval", seed=1)
            evaluator = ContractEvaluatorRuntime(store)
            metric = {"id": "quality", "evaluator": {"kind": "external_llm_judge"}}
            with patch("env_factory.sandbox_runtime.RuntimeLLMClient.json_chat",
                       side_effect=[OSError("temporary outage"), {"label": "pass"}]) as judge:
                first = evaluator.json_judge(metric, {"answer": "x"}, fallback={"label": "fail"})
                second = evaluator.json_judge(metric, {"answer": "x"}, fallback={"label": "different"})
                third = evaluator.json_judge(metric, {"answer": "x"}, fallback={"label": "different"})
            self.assertEqual(first, {"label": "fail"})
            self.assertEqual(second, {"label": "pass"})
            self.assertEqual(third, second)
            self.assertEqual(judge.call_count, 2)
            events = store.replay()["events"]
            self.assertEqual([item["event"] for item in events], ["evaluator_call", "evaluator_call"])
            self.assertTrue(events[0]["payload"]["used_fallback"])
            self.assertFalse(events[1]["payload"]["used_fallback"])
            self.assertFalse(events[0]["payload"]["judgment_obtained"])
            self.assertTrue(events[1]["payload"]["judgment_obtained"])

    def test_judge_preserves_business_time_fields_inside_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            store.reset(episode_id="business-time", seed=1)
            evaluator = ContractEvaluatorRuntime(store)
            context = {
                "business_data": {"orders": [{"created_at": "2025-02-01", "timestamp": "09:30"}]},
                "tool_results": {"trace_hash": "transport-hash", "events": [{
                    "event": "tool_call", "timestamp": 123.0,
                    "result": {"created_at": "2025-02-01", "timestamp": "09:30"},
                }]},
            }
            metric = {"id": "fact", "evaluator": {"score_mapping": {"pass": 1, "fail": 0}}}
            with patch("env_factory.sandbox_runtime.RuntimeLLMClient.json_chat",
                       return_value={"label": "pass"}) as judge:
                evaluator.json_judge(metric, context, fallback={"label": "fail"})
            sent = json.loads(judge.call_args.args[0][1]["content"])["runtime_evidence"]
            self.assertEqual(sent["business_data"]["orders"][0]["created_at"], "2025-02-01")
            self.assertEqual(sent["tool_results"]["events"][0]["result"]["timestamp"], "09:30")
            self.assertNotIn("timestamp", sent["tool_results"]["events"][0])
            self.assertNotIn("trace_hash", sent["tool_results"])

    def test_model_evaluator_limits_context_and_explains_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            store.reset(episode_id="eval", seed=1)
            contract = {
                "task": "Compare A and B",
                "public_input": {"initial_user_message": "Compare A and B for this project"},
                "metrics": [{
                "id": "quality", "rubric": "answer is correct",
                "criteria": ["contains the requested comparison"],
                "evaluation_inputs": ["final_agent_response"],
                "score_range": [0, 1],
                "evaluator": {
                    "kind": "external_llm_judge", "source": "external_llm",
                    "score_mapping": {"satisfied": 1, "partial": 0.5, "missing": 0},
                },
                }],
            }
            captured = {}
            def fake_chat(client, messages, **kwargs):
                captured["messages"] = messages
                return {"label": "satisfied"}
            with patch("env_factory.sandbox_runtime.RuntimeLLMClient.json_chat", new=fake_chat):
                scores = ContractModelMetricEvaluator(contract, store).evaluate_all(
                    {"final_agent_response": "A vs B", "business_state": {"huge": [1] * 1000}}, {}
                )
            self.assertEqual(scores, {"quality": 1.0})
            self.assertIn("label_scores", captured["messages"][0]["content"])
            self.assertIn("all assistant responses cumulatively", captured["messages"][0]["content"])
            self.assertIn("Compare A and B for this project", captured["messages"][1]["content"])
            self.assertNotIn("business_state", captured["messages"][1]["content"])

    def test_model_evaluator_uses_current_tool_results_for_factuality(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            store.reset(episode_id="current-data", seed=1)
            contract = {"metrics": [{
                "id": "accuracy", "rubric": "match current product",
                "criteria": ["All answer facts match current business records"],
                "evaluation_inputs": ["final_agent_response", "tool_results", "business_data"],
                "score_range": [0, 1],
                "evaluator": {"kind": "external_llm_judge", "source": "external_llm",
                              "score_mapping": {"correct": 1, "incorrect": 0}},
            }]}
            captured = {}
            def fake_chat(client, messages, **kwargs):
                captured["messages"] = messages
                return {"label": "incorrect"}
            context = {
                "final_agent_response": "The product is blue",
                "business_state": {"product": [{"color": "red"}]},
                "trajectory": {"events": [
                    {"event": "old transport event"},
                    {"event": "tool_call", "payload": {
                        "tool_name": "query_product", "arguments": {"id": "HF-2024"},
                    }, "result": {"records": [{"color": "red"}]}},
                ]},
                "observation": {"tool_results": [{"records": [{"color": "red"}]}]},
            }
            with patch("env_factory.sandbox_runtime.RuntimeLLMClient.json_chat", new=fake_chat):
                scores = ContractModelMetricEvaluator(contract, store).evaluate_all(context, {})
            self.assertEqual(scores, {"accuracy": 0.0})
            sent = json.loads(captured["messages"][1]["content"])["runtime_evidence"]
            self.assertEqual(sent["tool_results"], [{
                "tool_name": "query_product", "arguments": {"id": "HF-2024"},
                "result": {"records": [{"color": "red"}]},
            }])
            self.assertNotIn("trajectory", sent)
            self.assertIn("stale private-record value", captured["messages"][0]["content"])

    def test_wsgi_auth_precedes_malformed_json_and_preserves_request_id(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {"SANDBOX_TRAINER_API_KEY": "trainer-key"}
        ):
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            app = SandboxApplication(
                episode_store=store,
                tool_registry=ContractToolRegistry([], {}),
                observation=lambda: {}, reward=lambda: {"reward": 0},
                user_turn=lambda messages: {"user_query": "ok", "should_end": True},
            )
            unauthorized = self._wsgi_request(app, "/v1/reset")
            self.assertEqual(unauthorized["status"], "401 Unauthorized")
            authorized = self._wsgi_request(
                app, "/v1/reset", authorization="Bearer trainer-key"
            )
            self.assertEqual(authorized["status"], "400 Bad Request")
            self.assertEqual(authorized["body"]["error"]["code"], "INVALID_JSON")
            self.assertEqual(
                authorized["headers"]["X-Request-ID"],
                authorized["body"]["error"]["request_id"],
            )

    def test_manifest_data_store_supports_stateless_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            data = ManifestDataStore(
                {"environment_mode": "stateless", "tables": []}, directory, store
            )
            data.reset()
            self.assertEqual(data.baseline, {})
            self.assertEqual(data.snapshot_hash(), __import__("hashlib").sha256(b"{}").hexdigest())

    def test_manifest_data_store_enforces_sql_types_indexes_and_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schema = {
                "table_name": "samples",
                "columns": [
                    {"name": "id", "type": "BIGINT", "nullable": False},
                    {"name": "code", "type": "VARCHAR(20)", "nullable": False},
                    {"name": "efficiency", "type": "DECIMAL(5,2)", "nullable": False},
                ],
                "primary_key": ["id"],
                "foreign_keys": [],
                "indexes": [{"name": "uq_code", "columns": ["code"], "unique": True}],
                "constraints": [{
                    "name": "efficiency_range", "type": "CHECK",
                    "expression": "efficiency >= 0 AND efficiency <= 100",
                }],
            }
            (root / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
            (root / "rows.jsonl").write_text(
                json.dumps({"id": 1, "code": "S-1", "efficiency": 22.5}) + "\n",
                encoding="utf-8",
            )
            manifest = {"environment_mode": "stateful", "tables": [{
                "table_name": "samples", "schema_file": "schema.json",
                "rows_file": "rows.jsonl",
            }]}
            store = EpisodeStore(root / "episodes.sqlite3")
            data = ManifestDataStore(manifest, root, store)
            data.reset()
            with self.assertRaisesRegex(SandboxError, "CHECK constraint failed"):
                data.update("samples", {"id": 1}, {"efficiency": 101})
            with self.assertRaisesRegex(SandboxError, "must be number"):
                data.update("samples", {"id": 1}, {"efficiency": "22.5"})
            with self.assertRaisesRegex(SandboxError, "duplicate unique index"):
                data.insert("samples", {"id": 2, "code": "S-1", "efficiency": 20.0})

    def test_insert_generates_integer_primary_key_and_safe_nullable_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schema = {"columns": [
                {"name": "id", "type": "INTEGER", "nullable": False},
                {"name": "name", "type": "TEXT", "nullable": False},
                {"name": "note", "type": "TEXT", "nullable": True},
                {"name": "status", "type": "TEXT", "nullable": False},
            ], "primary_key": ["id"], "foreign_keys": [], "indexes": [], "constraints": []}
            (root / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
            (root / "rows.jsonl").write_text(json.dumps({
                "id": 7, "name": "old", "note": None, "status": "open",
            }) + "\n", encoding="utf-8")
            store = EpisodeStore(root / "episodes.sqlite3")
            data = ManifestDataStore({"environment_mode": "stateful", "tables": [{
                "table_name": "events", "schema_file": "schema.json", "rows_file": "rows.jsonl",
            }]}, root, store)
            data.reset()
            self.assertEqual(data.insert("events", {"name": "new", "status": "open"}), {
                "id": 8, "name": "new", "note": None, "status": "open",
            })
            with self.assertRaisesRegex(SandboxError, "does not match schema"):
                data.insert("events", {"name": "missing status"})

    def test_nullable_in_check_accepts_only_listed_values_or_null(self):
        expression = "is_matched IN (0, 1) OR is_matched IS NULL"
        self.assertTrue(ManifestDataStore._check_constraint({"is_matched": None}, expression))
        self.assertTrue(ManifestDataStore._check_constraint({"is_matched": 1}, expression))
        self.assertFalse(ManifestDataStore._check_constraint({"is_matched": 2}, expression))
        with self.assertRaises(SandboxError):
            ManifestDataStore._check_constraint({"is_matched": 1}, "is_matched = 1 OR 1 = 1")

    def test_nullable_numeric_check_accepts_null_or_positive_value(self):
        expression = "waitlist_seq IS NULL OR waitlist_seq > 0"
        self.assertTrue(ManifestDataStore._check_constraint({"waitlist_seq": None}, expression))
        self.assertTrue(ManifestDataStore._check_constraint({"waitlist_seq": 2}, expression))
        self.assertFalse(ManifestDataStore._check_constraint({"waitlist_seq": 0}, expression))

    def test_contract_user_simulator_failure_does_not_accept_completion_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="user", seed=3)
            simulator = ContractUserSimulator(
                store,
                profiles=[{"profile_id": "p1"}],
                scripts=[{
                    "script_id": "s1", "initial_state": "start", "variables": {},
                    "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。"},
                    "states": [
                        {"state_id": "start", "user_behavior": "第一问", "terminal": False},
                        {"state_id": "done", "user_behavior": "结束", "terminal": True},
                    ],
                    "transitions": [{
                        "transition_id": "finish", "from_state": "start", "to_state": "done",
                        "condition": "continue", "should_end": True, "updates": {},
                    }],
                }],
                renderer=lambda _: (_ for _ in ()).throw(RuntimeError("timeout")),
            )
            simulator.reset(episode)
            first = simulator.turn([{"role": "assistant", "content": "任务已完成，结果如下。"}])
            state = store.get_state(ContractUserSimulator.STATE_KEY)
            self.assertEqual(first["match_status"], "unmatched")
            self.assertEqual(first["outcome_category"], "unrecognized")
            self.assertFalse(first["should_end"])
            self.assertEqual(first["reason_code"], "simulator_unavailable")
            self.assertEqual(state["state_id"], "start")

    def test_contract_user_simulator_fallback_does_not_advance_on_irrelevant_reply(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="irrelevant", seed=3)
            simulator = ContractUserSimulator(
                store,
                profiles=[{"profile_id": "p1"}],
                scripts=[{
                    "script_id": "s1", "initial_state": "start", "variables": {},
                    "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。"},
                    "states": [
                        {"state_id": "start", "user_behavior": "提出任务", "terminal": False},
                        {"state_id": "review", "user_behavior": "检查结果", "terminal": False},
                        {"state_id": "done", "user_behavior": "结束", "terminal": True},
                    ],
                    "transitions": [
                        {"transition_id": "satisfied", "outcome_category": "goal_satisfied", "from_state": "start", "to_state": "review", "condition": "任务完成", "should_end": False, "updates": {}},
                        {"transition_id": "accepted", "outcome_category": "user_acceptance", "from_state": "review", "to_state": "done", "condition": "接受结果", "should_end": True, "updates": {}},
                    ],
                }],
                renderer=lambda _: (_ for _ in ()).throw(RuntimeError("offline")),
            )
            simulator.reset(episode)
            result = simulator.turn([{"role": "assistant", "content": "今天天气不错。"}])
            state = store.get_state(ContractUserSimulator.STATE_KEY)
            self.assertEqual(result["match_status"], "unmatched")
            self.assertEqual(state["state_id"], "start")
            self.assertEqual(state["recovery_count"], 1)

    def test_contract_user_simulator_tracks_declared_fsm_state(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="fsm", seed=1)
            scripts = [{
                "script_id": "s1", "initial_state": "start", "variables": {"known": False},
                "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。"},
                "states": [{"state_id": "start", "user_behavior": "开始", "terminal": False}, {"state_id": "done", "user_behavior": "结束", "terminal": True}],
                "transitions": [{
                    "transition_id": "t1", "from_state": "start", "to_state": "done",
                    "condition": "agent completes", "should_end": True, "updates": {"known": True},
                }],
            }]
            simulator = ContractUserSimulator(
                store, profiles=[{"profile_id": "p1"}], scripts=scripts,
                renderer=lambda _: {"user_query": "结束", "transition_id": "t1"},
            )
            simulator.reset(episode)
            result = simulator.turn([{"role": "assistant", "content": "完成"}])
            state = store.get_state(ContractUserSimulator.STATE_KEY)
            self.assertTrue(result["should_end"])
            self.assertEqual(result["fsm_script_id"], "s1")
            self.assertEqual(result["fsm_transition_id"], "t1")
            self.assertEqual(result["fsm_state_before"], "start")
            self.assertEqual(result["fsm_state_after"], "done")
            self.assertTrue(result["fsm_transition_applied"])
            self.assertEqual(result["fsm_recovery_count"], 0)
            self.assertEqual(state["state_id"], "done")
            self.assertTrue(state["variables"]["known"])

    def test_contract_user_simulator_does_not_advance_on_unmatched_dialogue(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="recovery", seed=1)
            script = {
                "script_id": "s1", "initial_state": "start", "variables": {},
                "recovery_policy": {"max_recoveries": 2, "user_behavior": "请重新回答。"},
                "states": [
                    {"state_id": "start", "user_behavior": "开始", "terminal": False},
                    {"state_id": "done", "user_behavior": "结束", "terminal": True},
                ],
                "transitions": [{
                    "transition_id": "finish", "from_state": "start", "to_state": "done",
                    "condition": "完成", "should_end": True, "updates": {},
                }],
            }
            simulator = ContractUserSimulator(
                store, profiles=[{"profile_id": "p1"}], scripts=[script],
                renderer=lambda _: {
                    "user_query": "这没有回答我的问题。", "match_status": "ambiguous",
                    "outcome_category": "unrecognized",
                    "reason_code": "off_topic_response",
                },
            )
            simulator.reset(episode)
            result = simulator.turn([{"role": "assistant", "content": "无关回复"}])
            state = store.get_state(ContractUserSimulator.STATE_KEY)
            self.assertEqual(state["state_id"], "start")
            self.assertEqual(state["recovery_count"], 1)
            self.assertFalse(result["should_end"])
            self.assertEqual(result["match_status"], "ambiguous")
            self.assertIsNone(result["fsm_transition_id"])
            self.assertEqual(result["fsm_state_before"], "start")
            self.assertEqual(result["fsm_state_after"], "start")
            self.assertFalse(result["fsm_transition_applied"])
            self.assertEqual(result["fsm_recovery_count"], 1)

    def test_contract_user_simulator_rejects_truncated_conversation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="conversation", seed=1)
            simulator = ContractUserSimulator(
                store, profiles=[{"profile_id": "p1"}], scripts=[{
                    "script_id": "s1", "initial_state": "start", "variables": {},
                    "recovery_policy": {"max_recoveries": 3},
                    "states": [{"state_id": "start", "terminal": False}],
                    "transitions": [],
                }], renderer=lambda _: {
                    "user_query": "请继续。", "match_status": "unmatched",
                    "outcome_category": "unrecognized", "reason_code": "retry",
                },
            )
            simulator.reset(episode)
            first = [{"role": "assistant", "content": "第一次回复"}]
            simulator.turn(first)
            with self.assertRaisesRegex(SandboxError, "complete ordered conversation"):
                simulator.turn([{"role": "assistant", "content": "第二次回复"}])
            complete = [
                *first,
                {"role": "user", "content": "请继续。"},
                {"role": "assistant", "content": "第二次回复"},
            ]
            self.assertEqual(simulator.turn(complete)["user_query"], "请继续。")
    def test_acceptance_scenario_runner_captures_and_reuses_values(self):
        calls = []
        def call(method, path, body, headers):
            calls.append((method, path, body, headers))
            if path == "/v1/reset":
                return 200, {"episode_id": "ep-1"}, {}
            return 200, {"records": [{"id": "1"}], "count": 1}, {}
        runner = AcceptanceScenarioRunner(call, trainer_headers={"Authorization": "Bearer x"})
        result = runner.run({
            "scenario_id": "query",
            "steps": [
                {"operation": "reset", "body": {"seed": 7}, "capture": {"episode": "$.episode_id"}},
                {"operation": "tool_call", "tool_name": "lookup", "arguments": {"episode": {"$ref": "episode"}}},
            ],
            "assertions": [{"path": "$.count", "operator": "eq", "expected": 1}],
        })
        self.assertEqual(result["variables"]["episode"], "ep-1")
        self.assertEqual(calls[1][2]["episode"], "ep-1")

    def test_acceptance_runner_submits_agent_response(self):
        calls = []
        def call(method, path, body, headers):
            calls.append((method, path, body, headers))
            return 200, {"accepted": True}, {}
        runner = AcceptanceScenarioRunner(call, trainer_headers={"Authorization": "Bearer x"})
        runner.run({
            "scenario_id": "answer",
            "steps": [{"operation": "agent_response", "content": "最终回答"}],
            "assertions": [{"path": "$.accepted", "operator": "eq", "expected": True}],
        })
        self.assertEqual(calls[0][0:3], ("POST", "/v1/agent_response", {"content": "最终回答"}))

    def test_acceptance_runner_supports_snapshots_mutation_and_step_assertions(self):
        state = {"status": "pending"}
        runner = AcceptanceScenarioRunner(
            lambda *args: (200, {}, {}),
            business_snapshot=lambda: dict(state),
            mutate_business_state=lambda mutation: state.update(mutation.get("changes", {})) or dict(state),
        )
        result = runner.run({
            "scenario_id": "counterfactual",
            "steps": [
                {"step_id": "before", "operation": "business_snapshot", "capture": {"before_status": "$.status"}},
                {"step_id": "mutated", "operation": "mutate_business_state", "mutation": {"changes": {"status": "failed"}}},
            ],
            "assertions": [
                {"source": "step:mutated", "path": "$.status", "operator": "changed", "expected": {"$ref": "before_status"}}
            ],
        })
        self.assertEqual(result["step_results"]["mutated"]["status"], "failed")
    def test_declarative_metric_evaluator_uses_runtime_context(self):
        evaluator = DeclarativeMetricEvaluator()
        specs = [
            {
                "metric_id": "enough_rows", "source": "business_state",
                "path": "$.results", "operator": "count_gte", "expected": 2,
                "score_mapping": {"pass": 1, "fail": 0},
            },
            {
                "metric_id": "no_noise", "source": "trajectory",
                "path": "$.noise_calls", "operator": "eq", "expected": 0,
                "score_mapping": {"pass": 0, "fail": -1},
            },
        ]
        scores = evaluator.evaluate_all(specs, {
            "business_state": {"results": [1, 2, 3]},
            "trajectory": {"noise_calls": 0},
        })
        self.assertEqual(scores, {"enough_rows": 1.0, "no_noise": 0.0})

    def test_declarative_metric_evaluator_detects_noise_tool_calls(self):
        evaluator = DeclarativeMetricEvaluator()
        spec = {
            "metric_id": "no_noise", "source": "trajectory", "path": "$.events",
            "operator": "none_tool_calls", "expected": ["weather"],
            "score_mapping": {"pass": 0, "fail": -1},
        }
        clean = evaluator.evaluate(spec, {"trajectory": {"events": []}})
        noisy = evaluator.evaluate(spec, {"trajectory": {"events": [
            {"event": "tool_call", "payload": {"tool_name": "weather"}}
        ]}})
        self.assertEqual((clean, noisy), (0.0, -1.0))

    def test_declarative_metric_contains_partial_business_row(self):
        evaluator = DeclarativeMetricEvaluator()
        spec = {
            "metric_id": "updated", "source": "business_state",
            "path": "$.samples", "operator": "contains",
            "expected": {"sample_id": "S-1", "efficiency": 22.8},
            "score_mapping": {"pass": 1, "fail": 0},
        }
        context = {"business_state": {"samples": [
            {"sample_id": "S-1", "material": "silicon", "efficiency": 22.8}
        ]}}
        self.assertEqual(evaluator.evaluate(spec, context), 1.0)
        context["business_state"]["samples"][0]["efficiency"] = 21.5
        self.assertEqual(evaluator.evaluate(spec, context), 0.0)

    def test_numeric_targets_recompute_from_private_business_rows(self):
        evaluator = DeclarativeMetricEvaluator()
        spec = {
            "metric_id": "purchase_total", "source": "final_agent_response",
            "path": "$", "operator": "numeric_targets",
            "expected": {"targets": [{
                "label": "总金额", "unit": "元", "tolerance": 0.01,
                "expression": {"op": "mul", "args": [
                    {"op": "sub", "args": [
                        {"literal": 3},
                        {"lookup": {"table": "inventory", "field": "remaining_kg",
                                    "where": {"ingredient_name": "木耳"}}},
                    ]},
                    {"literal": 60},
                ]},
            }]},
            "score_mapping": {"pass": 1, "fail": 0},
        }
        context = {
            "final_agent_response": "结论：总金额132元。",
            "business_state": {"inventory": [{"ingredient_name": "木耳", "remaining_kg": 0.8}]},
        }
        self.assertEqual(evaluator.evaluate(spec, context), 1.0)
        context["business_state"]["inventory"][0]["remaining_kg"] = 1.8
        self.assertEqual(evaluator.evaluate(spec, context), 0.0)
        context["final_agent_response"] = "结论：总金额72元。"
        self.assertEqual(evaluator.evaluate(spec, context), 1.0)
        context["final_agent_response"] = "结论：总金额999元。补充说明：总金额72元。"
        self.assertEqual(evaluator.evaluate(spec, context), 0.0)
        spec["expected"]["targets"][0]["expression"]["args"][0]["args"][0]["literal"] = 23
        context["final_agent_response"] = "结论：总金额1,332元。"
        context["business_state"]["inventory"][0]["remaining_kg"] = 0.8
        self.assertEqual(evaluator.evaluate(spec, context), 1.0)

    def test_numeric_targets_support_private_row_aggregation(self):
        evaluator = DeclarativeMetricEvaluator()
        spec = {
            "metric_id": "trade_total", "source": "final_agent_response",
            "path": "$", "operator": "numeric_targets",
            "expected": {"targets": [{
                "label": "合计", "unit": "元", "tolerance": 0,
                "expression": {"aggregate": {
                    "table": "trades", "field": "amount", "where": {"status": "settled"},
                    "op": "sum",
                }},
            }]},
            "score_mapping": {"pass": 1, "fail": 0},
        }
        context = {"final_agent_response": "合计300元", "business_state": {
            "trades": [{"amount": 100, "status": "settled"},
                       {"amount": 200, "status": "settled"},
                       {"amount": 900, "status": "pending"}],
        }}
        self.assertEqual(evaluator.evaluate(spec, context), 1.0)
        context["business_state"]["trades"][0]["amount"] = 101
        self.assertEqual(evaluator.evaluate(spec, context), 0.0)
        spec["expected"]["targets"][0]["expression"]["aggregate"]["op"] = "count"
        context["final_agent_response"] = "合计2元"
        self.assertEqual(evaluator.evaluate(spec, context), 1.0)
        context["business_state"]["trades"].pop(0)
        self.assertEqual(evaluator.evaluate(spec, context), 0.0)

    def test_numeric_targets_sum_product_includes_new_matching_rows(self):
        evaluator = DeclarativeMetricEvaluator()
        spec = {
            "source": "final_agent_response", "path": "$", "operator": "numeric_targets",
            "expected": {"targets": [{
                "label": "总面积", "unit": "平方英尺", "tolerance": 0,
                "expression": {"aggregate": {
                    "table": "inventory", "fields": ["area", "quantity"],
                    "where": {"batch": "A"}, "op": "sum_product",
                }},
            }]},
        }
        context = {"final_agent_response": "总面积14平方英尺", "business_state": {
            "inventory": [{"batch": "A", "area": 2, "quantity": 3},
                          {"batch": "A", "area": 4, "quantity": 2},
                          {"batch": "B", "area": 10, "quantity": 5}],
        }}
        self.assertEqual(evaluator.evaluate(spec, context), 1.0)
        context["business_state"]["inventory"].append(
            {"batch": "A", "area": 1, "quantity": 2}
        )
        self.assertEqual(evaluator.evaluate(spec, context), 0.0)
        context["final_agent_response"] = "总面积16平方英尺"
        self.assertEqual(evaluator.evaluate(spec, context), 1.0)
        malformed = copy.deepcopy(spec)
        malformed["expected"]["targets"][0]["expression"]["aggregate"]["field"] = "area"
        with self.assertRaisesRegex(SandboxError, "extra=\\['field'\\]"):
            DeclarativeMetricEvaluator.validate_numeric_targets(malformed["expected"])

    def test_declarative_metric_matches_exact_tool_call_with_capture(self):
        evaluator = DeclarativeMetricEvaluator()
        spec = {
            "metric_id": "write", "source": "trajectory", "path": "$.events",
            "operator": "contains_tool_call",
            "expected": {
                "tool_name": "update_item",
                "arguments": {"item_id": {"$ref": "selected_id"}, "quantity": 5},
                "captures": [{
                    "name": "selected_id", "tool_name": "find_item",
                    "path": "$.records[0].id",
                }],
            },
            "score_mapping": {"pass": 1, "fail": 0},
        }
        events = [
            {"event": "tool_call", "payload": {
                "tool_name": "find_item", "arguments": {"name": "tea"},
            }, "result": {"records": [{"id": 7}]}},
            {"event": "tool_call", "payload": {
                "tool_name": "update_item", "arguments": {"item_id": 7, "quantity": 5},
            }, "result": {"updated_count": 1}},
        ]
        self.assertEqual(evaluator.evaluate(spec, {"trajectory": {"events": events}}), 1.0)
        events[-1]["payload"]["arguments"]["quantity"] = 4
        self.assertEqual(evaluator.evaluate(spec, {"trajectory": {"events": events}}), 0.0)
    def test_declarative_tool_compiler_filters_projects_and_orders(self):
        class FakeData:
            def table(self, name):
                self.name = name
                return [
                    {"id": "2", "kind": "cdn", "enabled": False, "internal": "x"},
                    {"id": "1", "kind": "cdn", "enabled": True, "internal": "y"},
                    {"id": "3", "kind": "storage", "enabled": True, "internal": "z"},
                ]

        compiler = DeclarativeToolCompiler(FakeData())
        handler = compiler.compile({
            "tool_name": "query",
            "operation": "select",
            "table": "services",
            "filters": [
                {"argument": "kind", "column": "kind", "operator": "eq"},
                {"argument": "enabled", "column": "enabled", "operator": "eq"},
            ],
            "projection": ["id", "kind"],
            "order_by": ["id"],
            "result_field": "services",
        })
        self.assertEqual(handler({"kind": "cdn", "enabled": True}), {
            "services": [{"id": "1", "kind": "cdn"}], "count": 1
        })

    def test_declarative_tool_contains_accepts_multi_value_query(self):
        class FakeData:
            def table(self, name):
                return [
                    {"id": "1", "tags": ["neutral", "linen"]},
                    {"id": "2", "tags": ["formal"]},
                ]

        handler = DeclarativeToolCompiler(FakeData()).compile({
            "tool_name": "query",
            "operation": "select",
            "table": "items",
            "filters": [{"argument": "tags", "column": "tags", "operator": "contains"}],
            "projection": ["id"],
            "result_field": "records",
        })
        self.assertEqual(handler({"tags": ["neutral", "casual"]}), {
            "records": [{"id": "1"}], "count": 1,
        })

    def test_declarative_tool_filters_by_column_in_prior_record_array(self):
        class FakeData:
            def table(self, name):
                return [
                    {"sample_code": "A", "latest_batch_no": "B-1", "passed": True},
                    {"sample_code": "B", "latest_batch_no": "B-2", "passed": False},
                    {"sample_code": "C", "latest_batch_no": "B-3", "passed": True},
                ]

        handler = DeclarativeToolCompiler(FakeData()).compile({
            "tool_name": "fetch_inspection", "operation": "select",
            "table": "samples", "result_field": "records",
            "filters": [{"argument": "inventory_records", "column": "latest_batch_no", "operator": "in"}],
        })
        self.assertEqual(handler({"inventory_records": [
            {"sample_code": "A", "latest_batch_no": "B-1"},
            {"sample_code": "B", "latest_batch_no": "B-2"},
        ]})["records"], [
            {"sample_code": "A", "latest_batch_no": "B-1", "passed": True},
            {"sample_code": "B", "latest_batch_no": "B-2", "passed": False},
        ])
        with self.assertRaises(SandboxError):
            handler({"inventory_records": [{"sample_code": "A"}]})

    def test_declarative_tool_resolves_human_name_through_foreign_key(self):
        class FakeData:
            def table(self, name):
                return {
                    "brands": [{"id": 7, "name": "Cotton"}],
                    "products": [
                        {"id": 1, "brand_id": 7},
                        {"id": 2, "brand_id": 8},
                    ],
                }[name]

        handler = DeclarativeToolCompiler(FakeData()).compile({
            "tool_name": "count_products", "operation": "aggregate_count",
            "table": "products", "result_field": "count",
            "filters": [{
                "argument": "brand_name", "column": "brand_id", "operator": "eq",
                "resolve": {
                    "table": "brands", "match_column": "name", "value_column": "id",
                },
            }],
        })
        self.assertEqual(handler({"brand_name": "Cotton"}), {"count": 1})
        self.assertEqual(handler({"brand_name": "Unknown"}), {"count": 0})

    def test_declarative_tool_resolves_name_list_and_projects_public_alias(self):
        class FakeData:
            def table(self, name):
                return {
                    "brands": [{"id": 7, "name": "Cotton"}, {"id": 8, "name": "Wool"}],
                    "products": [
                        {"id": 1, "name": "Shirt", "brand_id": 7},
                        {"id": 2, "name": "Coat", "brand_id": 8},
                    ],
                }[name]

        handler = DeclarativeToolCompiler(FakeData()).compile({
            "tool_name": "find_products", "operation": "select",
            "table": "products", "result_field": "records",
            "filters": [{
                "argument": "brand_names", "column": "brand_id", "operator": "in",
                "resolve": {
                    "table": "brands", "match_column": "name", "value_column": "id",
                },
            }],
            "projection": ["id"], "projection_aliases": {"product_name": "name"},
        })
        self.assertEqual(handler({"brand_names": ["Cotton", "Wool"]}), {
            "records": [
                {"id": 1, "product_name": "Shirt"},
                {"id": 2, "product_name": "Coat"},
            ],
            "count": 2,
        })

    def test_manifest_store_write_operations(self):
        class MemoryStore:
            def __init__(self): self.rows = [{"id": "1", "value": "a"}]
            def table(self, name): return [dict(row) for row in self.rows]
            def insert(self, name, row): self.rows.append(dict(row)); return dict(row)
            def update(self, name, selector, changes):
                count = 0
                for row in self.rows:
                    if all(row.get(k) == v for k, v in selector.items()): row.update(changes); count += 1
                return count
            def delete(self, name, selector):
                before = len(self.rows); self.rows = [r for r in self.rows if not all(r.get(k) == v for k, v in selector.items())]; return before - len(self.rows)
        data = MemoryStore()
        compiler = DeclarativeToolCompiler(data)
        update = compiler.compile({"operation": "update", "table": "items", "result_field": "records", "filters": [], "projection": [], "order_by": [], "selector": {"id": "id"}, "changes": {"value": "value"}})
        self.assertEqual(update({"id": "1", "value": "b"})["updated_count"], 1)
        delete = compiler.compile({"operation": "delete", "table": "items", "result_field": "records", "filters": [], "projection": [], "order_by": [], "selector": {"id": "id"}})
        self.assertEqual(delete({"id": "1"})["deleted_count"], 1)

    def test_manifest_data_store_resets_isolated_business_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schemas").mkdir()
            (root / "rows").mkdir()
            schema = {
                "columns": [{"name": "id"}, {"name": "value"}],
                "primary_key": ["id"],
                "foreign_keys": [],
            }
            (root / "schemas" / "items.json").write_text(json.dumps(schema), encoding="utf-8")
            (root / "rows" / "items.jsonl").write_text('{"id":"1","value":"a"}\n', encoding="utf-8")
            manifest = {"tables": [{
                "table_name": "items",
                "schema_file": "schemas/items.json",
                "rows_file": "rows/items.jsonl",
            }]}
            episodes = EpisodeStore(root / "episodes.sqlite3")
            data = ManifestDataStore(manifest, root, episodes)
            episodes.reset(episode_id="a", seed=1, data_hash=data.data_hash)
            data.reset()
            data.replace_table("items", [{"id": "1", "value": "changed"}])
            self.assertEqual(data.select("items", id="1")[0]["value"], "changed")
            episodes.reset(episode_id="b", seed=1, data_hash=data.data_hash)
            data.reset()
            self.assertEqual(data.select("items", id="1")[0]["value"], "a")

    def test_manifest_validator_accepts_descriptive_foreign_key_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schemas").mkdir()
            (root / "rows").mkdir()
            parent_schema = {
                "columns": [{"name": "id"}],
                "primary_key": ["id"],
                "foreign_keys": [],
            }
            child_schema = {
                "columns": [{"name": "id"}, {"name": "parent_id"}],
                "primary_key": ["id"],
                "foreign_keys": [{
                    "column": "parent_id",
                    "references_table": "parents",
                    "references_column": "id",
                }],
            }
            (root / "schemas" / "parents.json").write_text(json.dumps(parent_schema), encoding="utf-8")
            (root / "schemas" / "children.json").write_text(json.dumps(child_schema), encoding="utf-8")
            (root / "rows" / "parents.jsonl").write_text('{"id":"p1"}\n', encoding="utf-8")
            (root / "rows" / "children.jsonl").write_text(
                '{"id":"c1","parent_id":"p1"}\n', encoding="utf-8"
            )
            manifest = {"tables": [
                {
                    "table_name": "parents",
                    "schema_file": "schemas/parents.json",
                    "rows_file": "rows/parents.jsonl",
                },
                {
                    "table_name": "children",
                    "schema_file": "schemas/children.json",
                    "rows_file": "rows/children.jsonl",
                },
            ]}
            validated = DataManifestValidator.validate(manifest, root)
            self.assertEqual(validated["tables"]["children"][0]["parent_id"], "p1")

    def test_manifest_validator_rejects_duplicate_tables_and_stale_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "schema.json").write_text(json.dumps({
                "table_name": "inventory", "columns": [{"name": "id"}],
                "primary_key": ["id"], "foreign_keys": [],
            }), encoding="utf-8")
            (root / "rows.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
            table = {"table_name": "inventory", "schema_file": "schema.json",
                     "rows_file": "rows.jsonl", "row_count": 1}
            self.assertEqual(len(DataManifestValidator.validate({"tables": [table]}, root)["tables"]), 1)
            with self.assertRaisesRegex(SandboxError, "duplicate table name"):
                DataManifestValidator.validate({"tables": [table, dict(table)]}, root)
            with self.assertRaisesRegex(SandboxError, "row_count differs"):
                DataManifestValidator.validate({"tables": [{**table, "row_count": 2}]}, root)
            with self.assertRaisesRegex(SandboxError, "schema name differs"):
                DataManifestValidator.validate({"tables": [{**table, "table_name": "orders"}]}, root)

    def test_manifest_validator_confines_files_to_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "data"
            root.mkdir()
            (root / "schema.json").write_text(json.dumps({
                "table_name": "inventory", "columns": [{"name": "id"}],
                "primary_key": ["id"], "foreign_keys": [],
            }), encoding="utf-8")
            (base / "outside.jsonl").write_text('{"id": 1}\n', encoding="utf-8")
            table = {"table_name": "inventory", "schema_file": "schema.json",
                     "rows_file": "../outside.jsonl"}
            with self.assertRaisesRegex(SandboxError, "escapes data root"):
                DataManifestValidator.validate({"tables": [table]}, root)
            (root / "linked.jsonl").symlink_to(base / "outside.jsonl")
            with self.assertRaisesRegex(SandboxError, "escapes data root"):
                DataManifestValidator.validate({"tables": [{**table, "rows_file": "linked.jsonl"}]}, root)
            with self.assertRaisesRegex(SandboxError, "not an object"):
                DataManifestValidator.validate({"tables": [None]}, root)
            (root / "schema.json").write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(SandboxError, "schema is not an object"):
                DataManifestValidator.validate({"tables": [{**table, "rows_file": "schema.json"}]}, root)

    def test_date_only_filter_bounds_include_full_timestamp_day(self):
        matches = DeclarativeToolCompiler._matches
        self.assertTrue(matches("2025-06-05 18:30:00", "2025-06-05", "lte"))
        self.assertTrue(matches("2025-06-05T00:15:00", "2025-06-05", "gte"))
        self.assertFalse(matches("2025-06-06 00:00:00", "2025-06-05", "lte"))
        self.assertFalse(matches("2025-06-04 23:59:59", "2025-06-05", "gte"))
        self.assertFalse(matches("2025-06-05 18:30:00", "2025-06-05 12:00:00", "lte"))

    def test_shared_application_routes_and_authenticates(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = os.environ.get("SANDBOX_TRAINER_API_KEY")
            os.environ["SANDBOX_TRAINER_API_KEY"] = "trainer-key"
            try:
                store = EpisodeStore(Path(directory) / "episodes.sqlite3")
                tools = [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                            "additionalProperties": False,
                        },
                    },
                }]
                registry = ContractToolRegistry(tools, {"lookup": lambda args: {"value": args["name"]}})
                app = SandboxApplication(
                    episode_store=store,
                    tool_registry=registry,
                    observation=lambda: {"public": True},
                    business_snapshot=lambda: {"private": "trainer-only"},
                    reward=lambda: {"reward": 1.0},
                    user_turn=lambda messages: {"user_query": "ok", "should_end": False},
                    data_hash="data-v1",
                )
                status, body, _ = app.handle("GET", "/health")
                self.assertEqual((status, body["status"]), (200, "ok"))
                status, _, _ = app.handle("GET", "/v1/observation")
                self.assertEqual(status, 401)
                auth = {"Authorization": "Bearer trainer-key"}
                status, reset, _ = app.handle("POST", "/v1/reset", {"seed": 7}, auth)
                self.assertEqual(reset["seed"], 7)
                status, state, _ = app.handle("GET", "/v1/state", headers=auth)
                self.assertEqual(
                    (status, state),
                    (200, {"business_state": {"private": "trainer-only"}}),
                )
                status, result, tool_headers = app.handle("POST", "/v1/tools/lookup", {"name": "x"})
                self.assertEqual((status, result), (200, {"value": "x"}))
                self.assertTrue(tool_headers["X-Tool-Call-ID"].startswith("call-"))
                status, accepted, _ = app.handle(
                    "POST", "/v1/agent_response", {"content": "最终回答"}, auth
                )
                self.assertEqual((status, accepted), (200, {"accepted": True}))
                self.assertEqual(store.get_state("final_agent_response"), "最终回答")
                status, reward, _ = app.handle("GET", "/v1/reward", headers=auth)
                self.assertEqual(reward["reward"], 1.0)
            finally:
                if previous is None:
                    os.environ.pop("SANDBOX_TRAINER_API_KEY", None)
                else:
                    os.environ["SANDBOX_TRAINER_API_KEY"] = previous

    def test_contract_tool_registry_validates_dispatches_and_records(self):
        events = []

        def record(*args, **kwargs):
            events.append((args, kwargs))

        tools = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }]
        registry = ContractToolRegistry(
            tools, {"lookup": lambda args: {"value": args["name"]}}, event_recorder=record
        )
        self.assertEqual(registry.execute("lookup", {"name": "x"}), {"value": "x"})
        self.assertEqual(len(events), 1)
        self.assertIn("tool_call_id", events[0][0][1])
        self.assertIn("duration_ms", events[0][0][1])
        with self.assertRaises(SandboxError):
            registry.execute("lookup", {})
        with self.assertRaises(SandboxError):
            registry.execute("lookup", {"name": "x", "extra": True})

    def test_tool_idempotency_prevents_reexecuting_business_handler(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            store.reset(episode_id="idem", seed=1)
            calls = []
            tools = [{"type": "function", "function": {
                "name": "write", "parameters": {
                    "type": "object", "properties": {"value": {"type": "string"}},
                    "required": ["value"], "additionalProperties": False,
                }
            }}]
            registry = ContractToolRegistry(
                tools, {"write": lambda args: calls.append(dict(args)) or {"saved": args["value"]}},
                event_recorder=store.event,
            )
            first = registry.execute("write", {"value": "a"}, idem_key="same")
            second = registry.execute("write", {"value": "a"}, idem_key="same")
            with self.assertRaisesRegex(SandboxError, "another request"):
                registry.execute("write", {"value": "different"}, idem_key="same")
            self.assertEqual(first, second)
            self.assertEqual(calls, [{"value": "a"}])

    def test_shared_runtime_owns_all_declared_mutation_seams(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "update",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
        }]
        calls = []
        registry = ContractToolRegistry(tools, {"update": lambda args: calls.append(args) or {"value": args.get("value")}})
        previous = os.environ.get("SANDBOX_MUTATION_MODE")
        previous_key = os.environ.get("SANDBOX_TRAINER_API_KEY")
        try:
            os.environ["SANDBOX_MUTATION_MODE"] = "skip_business_write"
            self.assertEqual(registry.execute("update", {"value": "x"})["mutation"], "skip_business_write")
            self.assertEqual(calls, [])
            os.environ["SANDBOX_MUTATION_MODE"] = "ignore_tool_arguments"
            self.assertEqual(registry.execute("update", {"unexpected": True}), {"value": None})

            with tempfile.TemporaryDirectory() as directory:
                store = EpisodeStore(Path(directory) / "episodes.sqlite3")
                app = SandboxApplication(
                    episode_store=store, tool_registry=registry,
                    observation=lambda: {}, reward=lambda: {"reward": 1.0},
                    user_turn=lambda messages: {"user_query": "ok", "should_end": False},
                )
                os.environ["SANDBOX_MUTATION_MODE"] = "bypass_trainer_auth"
                self.assertEqual(app.handle("POST", "/v1/reset", {})[0], 200)
                os.environ["SANDBOX_MUTATION_MODE"] = "constant_reward"
                os.environ["SANDBOX_TRAINER_API_KEY"] = "mutation-key"
                status, reward, _ = app.handle(
                    "GET", "/v1/reward", headers={"Authorization": "Bearer mutation-key"}
                )
                self.assertEqual(status, 200)
                self.assertEqual(
                    reward,
                    {"reward": 0.0, "raw_reward": 0.0, "components": {"__mutation__": 0.0}},
                )
        finally:
            if previous is None:
                os.environ.pop("SANDBOX_MUTATION_MODE", None)
            else:
                os.environ["SANDBOX_MUTATION_MODE"] = previous
            if previous_key is None:
                os.environ.pop("SANDBOX_TRAINER_API_KEY", None)
            else:
                os.environ["SANDBOX_TRAINER_API_KEY"] = previous_key

    def test_noise_tool_needs_no_task_handler_and_has_no_business_effect(self):
        events = []
        tools = [{
            "type": "function",
            "function": {
                "name": "weather",
                "parameters": {
                    "type": "object", "properties": {}, "required": [],
                    "additionalProperties": False,
                },
            },
        }]
        registry = ContractToolRegistry(
            tools,
            {},
            noise_tools=[{"name": "weather", "category": "unrelated"}],
            event_recorder=lambda *args, **kwargs: events.append((args, kwargs)),
        )
        result = registry.execute("weather", {})
        self.assertNotIn("category", result)
        self.assertEqual(result, {"records": [], "count": 0})
        self.assertNotIn("message", result)
        self.assertTrue(events[0][0][1]["noise"])

    def test_reward_aggregator_uses_contract_weights_and_ranges(self):
        aggregator = ContractRewardAggregator([
            {"id": "success", "weight": 0.8, "score_range": [0, 1]},
            {"id": "process", "weight": 0.2, "score_range": [0, 1]},
            {"id": "penalty", "weight": 1.0, "score_range": [-1, 0]},
        ])
        result = aggregator.aggregate({"success": 1, "process": 0.5, "penalty": -0.25})
        self.assertAlmostEqual(result["reward"], 0.65)
        self.assertEqual(set(result["components"]), {"success", "process", "penalty"})
        with self.assertRaises(SandboxError):
            aggregator.aggregate({"success": 2})

    def test_reward_aggregator_rejects_missing_declared_scores(self):
        aggregator = ContractRewardAggregator([
            {"id": "success", "weight": 1.0, "score_range": [0, 1]},
        ])
        with self.assertRaisesRegex(SandboxError, "declared metrics have no runtime score"):
            aggregator.aggregate({})

    def test_reward_gate_blocks_outcome_without_required_tool_progress(self):
        metrics = [
            {"id": "process", "category": "process", "score_range": [0, 1]},
            {"id": "outcome", "category": "outcome", "score_range": [0, 1]},
        ]
        gate = ContractRewardGate({
            "training_contract": {"category": "simple_agentic"},
            "capability_dag": {"nodes": ["lookup"]},
        }, metrics)
        blocked = gate.apply(
            {"process": 0.0, "outcome": 1.0},
            {"trajectory": {"events": []}},
        )
        self.assertEqual(blocked, {"process": 0.0, "outcome": 0.0})
        allowed = gate.apply(
            {"process": 1.0, "outcome": 1.0},
            {"trajectory": {"events": [{
                "event": "tool_call", "payload": {"tool_name": "lookup", "noise": False},
            }]}},
        )
        self.assertEqual(allowed["outcome"], 1.0)

        wrong_arguments = gate.apply(
            {"process": 0.0, "outcome": 1.0},
            {"trajectory": {"events": [{
                "event": "tool_call", "payload": {"tool_name": "lookup", "noise": False},
            }]}},
        )
        self.assertEqual(wrong_arguments["outcome"], 0.0)

    def test_reward_gate_enforces_multi_step_dependency_order(self):
        metrics = [{"id": "outcome", "category": "outcome", "score_range": [0, 1]}]
        gate = ContractRewardGate({
            "training_contract": {"category": "multi_step_agentic"},
            "capability_dag": {
                "nodes": ["lookup", "update"],
                "edges": [{"from_tool": "lookup", "to_tool": "update", "via": "record_id"}],
            },
        }, metrics)
        def context(names):
            return {"trajectory": {"events": [
                {"event": "tool_call", "payload": {"tool_name": name, "noise": False}}
                for name in names
            ]}}
        self.assertEqual(gate.apply({"outcome": 1.0}, context(["update", "lookup"]))["outcome"], 0.0)
        self.assertEqual(gate.apply({"outcome": 1.0}, context(["lookup", "update"]))["outcome"], 1.0)

    def test_reward_gate_removes_isolated_process_credit_before_chain_completion(self):
        metrics = [
            {"id": "lookup_process", "category": "process", "score_range": [0, 1]},
            {"id": "update_process", "category": "process", "score_range": [0, 1]},
            {"id": "outcome", "category": "outcome", "score_range": [0, 1]},
        ]
        gate = ContractRewardGate({
            "training_contract": {"category": "multi_step_agentic"},
            "capability_dag": {
                "nodes": ["lookup", "update"],
                "edges": [{"from_tool": "lookup", "to_tool": "update"}],
            },
        }, metrics)
        blocked = gate.apply(
            {"lookup_process": 0.0, "update_process": 1.0, "outcome": 0.0},
            {"trajectory": {"events": [{
                "event": "tool_call", "payload": {"tool_name": "update", "noise": False},
            }]}},
        )
        self.assertEqual(blocked, {
            "lookup_process": 0.0, "update_process": 0.0, "outcome": 0.0,
        })

    def test_reward_gate_leaves_unnecessary_tool_cost_to_penalty_metric(self):
        gate = ContractRewardGate({
            "training_contract": {"category": "direct_response"},
            "capability_dag": {"nodes": [], "edges": []},
        }, [{"id": "outcome", "category": "outcome", "score_range": [0, 1]}])
        result = gate.apply({"outcome": 1.0}, {"final_agent_response": "The answer is 42.", "trajectory": {"events": [{
            "event": "tool_call", "payload": {"tool_name": "roll_virtual_die", "noise": True},
        }]}})
        self.assertEqual(result["outcome"], 1.0)

    def test_direct_response_reward_requires_a_final_answer(self):
        gate = ContractRewardGate({
            "training_contract": {"category": "direct_response"},
            "capability_dag": {"nodes": [], "edges": []},
        }, [{"id": "outcome", "category": "outcome", "score_range": [0, 1]}])
        self.assertEqual(gate.apply(
            {"outcome": 1.0},
            {"final_agent_response": "", "trajectory": {"events": []}},
        )["outcome"], 0.0)

    def test_json_schema_validator_handles_nested_values(self):
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"enabled": {"type": "boolean"}},
                        "required": ["enabled"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["items"],
            "additionalProperties": False,
        }
        validate_json_schema(schema, {"items": [{"enabled": True}]})
        with self.assertRaises(SandboxError):
            validate_json_schema(schema, {"items": [{"enabled": "yes"}]})

    def test_schema_validates_typed_dynamic_object_fields(self):
        schema = {"type": "object", "additionalProperties": {"type": "number"}}
        validate_json_schema(schema, {"density": 2.15, "porosity": 0.3})
        with self.assertRaises(SandboxError):
            validate_json_schema(schema, {"density": "2.15"})

    def test_episode_reset_idempotency_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            episode = store.reset(episode_id="ep-1", seed=7, data_hash="data-v1")
            first = store.event("tool_call", {"name": "query"}, {"ok": True}, idem_key="req-1")
            second = store.event("tool_call", {"name": "query"}, {"ok": False}, idem_key="req-1")
            self.assertEqual(first, second)
            replay = store.replay()
            self.assertEqual(replay["episode_id"], episode.episode_id)
            self.assertEqual(replay["seed"], 7)
            self.assertEqual(len(replay["events"]), 1)
            self.assertTrue(replay["trace_hash"])
            store.set_state("user_state", {"node_id": "start", "known_facts": ["a"]})
            self.assertEqual(store.get_state("user_state")["node_id"], "start")
            store.reset(episode_id="ep-2", seed=7, data_hash="data-v1")
            self.assertEqual(store.get_state("user_state"), None)

    def test_reset_selects_older_episode_again(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EpisodeStore(Path(directory) / "episodes.sqlite3")
            store.reset(episode_id="older", seed=1)
            store.reset(episode_id="newer", seed=2)
            self.assertEqual(store.current().episode_id, "newer")
            store.reset(episode_id="older", seed=3)
            self.assertEqual(store.current().episode_id, "older")
            self.assertEqual(store.current().seed, 3)

    def test_trainer_authentication(self):
        previous = os.environ.get("SANDBOX_TRAINER_API_KEY")
        os.environ["SANDBOX_TRAINER_API_KEY"] = "test-secret"
        try:
            require_trainer("Bearer test-secret")
            with self.assertRaises(SandboxError) as context:
                require_trainer("Bearer wrong")
            self.assertEqual(context.exception.status, 401)
        finally:
            if previous is None:
                os.environ.pop("SANDBOX_TRAINER_API_KEY", None)
            else:
                os.environ["SANDBOX_TRAINER_API_KEY"] = previous

if __name__ == "__main__":
    unittest.main()
