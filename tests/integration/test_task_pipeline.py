import importlib.util
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from env_factory.tasks.task_portability import prepare_sandbox_task
from env_factory.sandbox_scoring import SCORE_RUBRIC, evidence_fingerprint


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    path = ROOT / "scripts" / f"{name}.py"
    if not path.is_file():
        path = ROOT / "scripts" / "rollout" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


from env_factory import task_pipeline as loop
rollout = load("run_live_rollout")


class PipelineBuildTest(unittest.TestCase):
    def test_direct_replay_gates_failed_delivery_score(self):
        with patch.object(loop, "_build_one_unfinalized", return_value={
            "passed": False, "score": 9.44,
            "offline_score": 9.38, "failure_class": "live_reward_calibration",
        }):
            result = loop._build_one(ROOT, Path("task.json"), Path("sandbox"), {})
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["diagnostic_score"], 9.44)
        self.assertEqual(result["delivery_score_kind"], "gated_final_10_point")

    def test_failed_delivery_has_zero_effective_score(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            loop, "_build_one", return_value={
                "passed": False, "score": 9.44,
                "offline_score": 9.38, "failure_class": "live_reward_calibration",
            },
        ):
            result = loop.build_one(
                ROOT, Path(directory) / "task.json", Path(directory) / "sandbox",
                {"sandbox_runtime": "none"},
            )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["diagnostic_score"], 9.44)
        self.assertEqual(result["offline_score"], 9.38)
        self.assertEqual(result["delivery_score_kind"], "gated_final_10_point")

    def test_training_ready_is_published_after_final_cleanup(self):
        for cleanup_code in (0, 1):
            with self.subTest(cleanup_code=cleanup_code), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                (output / "status.json").write_text(json.dumps({
                    "status": "success", "success": True, "exit_code": 0}))
                verified = {"passed": True, "score": 10, "build": {"exit_code": 0},
                    "live_rollout_verified": True, "live_reward_calibration_verified": True,
                    "data_governance_verified": True, "trajectory_privacy_verified": True,
                    "sandbox_score": {"passed": True}}
                with patch.object(loop, "_build_one", return_value=verified), patch.object(
                    loop, "run_process", return_value={"exit_code": cleanup_code}):
                    result = loop.build_one(ROOT, output / "task.json", output,
                                            {"sandbox_runtime": "docker"})
                self.assertEqual(result["training_ready"], cleanup_code == 0)
                self.assertEqual(json.loads((output / "status.json").read_text())["training_ready"], cleanup_code == 0)

    def test_builder_owner_codes_survive_failure_classification(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for phase, code, target in (
                ("task_contract_rejected", "TASK_CONTRACT", "task_pipeline"),
                ("platform_contract_rejected", "PLATFORM_RUNTIME", "platform_runtime_boundary"),
                ("runtime_timeout_rejected", "BUILD_RUNTIME_TIMEOUT", "platform_or_task_contract"),
                ("repair_no_progress", "BUILD_REPAIR_NO_PROGRESS", "sandbox_builder"),
                ("repair_budget_exhausted", "BUILD_REPAIR_BUDGET", "sandbox_builder"),
            ):
                (output / "status.json").write_text(json.dumps({
                    "status": "failed", "failed_phase": phase,
                    "failure_code": code,
                }), encoding="utf-8")
                result = loop.classify_build_failure(output)
                self.assertEqual(result["failure_code"], code)
                self.assertEqual(result["repair_target"], target)

    @staticmethod
    def write_task_lineage(output: Path, task_path: Path) -> None:
        prepare_sandbox_task(task_path, output)

    @staticmethod
    def mock_sandbox_score(output: Path, threshold: float = 8) -> dict:
        return {
            "checks": [{"name": name, "weight": weight, "passed": True,
                        "evidence": "verified", "critical": critical}
                       for name, weight, critical in SCORE_RUBRIC],
            "mode": "offline_executable", "network_used": False, "model_used": False,
            "live_rollout_verified": False, "threshold": threshold,
            "score": 10.0, "eligible": True, "passed": True,
            "failed_critical_gates": [],
            "model": loop.MODEL, "review_model": loop.MODEL,
            "evidence_fingerprint": evidence_fingerprint(output, ROOT),
        }

    def test_concurrent_attempts_receive_distinct_stable_container_tags(self):
        first = loop.sandbox_image_tag(Path("/tmp/experiment/sample-1/attempt-1"))
        second = loop.sandbox_image_tag(Path("/tmp/experiment/sample-2/attempt-1"))
        self.assertEqual(first, loop.sandbox_image_tag(
            Path("/tmp/experiment/sample-1/attempt-1")
        ))
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^envfactory-sandbox-[0-9a-f]{20}$")



    def test_timeout_and_nonzero_exit_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = loop.run_process([sys.executable, "-c", "import time; time.sleep(10)"], root, root / "timeout.log", .05)
            self.assertEqual(result["exit_code"], 124)
            self.assertTrue(result["timed_out"])
            result = loop.run_process([sys.executable, "-c", "raise SystemExit(7)"], root, root / "exit.log", 5)
            self.assertEqual(result["exit_code"], 7)
            self.assertFalse(loop.ACTIVE_PROCESSES)











    def test_failed_manifest_waits_for_failure_report_before_classification(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "task-1"
            task_dir.mkdir()
            manifest_path = task_dir / "sample_manifest.json"
            sample = {"batch_index": 1, "status": "failed", "training_category": "simple_agentic"}
            manifest_path.write_text(json.dumps(sample), encoding="utf-8")
            self.assertIsNone(loop.resolve_generated_job(
                1, (manifest_path, sample), generation_done=False,
            ))
            (task_dir / "failure.json").write_text(json.dumps({
                "failure_class": "GEN_SCHEMA", "message": "bad schema",
            }), encoding="utf-8")
            resolved = loop.resolve_generated_job(
                1, (manifest_path, sample), generation_done=False,
            )
            self.assertEqual(resolved["result"]["failure_code"], "GEN_SCHEMA")




    def test_buildability_failure_is_not_mislabeled_as_luna_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })
            def failed_build(*args, **kwargs):
                output.mkdir(exist_ok=True)
                (output / "buildability.json").write_text(json.dumps({
                    "buildable": False,
                    "issues": [{"code": "EXTERNAL_CAPABILITY_UNAVAILABLE"}],
                }))
                return {"exit_code": 4, "timed_out": False, "seconds": .1}
            config = {"score_threshold": 8, "max_attempts": 2, "build_timeout": 10,
                      "build_mode": "clean"}
            with (
                patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                patch.object(loop, "run_process", side_effect=failed_build),
            ):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertEqual(result["failure_class"], "task_quality")
            self.assertEqual(result["failure_code"], "EXTERNAL_CAPABILITY_UNAVAILABLE")

    def test_infrastructure_disconnect_retries_with_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}", encoding="utf-8")
            output = root / "sandbox"
            calls = []
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })

            def disconnected(command, *_args):
                calls.append(command)
                output.mkdir(exist_ok=True)
                (output / "status.json").write_text(json.dumps({
                    "status": "failed",
                    "failure_code": "INFRA", "failure_category": "infrastructure",
                    "failed_phase": "node_development",
                }), encoding="utf-8")
                return {"exit_code": 5, "timed_out": False, "seconds": .1}

            config = {
                "score_threshold": 8, "max_attempts": 2, "build_timeout": 10,
                "build_mode": "clean", "infrastructure_retries": 1,
            }
            with patch("env_factory.tasks.task_quality.score_file", return_value=quality), \
                    patch.object(loop, "run_process", side_effect=disconnected):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertEqual(len(calls), 2)
            self.assertNotIn("--resume", calls[0])
            self.assertIn("--resume", calls[1])
            self.assertEqual(result["failure_code"], "INFRA")


    def test_build_failure_uses_last_failed_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })

            def failed_build(*args, **kwargs):
                output.mkdir(exist_ok=True)
                (output / "status.json").write_text(json.dumps({
                    "status": "failed", "phase": "failed",
                    "failed_phase": "runtime_validation",
                }))
                return {"exit_code": 5, "timed_out": False, "seconds": .1}

            config = {"score_threshold": 8, "max_attempts": 2, "build_timeout": 10,
                      "build_mode": "clean"}
            with (
                patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                patch.object(loop, "run_process", side_effect=failed_build),
            ):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertEqual(result["failure_class"], "build")
            self.assertEqual(result["build_failed_phase"], "runtime_validation")
            self.assertEqual(result["failure_code"], "BUILD_RUNTIME_INTEGRITY")
            self.assertEqual(result["repair_target"], "platform_runtime_boundary")

    def test_scoring_requires_fresh_report_bound_to_current_sandbox(self):
        quality = SimpleNamespace(to_dict=lambda: {
            "eligible": True, "score": 9, "training_category": "simple_agentic",
        })
        for case in ("missing", "wrong_fingerprint", "bare_score"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                task_path = root / "task.json"
                task_path.write_text("{}")
                output = root / "sandbox"
                output.mkdir()
                report_path = output / "score_summary.json"
                report_path.write_text('{"sandboxes":[{"passed":true,"score":10}]}')

                def process(command, cwd, log, timeout):
                    rendered = " ".join(command)
                    if "develop_sandbox_with_agent.sh" in rendered:
                        self.write_task_lineage(output, task_path)
                    if "score_sandbox_offline.py" in rendered:
                        self.assertFalse(report_path.exists())
                        if case == "wrong_fingerprint":
                            report = self.mock_sandbox_score(output)
                            report["evidence_fingerprint"] = "stale"
                            report_path.write_text(json.dumps({"sandboxes": [report]}))
                        if case == "bare_score":
                            report_path.write_text(json.dumps({"sandboxes": [{
                                "passed": True, "score": 10,
                                "model": loop.MODEL, "review_model": loop.MODEL,
                                "evidence_fingerprint": evidence_fingerprint(output, ROOT),
                            }]}))
                    return {"exit_code": 0, "timed_out": False, "seconds": .1}

                config = {"score_threshold": 8, "max_attempts": 2, "build_timeout": 10,
                          "score_timeout": 10, "build_mode": "clean", "validation": "offline"}
                with (
                    patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                    patch.object(loop, "run_process", side_effect=process),
                ):
                    result = loop.build_one(ROOT, task_path, output, config)
                self.assertFalse(result["passed"])
                self.assertEqual(result["failure_class"], "offline_validation")
                if case == "missing":
                    self.assertFalse(report_path.exists())


    def test_executable_build_qualifies_without_live_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            output.mkdir()
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })
            commands = []
            def process(command, cwd, log, timeout):
                commands.append(command)
                rendered = " ".join(command)
                if "develop_sandbox_with_agent.sh" in rendered:
                    self.write_task_lineage(output, task_path)
                    (output / "status.json").write_text(json.dumps({
                        "status": "success", "success": True, "exit_code": 0}))
                elif "score_sandbox_offline.py" in rendered:
                    (output / "score_summary.json").write_text(json.dumps({
                        "sandboxes": [self.mock_sandbox_score(output)],
                    }))
                elif "audit_data_governance.py" in rendered:
                    (output / "data_governance.json").write_text(json.dumps({
                        "eligible_for_external_model_processing": True,
                    }))
                else:
                    self.assertEqual(command[:3], ["docker", "image", "rm"])
                return {"exit_code": 0, "timed_out": False, "seconds": .1}
            config = {"score_threshold": 8, "max_attempts": 2, "build_timeout": 10,
                      "score_timeout": 10, "sandbox_runtime": "docker"}
            with patch("env_factory.tasks.task_quality.score_file", return_value=quality), \
                    patch.object(loop, "run_process", side_effect=process):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertTrue(result["training_ready"])
            self.assertEqual(result["score"], result["offline_score"])
            self.assertEqual(result["qualification_basis"], "executable_v1")
            for name in ("live_rollout", "live_reward_calibration", "trajectory_privacy"):
                self.assertFalse(result[name + "_verified"])
                self.assertEqual(result["validation_checks"][name], "not_run")
            self.assertFalse(any("run_live_rollout.py" in " ".join(c) for c in commands))

    def test_data_governance_failure_prevents_live_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task_path = root / "task.json"
            task_path.write_text("{}")
            output = root / "sandbox"
            output.mkdir()
            quality = SimpleNamespace(to_dict=lambda: {
                "eligible": True, "score": 9, "training_category": "simple_agentic",
            })
            commands = []

            def process(command, cwd, log, timeout):
                commands.append(command)
                rendered = " ".join(command)
                if "develop_sandbox_with_agent.sh" in rendered:
                    self.write_task_lineage(output, task_path)
                if "score_sandbox_offline.py" in rendered:
                    (output / "score_summary.json").write_text(json.dumps({
                        "sandboxes": [self.mock_sandbox_score(output)],
                    }))
                    return {"exit_code": 0, "timed_out": False, "seconds": .1}
                if "audit_data_governance.py" in rendered:
                    (output / "data_governance.json").write_text(json.dumps({
                        "eligible_for_external_model_processing": False,
                        "credential_findings": [{"kind": "private_key", "path": "$.data"}],
                    }))
                    return {"exit_code": 1, "timed_out": False, "seconds": .1}
                return {"exit_code": 0, "timed_out": False, "seconds": .1}

            config = {
                "score_threshold": 8, "max_attempts": 2, "build_timeout": 10,
                "score_timeout": 10, "rollout_timeout": 10,
                "build_mode": "clean", "validation": "live",
                "rollout_episodes": 3, "rollout_steps": 20,
                "rollout_min_success_rate": 0,
            }
            with (
                patch("env_factory.tasks.task_quality.score_file", return_value=quality),
                patch.object(loop, "run_process", side_effect=process),
            ):
                result = loop.build_one(ROOT, task_path, output, config)
            self.assertFalse(result["passed"])
            self.assertEqual(result["failure_class"], "data_governance")
            self.assertFalse(any(
                "run_live_rollout.py" in " ".join(command) for command in commands
            ))






class FakeApp:
    def __init__(self, *, correct=True, fallback=False):
        self.correct, self.fallback = correct, fallback
        self.changed = False

    def business_snapshot(self):
        return {"items": [{"id": 1, "done": self.changed}]}

    def handle(self, method, path, body=None, headers=None):
        if path == "/v1/reset":
            self.changed = False
            value = {}
        elif path == "/v1/tools":
            value = {"tools": [{"type": "function", "function": {"name": "finish", "parameters": {"type": "object"}}}]}
        elif path == "/v1/state":
            value = {"business_state": self.business_snapshot()}
        elif path == "/v1/tools/finish":
            self.changed = self.correct
            value = {"updated": 1}
        elif path == "/v1/reward":
            value = {"reward": 1 if getattr(self, "answered", False) else 0}
        elif path == "/v1/agent_response":
            self.answered = True
            value = {"accepted": True}
        elif path == "/v1/user_simulator":
            value = {"user_query": "accepted", "should_end": True, "termination_reason": "completed"}
        elif path == "/v1/replay":
            value = {"events": [{"payload": {"used_fallback": self.fallback}}]}
        else:
            value = {}
        return 200, value, {}


class RolloutTest(unittest.TestCase):
    def test_live_transport_reuses_key_after_declared_response_loss(self):
        class FaultApp(FakeApp):
            keys = None
            def handle(self, method, path, body=None, headers=None):
                if path == '/v1/tools/finish':
                    if self.keys is None:
                        self.keys = []
                    self.keys.append(headers.get('Idempotency-Key'))
                    self.changed = True
                    if len(self.keys) == 1:
                        return 504, {'error': {'code': 'RESPONSE_LOST'}}, {}
                return super().handle(method, path, body, headers)
        app = FaultApp()
        actions = iter([{'kind': 'tool', 'name': 'finish', 'arguments': {}}] * 2
                       + [{'kind': 'respond', 'content': 'done'}])
        client = SimpleNamespace(chat=lambda *a, **k: SimpleNamespace(
            content=json.dumps(next(actions)), model='test', id='test', finish_reason='stop', usage={}))
        task = {'task': 'finish', 'business_lifecycle': {'dynamics': {'faults': [
            {'tool': 'finish', 'status': 504, 'phase': 'after_commit'}]}}}
        with patch.dict(os.environ, {'SANDBOX_TRAINER_API_KEY': 'test'}):
            result = rollout.episode(app, task, client, 0, 4)
        self.assertTrue(result['agent_success'])
        self.assertTrue(app.keys[0])
        self.assertEqual(app.keys[0], app.keys[1])
        self.assertEqual(result['transitions'][0]['result']['status'], 504)
        user_request = next(step for step in result['trajectory']
                            if step['path'] == '/v1/user_simulator')
        self.assertEqual(user_request['body']['messages'][-1],
                         {'role': 'assistant', 'content': 'done'})
        self.assertNotIn({'role': 'user', 'content': 'accepted'},
                         user_request['body']['messages'])

    def test_live_transport_does_not_hide_undeclared_server_errors(self):
        class BrokenApp(FakeApp):
            def handle(self, method, path, body=None, headers=None):
                if path == '/v1/tools/finish':
                    return 500, {'error': {'code': 'INTERNAL_ERROR'}}, {}
                return super().handle(method, path, body, headers)
        client = SimpleNamespace(chat=lambda *a, **k: SimpleNamespace(
            content=json.dumps({'kind': 'tool', 'name': 'finish', 'arguments': {}}),
            model='test', id='test', finish_reason='stop', usage={}))
        with patch.dict(os.environ, {'SANDBOX_TRAINER_API_KEY': 'test'}):
            with self.assertRaisesRegex(RuntimeError, 'status=500'):
                rollout.episode(BrokenApp(), {'task': 'finish'}, client, 0, 2)

    def test_http_sandbox_client_only_accepts_loopback_origins(self):
        self.assertEqual(
            rollout.HTTPSandboxClient("http://127.0.0.1:8000").base_url,
            "http://127.0.0.1:8000",
        )
        for value in (
            "https://127.0.0.1:8000",
            "http://sandbox.example:8000",
            "http://user:secret@127.0.0.1:8000",
            "http://127.0.0.1:8000/path",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rollout.HTTPSandboxClient(value)

    def test_release_gate_requires_two_of_three_clean_successes(self):
        episodes = [
            {"agent_success": True, "issues": []},
            {"agent_success": True, "issues": []},
            {"agent_success": False, "issues": []},
        ]
        report = rollout.summarize_episodes(episodes, 2 / 3)
        self.assertTrue(report["passed"])
        self.assertEqual(report["quality_score"], 1.0)
        episodes[1]["agent_success"] = False
        report = rollout.summarize_episodes(episodes, 2 / 3)
        self.assertTrue(report["passed"])
        self.assertTrue(report["environment_qualified"])
        self.assertFalse(report["agent_policy_qualified"])
        self.assertIsNone(report["failure_owner"])
        episodes[1]["issues"] = ["runtime_llm_fallback"]
        report = rollout.summarize_episodes(episodes, 0)
        self.assertFalse(report["all_episodes_fallback_free"])
        self.assertEqual(report["failure_owner"], "environment")

    def run_episode(self, app):
        task = {"task": "finish item", "acceptance_contract": {"secret": "do not show"},
                "public_input": {"initial_user_message": "finish the supplied item", "materials": [{
                    "name": "item", "mime_type": "application/json", "content": '{"id": 1}'
                }]},
                "task_spec": {"goal_contract": {"row_predicates": [{"table": "items", "where": {"id": 1},
                       "values": {"done": True}, "count": 1}], "requires_state_change": True}}}
        calls = []
        actions = iter([{"kind": "tool", "name": "finish", "arguments": {}}, {"kind": "respond", "content": "done"}])
        def chat(messages, **kwargs):
            calls.append(json.dumps(messages))
            return SimpleNamespace(
                content=json.dumps(next(actions)),
                model="actual-policy-snapshot",
                id="response-id",
                finish_reason="stop",
                usage={"total_tokens": 10},
            )
        with patch.dict(os.environ, {"SANDBOX_TRAINER_API_KEY": "test"}):
            result = rollout.episode(app, task, SimpleNamespace(chat=chat), 0, 4)
        self.assertTrue(all("do not show" not in call for call in calls))
        self.assertTrue(any("Public materials" in call and "application/json" in call for call in calls))
        return result

    def test_live_agent_uses_only_public_inputs_and_records_usage(self):
        result = self.run_episode(FakeApp())
        self.assertTrue(result["agent_success"])
        self.assertEqual(
            sum(item["token_usage"]["total_tokens"] for item in result["usage"]),
            20,
        )
        self.assertEqual(
            {item["response_model"] for item in result["usage"]},
            {"actual-policy-snapshot"},
        )
        self.assertEqual(result["schema_version"], "2.0")
        self.assertEqual(len(result["transitions"]), 2)
        self.assertEqual(result["transitions"][0]["action"]["kind"], "tool")
        self.assertIsInstance(result["transitions"][0]["next_observation"], dict)
        self.assertTrue(result["transitions"][-1]["terminated"])
        self.assertEqual(
            result["transitions"][-1]["result"],
            {"status": 200, "user_query": "accepted"},
        )
        self.assertTrue(
            result["transitions"][-1]["trainer_metadata"]["user_simulator"][
                "should_end"
            ]
        )

    def test_high_reward_with_wrong_state_is_rejected(self):
        result = self.run_episode(FakeApp(correct=False))
        self.assertFalse(result["agent_success"])
        self.assertIn("reward_state_disagreement", result["issues"])

    def test_runtime_fallback_is_not_live_success(self):
        result = self.run_episode(FakeApp(fallback=True))
        self.assertFalse(result["agent_success"])
        self.assertIn("runtime_llm_fallback", result["issues"])
