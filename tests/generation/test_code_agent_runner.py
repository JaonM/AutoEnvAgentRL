import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from env_factory.generation.code_agent import generate
from env_factory.generation.pipeline_errors import PipelineGenerationError
from env_factory.generation.semantic_review import SemanticReviewUnavailable


class CodeAgentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.review_patch = patch("env_factory.generation.code_agent.review_source", return_value={
            "status": "pass", "findings": [], "provenance": {
                "completed_turns": 1, "usage": {}, "events_sha256": "a" * 64}})
        self.review = self.review_patch.start()
        self.addCleanup(self.review_patch.stop)
    def test_invalid_json_gets_one_targeted_repair_with_shared_budget_and_both_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clock = [0.0]
            prompts, budgets = [], []
            def launch(command, **kwargs):
                def finish(prompt, timeout):
                    prompts.append(prompt)
                    budgets.append(timeout)
                    workspace = root / "authoring"
                    (workspace / "source.json").write_text('{"broken":' if len(prompts) == 1 else '{"version":"1.0"}')
                    kwargs["stdout"].write('{"type":"turn.completed","usage":{"output_tokens":50}}\n')
                    clock[0] += 200
                return Mock(returncode=0, communicate=Mock(side_effect=finish))
            with patch("env_factory.generation.code_agent.subprocess.Popen", side_effect=launch), patch(
                    "env_factory.generation.code_agent.time.monotonic", side_effect=lambda: clock[0]), patch(
                    "env_factory.generation.code_agent.compile_source", return_value={"generation_pipeline": {}}) as compiler, patch(
                    "env_factory.generation.code_agent.verify_delivery"):
                result = generate(request={"training_category": "multi_step_agentic"}, artifact_dir=root)
            self.assertEqual(budgets, [600, 400])
            self.assertIn("parent_validation.json", prompts[1])
            compiler.assert_called_once()
            evidence = json.loads((root / "code_agent_generation.json").read_text())
            self.assertEqual(evidence["agent_invocations"], 3)
            self.assertEqual(evidence["completed_turns"], 3)
            self.assertEqual(evidence["usage"]["output_tokens"], 100)
            self.assertEqual([x["status"] for x in evidence["attempts"]], ["validation_failed", "passed"])
            self.assertEqual((root / "code_agent_attempts/attempt-1/source.json").read_text(), '{"broken":')
            self.assertEqual(result["generation_pipeline"]["agent_invocations"], 3)

    def test_delivery_failure_is_returned_for_repair_but_never_retried_indefinitely(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def launch(command, **kwargs):
                def finish(prompt, timeout):
                    (root / "authoring/source.json").write_text('{"version":"1.0"}')
                    kwargs["stdout"].write('{"type":"turn.completed"}\n')
                return Mock(returncode=0, communicate=Mock(side_effect=finish))
            with patch("env_factory.generation.code_agent.subprocess.Popen", side_effect=launch) as popen, patch(
                    "env_factory.generation.code_agent.compile_source", return_value={"generation_pipeline": {}}), patch(
                    "env_factory.generation.code_agent.verify_delivery", side_effect=ValueError("reward not sensitive")), self.assertRaisesRegex(
                    PipelineGenerationError, "reward not sensitive"):
                generate(request={}, artifact_dir=root)
            self.assertEqual(popen.call_count, 2)
            self.assertIn("reward not sensitive", (root / "authoring/parent_validation.json").read_text())
            evidence = json.loads((root / "code_agent_generation.json").read_text())
            self.assertEqual(evidence["status"], "failed")
            self.assertEqual(evidence["attempts"][-1]["status"], "failed")

    def test_semantic_finding_repairs_same_source_and_requires_new_review(self):
        passed = self.review.return_value
        failed = {**passed, "status": "fail", "findings": [{"code": "self_attestation",
            "reason": "Boolean flags replace requested facts", "repair": "Return the facts"}]}
        self.review.side_effect = [failed, passed]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def launch(command, **kwargs):
                def finish(prompt, timeout):
                    (root / "authoring/source.json").write_text('{"version":"1.0"}')
                    kwargs["stdout"].write('{"type":"turn.completed"}\n')
                return Mock(returncode=0, communicate=Mock(side_effect=finish))
            with patch("env_factory.generation.code_agent.subprocess.Popen", side_effect=launch) as popen, patch(
                    "env_factory.generation.code_agent.compile_source", return_value={"generation_pipeline": {}}), patch(
                    "env_factory.generation.code_agent.verify_delivery"):
                result = generate(request={"training_category": "direct_response"}, artifact_dir=root)
            self.assertEqual(popen.call_count, 2)
            self.assertEqual(self.review.call_count, 2)
            self.assertIn("self_attestation", (root / "authoring/parent_validation.json").read_text())
            self.assertEqual(result["generation_pipeline"]["agent_invocations"], 4)
            self.assertEqual(result["generation_pipeline"]["review_invocations"], 2)

    def test_structure_repair_does_not_consume_semantic_repair(self):
        passed = self.review.return_value
        self.review.side_effect = [{**passed, "status": "fail", "findings": [{"code": "wrong_rule"}]}, passed]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launches = []
            def launch(command, **kwargs):
                def finish(prompt, timeout):
                    launches.append(prompt)
                    (root / "authoring/source.json").write_text('{"broken":' if len(launches) == 1 else '{"version":"1.0"}')
                    kwargs["stdout"].write('{"type":"turn.completed"}\n')
                return Mock(returncode=0, communicate=Mock(side_effect=finish))
            with patch("env_factory.generation.code_agent.subprocess.Popen", side_effect=launch), patch(
                    "env_factory.generation.code_agent.compile_source", return_value={"generation_pipeline": {}}), patch(
                    "env_factory.generation.code_agent.verify_delivery"):
                result = generate(request={}, artifact_dir=root)
            self.assertEqual(len(launches), 3)
            self.assertEqual(result["generation_pipeline"]["repair_budget"]["used"], {"structural": 1, "semantic": 1})
            self.assertIn("semantic phase", launches[2])

    def test_repeated_semantic_rejection_stops_after_one_semantic_repair(self):
        self.review.return_value = {**self.review.return_value, "status": "fail", "findings": [{"code": "wrong_rule"}]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def launch(command, **kwargs):
                def finish(prompt, timeout):
                    (root / "authoring/source.json").write_text('{"version":"1.0"}')
                    kwargs["stdout"].write('{"type":"turn.completed"}\n')
                return Mock(returncode=0, communicate=Mock(side_effect=finish))
            with patch("env_factory.generation.code_agent.subprocess.Popen", side_effect=launch) as popen, patch(
                    "env_factory.generation.code_agent.compile_source", return_value={"generation_pipeline": {}}), patch(
                    "env_factory.generation.code_agent.verify_delivery"), self.assertRaisesRegex(
                    PipelineGenerationError, "SOURCE_SEMANTIC_REVIEW_FAILED"):
                generate(request={}, artifact_dir=root)
            self.assertEqual(popen.call_count, 2)
            evidence = json.loads((root / "code_agent_generation.json").read_text())
            self.assertEqual(evidence["repair_budget"]["used"], {"structural": 0, "semantic": 1})

    def test_reviewer_infrastructure_failure_does_not_trigger_author_repair(self):
        self.review.side_effect = SemanticReviewUnavailable("SOURCE_REVIEW_TIMEOUT")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def launch(command, **kwargs):
                def finish(prompt, timeout):
                    (root / "authoring/source.json").write_text('{"version":"1.0"}')
                    kwargs["stdout"].write('{"type":"turn.completed"}\n')
                return Mock(returncode=0, communicate=Mock(side_effect=finish))
            with patch("env_factory.generation.code_agent.subprocess.Popen", side_effect=launch) as popen, patch(
                    "env_factory.generation.code_agent.compile_source", return_value={"generation_pipeline": {}}), patch(
                    "env_factory.generation.code_agent.verify_delivery"), self.assertRaisesRegex(
                    PipelineGenerationError, "SOURCE_REVIEW_TIMEOUT"):
                generate(request={}, artifact_dir=root)
            self.assertEqual(popen.call_count, 1)

    def test_parent_recompiles_source_instead_of_trusting_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock(returncode=0)
            def finish(prompt, timeout):
                workspace = root / "authoring"
                (workspace / "source.json").write_text('{"version":"1.0"}')
                (root / "code_agent_attempts/attempt-1/events.jsonl").write_text('{"type":"turn.completed","usage":{"output_tokens":50}}\n')
                (workspace / "compiled_artifacts.json").write_text('{"forged":true}')
            process.communicate.side_effect = finish
            with patch("env_factory.generation.code_agent.subprocess.Popen", return_value=process) as popen, patch(
                    "env_factory.generation.code_agent.compile_source", return_value={"generation_pipeline": {}}) as compiler, patch(
                    "env_factory.generation.code_agent.verify_delivery") as delivery:
                result = generate(request={"training_category": "direct_response"}, artifact_dir=root)
            self.assertEqual(compiler.call_args.args[0], {"version": "1.0"})
            self.assertEqual(compiler.call_args.kwargs["root"], root.resolve())
            self.assertIn("gpt-6-luna", popen.call_args.args[0])
            self.assertEqual(result["generation_pipeline"]["completed_turns"], 2)
            self.assertIsNone(result["generation_pipeline"]["llm_calls"])
            delivery.assert_called_once()

    def test_cli_exit_zero_without_completed_model_turn_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock(returncode=0)
            with patch("env_factory.generation.code_agent.subprocess.Popen", return_value=process), self.assertRaisesRegex(
                    PipelineGenerationError, "NO_COMPLETION"):
                generate(request={}, artifact_dir=root)
            self.assertEqual(json.loads((root / "code_agent_generation.json").read_text())["status"], "failed")

    def test_timeout_terminates_process_group_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock(pid=12345)
            process.communicate.side_effect = subprocess.TimeoutExpired("codex", .1)
            with patch("env_factory.generation.code_agent.subprocess.Popen", return_value=process), patch(
                    "env_factory.generation.code_agent.os.killpg") as kill, self.assertRaisesRegex(
                    PipelineGenerationError, "CODE_AGENT_TIMEOUT"):
                generate(request={}, artifact_dir=root, timeout=.1)
            kill.assert_called_once()
            self.assertEqual(json.loads((root / "code_agent_generation.json").read_text())["status"], "failed")

    def test_protected_request_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = Mock(returncode=0)
            def finish(prompt, timeout):
                workspace = root / "authoring"
                (root / "code_agent_attempts/attempt-1/events.jsonl").write_text('{"type":"turn.completed"}\n')
                (workspace / "request.json").write_text('{}')
            process.communicate.side_effect = finish
            with patch("env_factory.generation.code_agent.subprocess.Popen", return_value=process), self.assertRaisesRegex(
                    PipelineGenerationError, "CONTRACT_CHANGED"):
                generate(request={"training_category": "direct_response"}, artifact_dir=root)


if __name__ == "__main__":
    unittest.main()
