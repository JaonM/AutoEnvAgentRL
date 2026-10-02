import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from env_factory.generation.semantic_review import (CHECKS, SemanticReviewUnavailable,
    review_source, validate_report)


def passing_report():
    return {"status": "pass", "checks": {name: {"passed": True, "evidence": "Concrete source evidence."}
            for name in CHECKS}, "findings": []}


class SemanticReviewTest(unittest.TestCase):
    def test_privacy_review_is_required_even_when_other_checks_pass(self):
        report = passing_report()
        report["checks"].pop("interaction_information_privacy")
        with self.assertRaisesRegex(SemanticReviewUnavailable, "incomplete checks"):
            validate_report(report)

    def test_privacy_failure_cannot_be_overridden_by_other_passing_checks(self):
        report = passing_report()
        report["checks"]["interaction_information_privacy"] = {
            "passed": False,
            "evidence": "compiled_public_input.initial_user_message already states the final choice."}
        report["findings"] = [{
            "code": "selected_option_disclosed",
            "source_paths": ["compiled_public_input.initial_user_message"],
            "reason": "The initial message explicitly selects the private option before stage 0.",
            "repair": "Keep the candidate options public and disclose the selection at stage 0."}]
        with self.assertRaisesRegex(SemanticReviewUnavailable, "contradicts"):
            validate_report(report)
        report["status"] = "fail"
        self.assertEqual(validate_report(report)["status"], "fail")

    def test_missing_checks_and_unsupported_verdicts_fail_closed(self):
        for mutate in (lambda r: r["checks"].pop("answerability"),
                       lambda r: r.update(status="fail"),
                       lambda r: r["checks"]["answerability"].update(passed=False),
                       lambda r: r["checks"]["answerability"].update(evidence="")):
            report = passing_report()
            mutate(report)
            with self.assertRaises(SemanticReviewUnavailable):
                validate_report(report)

    def test_actionable_semantic_rejection_is_preserved(self):
        report = passing_report()
        report["status"] = "fail"
        report["checks"]["answerability"]["passed"] = False
        report["findings"] = [{"code": "hidden_capacity", "source_paths": ["source.business_tools[0]"],
            "reason": "No tool returns requested capacity.", "repair": "Expose capacity in the business interface."}]
        self.assertEqual(validate_report(report), report)

    def test_real_cli_boundary_uses_readonly_mode_and_records_completed_turn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def launch(command, **kwargs):
                self.assertIn("read-only", command)
                schema_path = Path(command[command.index("--output-schema") + 1])
                schema = json.loads(schema_path.read_text())
                self.assertEqual(set(schema["required"]), {"status", "checks", "findings"})
                self.assertFalse(schema["additionalProperties"])
                self.assertEqual(set(schema["properties"]["checks"]["required"]), set(CHECKS))
                self.assertEqual(schema["properties"]["status"]["enum"], ["pass", "fail"])
                def finish(prompt, timeout):
                    self.assertIn("finite scripted environment", prompt)
                    self.assertIn("not an imagined unimplemented reply", prompt)
                    self.assertIn("interaction_information_privacy", prompt)
                    self.assertIn("A substring match alone is not semantic evidence", prompt)
                    self.assertIn("<candidate_json>", prompt)
                    self.assertIn('"compiled_reward_rules"', prompt)
                    (root / "response.json").write_text(json.dumps(passing_report()))
                    kwargs["stdout"].write('{"type":"turn.completed","usage":{"output_tokens":32}}\n')
                return Mock(returncode=0, communicate=Mock(side_effect=finish))
            with patch("env_factory.generation.semantic_review.subprocess.Popen", side_effect=launch):
                result = review_source(source={}, request={}, artifacts={"public_input": {}, "tools": [],
                    "metrics": [], "metric_implementations": []}, root=root, model="gpt-6-luna", timeout=60)
            self.assertEqual(result["provenance"]["completed_turns"], 1)
            self.assertEqual(result["provenance"]["usage"]["output_tokens"], 32)
            self.assertIn("not_executable_proof", result["provenance"]["scope"])

    def test_unavailable_candidate_is_not_an_author_semantic_defect(self):
        report = passing_report()
        report["status"] = "fail"
        report["checks"]["answerability"]["passed"] = False
        report["findings"] = [{"code": "REVIEW_SOURCE_UNAVAILABLE",
            "source_paths": ["candidate.json"], "reason": "Cannot inspect candidate.",
            "repair": "Provide candidate."}]
        with self.assertRaisesRegex(SemanticReviewUnavailable, "could not inspect"):
            validate_report(report)

    def test_timeout_terminates_reviewer_process_group(self):
        with tempfile.TemporaryDirectory() as directory:
            process = Mock(pid=4567)
            process.communicate.side_effect = subprocess.TimeoutExpired("codex", 1)
            with patch("env_factory.generation.semantic_review.subprocess.Popen", return_value=process), patch(
                    "env_factory.generation.semantic_review.os.killpg") as kill, self.assertRaisesRegex(
                    SemanticReviewUnavailable, "SOURCE_REVIEW_TIMEOUT"):
                review_source(source={}, request={}, artifacts={"public_input": {}, "tools": [],
                    "metrics": [], "metric_implementations": []}, root=Path(directory), model="gpt-6-luna", timeout=1)
            kill.assert_called_once()
