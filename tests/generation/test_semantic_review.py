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
                    self.assertIn('"compiled_tool_implementations"', prompt)
                    self.assertIn("compiler expands source.business_conditions", prompt)
                    self.assertIn('"compiled_reference_scenarios"', prompt)
                    self.assertIn("superseded value", prompt)
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


def test_invalid_report_gets_one_repair_with_shared_budget(tmp_path, monkeypatch):
    import types
    from env_factory.generation import semantic_review as module
    clock=[0.];calls=[]
    monkeypatch.setattr(module,'time',types.SimpleNamespace(monotonic=lambda:clock[0]))
    def once(**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            (tmp_path/'events.jsonl').write_text('{"type":"turn.completed","usage":{"output_tokens":7}}\n')
            (tmp_path/'response.json').write_text('{}')
            clock[0]+=40
            raise SemanticReviewUnavailable('SOURCE_REVIEW_INVALID: verdict contradicts checks/findings')
        result=passing_report()
        result['provenance']={'agent_invocations':1,'completed_turns':1,'usage':{'output_tokens':3},'events_sha256':'b'*64}
        return result
    monkeypatch.setattr(module,'_review_source_once',once)
    result=module.review_source(source={},artifacts={},request={},root=tmp_path,model='test',timeout=60)
    assert calls[1]['timeout']==20
    assert calls[1]['report_repair_reason'].startswith('SOURCE_REVIEW_INVALID:')
    assert result['provenance']['agent_invocations']==2
    assert result['provenance']['usage']['output_tokens']==10


def test_valid_business_failure_is_not_retried(tmp_path, monkeypatch):
    from env_factory.generation import semantic_review as module
    report=passing_report();report['status']='fail'
    report['checks']['answerability']['passed']=False
    report['findings']=[{'code':'missing','source_paths':['source.tools'],'reason':'Required input absent','repair':'Expose input'}]
    mocked=Mock(return_value=report);monkeypatch.setattr(module,'_review_source_once',mocked)
    assert module.review_source(source={},artifacts={},request={},root=tmp_path,model='test',timeout=60)==report
    assert mocked.call_count==1


def test_invalid_report_retry_limit_and_deadline(tmp_path, monkeypatch):
    import types
    import pytest
    from env_factory.generation import semantic_review as module
    for duration,expected_calls in [(1,2),(60,1)]:
        clock=[0.];calls=[]
        monkeypatch.setattr(module,'time',types.SimpleNamespace(monotonic=lambda:clock[0]))
        def once(**kwargs):
            calls.append(kwargs);clock[0]+=duration
            raise SemanticReviewUnavailable('SOURCE_REVIEW_INVALID: inconsistent')
        monkeypatch.setattr(module,'_review_source_once',once)
        with pytest.raises(SemanticReviewUnavailable):
            module.review_source(source={},artifacts={},request={},root=tmp_path,model='test',timeout=60)
        assert len(calls)==expected_calls
