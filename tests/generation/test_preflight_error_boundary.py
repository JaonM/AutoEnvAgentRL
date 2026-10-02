import json
from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source, verify_execution
from env_factory.sandbox_runtime import AcceptanceScenarioRunner
from .test_agent_authoring import fixture


def test_authored_negative_can_explicitly_expect_business_http_rejection(tmp_path):
    source, request = fixture("simple_agentic")
    artifacts = compile_source(source, root=tmp_path, request=request)
    scenarios = artifacts["acceptance_contract"]["executable_scenarios"]
    failure = next(s for s in scenarios if s.get("kind") == "goal_failure")
    failure["steps"] = [{"operation": "tool_call", "tool_name": "read_temperature",
                         "arguments": {}, "expected_status": 400}]
    failure["assertions"] = []
    assert verify_execution(artifacts, tmp_path)["passed"]


def test_harness_error_is_not_evidence_that_no_tools_policy_was_rejected(tmp_path):
    source, request = fixture("simple_agentic")
    artifacts = compile_source(source, root=tmp_path, request=request)
    original = AcceptanceScenarioRunner.run

    def run(self, scenario):
        steps = scenario.get("steps", [])
        if (not scenario.get("scenario_id")
                and not any(s.get("operation") == "tool_call" for s in steps)
                and any(s.get("operation") == "agent_response" and "18" in s.get("content", "") for s in steps)):
            raise KeyError("synthetic harness defect")
        return original(self, scenario)

    with patch.object(AcceptanceScenarioRunner, "run", run), pytest.raises(KeyError, match="synthetic harness defect"):
        verify_execution(artifacts, tmp_path)


def test_failure_diagnostic_preserves_original_judgment_without_rejudging(tmp_path):
    source, request = fixture("simple_agentic")
    source["scenarios"][0]["steps"].append({"operation": "reward", "step_id": "reward"})
    source["scenarios"][0]["assertions"] = [
        {"source": "step:reward", "path": "$.reward", "operator": "gte", "expected": 1}]
    calls = []

    def reject(self, context, scores):
        calls.append(context)
        self.store.event("evaluator_call", {"metric_id": "temperature_answer",
            "used_fallback": False, "judgment_obtained": True}, {"label": "fail"})
        return {"temperature_answer": 0.0}

    with patch("env_factory.sandbox_runtime.ContractModelMetricEvaluator.evaluate_all", reject):
        with pytest.raises(ValueError, match="AGENT_SCENARIO_FAILED"):
            compile_source(source, root=tmp_path, request=request)
    report = json.loads((tmp_path / "preflight_failure.json").read_text())["diagnostic"]
    assert report["last_evaluated_components"]["temperature_answer"] == 0.0
    assert report["model_evidence"][0]["result"]["label"] == "fail"
    assert len(calls) == 1
