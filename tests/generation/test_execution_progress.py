import json
import pytest

from env_factory.generation.execution_progress import ExecutionProgress


def test_availability_feedback_does_not_claim_an_actionable_business_defect(tmp_path):
    progress = ExecutionProgress(tmp_path)
    progress.feedback('validation_branch', error_type='SemanticCalibrationUnavailable')
    progress.feedback('semantic_calibration', code='SEMANTIC_CALIBRATION_UNAVAILABLE')
    progress.feedback('reference', code='DETERMINISTIC_REWARD_UNREACHABLE')
    rows = [json.loads(line) for line in progress.path.read_text().splitlines()]
    assert [row['status'] for row in rows] == ['availability_feedback', 'availability_feedback', 'actionable_feedback']


def test_progress_is_visible_during_work_and_preserves_failed_attempt(tmp_path):
    progress = ExecutionProgress(tmp_path)
    with pytest.raises(RuntimeError, match="private payload"):
        with progress.measure("reward_evaluation", scenario_id="s1"):
            rows = [json.loads(line) for line in progress.path.read_text().splitlines()]
            assert rows[-1]["status"] == "running"
            raise RuntimeError("private payload")
    with ExecutionProgress(tmp_path).measure("reference_scenario", scenario_id="s2"):
        pass
    rows = [json.loads(line) for line in progress.path.read_text().splitlines()]
    assert [row["status"] for row in rows] == ["running", "failed", "running", "completed"]
    assert rows[0]["operation_id"] == rows[1]["operation_id"]
    assert rows[0]["run_id"] != rows[2]["run_id"]
    assert rows[1]["error_type"] == "RuntimeError"
    assert rows[1]["seconds"] >= 0
    assert "private payload" not in progress.path.read_text()


def test_compiler_emits_executed_scenario_and_reward_progress(tmp_path):
    from .test_interaction_contract import interactive_source
    from env_factory.generation.agent_authoring import compile_source
    source, request = interactive_source()
    artifacts = compile_source(source, root=tmp_path, request=request)
    rows = [json.loads(line) for line in (tmp_path / 'execution_progress.jsonl').read_text().splitlines()]
    expected = {s['scenario_id'] for s in artifacts['acceptance_contract']['executable_scenarios']
                if s.get('kind') in {'goal_success', 'goal_failure'}}
    assert expected <= {row['scenario_id'] for row in rows
                       if row['phase'] == 'reference_scenario' and row['status'] == 'completed'}
    assert any(row['phase'] == 'reward_evaluation' for row in rows)
