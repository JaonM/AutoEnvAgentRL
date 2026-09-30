import json
from copy import deepcopy

from env_factory.evidence.training_status import write_training_status


def qualified():
    return {key: True for key in ('passed', 'live_rollout_verified',
        'live_reward_calibration_verified', 'data_governance_verified',
        'trajectory_privacy_verified')} | {'sandbox_score': {'passed': True}}


def test_only_final_verified_delivery_is_training_ready(tmp_path):
    path = tmp_path / 'status.json'
    original = {'status': 'success', 'success': True, 'exit_code': 0, 'phase': 'completed'}
    path.write_text(json.dumps(original))
    result = qualified()
    write_training_status(tmp_path, result)
    assert result['training_ready'] is True
    assert json.loads(path.read_text()) == {**original, 'training_ready': True}
    for field in qualified():
        incomplete = deepcopy(qualified())
        del incomplete[field]
        write_training_status(tmp_path, incomplete)
        assert incomplete['training_ready'] is False
        assert json.loads(path.read_text())['training_ready'] is False


def test_new_attempt_and_final_failure_revoke_readiness(tmp_path):
    path = tmp_path / 'status.json'
    path.write_text(json.dumps({'status': 'success', 'success': True, 'exit_code': 0, 'training_ready': True}))
    write_training_status(tmp_path)
    assert json.loads(path.read_text())['training_ready'] is False
    result = qualified()
    result['passed'] = False
    write_training_status(tmp_path, result)
    assert result['training_ready'] is False
    path.write_text(json.dumps({'status': 'failed', 'success': False, 'exit_code': 5}))
    write_training_status(tmp_path, qualified())
    assert json.loads(path.read_text())['training_ready'] is False


def test_missing_build_status_cannot_claim_ready(tmp_path):
    result = qualified()
    write_training_status(tmp_path, result)
    assert result['training_ready'] is False
    assert not (tmp_path / 'status.json').exists()
