import copy
import json

from env_factory_v2.examples import simple
from env_factory_v2.validation.verification import qualify
from env_factory_v2.delivery.export import export_bundle, verify_bundle
from env_factory_v2.validation.api_verification import verify_api


def test_compact_qualification_keeps_actual_gates_answers_and_api_execution():
    compiled, full = qualify(simple())
    _, compact = qualify(simple(), compact=True)
    assert compact['gates'] == full['gates']
    assert len(compact['records']) == len(full['records'])
    for a, b in zip(full['records'], compact['records']):
        assert a['passed'] == b['passed']
        assert a['replay_passed'] == b['replay_passed']
        assert a['first']['answer'] == b['first']['answer']
        assert a['first']['outcome'] == b['first']['outcome']
        assert 'trace' not in b['first'] and 'final_state' not in b['first']
    assert verify_api(compiled['source'], compact)['passed']


def test_export_compacts_evidence_without_mutating_full_host_report(tmp_path):
    compiled, report = qualify(simple())
    before = copy.deepcopy(report)
    root = tmp_path / 'bundle'
    export_bundle(compiled, report, root)
    stored = json.loads((root / 'verification.json').read_text())
    assert stored['full_execution_evidence_identity'] == report['evidence_identity']
    assert stored['gates'] == report['gates']
    assert 'final_state' not in stored['records'][0]['first']
    assert report == before
    verify_bundle(root)


def test_compact_mode_checks_full_world_replay_before_discarding_detail(monkeypatch):
    from env_factory_v2.validation import verification
    original = verification.run_scenario
    calls = 0

    def changed_world(*args, **kwargs):
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == 2:
            result['final_state']['unexpected_difference'] = True
        return result

    monkeypatch.setattr(verification, 'run_scenario', changed_world)
    _, report = verification.qualify(simple(), compact=True)
    assert report['gates']['joint_execution_and_counterexamples']
    assert not report['gates']['deterministic_reset_and_replay']
    assert not report['passed']
    assert not report['records'][0]['passed']
    assert not report['records'][0]['replay_passed']
