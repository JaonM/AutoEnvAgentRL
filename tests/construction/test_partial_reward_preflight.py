from scripts.sandbox import validate_agentic_training_value as module
from scripts.sandbox.validate_agentic_training_value import meaningful_result, negative_requirement_evidence, sensitive_outcomes
import pytest


@pytest.mark.parametrize('failed_requirement', [False, True])
def test_delivery_accepts_the_component_negative_already_verified_by_compiler(tmp_path, monkeypatch, failed_requirement):
    import importlib.util
    import json
    import sys
    from env_factory import sandbox_runtime
    from env_factory.generation.agent_authoring import compile_source
    from scripts.sandbox.generate_sandbox_scaffold import generate
    from tests.generation.test_component_negative_preflight import partial_failure_source

    source, request = partial_failure_source()
    for scenario in source['scenarios']:
        scenario['steps'].insert(0, {'operation': 'reset', 'body': {'episode_id': 'partial', 'seed': 17}})
        if scenario['steps'][-1]['operation'] != 'reward':
            scenario['steps'].append({'operation': 'reward', 'step_id': 'score'})
    artifact = compile_source(source, root=tmp_path, request=request)
    assert artifact['generation_pipeline']['verification']['checks']['authored_failure_1']
    (tmp_path / 'task.json').write_text(json.dumps(artifact))
    (tmp_path / 'BUILD_CONTRACT.json').write_text(json.dumps({**artifact, 'artifacts': artifact}))
    generate(tmp_path)
    monkeypatch.setitem(sys.modules, 'sandbox_runtime', sandbox_runtime)
    spec = importlib.util.spec_from_file_location('task_impl', tmp_path / 'task_impl.py')
    hooks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hooks)
    monkeypatch.setitem(sys.modules, 'task_impl', hooks)
    if failed_requirement:
        monkeypatch.setattr(module, 'negative_requirement_evidence', lambda *args: {'passed': False})
    report = module.validate(tmp_path)
    assert 'success_trajectory' not in report['failed_gates'], report['failures']
    assert ('failure_trajectory' in report['failed_gates']) is failed_requirement, report['failures']

def test_zero_count_is_evidence_only_for_declared_aggregation():
    assert meaningful_result({'count': 0}, operation='aggregate_count')
    assert not meaningful_result({'count': -1}, operation='aggregate_count')
    assert meaningful_result({'count': 0, 'records': []}, operation='select')
    assert not meaningful_result({'count': 1, 'records': []}, operation='select')
    assert not meaningful_result({'count': 0, 'records': [{}]}, operation='select')
    assert not meaningful_result({'count': 0}, operation='select')
    assert not meaningful_result({'count': 0, 'records': []})
    assert not meaningful_result({'count': False}, operation='aggregate_count')


def test_partial_negative_requires_fresh_bound_component_failure():
    task = {'metrics': [{'id': 'count', 'score_range': [0, 1]}], 'requirement_coverage': {
        'status': 'structurally_valid', 'paired_execution_verified': True, 'requirements': [{
            'id': 'correct_count', 'outcome_ids': ['count'], 'positive_scenarios': ['good'],
            'negative_scenarios': ['bad']}]}}
    scenario = {'scenario_id': 'bad'}
    run = {'history': [{'operation': 'reward', 'body': {'reward': .7, 'components': {'count': 0}}}]}
    assert negative_requirement_evidence(task, scenario, run)['passed']
    run['history'][0]['body']['components']['count'] = 1
    assert not negative_requirement_evidence(task, scenario, run)['passed']
    assert not negative_requirement_evidence(task, scenario, {})['passed']
    assert negative_requirement_evidence(task, {'scenario_id': 'unbound'}, run) is None


def test_data_sensitivity_requires_component_and_weighted_total_loss():
    metrics = [{'id': key, 'category': 'outcome', 'weight': 1, 'score_range': [0, 1]}
               for key in ('period', 'topic', 'count')]
    def run(reward, count):
        return {'history': [{'operation': 'reward', 'body': {'reward': reward,
            'components': {'period': 1, 'topic': 1, 'count': count}}}]}
    baseline = run(1, 1)
    assert sensitive_outcomes(baseline, run(2 / 3, 0), metrics) == ['count']
    assert sensitive_outcomes(baseline, run(1, 0), metrics) == []
    assert sensitive_outcomes(baseline, run(2 / 3, 1), metrics) == []
    assert sensitive_outcomes(baseline, run(2 / 3, float('nan')), metrics) == []


def test_live_audit_does_not_accept_fallback_as_negative_evidence():
    class App:
        payload = {'judgment_obtained': False, 'used_fallback': True}
        def handle(self, method, path, body, headers):
            if path == '/v1/replay':
                return 200, {'events': [{'event': 'evaluator_call', 'payload': self.payload}]}, {}
            return 200, {'reward': 0}, {}
    failures = []
    app = App()
    audited = module.EvaluatorEvidenceAudit(app, {}, failures)
    assert audited.handle('GET', '/v1/reward', None, {})[1]['reward'] == 0
    assert failures[0]['gate'] == 'evaluator_availability'
    audited.handle('GET', '/v1/reward', None, {})
    assert len(failures) == 1
    app.payload = {'judgment_obtained': True, 'used_fallback': False}
    failures.clear()
    audited.handle('GET', '/v1/reward', None, {})
    assert not failures
