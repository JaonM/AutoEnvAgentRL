import copy
import json
import threading
from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source, verify_execution, prepare_route_probe
from env_factory.generation.deterministic_preflight import check_all_routes
from env_factory.generation.validation_batch import verify_independent_routes
from .test_agent_authoring import fixture


def compiled(tmp_path):
    source, request = fixture('direct_response')
    return compile_source(source, root=tmp_path, request=request)


def test_deterministic_pass_collects_main_and_stop_errors_without_models(tmp_path):
    artifacts = compiled(tmp_path)
    scenarios = [s for s in artifacts['acceptance_contract']['executable_scenarios']
                 if s.get('kind') in {'goal_success', 'goal_failure'}]
    for scenario in scenarios:
        if scenario['kind'] == 'goal_success':
            for step in scenario['steps']:
                if step['operation'] == 'agent_response':
                    step['content'] = '999'
    artifacts['execution_route_probes'] = [{'route_id': 'stop', 'initial_changes': [],
                                           'scenarios': copy.deepcopy(scenarios)}]
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as model:
        with pytest.raises(ValueError, match='DETERMINISTIC_PREFLIGHT_FAILED'):
            check_all_routes(artifacts, tmp_path, verify_execution, prepare_route_probe)
    model.assert_not_called()
    report = json.loads((tmp_path / 'deterministic_preflight_all_routes.json').read_text())
    assert [r['route_id'] for r in report['routes'] if r['findings']] == ['references', 'stop']
    assert report['training_ready'] is False
    assert 'actionable_feedback' in (tmp_path / 'execution_progress.jsonl').read_text()


def test_unknown_semantics_are_not_certified_by_optimistic_credit(tmp_path):
    artifacts = compiled(tmp_path)
    artifacts['metrics'][0]['weight'] = .5
    artifacts['metrics'].append({'id': 'meaning', 'category': 'outcome', 'type': 'model-based',
                                 'score_range': [0, 1], 'weight': .5})
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as model:
        report = verify_execution(artifacts, tmp_path, deterministic_only=True)
    model.assert_not_called()
    assert not report['findings']
    assert report['training_ready'] is False
    assert 'semantic_calibration' in report['pending']


def test_parallel_independent_failures_are_all_reported(tmp_path):
    barrier = threading.Barrier(2)
    executed = []
    artifacts = {'execution_route_probes': [
        {'route_id': name, 'initial_changes': []} for name in ['stop', 'other']]}
    def prepare(artifacts, root, probe):
        return {'identity': probe['route_id']}, root / probe['route_id']
    def verify(candidate, root, **kwargs):
        identity = candidate.get('identity', 'references')
        executed.append(identity)
        if identity in {'references', 'stop'}:
            barrier.wait(timeout=3)
        raise ValueError(identity + '_failed')
    with pytest.raises(ValueError, match='VALIDATION_BATCH_FAILED'):
        verify_independent_routes(artifacts, tmp_path, verify, prepare)
    assert set(executed) == {'references', 'stop', 'other'}
    report = json.loads((tmp_path / 'validation_findings.json').read_text())
    assert len(report['findings']) == 3


def test_serial_and_parallel_route_verdicts_and_checks_agree(tmp_path):
    from .test_execution_routes import source_with_routes
    from env_factory.generation.agent_authoring import _verify_execution
    source, request = source_with_routes()
    artifacts = compile_source(source, root=tmp_path, request=request, structural_preview=True)
    reports = [verify_independent_routes(copy.deepcopy(artifacts), tmp_path, _verify_execution,
               prepare_route_probe, workers=workers) for workers in (1, 2)]
    assert all(report['passed'] for report in reports)
    assert reports[0]['checks'] == reports[1]['checks']
    assert reports[0]['execution_route_coverage'] == reports[1]['execution_route_coverage']
    assert reports[0]['reference_observations'] == reports[1]['reference_observations']


def test_author_deterministic_preview_never_calls_models_or_exports_task(tmp_path, monkeypatch):
    from env_factory.generation.agent_authoring import main
    source, request = fixture('direct_response')
    source_path, request_path = tmp_path / 'source.json', tmp_path / 'request.json'
    source_path.write_text(json.dumps(source))
    request_path.write_text(json.dumps(request))
    output = tmp_path / 'preview'
    monkeypatch.setattr('sys.argv', ['preview', '--source', str(source_path), '--request',
        str(request_path), '--output', str(output), '--deterministic-preview'])
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as model:
        main()
    model.assert_not_called()
    report = json.loads((output / 'deterministic_preflight_all_routes.json').read_text())
    assert report['training_ready'] is False
    assert report['model_calls'] == 0
    assert not any((output / name).exists() for name in ['task.json', 'compiled_artifacts.json', 'status.json'])
    (output / 'task.json').write_text('original')
    with pytest.raises(SystemExit):
        main()
    assert (output / 'task.json').read_text() == 'original'
