import copy
import json
from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.semantic_reward import compile_semantic_outcome, calibrate_semantic_outcomes
from env_factory.sandbox_runtime import EpisodeStore
from .test_semantic_reward import fixture
from .test_agent_authoring import fixture as task_fixture


def test_reference_identity_preserves_context_dependent_labels():
    item, schema = fixture()
    first = item['semantic']['cases'][0]
    second = copy.deepcopy(first)
    first['reference_id'] = 'warm'
    second.update(reference_id='cold', expected='fail')
    item['semantic']['cases'].append(second)
    _, cases = compile_semantic_outcome(item, schema=schema, weight=1)
    assert cases[-1]['reference_id'] == 'cold'
    item['semantic']['cases'][-1]['reference_id'] = 'warm'
    with pytest.raises(ValueError, match='duplicate'):
        compile_semantic_outcome(item, schema=schema, weight=1)


@pytest.mark.parametrize('value', ['', ' ', 0, [], None])
def test_rejects_invalid_reference_identity(value):
    item, schema = fixture()
    item['semantic']['cases'][0]['reference_id'] = value
    with pytest.raises(ValueError, match='SEMANTIC_REWARD_CASES'):
        compile_semantic_outcome(item, schema=schema, weight=1)


def test_missing_reference_cannot_fall_back_to_primary(tmp_path, monkeypatch):
    item, schema = fixture()
    item['semantic']['cases'][0]['reference_id'] = 'missing'
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=1)
    monkeypatch.delenv('SANDBOX_EVALUATOR_MOCK', raising=False)
    store = EpisodeStore(tmp_path / 'missing.sqlite3')
    store.reset(episode_id='missing', seed=1)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as judge:
        with pytest.raises(ValueError, match='SEMANTIC_CALIBRATION_REFERENCE'):
            calibrate_semantic_outcomes({'metrics': [metric]}, {metric['id']: cases}, context={}, store=store)
        judge.assert_not_called()


def test_failed_calibration_reports_exact_answer_without_changing_label(tmp_path, monkeypatch):
    item, schema = fixture()
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=1)
    monkeypatch.delenv('SANDBOX_EVALUATOR_MOCK', raising=False)
    monkeypatch.setenv('SANDBOX_LLM_API_KEY', 'offline-test')
    monkeypatch.setenv('SANDBOX_LLM_MODEL', 'offline-test')
    store = EpisodeStore(tmp_path / 'failed.sqlite3')
    store.reset(episode_id='failed', seed=1)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'fail'}):
        report = calibrate_semantic_outcomes({'metrics': [metric]}, {metric['id']: cases},
                                            context={}, store=store)
    assert not report['passed']
    failed = next(row for row in report['cases'] if not row['passed'])
    assert failed['expected_label'] == 'pass' and failed['actual_score'] == 0
    assert failed['failed_case_evidence']['answer'] == cases[failed['index']]['answer']
    assert failed['failed_case_evidence']['criteria'] == metric['criteria']
    assert all('failed_case_evidence' not in row for row in report['cases'] if row['passed'])


def test_compiler_calibrates_each_branch_against_its_actual_tool_result(tmp_path, monkeypatch):
    source, request = task_fixture('simple_agentic')
    item, schema = fixture()
    item['rubric'] = 'Explain the temperature reported by the tool.'
    item['semantic']['criteria'] = ['State the observed temperature, with faithful paraphrases accepted.']
    item['semantic']['cases'] = [
        {'answer': {'why': text}, 'expected': label, 'reference_id': ref}
        for ref, text, label in [
            ('success', '18 degrees', 'pass'), ('success', 'Reading: 18', 'pass'),
            ('alternate', '23 degrees', 'pass'), ('alternate', 'Reading: 23', 'pass'),
            ('success', '23 degrees', 'fail'), ('alternate', '18 degrees', 'fail')]]
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=1)
    source.update(metrics=[metric], metric_implementations=[],
                  semantic_calibration={metric['id']: cases},
                  answer_contract={'format': 'json_object', 'schema': schema})
    source['description']['public_input']['initial_user_message'] = 'Query 青松库 or 白杨库 and explain its observed temperature in JSON why.'
    source['scenarios'][0]['steps'][-1]['content'] = json.dumps({'why': '18 degrees'})
    source['scenarios'][1]['steps'][-1]['content'] = json.dumps({'why': '999 degrees'})
    alternate = copy.deepcopy(source['scenarios'][0])
    alternate['scenario_id'] = 'alternate'
    alternate['steps'][0]['arguments']['name'] = '白杨库'
    alternate['steps'][-1]['content'] = json.dumps({'why': '23 degrees'})
    source['scenarios'].insert(1, alternate)
    monkeypatch.delenv('SANDBOX_EVALUATOR_MOCK', raising=False)
    monkeypatch.setenv('SANDBOX_LLM_API_KEY', 'offline-test')
    monkeypatch.setenv('SANDBOX_LLM_MODEL', 'offline-test')
    observed = []

    def judge(messages, **kwargs):
        evidence = json.loads(messages[1]['content'])['runtime_evidence']
        serialized = json.dumps(evidence.get('tool_results', {}), ensure_ascii=False)
        answer = evidence['final_agent_response']
        temperature = 23 if '23' in serialized else 18 if '18' in serialized else None
        observed.append((temperature, answer))
        return {'label': 'pass' if temperature is not None and str(temperature) in answer else 'fail'}

    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', side_effect=judge):
        result = compile_source(source, root=tmp_path, request=request)
    assert result['task_readiness']['ready']
    report = json.loads((tmp_path / 'semantic_calibration.json').read_text())
    assert report['passed']
    assert [case['actual_score'] for case in report['cases']] == [1, 1, 1, 1, 0, 0]
    assert report['reference_scenarios']['alternate'] == 'alternate'
    assert any(t == 23 and '18 degrees' in answer for t, answer in observed)


def test_reference_family_calibrates_every_expanded_script_without_changing_labels():
    from env_factory.generation.semantic_reward import scope_calibration_references
    successes = [{'scenario_id': f'success-script-{i}'} for i in range(1, 4)]
    calibration = {'explanation': [
        {'reference_id': 'success', 'answer': {'why': 'faithful'}, 'expected': 'pass'},
        {'reference_id': 'success', 'answer': {'why': 'contradictory'}, 'expected': 'fail'}]}
    original = copy.deepcopy(calibration)
    scoped, contexts = scope_calibration_references(calibration, successes)
    assert calibration == original
    assert len(scoped['explanation']) == 6
    for label in ['pass', 'fail']:
        assert {case['reference_id'] for case in scoped['explanation'] if case['expected'] == label} == {
            'success-script-1', 'success-script-2', 'success-script-3'}
    assert all(contexts[case['reference_id']]['scenario_id'] == case['reference_id'] for case in scoped['explanation'])
    # Explicit scope stays explicit, including when other scripts exist.
    calibration['explanation'][0]['reference_id'] = 'success-script-2'
    scoped, _ = scope_calibration_references(calibration, successes)
    assert len(scoped['explanation']) == 4


def test_scoped_calibration_rejects_unknown_and_overlapping_contexts():
    from env_factory.generation.semantic_reward import scope_calibration_references
    successes = [{'scenario_id': f'success-script-{i}'} for i in range(1, 4)]
    case = {'reference_id': 'unknown', 'answer': {'why': 'claim'}, 'expected': 'pass'}
    with pytest.raises(ValueError, match='unknown successful reference'):
        scope_calibration_references({'explanation': [case]}, successes)
    case['reference_id'] = 'success'
    conflict = {**case, 'reference_id': 'success-script-2', 'expected': 'fail'}
    with pytest.raises(ValueError, match='overlapping or contradictory'):
        scope_calibration_references({'explanation': [case, conflict]}, successes)
