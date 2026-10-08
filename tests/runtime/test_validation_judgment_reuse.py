import copy
from unittest.mock import patch

import pytest

from env_factory.sandbox_runtime import ContractEvaluatorRuntime, EpisodeStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK', '0')
    monkeypatch.setenv('SANDBOX_LLM_API_KEY', 'offline-test-key')
    monkeypatch.setenv('SANDBOX_LLM_MODEL', 'offline-test-model')
    value = EpisodeStore(tmp_path / 'episodes.sqlite3')
    value.reset(episode_id='first', seed=17)
    return value


METRIC = {'id': 'explanation', 'rubric': 'Explain the evidence', 'criteria': ['Be factual'],
          'evaluator': {'kind': 'external_llm_judge', 'score_mapping': {'pass': 1, 'fail': 0}}}
CONTEXT = {'final_agent_response': 'The measured value is 17.', 'business_data': {'value': 17}}


@pytest.mark.parametrize('reuse,expected_calls', [(False, 2), (True, 1)])
def test_only_opted_in_validation_reuses_judgments_across_resets(store, reuse, expected_calls):
    evaluator = ContractEvaluatorRuntime(store, reuse_across_resets=reuse)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'pass'}) as judge:
        first = evaluator.json_judge(METRIC, CONTEXT, fallback={'label': 'fail'})
        original = store.replay()['events'][-1]['payload']
        store.reset(episode_id='second', seed=17)
        second = evaluator.json_judge(METRIC, CONTEXT, fallback={'label': 'fail'})
    assert first == second == {'label': 'pass'}
    assert judge.call_count == expected_calls
    event = store.replay()['events'][-1]['payload']
    assert event['cached'] is reuse
    if reuse:
        assert event['cache_scope'] == 'validation_run'
        assert event['origin_judgment_id'] == original['judgment_id']
        report = evaluator.validation_cache_report()
        assert report['reused_judgments'] == 1
        assert report['original_judgments'][0]['judgment_id'] == event['origin_judgment_id']
        assert report['original_judgments'][0]['context_hash'] == event['context_hash']
        assert 'offline-test-key' not in str(report)


@pytest.mark.parametrize('change', ['answer', 'data', 'rubric', 'criteria', 'schema', 'model', 'endpoint', 'key'])
def test_changed_evidence_contract_or_provider_requires_new_judgment(store, monkeypatch, change):
    evaluator = ContractEvaluatorRuntime(store, reuse_across_resets=True)
    metric, context = copy.deepcopy(METRIC), copy.deepcopy(CONTEXT)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'pass'}) as judge:
        evaluator.json_judge(metric, context, fallback={'label': 'fail'})
        store.reset(episode_id='changed', seed=17)
        schema = None
        if change == 'answer': context['final_agent_response'] = 'A different claim.'
        elif change == 'data': context['business_data']['value'] = 18
        elif change == 'rubric': metric['rubric'] += ' and justify the boundary'
        elif change == 'criteria': metric['criteria'].append('Explain the boundary')
        elif change == 'schema': schema = {'type': 'object', 'properties': {'label': {'type': 'string'}}}
        elif change == 'model': monkeypatch.setenv('SANDBOX_LLM_MODEL', 'changed-model')
        elif change == 'endpoint': monkeypatch.setenv('SANDBOX_LLM_BASE_URL', 'https://example.invalid')
        elif change == 'key': monkeypatch.setenv('SANDBOX_LLM_API_KEY', 'changed-offline-key')
        evaluator.json_judge(metric, context, fallback={'label': 'fail'}, response_schema=schema)
    assert judge.call_count == 2
    assert store.replay()['events'][-1]['payload']['cached'] is False


def test_fallback_is_never_reused_as_a_judgment(store):
    evaluator = ContractEvaluatorRuntime(store, reuse_across_resets=True)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat',
               side_effect=[OSError('offline'), {'label': 'pass'}]) as judge:
        assert evaluator.json_judge(METRIC, CONTEXT, fallback={'label': 'fail'}) == {'label': 'fail'}
        store.reset(episode_id='retry', seed=17)
        assert evaluator.json_judge(METRIC, CONTEXT, fallback={'label': 'fail'}) == {'label': 'pass'}
        store.reset(episode_id='reuse', seed=17)
        assert evaluator.json_judge(METRIC, CONTEXT, fallback={'label': 'fail'}) == {'label': 'pass'}
    assert judge.call_count == 2


def test_separate_validation_runs_do_not_share_judgments(store):
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'pass'}) as judge:
        ContractEvaluatorRuntime(store, reuse_across_resets=True).json_judge(METRIC, CONTEXT, fallback={})
        store.reset(episode_id='other-run', seed=17)
        ContractEvaluatorRuntime(store, reuse_across_resets=True).json_judge(METRIC, CONTEXT, fallback={})
    assert judge.call_count == 2


def test_calibration_still_requests_fresh_judgments_for_each_case(store):
    from env_factory.generation.semantic_reward import compile_semantic_outcome, calibrate_semantic_outcomes
    from tests.generation.test_semantic_reward import fixture
    item, schema = fixture()
    cases = item['semantic']['cases']
    cases[1]['answer'] = copy.deepcopy(cases[0]['answer'])
    cases[3]['answer'] = copy.deepcopy(cases[2]['answer'])
    for index, case in enumerate(cases): case['reference_id'] = f'ref-{index % 2}'
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=1)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', side_effect=[
            {'label': label} for label in ['pass', 'pass', 'fail', 'fail']]) as judge:
        report = calibrate_semantic_outcomes({'metrics': [metric]}, {metric['id']: cases},
            context={}, reference_contexts={'ref-0': {}, 'ref-1': {}}, store=store)
    assert report['passed']
    assert judge.call_count == 4
    assert all(not event['payload']['cached'] for event in store.replay()['events'])


def test_runner_cache_reuses_unchanged_repairs_and_invalidates_conflicting_fresh_judgments(store):
    from env_factory.sandbox_runtime import ValidationJudgmentCache
    shared = ValidationJudgmentCache()
    def evaluate(reuse):
        store.reset(seed=17)
        evaluator = ContractEvaluatorRuntime(store, reuse_across_resets=reuse, judgment_cache=shared)
        return evaluator.json_judge(METRIC, CONTEXT, fallback={'label': 'fail'}), evaluator
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', side_effect=[
            {'label': 'pass'}, {'label': 'fail'}, {'label': 'pass'}, {'label': 'fail'}]) as judge:
        assert evaluate(True)[0]['label'] == 'pass'
        assert evaluate(True)[0]['label'] == 'pass'
        assert judge.call_count == 1
        result, fresh = evaluate(False)
        assert result['label'] == 'fail'
        assert fresh.validation_cache_report()['conflicting_judgments']
        assert evaluate(True)[0]['label'] == 'pass'
        assert evaluate(True)[0]['label'] == 'fail'
    assert judge.call_count == 4
    assert len(shared) == 0


def test_provider_change_invalidates_episode_cache_without_reset(store, monkeypatch):
    evaluator = ContractEvaluatorRuntime(store, reuse_across_resets=True)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'pass'}) as judge:
        evaluator.json_judge(METRIC, CONTEXT, fallback={})
        monkeypatch.setenv('SANDBOX_LLM_MODEL', 'another-model')
        evaluator.json_judge(METRIC, CONTEXT, fallback={})
    assert judge.call_count == 2
