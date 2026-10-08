import json
from unittest.mock import patch

import pytest

from env_factory.sandbox_runtime import (
    ContractModelMetricEvaluator, DeclarativeMetricEvaluator, EpisodeStore, SandboxError,
)


def fixture(tmp_path):
    fields = {'count': {'type': 'integer', 'description': 'Number of observed candidates'},
              'unique': {'type': 'boolean', 'description': 'Exactly one candidate'},
              'other': {'type': 'integer'}, 'why': {'type': 'string'}, 'separate': {'type': 'string'}}
    schema = {'type': 'object', 'properties': fields, 'required': list(fields), 'additionalProperties': False}
    count = {'observed_value': {'value': {'tool_result': {'name': 'read', 'path': '$.count'}},
                                'missing': {'literal': 0}}}
    rule = {'metric_id': 'facts', 'source': 'final_agent_response', 'path': '$', 'operator': 'value_targets',
            'score_mapping': {'pass': 1, 'fail': 0}, 'expected': {'answer_format': 'json_object',
            'answer_schema': schema, 'targets': [
                {'key': 'count', 'expression': count},
                {'key': 'unique', 'expression': {'op': 'eq', 'args': [count, {'literal': 1}]}},
                {'key': 'other', 'expression': {'literal': 42}}]}}
    metric = {'id': 'reasoning', 'category': 'outcome', 'scope': 'terminal', 'semantic_fields': ['why'],
        'semantic_context_fields': ['count', 'unique'], 'criteria': ['Explain why selection is ambiguous'],
        'evaluation_inputs': ['final_agent_response', 'tool_results', 'business_data'],
        'evaluator': {'kind': 'external_llm_judge', 'score_mapping': {'pass': 1, 'fail': 0}}}
    contract = {'metrics': [metric], 'metric_implementations': [rule],
                'public_input': {'answer_contract': {'schema': schema}}}
    context = {'final_agent_response': json.dumps({'count': 2, 'unique': False, 'other': 0,
                'why': 'Several candidates remain, so a unique selection is impossible.',
                'separate': 'This other explanation must not enter the judge context.'}),
               'business_state': {}, 'reward_state_snapshots': {'$tool_results': {'read': {'count': 2}}}}
    store = EpisodeStore(tmp_path / 'context.sqlite3'); store.reset(seed=17)
    return ContractModelMetricEvaluator(contract, store), metric, context


def submitted(context, **changes):
    return {**context, 'final_agent_response': json.dumps({**json.loads(context['final_agent_response']), **changes})}


def test_verified_context_is_per_field_and_never_trusts_aggregate_scores(tmp_path, monkeypatch):
    monkeypatch.delenv('SANDBOX_EVALUATOR_MOCK', raising=False)
    judge, metric, context = fixture(tmp_path)
    assert DeclarativeMetricEvaluator().evaluate_all(judge.contract['metric_implementations'], context)['facts'] == 0
    sent = []
    def model(messages, **kwargs):
        sent.append(json.loads(messages[1]['content'])['runtime_evidence'])
        return {'label': 'pass'}
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', side_effect=model):
        assert judge.evaluate_all(context, {'facts': 0}) == {'reasoning': 1}
        wrong = submitted(context, count=1, unique=True)
        # A caller-supplied score and claimed verification cannot promote false facts.
        wrong['verified_answer_context'] = {'count': {'value': 1}}
        assert judge.evaluate_all(wrong, {'facts': 1}) == {'reasoning': 1}
    assert sent[0]['verified_answer_context'] == {
        'count': {'value': 2, 'meaning': 'Number of observed candidates'},
        'unique': {'value': False, 'meaning': 'Exactly one candidate'}}
    assert sent[0]['unverified_answer_context_fields'] == []
    assert set(json.loads(sent[0]['final_agent_response'])) == {'why'}
    assert 'separate' not in sent[0]['verified_answer_context']
    assert sent[1]['verified_answer_context'] == {}
    assert sent[1]['unverified_answer_context_fields'] == ['count', 'unique']
    assert DeclarativeMetricEvaluator().evaluate_all(judge.contract['metric_implementations'], wrong)['facts'] == 0


def test_context_uses_observation_checkpoint_not_current_state_and_invalidates_cache(tmp_path, monkeypatch):
    monkeypatch.delenv('SANDBOX_EVALUATOR_MOCK', raising=False)
    judge, metric, context = fixture(tmp_path)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'pass'}) as call:
        judge.evaluate_all(context, {})
        judge.evaluate_all(context, {})
        context['reward_state_snapshots']['$tool_results']['read']['count'] = 1
        judge.evaluate_all(context, {})
        assert call.call_count == 2
        sent = json.loads(call.call_args.args[0][1]['content'])['runtime_evidence']
        assert sent['verified_answer_context'] == {}
        assert sent['unverified_answer_context_fields'] == ['count', 'unique']


@pytest.mark.parametrize('change', ['missing_target', 'duplicate_target', 'semantic_reference', 'unknown', 'duplicate', 'no_schema'])
def test_malformed_context_contract_stops_before_model(tmp_path, change):
    judge, metric, context = fixture(tmp_path)
    if change == 'missing_target': judge.contract['metric_implementations'] = []
    if change == 'duplicate_target': judge.contract['metric_implementations'] *= 2
    if change == 'semantic_reference': metric['semantic_context_fields'] = ['why']
    if change == 'unknown': metric['semantic_context_fields'] = ['unknown']
    if change == 'duplicate': metric['semantic_context_fields'] = ['count', 'count']
    if change == 'no_schema': judge.contract['public_input'] = {}
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as model:
        with pytest.raises(SandboxError, match='semantic context'):
            judge.evaluate_all(context, {})
    model.assert_not_called()


def test_verification_reuses_numeric_tolerance_and_array_membership(tmp_path):
    judge, metric, context = fixture(tmp_path)
    rule = judge.contract['metric_implementations'][0]
    schema = rule['expected']['answer_schema']
    schema['properties']['count'] = {'type': 'number'}
    schema['properties']['unique'] = {'type': 'array', 'items': {'type': 'integer'}}
    rule['expected']['targets'][:2] = [
        {'key': 'count', 'expression': {'literal': 2.0}, 'tolerance': 0.1},
        {'key': 'unique', 'expression': {'literal': [1, 2, 2]}, 'array_order': []}]
    valid = submitted(context, count=2.1, unique=[2, 1, 2])
    assert set(judge._verified_answer_context(metric, valid)['verified_answer_context']) == {'count', 'unique'}
    invalid = submitted(context, count=2.11, unique=[1, 1, 2])
    assert judge._verified_answer_context(metric, invalid)['unverified_answer_context_fields'] == ['count', 'unique']


def test_duplicate_submitted_keys_never_get_model_credit(tmp_path):
    judge, _, context = fixture(tmp_path)
    context['final_agent_response'] = context['final_agent_response'].replace('"count": 2', '"count": 2, "count": 1')
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as model:
        assert judge.evaluate_all(context, {}) == {'reasoning': 0}
    model.assert_not_called()
