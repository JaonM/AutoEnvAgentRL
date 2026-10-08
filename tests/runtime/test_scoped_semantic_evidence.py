import json
from unittest.mock import patch

import pytest

from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, SandboxError
from .test_verified_answer_context import fixture, submitted


def scoped(tmp_path):
    judge, metric, context = fixture(tmp_path)
    metric['semantic_evidence_mode'] = 'fields'
    return judge, metric, context


def capture(judge, context):
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'pass'}) as model:
        judge.evaluate_all(context, {})
    return json.loads(model.call_args.args[0][1]['content'])['runtime_evidence']


def test_ten_thousand_rows_stay_local_and_submitted_facts_cannot_poison_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK', '0')
    judge, metric, context = scoped(tmp_path)
    rows = [{'id': i, 'text': 'UNRELATED_PRIVATE_ROW_' + str(i)} for i in range(10000)]
    context['business_state'] = {'large': rows}
    context['trajectory'] = {'events': [
        {'event': 'tool_call', 'payload': {'tool_name': 'read', 'arguments': {'large': rows}}, 'result': rows},
        {'event': 'tool_attempt', 'payload': {'tool_name': 'read', 'arguments': {'large': rows}}, 'result': {'status': 200}},
        {'event': 'user_turn', 'payload': {'messages': [{'role': 'assistant', 'content': 'Which group?'}]},
         'result': {'user_query': 'Group A'}}]}
    wrong = submitted(context, count=1, unique=True)
    wrong['executable_field_facts'] = {'count': {'value': 1}}
    sent = capture(judge, wrong)
    assert sent['executable_field_facts']['count']['value'] == 2
    assert sent['executable_field_facts']['unique']['value'] is False
    assert sent['verified_answer_context'] == {}
    assert sent['unverified_answer_context_fields'] == ['count', 'unique']
    assert sent['observed_tool_statuses'] == [{'tool_name': 'read', 'status': 200}]
    assert sent['observed_user_turns'] == [{'assistant_response': 'Which group?', 'user_response': 'Group A'}]
    packet = json.dumps(sent)
    assert 'UNRELATED_PRIVATE_ROW' not in packet and len(packet.encode()) < 4000
    assert not {'business_data', 'business_state', 'tool_results', 'trajectory'} & set(sent)
    assert DeclarativeMetricEvaluator().evaluate_all(judge.contract['metric_implementations'], wrong)['facts'] == 0


def test_defaulted_observation_is_marked_even_when_used_in_derived_fact(tmp_path):
    judge, metric, context = scoped(tmp_path)
    context['reward_state_snapshots'] = {}
    facts = judge._verified_answer_context(metric, context)['executable_field_facts']
    for name in ('count', 'unique'):
        assert facts[name]['observation_availability'] == [
            {'name': 'read', 'path': '$.count', 'present': False, 'missing_value': 0}]
    assert facts['count']['value'] == 0 and facts['unique']['value'] is False
    context['reward_state_snapshots'] = {'$tool_results': {'read': {'count': 0}}}
    assert judge._verified_answer_context(metric, context)['executable_field_facts']['count'][
        'observation_availability'][0]['present'] is True


def test_unresolved_is_not_a_fact_and_decimal_value_retains_precision(tmp_path):
    judge, metric, context = scoped(tmp_path)
    rule = judge.contract['metric_implementations'][0]
    rule['expected']['targets'][0]['expression'] = {'op': 'div', 'args': [{'literal': 1}, {'literal': 3}]}
    rule['expected']['answer_schema']['properties']['count']['type'] = 'number'
    rule['expected']['targets'][1]['expression'] = {'tool_result': {'name': 'absent', 'path': '$.value'}}
    evidence = judge._verified_answer_context(metric, context)
    assert evidence['unresolved_field_facts'] == ['unique']
    assert 'unique' not in evidence['executable_field_facts']
    count = evidence['executable_field_facts']['count']
    assert count['value'] == pytest.approx(1/3)
    assert count['decimal_values'][0]['value'].startswith('0.333333333333333333')
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize('change', ['bad_mode', 'no_context', 'no_owned', 'no_answer', 'no_inputs', 'too_large'])
def test_invalid_or_oversized_evidence_fails_before_model(tmp_path, monkeypatch, change):
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK', '0')
    judge, metric, context = scoped(tmp_path)
    if change == 'bad_mode': metric['semantic_evidence_mode'] = 'guess'
    if change == 'no_context': metric['semantic_context_fields'] = []
    if change == 'no_owned': metric['semantic_fields'] = []
    if change == 'no_answer': metric['evaluation_inputs'] = ['business_data']
    if change == 'no_inputs': metric['evaluation_inputs'] = None
    if change == 'too_large': judge.contract['public_input']['text'] = '事' * 30000
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as model:
        with pytest.raises(SandboxError, match='evidence mode|65536'):
            judge.evaluate_all(context, {})
    model.assert_not_called()


def test_modes_and_changed_runtime_facts_invalidate_judgment_cache(tmp_path, monkeypatch):
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK', '0')
    judge, metric, context = scoped(tmp_path)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', return_value={'label': 'pass'}) as model:
        judge.evaluate_all(context, {})
        judge.evaluate_all(context, {})
        assert model.call_count == 1
        context['reward_state_snapshots']['$tool_results']['read']['count'] = 3
        judge.evaluate_all(context, {})
        assert model.call_count == 2
        metric['semantic_evidence_mode'] = 'runtime'
        judge.evaluate_all(context, {})
        assert model.call_count == 3
    evidence = json.loads(model.call_args.args[0][1]['content'])['runtime_evidence']
    assert 'business_data' in evidence and 'executable_field_facts' not in evidence
