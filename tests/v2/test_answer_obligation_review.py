import copy
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from env_factory_v2.expressions import ContractError
from v2_answer_obligations import review_answer_obligations, generation_output_scope, review_case_answer_obligations
from v2_business_acceptance import business_truth


def fixture():
    return {'user_request': '执行操作，报告未完成原因。', 'business': {'policy': '仅在资格满足且授权后执行。'},
            'truth': {'answer': {'changed': ['hidden-id'], 'reasons': ['private-reason']},
                      'initial': {'items': []}, 'final_state': {'items': []}}}


def review():
    citation = {'source': 'user_request', 'text': '执行操作，报告未完成原因。'}
    return {'roles': [{'role': 'changed', 'kind': 'state_only', 'reason': 'State change only.', 'citations': [citation]},
                      {'role': 'reasons', 'kind': 'answer', 'reason': 'Explicit report.', 'citations': [citation]}],
            'process_requirements': [], 'uncovered_answers': []}


def test_scope_is_independent_of_generated_layout_and_expected_values():
    agent = Mock(); agent.ask.return_value = (review(), {'backend': 'test'})
    data = fixture(); original = copy.deepcopy(data)
    result = review_answer_obligations(agent, **data)
    assert result['passed'] and [v['role'] for v in result['answer_obligations']] == ['reasons']
    assert data == original
    payload = agent.ask.call_args.args[1]
    assert set(payload) == {'original_sources', 'proposed_role_shapes', 'state_tables'}
    assert 'hidden-id' not in str(payload) and 'private-reason' not in str(payload)


@pytest.mark.parametrize('invalid', ['missing', 'duplicate', 'fabricated_quote'])
def test_scope_needs_complete_grounded_role_review(invalid):
    value = review()
    if invalid == 'missing': value['roles'].pop()
    elif invalid == 'duplicate': value['roles'][1] = value['roles'][0]
    else: value['roles'][0]['citations'][0]['text'] = 'invented obligation'
    agent = Mock(); agent.ask.return_value = (value, {})
    with pytest.raises(ContractError): review_answer_obligations(agent, **fixture())


@pytest.mark.parametrize('unresolved', ['role', 'missing_answer'])
def test_unresolved_scope_cannot_be_frozen(unresolved):
    value = review()
    if unresolved == 'role': value['roles'][0]['kind'] = 'unsupported'
    else: value['uncovered_answers'] = [{'source': 'user_request', 'text': '报告未完成原因'}]
    agent = Mock(); agent.ask.return_value = (value, {})
    assert not review_answer_obligations(agent, **fixture())['passed']


def test_timed_truth_respects_answer_scope_without_losing_state_or_timing():
    truth = {'answer': {'effect': 7, 'remaining': 3}, 'initial': {'items': [1]},
             'final_state': {'items': [2]}, 'runtime_evidence': {'virtual_seconds': 42}}
    world = Mock(); world.trace = [{'kind': 'tool', 'status': 'ok', 'tool': 'read', 'id': 'observed'}]
    world.tools = {'read': {'kind': 'query', 'table': 'quotes'}}
    spec = {'kind': 'procurement_partial', 'answer_obligations': [
        {'role': 'remaining', 'source': 'user_request', 'text': '报告余量'}]}
    with patch('v2_business_acceptance.runtime_business_truth', return_value=copy.deepcopy(truth)):
        actual = business_truth(spec, {'policy': ''}, {}, user_request='执行后报告余量', world=world)
    assert actual['answer'] == {'remaining': 3}
    assert actual['state_result_values'] == truth['answer']
    assert actual['initial'] == truth['initial'] and actual['final_state'] == truth['final_state']
    assert actual['runtime_evidence'] == truth['runtime_evidence']


def test_process_requirements_remain_visible_without_becoming_answer_fields():
    value = review()
    value['process_requirements'] = [{'source': 'business_policy', 'text': '授权后执行'}]
    agent = Mock(); agent.ask.return_value = (value, {})
    result = review_answer_obligations(agent, **fixture())
    assert result['passed']
    assert result['review']['process_requirements'] == value['process_requirements']
    assert len(result['answer_obligations']) == 1


def test_same_review_supplies_generation_scope_without_oracle_values():
    agent = Mock(); agent.ask.return_value = (review(), {})
    data = fixture()
    result = review_answer_obligations(agent, **data)
    spec = {key: result[key] for key in ('answer_obligations', 'generation_output_scope')}
    scope = generation_output_scope(spec, user_request=data['user_request'], business=data['business'])
    assert scope['roles'] == review()['roles']
    assert 'hidden-id' not in str(scope) and 'private-reason' not in str(scope)
    assert 'truth' not in scope and 'expected' not in str(scope)
    scope['roles'].clear()
    assert len(spec['generation_output_scope']['roles']) == 2


@pytest.mark.parametrize('changed', ['request', 'policy', 'acceptance', 'extra_value', 'unresolved'])
def test_generation_scope_cannot_drift_from_sources_or_acceptance(changed):
    agent = Mock(); agent.ask.return_value = (review(), {})
    data = fixture()
    result = review_answer_obligations(agent, **data)
    spec = {key: result[key] for key in ('answer_obligations', 'generation_output_scope')}
    if changed == 'request': data['user_request'] += ' changed'
    elif changed == 'policy': data['business']['policy'] += ' changed'
    elif changed == 'acceptance': spec['answer_obligations'] = []
    elif changed == 'extra_value': spec['generation_output_scope']['roles'][0]['expected'] = 42
    else: spec['generation_output_scope']['roles'][0]['kind'] = 'unsupported'
    with pytest.raises(ContractError):
        generation_output_scope(spec, user_request=data['user_request'], business=data['business'])


def test_historical_scope_is_not_fabricated_from_truth():
    assert generation_output_scope({'expected': 42}, user_request='old', business={'policy': ''}) is None


def test_declared_shapes_do_not_invoke_runtime_oracle_or_fabricate_clock():
    data = fixture()
    data['business']['tables'] = [{'id': 'items', 'rows': []}]
    shapes = {'changed': {'type': 'array', 'item_shapes': [{'type': 'str'}]},
              'reasons': {'type': 'array', 'item_shapes': [{'type': 'str'}]}}
    agent = Mock(); agent.ask.return_value = (review(), {})
    with patch('audit_v2_natural.canonical_truth', side_effect=AssertionError('runtime truth must not run')):
        result = review_case_answer_obligations(agent, user_request=data['user_request'], business=data['business'],
            user={'facts': {'secret': 'never disclose'}}, spec={'kind': 'arbitrary_timed_task', 'answer_role_shapes': shapes})
    assert result['passed']
    payload = agent.ask.call_args.args[1]
    assert payload['proposed_role_shapes'] == shapes
    assert payload['state_tables'] == ['items']
    assert 'never disclose' not in str(payload)


@pytest.mark.parametrize('shape', [{'type': 'int', 'expected': 42},
    {'type': 'array', 'item_shapes': [{'type': 'str', 'enum': ['secret']}]},
    {'type': 'object', 'properties': {'nested': {'type': 'int', 'minimum': 12}}}])
def test_declared_shapes_reject_values_constraints_before_model_call(shape):
    agent = Mock()
    with pytest.raises(ContractError):
        review_answer_obligations(agent, user_request='request', business={'policy': ''},
                                 role_shapes={'result': shape}, state_tables=[])
    agent.ask.assert_not_called()


def split_review():
    value = review()
    value['communication_requirements'] = [{'meaning': 'Explain why requested work remains incomplete.',
        'citations': [{'source': 'user_request', 'text': '报告未完成原因'}]}]
    value['behavior_requirements'] = [{'meaning': 'Only execute eligible authorized operations.',
        'citations': [{'source': 'business_policy', 'text': '仅在资格满足且授权后执行。'}]}]
    return value


def test_generation_receives_distinct_grounded_communication_and_behavior():
    value = split_review(); agent = Mock(); agent.ask.return_value = (value, {})
    data = fixture(); original = copy.deepcopy(data)
    result = review_answer_obligations(agent, **data)
    assert result['passed'] and data == original
    scope = generation_output_scope(result, user_request=data['user_request'], business=data['business'])
    for key in ('communication_requirements', 'behavior_requirements'):
        assert scope[key] == value[key]
        assert key in agent.ask.call_args.args[2]['required']
    assert result['answer_obligations'] == [{'role': 'reasons', 'source': 'user_request', 'text': data['user_request']}]
    scope['behavior_requirements'].clear()
    assert result['generation_output_scope']['behavior_requirements']


@pytest.mark.parametrize('invalid', ['missing_kind', 'invented_citation', 'extra_value', 'not_list'])
def test_split_scope_cannot_be_partial_or_leak_uncited_extra_data(invalid):
    value = split_review()
    if invalid == 'missing_kind': value.pop('behavior_requirements')
    elif invalid == 'invented_citation': value['communication_requirements'][0]['citations'][0]['text'] = 'invented'
    elif invalid == 'extra_value': value['behavior_requirements'][0]['expected'] = 'secret'
    else: value['behavior_requirements'] = {}
    agent = Mock(); agent.ask.return_value = (value, {})
    with pytest.raises(ContractError): review_answer_obligations(agent, **fixture())
