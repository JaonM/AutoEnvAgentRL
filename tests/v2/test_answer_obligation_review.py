import copy
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from env_factory_v2.expressions import ContractError
from v2_answer_obligations import review_answer_obligations, generation_output_scope
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
