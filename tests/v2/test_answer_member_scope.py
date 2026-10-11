import copy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from env_factory_v2.expressions import ContractError, canonical
from v2_answer_members import member_paths, project_members, project_record_roles
from v2_answer_obligations import review_answer_obligations, generation_output_scope
from v2_business_acceptance import apply_answer_obligations
from v2_record_semantics import record_projection, selected_boolean


def reviewed(required_effect=False):
    request = 'Report permissions for each object; perform the authorized changes.'
    citation = {'source': 'user_request', 'text': request}
    answer = {'results': [{'id': 'a', 'allowed': True, 'committed': False},
                          {'id': 'b', 'allowed': False, 'committed': True}]}
    truth = {'answer': answer, 'initial': {'items': [1]}, 'final_state': {'items': [2]},
             'runtime_evidence': {'virtual_seconds': 42}}
    members = [{'path': canonical(['results', None, field]),
                'kind': 'state_only' if field == 'committed' and not required_effect else 'answer',
                'reason': 'Original requirement interpretation.', 'citations': [citation]}
               for field in ('id', 'allowed', 'committed')]
    value = {'roles': [{'role': 'results', 'kind': 'answer', 'reason': 'Requested report.',
                       'citations': [citation]}], 'process_requirements': [citation],
             'uncovered_answers': [], 'member_reviews': members}
    agent = Mock(); agent.ask.return_value = (value, {})
    return request, truth, value, agent


def scope_spec(request, truth, agent):
    result = review_answer_obligations(agent, user_request=request, business={'policy': ''}, truth=truth)
    return {key: result[key] for key in ('answer_obligations', 'generation_output_scope')}


def test_nested_scope_preserves_all_rows_and_full_state_evidence():
    request, truth, _, agent = reviewed()
    original = copy.deepcopy(truth)
    spec = scope_spec(request, truth, agent)
    assert 'committed' in str(agent.ask.call_args.args[1]['nested_member_paths'])
    assert agent.ask.call_args.args[2]['properties']['member_reviews']['items']['properties']['path'] == {'type': 'string'}
    assert 'virtual_seconds' not in str(agent.ask.call_args.args[1])
    actual = apply_answer_obligations(truth, spec, user_request=request, policy='')
    assert actual['answer'] == {'results': [{'id': 'a', 'allowed': True}, {'id': 'b', 'allowed': False}]}
    assert actual['state_result_values'] == original['answer']
    for key in ('initial', 'final_state', 'runtime_evidence'):
        assert actual[key] == original[key]
    actual['answer']['results'][0]['allowed'] = False
    assert actual['state_result_values'] == original['answer']


def test_execution_result_remains_required_when_review_requires_it():
    request, truth, _, agent = reviewed(required_effect=True)
    spec = scope_spec(request, truth, agent)
    assert project_members(truth['answer'], spec['generation_output_scope']) == truth['answer']


@pytest.mark.parametrize('error', ['missing', 'duplicate', 'fabricated_quote', 'parent_not_answer'])
def test_incomplete_or_ungrounded_member_reviews_fail(error):
    request, truth, value, agent = reviewed()
    if error == 'missing': value['member_reviews'].pop()
    elif error == 'duplicate': value['member_reviews'][1] = value['member_reviews'][0]
    elif error == 'fabricated_quote': value['member_reviews'][0]['citations'] = [{'source': 'user_request', 'text': 'invented'}]
    else: value['roles'][0]['kind'] = 'optional'
    with pytest.raises(ContractError): scope_spec(request, truth, agent)


def test_unresolved_member_cannot_be_frozen():
    request, truth, value, agent = reviewed()
    value['member_reviews'][0]['kind'] = 'unsupported'
    result = review_answer_obligations(agent, user_request=request, business={'policy': ''}, truth=truth)
    assert not result['passed'] and result['generation_output_scope'] is None


def test_declared_structure_is_bound_to_frozen_member_scope():
    request, truth, _, agent = reviewed()
    spec = scope_spec(request, truth, agent)
    spec['answer_role_shapes'] = {'results': {'type': 'array', 'item_shapes': []}}
    with pytest.raises(ContractError, match='declared answer structure'):
        generation_output_scope(spec, user_request=request, business={'policy': ''})


def test_nested_arrays_nulls_and_special_property_names_are_lossless():
    paths = [['a.b', None, 'x/y', None, '*'], ['a.b', None, 'secret']]
    scope = {'member_reviews': [{'path': canonical(p), 'kind': kind}
             for p, kind in zip(paths, ['answer', 'state_only'])]}
    value = {'a.b': [{'x/y': [{'*': 0, 'extra': 2}, None], 'secret': 3},
                     {'x/y': [], 'secret': 4}]}
    assert project_members(value, scope) == {'a.b': [{'x/y': [{'*': 0}, None]}, {'x/y': []}]}
    value['a.b'][0]['x/y'][0].pop('*')
    with pytest.raises(ContractError, match='lacks a required member'): project_members(value, scope)


def test_heterogeneous_overlapping_scopes_fail_instead_of_leaking_excluded_member():
    shape = {'role': {'type': 'array', 'item_shapes': [{'type': 'str'},
        {'type': 'object', 'properties': {'private': {'type': 'str'}}}]}}
    paths = member_paths(shape)
    scope = {'member_reviews': [{'path': p, 'kind': 'answer' if p == canonical(['role', None]) else 'optional'} for p in paths]}
    with pytest.raises(ContractError, match='overlapping'):
        project_members({'role': [{'private': 'hidden'}]}, scope)


def test_record_adapter_no_longer_requires_state_only_columns():
    request, truth, _, agent = reviewed()
    truth['record_roles'] = {'results': {'identity': 'id', 'fields': ['allowed', 'committed'], 'scope': 'read'}}
    spec = scope_spec(request, truth, agent)
    actual = apply_answer_obligations(truth, spec, user_request=request, policy='')
    descriptor = actual['record_roles']['results']
    assert descriptor['fields'] == ['allowed']
    layout = {'columns': [{'name': 'allowed', 'path': 'permission'}],
              'identity_path': 'identifier', 'details_path': '', 'scope_field': ''}
    normalized = record_projection([{'identifier': 'a', 'permission': True},
                                   {'identifier': 'b', 'permission': False}], layout, descriptor)
    assert normalized == actual['answer']['results']
    normalized[1]['allowed'] = True
    assert normalized != actual['answer']['results']


def test_scalar_boolean_adapter_retains_complete_independent_evidence():
    descriptor = {'kind': 'selected_boolean', 'identity': 'id', 'fields': ['allowed'], 'member': 'a',
                  'complete_records': [{'id': 'a', 'allowed': True}, {'id': 'b', 'allowed': False}]}
    actual = project_record_roles({'eligible': descriptor}, {'eligible': True}, {'member_reviews': []})
    assert actual['eligible'] == descriptor
    with pytest.raises(ContractError, match='complete qualification'):
        selected_boolean([{'id': 'a', 'allowed': True}, {'id': 'b', 'allowed': True}], actual['eligible'])
