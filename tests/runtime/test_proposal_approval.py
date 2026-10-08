import copy
import json

import pytest

from env_factory.sandbox_runtime import ApprovalTerms, ContractToolRegistry, ContractUserSimulator, SandboxError
from .test_approval_terms import setup_booking


def proposal_booking(tmp_path):
    store, data, stage, lifecycle, tool = setup_booking(tmp_path)
    stage.update(id='approve', user_reply='Confirm this displayed booking.', retry_reply='Show the proposed parameters.')
    term = stage['approval_terms']['book']
    term['proposal_arguments'] = ['fee']
    term['postconditions'][0]['values']['daily_fee'] = {'proposal_argument': 'fee'}
    stage['reference_proposal'] = {'id': 1, 'fee': 900}
    tool['function']['parameters']['properties']['fee'] = {'type': 'integer'}
    tool['function']['parameters']['required'].append('fee')
    store.set_state('business_grants', [])
    store.set_state(ApprovalTerms.KEY, {})
    script = {'script_id': 's', 'states': [], 'transitions': [], 'interaction_protocol': {
        'request_mode': 'structured_v1', 'stages': [stage]}}
    user = ContractUserSimulator(store, profiles=[{'profile_id': 'p'}], scripts=[script])
    return store, data, stage, lifecycle, tool, script, user


def ask(user, script, state, proposal):
    request = {'intent': 'confirm_execution', 'tool': 'book', 'arguments': []}
    if proposal is not None:
        request['proposal'] = proposal
    content = json.dumps({'question': 'Please confirm these exact displayed parameters.', 'request': request})
    return user._protocol_turn(script, state, [{'role': 'assistant', 'content': content}])


def test_actual_proposal_is_frozen_and_echoed_by_user_confirmation(tmp_path):
    store, data, stage, lifecycle, tool, script, user = proposal_booking(tmp_path)
    state = {'script_id': 's', 'protocol_index': 0, 'turn_index': 0}
    proposal = {'id': 1, 'fee': 950}  # Different from the authored reference's 900.
    response = ask(user, script, state, proposal)
    assert response['interaction_stage'] == 'approve'
    assert '950' in response['user_query']
    proposal['fee'] = 1000
    assert store.get_state(ApprovalTerms.KEY)['book']['arguments'] == {'id': 1, 'fee': 950}
    registry = ContractToolRegistry([tool], {'book': lambda args: {
        'updated': data.update('bookings', {'id': args['id']}, {'daily_fee': args['fee']})}},
        event_recorder=store.event, lifecycle=lifecycle)
    for arguments in [{'id': 1, 'fee': 1000}, {'id': 2, 'fee': 950}, {'id': 1, 'fee': 950, 'extra': True}]:
        with pytest.raises(SandboxError):
            lifecycle.before('book', arguments, None)
    registry.execute('book', {'id': 1, 'fee': 950})
    assert data.table('bookings')[0]['daily_fee'] == 950


@pytest.mark.parametrize('proposal', [None, {'id': 1}, {'id': 1, 'fee': 950, 'extra': 0},
    {'id': True, 'fee': 950}, {'id': 2, 'fee': 950}, {'id': 1, 'fee': None},
    {'id': 1, 'fee': float('nan')}, {'id': 1, 'fee': {'$ref': 'offer'}}])
def test_missing_or_unbounded_proposal_never_grants_consent(tmp_path, proposal):
    store, data, stage, lifecycle, tool, script, user = proposal_booking(tmp_path)
    response = ask(user, script, {'script_id': 's', 'protocol_index': 0, 'turn_index': 0}, proposal)
    assert response['interaction_stage'] is None
    assert store.get_state('business_grants') == []
    assert store.get_state(ApprovalTerms.KEY) == {}
    with pytest.raises(SandboxError, match='authorization'):
        lifecycle.before('book', {'id': 1, 'fee': 900}, None)


def test_server_cannot_replace_approved_value_before_commit(tmp_path):
    store, data, stage, lifecycle, tool, script, user = proposal_booking(tmp_path)
    ask(user, script, {'script_id': 's', 'protocol_index': 0, 'turn_index': 0}, {'id': 1, 'fee': 950})
    registry = ContractToolRegistry([tool], {'book': lambda args: {
        'updated': data.update('bookings', {'id': 1}, {'daily_fee': 1000})}},
        event_recorder=store.event, lifecycle=lifecycle)
    with pytest.raises(SandboxError) as error:
        registry.execute('book', {'id': 1, 'fee': 950})
    assert error.value.code == 'AUTHORIZATION_TERMS_MISMATCH'
    assert data.table('bookings')[0]['daily_fee'] == 0


def test_revocation_clears_proposal_and_new_consent_replaces_it(tmp_path):
    store, data, stage, lifecycle, tool, script, user = proposal_booking(tmp_path)
    ApprovalTerms.apply(store, stage, proposal={'id': 1, 'fee': 950})
    ApprovalTerms.apply(store, {'grant_permissions': [], 'revoke_permissions': ['book']})
    assert store.get_state(ApprovalTerms.KEY) == {}
    ApprovalTerms.apply(store, stage, proposal={'id': 1, 'fee': 975})
    assert store.get_state(ApprovalTerms.KEY)['book']['postconditions'][0]['values']['daily_fee'] == 975


def test_postcondition_cannot_reference_an_undeclared_proposal_argument(tmp_path):
    _, _, stage, _, _, _, _ = proposal_booking(tmp_path)
    stage['approval_terms']['book']['postconditions'][0]['values']['daily_fee'] = {'proposal_argument': 'other'}
    with pytest.raises(ValueError, match='AUTHORIZATION_TERMS_INVALID'):
        ApprovalTerms.validate(stage)


def test_explicit_rejection_accepts_displayed_proposal_without_granting(tmp_path):
    store, data, stage, lifecycle, tool, script, user = proposal_booking(tmp_path)
    stage.update(grant_permissions=[], revoke_permissions=['book'], approval_terms={},
                 user_reply='Reject this booking.')
    response = ask(user, script, {'script_id': 's', 'protocol_index': 0, 'turn_index': 0},
                   {'id': 1, 'fee': 950})
    assert response['interaction_stage'] == 'approve'
    assert response['user_query'] == 'Reject this booking.'
    assert store.get_state('business_grants') == []
    assert store.get_state(ApprovalTerms.KEY) == {}
    with pytest.raises(SandboxError, match='authorization'):
        lifecycle.before('book', {'id': 1, 'fee': 950}, None)


def test_rejection_branch_can_refuse_proposal_outside_base_approval_limits(tmp_path):
    store, data, stage, lifecycle, tool, script, user = proposal_booking(tmp_path)
    stage['branches'] = [{'id': 'reject', 'when': [{'source': 'assistant_text', 'path': '$',
        'operator': 'contains', 'expected': 'Please'}], 'user_reply': 'Reject this booking.',
        'outcome': 'user_rejection', 'advance': True, 'grant_permissions': [],
        'revoke_permissions': ['book'], 'approval_terms': {}}]
    response = ask(user, script, {'script_id': 's', 'protocol_index': 0, 'turn_index': 0},
                   {'id': 2, 'fee': 950})
    assert response['interaction_branch'] == 'reject'
    assert response['outcome_category'] == 'user_rejection'
    assert store.get_state('business_grants') == []
    assert store.get_state(ApprovalTerms.KEY) == {}


@pytest.mark.parametrize('identifier,approved', [(1, False), (2, True)])
def test_grant_uses_effective_branch_terms_before_permission_is_written(tmp_path, identifier, approved):
    store, data, stage, lifecycle, tool, script, user = proposal_booking(tmp_path)
    terms = copy.deepcopy(stage['approval_terms'])
    terms['book']['arguments']['id'] = 2
    stage['branches'] = [{'id': 'alternative', 'when': [{'source': 'assistant_text', 'path': '$',
        'operator': 'contains', 'expected': 'Please'}], 'user_reply': 'Confirm this alternative.',
        'outcome': 'user_acceptance', 'advance': True, 'grant_permissions': ['book'],
        'approval_terms': terms}]
    response = ask(user, script, {'script_id': 's', 'protocol_index': 0, 'turn_index': 0},
                   {'id': identifier, 'fee': 950})
    assert bool(response['interaction_stage']) == approved
    assert bool(store.get_state('business_grants')) == approved
    if approved:
        assert store.get_state(ApprovalTerms.KEY)['book']['arguments'] == {'id': 2, 'fee': 950}
    else:
        assert response['interaction_branch'] is None
        assert store.get_state(ApprovalTerms.KEY) == {}
