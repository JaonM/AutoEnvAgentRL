"""One actual user decision grants distinct, immutable operation scopes."""
from copy import deepcopy
import json

import pytest

from env_factory.sandbox_runtime import (ApprovalTerms, BusinessLifecycle, ContractToolRegistry,
                                         ContractUserSimulator, SandboxError)
from tests.runtime.test_approval_terms import setup_booking
from rl.environments.user_interaction import interaction_tool, interaction_content


def group_booking(tmp_path):
    store, data, _, _, book = setup_booking(tmp_path)
    data.replace_table('bookings', [{'id': i, 'daily_fee': 0} for i in [1, 2, 3]])
    store.set_state('business_grants', [])
    store.set_state(ApprovalTerms.KEY, {})
    stage = {'id': 'approve', 'kind': 'execution_confirmation', 'before_tool': 'book',
        'grant_permissions': ['book', 'release'], 'user_reply': 'Approve exactly these two operations.',
        'retry_reply': 'Show both operations and their exact targets.',
        'approval_terms': {
            'book': {'scope': 'terms', 'tool': 'book', 'arguments': {'id': 1},
                     'proposal_arguments': ['fee'], 'postconditions': [
                         {'table': 'bookings', 'where': {'id': 1},
                          'values': {'daily_fee': {'proposal_argument': 'fee'}}, 'count': 1}]},
            'release': {'scope': 'terms', 'tool': 'release', 'proposal_arguments': ['ids', 'fee'],
                        'postconditions': [{'table': 'bookings', 'where': {'id': i},
                                            'values': {'daily_fee': {'proposal_argument': 'fee'}}, 'count': 1}
                                           for i in [2, 3]]}}}
    proposal = {'book': {'id': 1, 'fee': 900}, 'release': {'ids': [2, 3], 'fee': 100}}
    stage['reference_proposal'] = deepcopy(proposal)
    book['function']['parameters']['properties']['fee'] = {'type': 'integer'}
    book['function']['parameters'].update(required=['id', 'fee'], additionalProperties=False)
    release = {'type': 'function', 'function': {'name': 'release', 'parameters': {
        'type': 'object', 'properties': {'ids': {'type': 'array', 'items': {'type': 'integer'}},
                                       'fee': {'type': 'integer'}},
        'required': ['ids', 'fee'], 'additionalProperties': False}}}
    lifecycle = BusinessLifecycle({'version': '1.0', 'tools': {
        'book': {'requires_grants': ['book']}, 'release': {'requires_grants': ['release']}}},
        data, store, ['book', 'release'])
    script = {'script_id': 's', 'states': [], 'transitions': [], 'interaction_protocol': {
        'request_mode': 'structured_v1', 'stages': [stage]}}
    user = ContractUserSimulator(store, profiles=[{'profile_id': 'p'}], scripts=[script])
    tools = [book, release]
    registry = ContractToolRegistry(tools, {
        'book': lambda args: {'updated': data.update('bookings', {'id': args['id']}, {'daily_fee': args['fee']})},
        'release': lambda args: {'updated': sum(data.update('bookings', {'id': i}, {'daily_fee': args['fee']})
                                              for i in args['ids'])}},
        event_recorder=store.event, lifecycle=lifecycle)
    return store, data, stage, proposal, script, user, tools, registry


def ask(user, script, proposal, tools):
    native = interaction_tool({'materials': [{'name': 'user_interaction_format'}]}, tools)
    text = interaction_content({'question': 'Approve the displayed operations and exact parameters?',
        'business_tool': 'book', 'requested_fields': [], 'intent': 'confirm_execution',
        'proposal': proposal}, native['function']['parameters'])
    return user._protocol_turn(script, {'script_id': 's', 'protocol_index': 0, 'turn_index': 0},
                               [{'role': 'assistant', 'content': text}])


def test_native_group_consent_executes_both_operations_with_different_values(tmp_path):
    store, data, stage, proposal, script, user, tools, registry = group_booking(tmp_path)
    response = ask(user, script, proposal, tools)
    assert response['interaction_stage'] == 'approve'
    assert json.loads(response['user_query'].split('\n')[-1])['approved_arguments'] == proposal
    registry.execute('book', proposal['book'])
    registry.execute('release', proposal['release'])
    assert [r['daily_fee'] for r in data.table('bookings')] == [900, 100, 100]
    terms = store.get_state(ApprovalTerms.KEY)
    assert terms['book']['tool'] == 'book' and terms['release']['tool'] == 'release'
    assert terms['release']['arguments']['ids'] == [2, 3]


@pytest.mark.parametrize('change', ['target', 'fee', 'membership', 'order', 'extra', 'cross_operation'])
def test_group_approval_cannot_be_reused_for_changed_operation(tmp_path, change):
    store, data, stage, proposal, script, user, tools, registry = group_booking(tmp_path)
    ask(user, script, proposal, tools)
    original = deepcopy(proposal)
    if change == 'target': proposal['book']['id'] = 2
    if change == 'fee': proposal['book']['fee'] = 901
    if change == 'membership': proposal['release']['ids'].append(1)
    if change == 'order': proposal['release']['ids'].reverse()
    if change == 'extra': proposal['release']['extra'] = True
    if change == 'cross_operation': proposal['book']['fee'] = proposal['release']['fee']
    name = 'book' if change in {'target', 'fee', 'cross_operation'} else 'release'
    with pytest.raises(SandboxError):
        registry.execute(name, proposal[name])
    assert [r['daily_fee'] for r in data.table('bookings')] == [0, 0, 0]
    assert store.get_state(ApprovalTerms.KEY)['release']['arguments'] == original['release']


@pytest.mark.parametrize('change', ['missing_operation', 'extra_operation', 'missing_argument', 'extra_argument'])
def test_incomplete_or_overbroad_group_never_grants(tmp_path, change):
    store, _, _, proposal, script, user, tools, _ = group_booking(tmp_path)
    if change == 'missing_operation': proposal.pop('release')
    if change == 'extra_operation': proposal['delete'] = {'id': 1}
    if change == 'missing_argument': proposal['release'].pop('ids')
    if change == 'extra_argument': proposal['release']['other'] = 1
    assert ask(user, script, proposal, tools)['interaction_stage'] is None
    assert store.get_state('business_grants') == []
    assert store.get_state(ApprovalTerms.KEY) == {}


def test_group_postcondition_failure_rolls_back_all_rows_of_failing_operation(tmp_path):
    store, data, _, proposal, script, user, tools, registry = group_booking(tmp_path)
    ask(user, script, proposal, tools)
    registry.execute('book', proposal['book'])
    wrong = ContractToolRegistry(tools, {**registry.handlers, 'release': lambda args: {
        'updated': sum(data.update('bookings', {'id': i}, {'daily_fee': 101}) for i in args['ids'])}},
        event_recorder=store.event, lifecycle=registry.lifecycle)
    with pytest.raises(SandboxError, match='approved terms'):
        wrong.execute('release', proposal['release'])
    assert [r['daily_fee'] for r in data.table('bookings')] == [900, 0, 0]
    registry.execute('release', proposal['release'])
    assert [r['daily_fee'] for r in data.table('bookings')] == [900, 100, 100]


def test_revoking_group_member_preserves_other_operation_terms(tmp_path):
    store, _, stage, proposal, script, user, tools, registry = group_booking(tmp_path)
    ask(user, script, proposal, tools)
    stage.update(grant_permissions=[], revoke_permissions=['release'], approval_terms={})
    assert ask(user, script, proposal, tools)['interaction_stage'] == 'approve'
    assert set(store.get_state(ApprovalTerms.KEY)) == {'book'}
    registry.execute('book', proposal['book'])
    with pytest.raises(SandboxError, match='authorization'):
        registry.execute('release', proposal['release'])


@pytest.mark.parametrize('value', [float('nan'), {'$ref': 'ids'}, {'tool_result': {'name': 'offer'}},
                                   list(range(1025)), {'nested': [[[[[[[[[1]]]]]]]]]}])
def test_native_approval_rejects_expressions_and_unbounded_values(tmp_path, value):
    _, _, _, proposal, script, user, tools, _ = group_booking(tmp_path)
    proposal['release']['ids'] = value
    with pytest.raises(SandboxError):
        ask(user, script, proposal, tools)


@pytest.mark.parametrize('value', [10 ** 5000, '\ud800', 'x' * 32769, {'a': list(range(1024)), 'b': list(range(1024))}],
                         ids=['oversize_integer', 'invalid_unicode', 'oversize_string', 'total_node_budget'])
def test_literal_bounds_reject_unserializable_or_oversize_data_without_crashing(value):
    assert ApprovalTerms.literal(value) is False


def test_nested_literal_objects_are_frozen_without_becoming_expressions(tmp_path):
    store, data, stage, proposal, script, user, tools, registry = group_booking(tmp_path)
    term = stage['approval_terms']['release']
    term['proposal_arguments'].append('metadata')
    proposal['release']['metadata'] = {'reason': 'rebalance', 'labels': ['A', 'B']}
    ask(user, script, proposal, tools)
    proposal['release']['metadata']['labels'].append('C')
    assert store.get_state(ApprovalTerms.KEY)['release']['arguments']['metadata']['labels'] == ['A', 'B']
    with pytest.raises(SandboxError, match='approved terms'):
        ApprovalTerms.check(store, data, 'release', ['release'], arguments=proposal['release'])
