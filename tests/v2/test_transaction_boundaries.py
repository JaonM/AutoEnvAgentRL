"""Transaction preconditions inspect pre-state; final invariants inspect post-state."""
import copy

import pytest

from env_factory_v2.examples import transaction, ref, lit
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.contracts.expressions import ContractError


def guarded_world(balance):
    source = transaction()
    source['tables'][0]['rows'][0]['balance'] = balance
    source['rules'].append({'id': 'sufficient', 'public': '源账户余额至少等于转账金额。',
        'predicate': {'ge': [{'sum': [{'map': {'items': {'filter': {
            'items': ref('data.accounts'), 'value': {'eq': [ref('row.id'), ref('args.from_id')]}}},
            'value': ref('row.balance')}}]}, ref('args.amount')]}})
    source['tools'][1]['rule_ids'].append('sufficient')
    world = Sandbox(source)
    world.call('accounts', {}, call_id='read')
    args = {'from_id': {'$result': 'read', 'path': 'items.0.id'},
            'to_id': {'$result': 'read', 'path': 'items.1.id'}, 'amount': 20}
    world.interact('approve', question='批准？', proposal={'tool': 'transfer', 'arguments': args})
    return world, args


def test_exact_funds_commit_and_insufficient_funds_rollback():
    world, args = guarded_world(20)
    world.call('transfer', args, call_id='write')
    assert [r['balance'] for r in world.data['accounts']] == [0, 30]
    world, args = guarded_world(19)
    before = copy.deepcopy(world.data)
    with pytest.raises(ContractError, match='transaction business precondition failed: sufficient'):
        world.call('transfer', args, call_id='write')
    assert world.data == before and world.user.states['approve'] == 'consent.authorized'


def test_post_state_invariants_still_block_negative_balance():
    source = transaction()
    source['tables'][0]['rows'][0]['balance'] = 19
    world = Sandbox(source)
    world.call('accounts', {}, call_id='read')
    args = {'from_id': {'$result': 'read', 'path': 'items.0.id'},
            'to_id': {'$result': 'read', 'path': 'items.1.id'}, 'amount': 20}
    world.interact('approve', question='批准？', proposal={'tool': 'transfer', 'arguments': args})
    before = copy.deepcopy(world.data)
    with pytest.raises(ContractError, match='invariant'):
        world.call('transfer', args, call_id='write')
    assert world.data == before
