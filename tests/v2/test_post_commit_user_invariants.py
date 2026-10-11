import copy

import pytest

from env_factory_v2.examples import transaction, ref, lit
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.contracts.contract import ContractError


def test_invariant_projects_consumption_without_premature_live_mutation():
    source = transaction()
    source['rules'].append({'id': 'consumed', 'public': '提交同时消费本次授权。',
        'predicate': {'eq': [ref('user_state.regions.approve'), lit('consent.consumed')]}})
    source['invariants'].append({'rule_id': 'consumed',
        'predicate': {'rule': {'id': 'consumed', 'arguments': {}}}})
    for balance, succeeds in [(100, True), (19, False)]:
        candidate = copy.deepcopy(source)
        candidate['tables'][0]['rows'][0]['balance'] = balance
        world = Sandbox(candidate)
        world.call('accounts', {}, call_id='read')
        args = {'from_id': {'$result': 'read', 'path': 'items.0.id'},
                'to_id': {'$result': 'read', 'path': 'items.1.id'}, 'amount': 20}
        world.interact('approve', question='批准？', proposal={'tool': 'transfer', 'arguments': args})
        before, history = copy.deepcopy(world.data), copy.deepcopy(world.user.history)
        if succeeds:
            world.call('transfer', args, call_id='write')
            assert world.user.states['approve'] == 'consent.consumed'
            assert world.data['accounts'][0]['balance'] == balance - 20
        else:
            with pytest.raises(ContractError, match='invariant'):
                world.call('transfer', args, call_id='write')
            assert world.data == before
            assert world.user.history == history
            assert world.user.states['approve'] == 'consent.authorized'
