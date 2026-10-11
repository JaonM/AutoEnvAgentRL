import copy

import pytest

from env_factory_v2.examples import transaction, lit, ref
from env_factory_v2.expressions import ContractError
from env_factory_v2.runtime import Sandbox
from env_factory_v2.transaction_semantics import prepare_transaction, TransactionUnavailable
from env_factory_v2.transaction_targets import project_transaction


def ready(source):
    world = Sandbox(source)
    args = {'from_id': 'A', 'to_id': 'B', 'amount': 20}
    world.interact('approve', question='Confirm?', proposal={'tool': 'transfer', 'arguments': args})
    return world, args


def test_sequential_create_then_update_has_one_shared_pure_result():
    source = transaction()
    source['invariants'] = []
    source['tools'][1]['operations'] = [
        {'table': 'accounts', 'key': lit('C'), 'create': True, 'set': {'balance': ref('args.amount')}},
        {'table': 'accounts', 'key': lit('C'), 'set': {'balance': {'add': [ref('row.balance'), lit(1)]}}}]
    world, args = ready(source)
    before = copy.deepcopy(world.data)
    context = world.context(args=args)
    snapshot = copy.deepcopy(context)
    candidate, changes = prepare_transaction(world.tools['transfer'], world.source, before, context)
    assert candidate['accounts'][-1] == {'id': 'C', 'balance': 21}
    assert changes == [{'table': 'accounts', 'key': 'C', 'set': {'balance': 20}, 'create': True},
                       {'table': 'accounts', 'key': 'C', 'set': {'balance': 21}}]
    assert context == snapshot and world.data == before
    projected, eligible = project_transaction(world.tools['transfer'], world.source, before, context,
                                              check_eligibility=True)
    assert eligible and projected == candidate
    world._transaction(world.tools['transfer'], args)
    assert world.data == candidate and world.user.states['approve'] == 'consent.consumed'


@pytest.mark.parametrize('failure', ['missing_target', 'existing_create', 'false_guard'])
def test_real_target_failures_reject_execution_and_do_not_change_projected_world(failure):
    source = transaction()
    if failure == 'missing_target': source['tools'][1]['operations'][0]['key'] = lit('absent')
    elif failure == 'existing_create':
        source['tools'][1]['operations'][0].update(create=True, set={'balance': lit(0)})
    else: source['rules'][0]['predicate'] = lit(False)
    world, args = ready(source)
    before, lifecycle = copy.deepcopy(world.data), copy.deepcopy(vars(world.user))
    projected, eligible = project_transaction(world.tools['transfer'], world.source, before,
        world.context(args=args), check_eligibility=True)
    assert not eligible and projected == before
    with pytest.raises(TransactionUnavailable): world._transaction(world.tools['transfer'], args)
    assert world.data == before and vars(world.user) == lifecycle


@pytest.mark.parametrize('predicate', [lit(1), {'eq': [ref('data.missing'), lit(0)]}])
def test_broken_guard_is_not_reclassified_as_legitimate_unavailability(predicate):
    source = transaction(); source['rules'][0]['predicate'] = predicate
    world, args = ready(source)
    for run in [lambda: world._transaction(world.tools['transfer'], args),
                lambda: project_transaction(world.tools['transfer'], world.source, world.data,
                    world.context(args=args), check_eligibility=True)]:
        with pytest.raises(ContractError) as error: run()
        assert not isinstance(error.value, TransactionUnavailable)
    assert world.user.states['approve'] == 'consent.authorized'


def test_pure_precondition_success_never_waives_commit_invariants():
    source = transaction(); source['tables'][0]['rows'][0]['balance'] = 19
    world, args = ready(source)
    before, lifecycle = copy.deepcopy(world.data), copy.deepcopy(vars(world.user))
    projected, eligible = project_transaction(world.tools['transfer'], world.source, before,
        world.context(args=args), check_eligibility=True)
    assert eligible and projected['accounts'][0]['balance'] == -1
    with pytest.raises(ContractError, match='invariant'): world._transaction(world.tools['transfer'], args)
    assert world.data == before and vars(world.user) == lifecycle
