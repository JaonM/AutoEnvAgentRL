"""Conditional partial writes retain guards, atomicity, consent and shared truth."""
import copy

import pytest

from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import transaction, lit, ref
from env_factory_v2.expressions import ContractError, evaluate
from env_factory_v2.runtime import Sandbox
from env_factory_v2.transaction_semantics import TransactionUnavailable
from env_factory_v2.transaction_targets import project_transaction


def fixture():
    source = transaction()
    source['invariants'] = []
    tool = source['tools'][1]
    # Generic optional ledger update, not a transfer with an omitted debit.
    tool['operations'][0]['when'] = lit(False)
    tool['operations'][0]['key'] = ref('data.missing')
    source['rules'].append({'id': 'eligible_row', 'public': 'Only B may be updated.',
                           'predicate': {'eq': [ref('row.id'), lit('B')]}})
    tool['rule_ids'].append('eligible_row')
    world = Sandbox(source)
    args = {'from_id': 'A', 'to_id': 'B', 'amount': 20}
    world.interact('approve', question='Confirm update?', proposal={'tool': 'transfer', 'arguments': args})
    return world, args


def eligibility(world, args):
    return evaluate({'transaction_preconditions': {'tool': 'transfer', 'basis': 'current',
        'arguments': {key: lit(value) for key, value in args.items()}}}, world.context(private=True))


def test_skipped_target_and_guard_do_not_block_real_partial_commit():
    world, args = fixture()
    before = copy.deepcopy(world.data)
    assert eligibility(world, args)
    projected, eligible = project_transaction(world.tools['transfer'], world.source, before,
        world.context(args=args), check_eligibility=True)
    assert eligible and world.data == before
    result = world._transaction(world.tools['transfer'], args)
    assert world.data == projected and world.data['accounts'][0] == before['accounts'][0]
    assert len(result['changes']) == 1 and result['changes'][0]['key'] == 'B'
    assert world.user.states['approve'] == 'consent.consumed'


def test_all_skipped_cannot_fake_commit_or_consume_authorization():
    world, args = fixture()
    world.tools['transfer']['operations'][1]['when'] = lit(False)
    before, state, version = copy.deepcopy(world.data), copy.deepcopy(vars(world.user)), world.version
    assert not eligibility(world, args)
    projected, eligible = project_transaction(world.tools['transfer'], world.source, before,
        world.context(args=args), check_eligibility=True)
    assert projected == before and not eligible
    with pytest.raises(TransactionUnavailable, match='no applicable'):
        world._transaction(world.tools['transfer'], args)
    assert world.data == before and vars(world.user) == state and world.version == version


@pytest.mark.parametrize('condition', [lit(1), ref('data.missing')])
def test_bad_condition_remains_an_error_not_legitimate_unavailability(condition):
    world, args = fixture()
    world.tools['transfer']['operations'][1]['when'] = condition
    with pytest.raises(ContractError) as error:
        eligibility(world, args)
    assert not isinstance(error.value, TransactionUnavailable)


def test_conditions_use_pre_state_even_after_an_earlier_write():
    source = transaction(); source['invariants'] = []
    initial_balance = source['tables'][0]['rows'][0]['balance']
    source['tools'][1]['operations'][1]['when'] = {'eq': [ref('data.accounts.0.balance'), lit(initial_balance)]}
    world = Sandbox(source)
    args = {'from_id': 'A', 'to_id': 'B', 'amount': 20}
    world.interact('approve', question='Confirm?', proposal={'tool': 'transfer', 'arguments': args})
    assert len(world._transaction(world.tools['transfer'], args)['changes']) == 2


def test_later_enabled_failure_rolls_back_earlier_write_and_consent():
    source = transaction(); source['tools'][1]['operations'][1].update(when=lit(True), key=lit('missing'))
    world = Sandbox(source); args = {'from_id': 'A', 'to_id': 'B', 'amount': 20}
    world.interact('approve', question='Confirm?', proposal={'tool': 'transfer', 'arguments': args})
    before, state = copy.deepcopy(world.data), copy.deepcopy(vars(world.user))
    with pytest.raises(TransactionUnavailable): world._transaction(world.tools['transfer'], args)
    assert world.data == before and vars(world.user) == state


@pytest.mark.parametrize('condition', [ref('row.id'), {'transaction_preconditions': {
    'tool': 'transfer', 'basis': 'current', 'arguments': {}}}])
def test_condition_cannot_read_unresolved_row_or_recurse_into_transactions(condition):
    source = transaction(); source['tools'][1]['operations'][0]['when'] = condition
    with pytest.raises(ContractError): compile_contract(source)


def test_argument_used_only_in_condition_is_a_real_consumed_parameter():
    source = transaction(); tool = source['tools'][1]
    tool['parameters']['properties']['enabled'] = {'type': 'boolean'}
    tool['parameters']['required'].append('enabled')
    tool['operations'][0]['when'] = ref('args.enabled')
    compile_contract(source)


def test_partial_applicability_does_not_waive_global_invariants():
    world, args = fixture()
    original = transaction()
    world.source['invariants'] = original['invariants']
    before = copy.deepcopy(world.data)
    with pytest.raises(ContractError, match='invariant'):
        world._transaction(world.tools['transfer'], args)
    assert world.data == before and world.user.states['approve'] == 'consent.authorized'


def test_skipped_create_conflict_is_irrelevant_but_enabled_conflict_is_atomic():
    world, args = fixture()
    operation = world.tools['transfer']['operations'][0]
    operation.update(create=True, key=lit('A'), set={'balance': lit(0)})
    assert eligibility(world, args)
    operation['when'] = lit(True)
    assert not eligibility(world, args)
    with pytest.raises(TransactionUnavailable, match='already exists'):
        world._transaction(world.tools['transfer'], args)
    assert world.user.states['approve'] == 'consent.authorized'


def test_conditional_write_still_requires_live_authorization():
    world, args = fixture()
    world.revoke('approve')
    before = copy.deepcopy(world.data)
    with pytest.raises(ContractError): world._transaction(world.tools['transfer'], args)
    assert world.data == before and world.user.states['approve'] == 'consent.revoked'


@pytest.mark.parametrize('hidden', [False, True])
def test_static_plan_cannot_hide_time_dependency_in_operation_condition(hidden):
    source = transaction()
    condition = {'lt': [ref('time.elapsed_seconds'), lit(10)]}
    if hidden:
        source['rules'].append({'id': 'window', 'public': 'Before second ten.', 'predicate': condition})
        condition = {'rule': {'id': 'window', 'arguments': {}}}
    source['tools'][1]['operations'][0]['when'] = condition
    source['goals'] = [{'id': 'outcome', 'rule_ids': ['conservation'], 'transaction_plan': [
        {'tool': 'transfer', 'arguments': {'from_id': lit('A'), 'to_id': lit('B'), 'amount': lit(20)}}]}]
    with pytest.raises(ContractError, match='dependencies'):
        compile_contract(source)
    goal = source['goals'][0]
    goal['transaction_timeline'] = goal.pop('transaction_plan')
    compile_contract(source)
