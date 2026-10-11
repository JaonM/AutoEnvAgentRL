import copy

import pytest

from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.examples import transaction, lit, ref
from env_factory_v2.contracts.expressions import ContractError, evaluate
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.execution.reward import assess_process


def check(basis='initial', **arguments):
    return {'transaction_preconditions': {'tool': 'transfer', 'basis': basis,
        'arguments': {'from_id': lit('A'), 'to_id': lit('B'), 'amount': lit(20), **arguments}}}


def source():
    value = transaction()
    value['rules'].append({'id': 'window', 'public': 'New transfers before second 10 only.',
                           'predicate': {'lt': [ref('time.elapsed_seconds'), lit(10)]}})
    value['tools'][1]['rule_ids'].append('window')
    value['goals'].append({'id': 'resolution', 'rule_ids': ['window'], 'predicate': {'or': [
        {'eq': [ref('user_state.regions.approve'), lit('consent.consumed')]}, {'not': [check()]}]}})
    return value


def test_initial_and_current_are_explicit_clock_bases_not_completion_claims():
    world = Sandbox(source())
    before = copy.deepcopy({**vars(world), 'user': vars(world.user)})
    assert evaluate(check(), world.context(private=True)) is True
    assert {**vars(world), 'user': vars(world.user)} == before
    assert world.user.states['approve'] != 'consent.consumed'
    world.wait(10)
    assert evaluate(check('current'), world.context(private=True)) is False
    assert evaluate(check('initial'), world.context(private=True)) is True
    # Waiting out the opportunity cannot satisfy the initial-basis completion goal.
    assert not next(g for g in assess_process(world)['goals'] if g['id'] == 'resolution')['passed']


def test_initial_snapshot_includes_reset_mutations_and_zero_time_events():
    value = source()
    value['events'] = [{'at_seconds': 0, 'table': 'accounts', 'key': 'A', 'set': {'balance': 5}}]
    value['rules'].append({'id': 'funds', 'public': 'The source account has enough funds.',
        'predicate': {'ge': [ref('data.accounts.0.balance'), ref('args.amount')]}})
    value['tools'][1]['rule_ids'].append('funds')
    world = Sandbox(value)
    assert world.initial['accounts'][0]['balance'] != 5
    assert evaluate(check(), world.context(private=True)) is False
    value.pop('events')
    world = Sandbox(value)
    world.reset([{'table': 'accounts', 'key': 'A', 'column': 'balance', 'value': 5}])
    assert evaluate(check(), world.context(private=True)) is False
    world.data['accounts'][0]['balance'] = 100
    assert evaluate(check('current'), world.context(private=True)) is True
    assert evaluate(check(), world.context(private=True)) is False


def test_selected_record_constraints_are_taken_from_actual_transaction():
    value = source()
    value['rules'].append({'id': 'selected_record', 'public': 'Only B is an eligible destination.',
        'predicate': {'eq': [ref('args.to_id'), lit('B')]}})
    value['tools'][1]['rule_ids'].append('selected_record')
    world = Sandbox(value)
    assert evaluate(check(), world.context(private=True))
    assert not evaluate(check(to_id=lit('A')), world.context(private=True))
    assert not evaluate(check(to_id=lit('missing')), world.context(private=True))
    assert world.data == world.initial and world.trace == []


@pytest.mark.parametrize('location', ['rule', 'operation', 'invariant'])
def test_recursive_execution_contexts_cannot_invoke_task_precondition_queries(location):
    value = source()
    if location == 'rule': value['rules'][0]['predicate'] = check()
    elif location == 'operation': value['tools'][1]['operations'][0]['set']['balance'] = check()
    else: value['invariants'][0]['predicate'] = check()
    with pytest.raises(ContractError, match='allowed existing transaction'): compile_contract(value)


@pytest.mark.parametrize('mutation', ['unknown', 'query', 'missing_argument', 'extra_argument', 'implicit_basis', 'invalid_basis'])
def test_invalid_task_queries_fail_at_compilation(mutation):
    value = source(); entry = value['goals'][-1]['predicate']['or'][1]['not'][0]['transaction_preconditions']
    if mutation == 'unknown': entry['tool'] = 'unknown'
    elif mutation == 'query': entry['tool'] = 'accounts'
    elif mutation == 'missing_argument': entry['arguments'].pop('amount')
    elif mutation == 'extra_argument': entry['arguments']['unused'] = lit(1)
    elif mutation == 'invalid_basis': entry['basis'] = []
    else: entry.pop('basis')
    with pytest.raises(ContractError): compile_contract(value)


def test_invalid_arguments_and_malformed_rules_are_errors_not_unavailability():
    world = Sandbox(source())
    with pytest.raises(ContractError): evaluate(check(amount=lit('wrong')), world.context(private=True))
    world.rules['window']['predicate'] = lit(1)
    with pytest.raises(ContractError, match='boolean'): evaluate(check(), world.context(private=True))


def test_preconditions_do_not_bypass_authorization_or_final_invariants():
    value = source()
    world = Sandbox(value)
    args = {'from_id': {'$result': 'read', 'path': 'items.0.id'},
            'to_id': {'$result': 'read', 'path': 'items.1.id'}, 'amount': 20}
    assert evaluate(check(), world.context(private=True))
    world.call('accounts', {}, call_id='read')
    with pytest.raises(ContractError, match='authorization'): world.call('transfer', args, call_id='unauthorized')
    value['tables'][0]['rows'][0]['balance'] = 19
    world = Sandbox(value)
    assert evaluate(check(), world.context(private=True))  # No sufficient-funds precondition was declared.
    world.call('accounts', {}, call_id='read')
    world.interact('approve', question='Confirm?', proposal={'tool': 'transfer', 'arguments': args})
    with pytest.raises(ContractError, match='invariant'): world.call('transfer', args, call_id='write')
    assert world.user.states['approve'] == 'consent.authorized'
