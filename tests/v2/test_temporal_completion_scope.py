"""Scope comes from declared business time, never from delayed final state."""
import copy
import pytest

from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.contracts.expressions import ContractError
from env_factory_v2.contracts.transaction_targets import matches_plan
from env_factory_v2.examples import lit
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.validation.verification import run_scenario
from test_transaction_targets import consent_plan


def anchored(*, at=0, opens=False):
    source = consent_plan()
    plan = source['goals'][0].pop('transaction_plan')
    plan[0]['eligibility_at_seconds'] = at
    source['goals'][0]['transaction_timeline'] = plan
    source['rules'].append({'id': 'window', 'public': 'Transfer during the declared time window.',
        'predicate': {('ge' if opens else 'lt'): [{'ref': 'time.elapsed_seconds'}, lit(10)]}})
    source['tools'][1]['rule_ids'].append('window')
    return source


def matches(sandbox):
    return matches_plan(sandbox.source['goals'][0]['transaction_timeline'], sandbox.source,
                        sandbox.context(private=True), trace=sandbox.trace)


def test_agent_delay_cannot_erase_initial_obligation():
    sandbox = Sandbox(compile_contract(anchored()))
    sandbox.wait(20)
    with pytest.raises(ContractError, match='completed consent'):
        matches(sandbox)


def test_completed_before_deadline_remains_complete_after_wait():
    source = anchored()
    scenario = copy.deepcopy(source['scenarios'][0])
    scenario['steps'].append({'kind': 'wait', 'seconds': 20})
    run = run_scenario(compile_contract(source), scenario)
    assert run['outcome']['success']


def test_legitimate_wait_then_execution_uses_future_scope():
    source = anchored(at=10, opens=True)
    scenario = copy.deepcopy(source['scenarios'][0])
    scenario['steps'].insert(0, {'kind': 'wait', 'seconds': 10})
    assert run_scenario(compile_contract(source), scenario)['outcome']['success']
    idle = Sandbox(compile_contract(source))
    idle.wait(20)
    with pytest.raises(ContractError, match='completed consent'):
        matches(idle)


def test_initial_unavailability_is_not_replaced_by_later_availability():
    sandbox = Sandbox(compile_contract(anchored(opens=True)))
    sandbox.wait(20)
    assert matches(sandbox)


def test_future_external_event_in_scope_requires_action():
    source = anchored(at=10, opens=True)
    source['rules'][-1]['predicate'] = {'ge': [{'ref': 'row.balance'}, lit(20)]}
    source['tables'][0]['rows'][1]['balance'] = 5
    source['events'] = [{'at_seconds': 10, 'table': 'accounts', 'key': 'B', 'set': {'balance': 30}}]
    sandbox = Sandbox(compile_contract(source))
    sandbox.wait(20)
    with pytest.raises(ContractError, match='completed consent'):
        matches(sandbox)


def test_initially_blocked_target_can_complete_without_pointless_authorization():
    source = anchored()
    source['rules'][-1]['predicate'] = lit(False)
    assert matches(Sandbox(compile_contract(source)))


@pytest.mark.parametrize('approved', [False, True])
def test_explicit_refusal_or_revocation_discharges_anchored_obligation(approved):
    source = anchored()
    source['user']['facts']['approve_approve'] = approved
    sandbox = Sandbox(compile_contract(source))
    sandbox.interact('approve', question='Confirm?', proposal={'tool': 'transfer', 'arguments': {
        'from_id': 'A', 'to_id': 'B', 'amount': 20}})
    if approved:
        sandbox.revoke('approve')
    sandbox.wait(20)
    assert matches(sandbox)


@pytest.mark.parametrize('value', [None, -1, True, 0.5, 'initial'])
def test_temporal_completion_requires_explicit_nonnegative_anchor(value):
    source = anchored()
    source['goals'][0]['transaction_timeline'][0]['eligibility_at_seconds'] = value
    with pytest.raises(ContractError, match='eligibility_at_seconds'):
        compile_contract(source)


def test_fixed_second_scope_rejects_action_tick_dependency():
    source = anchored()
    source['rules'][-1]['predicate'] = {'lt': [{'ref': 'time.tick'}, lit(10)]}
    with pytest.raises(ContractError, match='action ticks'):
        compile_contract(source)


def test_missing_execution_cannot_be_forged_by_consumed_lifecycle():
    sandbox = Sandbox(compile_contract(anchored()))
    sandbox.user.states['approve'] = 'consent.consumed'
    sandbox.wait(20)
    with pytest.raises(ContractError, match='no successful execution'):
        matches(sandbox)
