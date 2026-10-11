import copy

import pytest

from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import lit, transaction
from env_factory_v2.expressions import ContractError
from env_factory_v2.runtime import Sandbox
from env_factory_v2.transaction_targets import matches_plan, project_plan
from env_factory_v2.verification import expected_result, run_scenario


def planned():
    source = transaction()
    source['rules'] = [rule for rule in source['rules'] if rule['id'] != 'transferred']
    source['goals'] = [{'id': 'transfer_outcome', 'rule_ids': ['nonnegative', 'conservation'],
        'transaction_plan': [{'tool': 'transfer', 'arguments': {
            'from_id': lit('A'), 'to_id': lit('B'), 'amount': lit(20)}}]}]
    return source


def consent_plan(approved=True):
    source = planned()
    source['goals'][0]['transaction_plan'][0]['completion'] = 'consent_or_unavailable'
    source['user']['facts']['approve_approve'] = approved
    return source


def consent_matches(sandbox):
    return matches_plan(sandbox.source['goals'][0]['transaction_plan'], sandbox.source,
                        sandbox.context(private=True), trace=sandbox.trace)


@pytest.mark.parametrize('approved', [True, False])
@pytest.mark.parametrize('wrong_proposal', [True, False])
def test_completion_binds_refusal_and_revocation_to_intended_arguments(approved, wrong_proposal):
    sandbox = Sandbox(compile_contract(consent_plan(approved)))
    sandbox.interact('approve', question='Confirm this transfer?', proposal={'tool': 'transfer',
        'arguments': {'from_id': 'B' if wrong_proposal else 'A',
                      'to_id': 'A' if wrong_proposal else 'B', 'amount': 20}})
    if approved:
        sandbox.revoke('approve')
    if wrong_proposal:
        with pytest.raises(ContractError, match='intended proposal'):
            consent_matches(sandbox)
    else:
        assert consent_matches(sandbox)


def test_eligible_completion_requires_execution_even_after_approval():
    sandbox = Sandbox(compile_contract(consent_plan()))
    with pytest.raises(ContractError, match='completed consent'):
        consent_matches(sandbox)
    sandbox.interact('approve', question='Confirm transfer?', proposal={'tool': 'transfer',
        'arguments': {'from_id': 'A', 'to_id': 'B', 'amount': 20}})
    with pytest.raises(ContractError, match='completed consent'):
        consent_matches(sandbox)


def test_completion_accepts_real_commit_and_rejects_collateral_writes():
    source = consent_plan()
    compiled = compile_contract(source)
    assert run_scenario(compiled, source['scenarios'][0])['outcome']['success']
    original = Sandbox(compiled)
    original.user.states['approve'] = 'consent.consumed'
    with pytest.raises(ContractError, match='no successful execution'):
        consent_matches(original)


@pytest.mark.parametrize('missing', [False, True])
def test_unavailable_transaction_needs_no_pointless_consent(missing):
    source = consent_plan()
    if missing:
        source['goals'][0]['transaction_plan'][0]['arguments']['from_id'] = lit('missing')
    else:
        source['rules'][0]['predicate'] = lit(False)
    sandbox = Sandbox(compile_contract(source))
    assert consent_matches(sandbox)
    sandbox.data['accounts'][0]['balance'] -= 20
    assert not consent_matches(sandbox)


@pytest.mark.parametrize('mutation', ['when', 'state_argument', 'time_guard'])
def test_derived_completion_rejects_circular_or_temporal_eligibility(mutation):
    source = consent_plan()
    intent = source['goals'][0]['transaction_plan'][0]
    if mutation == 'when':
        intent['when'] = lit(False)
    elif mutation == 'state_argument':
        intent['arguments']['from_id'] = {'ref': 'user_state.regions.approve'}
    else:
        source['rules'][0]['predicate'] = {'gt': [{'ref': 'time.tick'}, lit(1)]}
    with pytest.raises(ContractError):
        compile_contract(source)


def test_plan_replaces_duplicate_state_formula_and_preserves_negative_checks():
    compiled = compile_contract(planned())
    for scenario in compiled['source']['scenarios']:
        assert expected_result(scenario, run_scenario(compiled, scenario))


def test_plan_uses_actual_initial_state_and_detects_collateral_writes():
    source = planned()
    sandbox = Sandbox(compile_contract(source))
    sandbox.reset([{'table': 'accounts', 'key': 'A', 'column': 'balance', 'value': 80}])
    context = sandbox.context(private=True)
    before = copy.deepcopy(context)
    plan = source['goals'][0]['transaction_plan']
    target = project_plan(plan, source, context)
    assert target['accounts'] == [{'id': 'A', 'balance': 60}, {'id': 'B', 'balance': 30}]
    assert context == before
    context['data'] = copy.deepcopy(target)
    context['data']['accounts'].reverse()
    assert matches_plan(plan, source, context)
    context['data']['accounts'][0]['balance'] += 1
    assert not matches_plan(plan, source, context)


def test_plan_composes_multiple_intentions_and_conditional_no_write():
    source = planned()
    plan = source['goals'][0]['transaction_plan']
    plan.append(copy.deepcopy(plan[0]))
    context = Sandbox(compile_contract(source)).context(private=True)
    assert project_plan(plan, source, context)['accounts'] == [
        {'id': 'A', 'balance': 60}, {'id': 'B', 'balance': 50}]
    for intent in plan:
        intent['when'] = lit(False)
    assert matches_plan(plan, source, context)
    context['data']['accounts'][0]['balance'] -= 20
    assert not matches_plan(plan, source, context)


def test_runtime_claim_of_commit_cannot_replace_actual_state(monkeypatch):
    source = planned()
    compiled = compile_contract(source)
    original = Sandbox._transaction
    def corrupt(self, tool, args):
        result = original(self, tool, args)
        self.data['accounts'][0]['balance'] += 1
        return result
    monkeypatch.setattr(Sandbox, '_transaction', corrupt)
    result = run_scenario(compiled, source['scenarios'][0])
    assert not result['outcome']['success']


@pytest.mark.parametrize('expression', [{'ref': 'data.accounts.0.balance'}, {'ref': 'time.tick'},
                                      {'ref': 'evidence.transfer_result.committed'}])
def test_intention_cannot_copy_actual_results_or_infer_commit_time(expression):
    source = planned()
    source['goals'][0]['transaction_plan'][0]['arguments']['amount'] = expression
    with pytest.raises(ContractError):
        compile_contract(source)


def test_time_dependency_hidden_in_rule_is_rejected():
    source = planned()
    source['rules'].append({'id': 'clock_value', 'public': 'Time dependent value.',
                            'predicate': {'ref': 'time.tick'}})
    source['tools'][1]['operations'][0]['set']['balance'] = {
        'rule': {'id': 'clock_value', 'arguments': {}}}
    with pytest.raises(ContractError, match='explicit state goals'):
        compile_contract(source)


def test_create_projection_uses_operations_and_detects_missing_rows():
    source = planned()
    source['tools'][1]['operations'] = [{'table': 'accounts', 'key': lit('C'), 'create': True,
                                        'set': {'balance': {'ref': 'args.amount'}}}]
    source['invariants'] = []
    context = Sandbox(compile_contract(source)).context(private=True)
    plan = source['goals'][0]['transaction_plan']
    target = project_plan(plan, source, context)
    assert target['accounts'][-1] == {'id': 'C', 'balance': 20}
    assert not matches_plan(plan, source, context)
    context['data'] = target
    assert matches_plan(plan, source, context)
