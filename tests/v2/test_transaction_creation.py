"""Atomic insertion is a platform capability, independent of yield measurement."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from measure_v2_natural import execute_reference
from v2_runtime_evidence import state_before_event
from env_factory_v2.examples import transaction, obj, lit
from env_factory_v2.contract import compile_contract, ContractError


def source_with_ledger():
    source = transaction()
    source['tables'].append({'id': 'ledger', 'primary_key': 'id',
        'schema': obj({'id': {'type': 'string'}, 'balance': {'type': 'integer'}, 'amount': {'type': 'integer'}}),
        'rows': []})
    write = next(tool for tool in source['tools'] if tool['kind'] == 'transaction')
    write['operations'].append({'table': 'ledger', 'key': lit('L1'), 'create': True,
                                'set': {'balance': lit(0), 'amount': lit(20)}})
    return source


def test_insert_commits_with_balance_updates_and_historical_reconstruction():
    compiled = compile_contract(source_with_ledger())
    scenario = next(s for s in compiled['source']['scenarios'] if s['expect_success'])
    world = execute_reference(compiled, scenario)
    assert world.data['ledger'] == [{'id': 'L1', 'balance': 0, 'amount': 20}]
    assert world.data['accounts'][0]['balance'] == 80
    world.wait(0)
    assert state_before_event(world, len(world.trace) - 1) == world.data
    refused = execute_reference(compiled, scenario, refuse=True)
    assert refused.data == refused.initial


def test_duplicate_insert_rolls_back_earlier_balance_updates_and_consent():
    from env_factory_v2.runtime import Sandbox
    source = source_with_ledger()
    source['tables'][-1]['rows'] = [{'id': 'L1', 'balance': 0, 'amount': 99}]
    world = Sandbox(source)
    scenario = next(s for s in world.source['scenarios'] if s['expect_success'])
    read = scenario['steps'][0]
    world.call(read['tool'], read['arguments'], call_id=read['id'])
    approval = next(step for step in scenario['steps'] if step['kind'] == 'user')
    world.interact(approval['stage'], question=approval['question'], proposal=approval['proposal'])
    before = copy.deepcopy(world.data)
    write = next(step for step in scenario['steps'] if step.get('tool') == 'transfer')
    with pytest.raises(ContractError, match='already exists'):
        world.call(write['tool'], write['arguments'], call_id='duplicate')
    assert world.data == before
    assert world.user.states[approval['stage']] == 'consent.authorized'


def test_create_requires_complete_row_and_forbids_nonexistent_row_reads():
    source = source_with_ledger()
    write = next(tool for tool in source['tools'] if tool['kind'] == 'transaction')
    write['operations'][-1]['set'].pop('amount')
    with pytest.raises(ContractError, match='every required'):
        compile_contract(source)
    source = source_with_ledger()
    next(t for t in source['tools'] if t['kind'] == 'transaction')['operations'][-1]['set']['amount'] = {'ref': 'row.amount'}
    with pytest.raises(ContractError):
        compile_contract(source)


def test_missing_created_record_is_an_independent_reward_counterexample(monkeypatch):
    import v2_business_acceptance as acceptance
    compiled = compile_contract(source_with_ledger())
    scenario = next(s for s in compiled['source']['scenarios'] if s['expect_success'])
    world = execute_reference(compiled, scenario)
    truth = {'initial': world.initial, 'final_state': world.data}
    inspected = []
    def judge_state(world, answer, **kwargs):
        inspected.append(copy.deepcopy(world.data['ledger']))
        return {'success': bool(world.data['ledger'])}
    monkeypatch.setattr(acceptance, 'grade', judge_state)
    checks = acceptance.state_counterexamples(compiled, scenario, truth, None)
    omitted = next(c for c in checks if c['id'] == 'missing_created_record_ledger_L1')
    assert omitted['passed']
    assert [] in inspected
