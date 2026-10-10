"""Evidence extraction against actual sandbox execution, not model yield."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_runtime_evidence import observed_time, authorization_decisions, state_before_event, authorized_call_context
from measure_v2_natural import execute_reference
from env_factory_v2.examples import transaction
from env_factory_v2.contract import compile_contract


def test_time_is_bound_to_actual_call_and_survives_later_wait():
    compiled = compile_contract(transaction())
    scenario = next(s for s in compiled['source']['scenarios'] if s['expect_success'])
    world = execute_reference(compiled, scenario)
    event = next(t for t in world.trace if t.get('kind') == 'tool' and t.get('status') == 'ok')
    timestamp = observed_time(world, event['id'])
    world.wait(12.5)
    assert observed_time(world, event['id']) == timestamp
    assert world.time()['elapsed_seconds'] > timestamp
    with pytest.raises(ValueError): observed_time(world, 'missing_call')


@pytest.mark.parametrize('refuse', [False, True])
def test_actual_approval_or_refusal_and_exact_proposal_are_preserved(refuse):
    compiled = compile_contract(transaction())
    scenario = next(s for s in compiled['source']['scenarios'] if s['expect_success'])
    world = execute_reference(compiled, scenario, refuse=refuse)
    decisions = authorization_decisions(world)
    assert decisions
    assert all(d['approved'] is (not refuse) for d in decisions)
    first = decisions[0]
    assert first['proposal'] == world.trace[first['trace_index']]['proposal']
    first['proposal']['tool'] = 'tampered'
    assert authorization_decisions(world)[0]['proposal']['tool'] != 'tampered'


def test_historical_state_uses_actual_commit_deltas_without_reexecuting_rules():
    compiled = compile_contract(transaction())
    scenario = next(s for s in compiled['source']['scenarios'] if s['expect_success'])
    world = execute_reference(compiled, scenario)
    write_index = next(i for i, event in enumerate(world.trace) if event.get('kind') == 'tool'
        and world.tools[event['tool']]['kind'] == 'transaction')
    assert state_before_event(world, write_index) == world.initial
    world.wait(0.5)
    assert state_before_event(world, len(world.trace) - 1) == world.data


def test_virtual_event_history_and_failed_write_do_not_distort_snapshot():
    from env_factory_v2.expressions import ContractError
    source = transaction()
    source['events'] = [{'at_seconds': 8, 'table': 'accounts', 'key': 'A', 'set': {'balance': 77}}]
    compiled = compile_contract(source)
    scenario = next(s for s in compiled['source']['scenarios'] if s['expect_success'])
    world = execute_reference(compiled, scenario)
    write = next(t for t in world.trace if t.get('kind') == 'tool'
                 and world.tools[t['tool']]['kind'] == 'transaction')
    with pytest.raises(ContractError):
        world.call(write['tool'], write['arguments'], call_id='repeat_consumed_authorization')
    assert world.trace[-1]['status'] == 'error'
    world.wait(10.5)
    assert state_before_event(world, len(world.trace) - 1) == world.data
    assert next(r for r in world.data['accounts'] if r['id'] == 'A')['balance'] == 77
    world.tools[write['tool']]['operations'] = []
    assert state_before_event(world, len(world.trace) - 1) == world.data


def test_authorized_context_binds_exact_actual_proposal_and_historical_state():
    compiled = compile_contract(transaction())
    scenario = next(s for s in compiled['source']['scenarios'] if s['expect_success'])
    world = execute_reference(compiled, scenario)
    decision = authorization_decisions(world)[0]
    write = next(t for t in world.trace if t.get('kind') == 'tool'
                 and world.tools[t['tool']]['kind'] == 'transaction')
    context = authorized_call_context(world, write['id'], decision['trace_index'])
    assert context['authorized_state'] == context['precommit_state'] == world.initial
    assert context['authorization_time'] < context['call_time']
    world.trace[decision['trace_index']]['proposal']['arguments']['amount'] = 21
    with pytest.raises(ValueError):
        authorized_call_context(world, write['id'], decision['trace_index'])


def test_real_query_clock_flows_to_independent_quote_truth_and_preserves_other_tables():
    from env_factory_v2.examples import obj, query
    from env_factory_v2.runtime import Sandbox
    from v2_business_acceptance import runtime_business_truth, business_truth
    source = transaction()
    rows = [dict(id='Q1', starts_at=0, ends_at=1, pack=6, minimum_boxes=2)]
    properties = {key: {'type': 'string' if key == 'id' else 'integer'} for key in rows[0]}
    source['tables'].append({'id': 'quotes', 'primary_key': 'id', 'schema': obj(properties), 'rows': rows})
    source['tools'].append(query('read_quotes', 'quotes', obj({}), list(properties)))
    world = Sandbox(source)
    world.call('read_quotes', {}, call_id='actual_quote_read')
    world.wait(12.5)
    spec = {'kind': 'procurement_quote', 'params': {}, 'observations': {}}
    business = {'tables': [{'id': 'quotes', 'rows': rows}], 'policy': '报价起止边界均包含。'}
    truth = runtime_business_truth(spec, business, {'facts': {'quote_id': 'Q1'}},
        user_request='报价有效吗？', world=world, time_call_id='actual_quote_read')
    assert truth['answer'] == {'valid': True, 'minimum_units': 12}
    assert truth['runtime_evidence']['virtual_seconds'] == 1
    assert truth['initial'] == truth['final_state'] == world.initial
    assert world.time()['elapsed_seconds'] == 13.5
    assert business_truth(spec, business, {'facts': {'quote_id': 'Q1'}},
        user_request='报价有效吗？', world=world) == truth
    with pytest.raises(ValueError):
        runtime_business_truth(spec, business, {'facts': {'quote_id': 'Q1'}},
            user_request='报价有效吗？', world=world, time_call_id='invented_read')
