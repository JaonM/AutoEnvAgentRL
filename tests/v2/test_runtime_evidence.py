"""Evidence extraction against actual sandbox execution, not model yield."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_runtime_evidence import (observed_time, authorization_decisions, state_before_event,
                                authorized_call_context, target_attempts, individual_submission_times)
from v2_runtime_evidence import target_authorizations
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


def test_target_times_use_actual_success_and_failure_attempts():
    from env_factory_v2.expressions import ContractError
    compiled = compile_contract(transaction())
    scenario = next(s for s in compiled['source']['scenarios'] if s['expect_success'])
    world = execute_reference(compiled, scenario)
    write = next(t for t in world.trace if t.get('kind') == 'tool'
                 and world.tools[t['tool']]['kind'] == 'transaction')
    times = individual_submission_times(world, 'accounts', ['A', 'B'])
    assert times == {'A': write['time']['elapsed_seconds'], 'B': write['time']['elapsed_seconds']}
    with pytest.raises(ContractError):
        world.call(write['tool'], write['arguments'], call_id='failed_second_attempt')
    attempts = target_attempts(world, 'accounts', 'A')
    assert [a['status'] for a in attempts] == ['ok', 'error']
    assert attempts[1]['time'] > attempts[0]['time']
    with pytest.raises(ValueError, match='one actual attempt'):
        individual_submission_times(world, 'accounts', ['A'])


def test_refusal_is_actual_target_evidence_without_requiring_a_write_attempt():
    compiled = compile_contract(transaction())
    scenario = next(s for s in compiled['source']['scenarios'] if s['expect_success'])
    world = execute_reference(compiled, scenario, refuse=True)
    assert target_attempts(world, 'accounts', 'A') == []
    decisions = target_authorizations(world, 'accounts', 'A')
    assert len(decisions) == 1 and decisions[0]['approved'] is False
    assert world.data == world.initial
    assert target_authorizations(world, 'accounts', 'unrelated') == []


def test_temporary_grant_runtime_adapter_handles_one_write_and_one_blocked_target():
    from prepare_v2_cross_business import access_cases
    from env_factory_v2.examples import obj, query, ref, lit
    from env_factory_v2.runtime import Sandbox
    from v2_business_acceptance import business_truth
    row = access_cases()[3]
    source = transaction()
    for table in row['business']['tables']:
        properties = {}
        for key, value in table['rows'][0].items():
            value = next((r[key] for r in table['rows'] if r[key] is not None), None)
            properties[key] = {'type': 'boolean' if type(value) is bool else
                               'integer' if type(value) is int or value is None else 'string'}
            if any(r[key] is None for r in table['rows']): properties[key]['nullable'] = True
        source['tables'].append({**table, 'primary_key': 'id', 'schema': obj(properties)})
    source['tools'].append(query('read_grants', 'grants', obj({}), list(row['business']['tables'][1]['rows'][0])))
    source['tools'].append({'id': 'grant_access', 'kind': 'transaction', 'description': 'Activate approved temporary access.',
        'parameters': obj({'grant_id': {'type': 'string'}}), 'rule_ids': [], 'authorization': 'grants',
        'result_bindings': [], 'user_bindings': {}, 'operations': [{'table': 'grants', 'key': ref('args.grant_id'),
            'set': {'active': lit(True), 'effect': lit('allow'), 'starts_at': ref('time.elapsed_seconds'), 'expires_at': lit(50)}}]})
    source['user']['facts']['approve_grants'] = True
    source['user']['stages'].append({'id': 'grants', 'kind': 'authorization', 'tool': 'grant_access', 'depends_on': []})
    world = Sandbox(source)
    world.call('read_grants', {}, call_id='grant_read')
    world.interact('grants', question='确认临时编辑到50秒吗？',
                   proposal={'tool': 'grant_access', 'arguments': {'grant_id': 'T0'}})
    world.call('grant_access', {'grant_id': 'T0'}, call_id='grant_write')
    truth = business_truth(row['acceptance'], row['business'], row['user'],
                           user_request=row['raw_request'], world=world)
    assert truth['final_state'] == world.data
    assert truth['runtime_evidence']['individual_times']['T0'] == 3
    assert truth['runtime_evidence']['individual_time_sources']['T1']['kind'] == 'no_transaction_or_consent'
    assert [r['committed'] for r in truth['answer']['results']] == [True, False]
    refused_world = Sandbox(source)
    refused_world.call('read_grants', {}, call_id='grant_read')
    refused = business_truth(row['acceptance'], row['business'], row['user'],
                             user_request=row['raw_request'], world=refused_world, refuse=True)
    assert refused['final_state'] == refused_world.data


@pytest.mark.parametrize('refuse', [False, True])
def test_large_purchase_runtime_adapter_reads_real_proposal_snapshot_and_created_order(refuse):
    from prepare_v2_cross_business import procurement_cases
    from env_factory_v2.examples import obj, query, ref, lit, array
    from env_factory_v2.runtime import Sandbox
    from v2_business_acceptance import business_truth
    row = procurement_cases()[4]
    source = transaction()
    for table in row['business']['tables']:
        if table['id'] == 'orders':
            schema = obj({'id': {'type': 'string'}, 'cost_cents': {'type': 'integer'},
                'lines': array(obj({'quote_id': {'type': 'string'}, 'quantity': {'type': 'integer'}}))})
        else:
            schema = obj({key: {'type': 'string' if isinstance(value, str) else 'integer'}
                          for key, value in table['rows'][0].items()})
        source['tables'].append({**table, 'primary_key': 'id', 'schema': schema})
    source['tools'].append(query('read_quotes', 'quotes', obj({}),
                                list(row['business']['tables'][0]['rows'][0]), page_size=128))
    source['tools'].append({'id': 'purchase', 'kind': 'transaction', 'description': 'Submit exact confirmed plan.',
        'parameters': obj({'order_id': {'type': 'string'}}), 'rule_ids': [], 'authorization': 'purchase',
        'result_bindings': [], 'user_bindings': {}, 'operations': [
            {'table': 'quotes', 'key': lit('V1'), 'set': {'stock': {'sub': [ref('row.stock'), lit(12)]}}},
            {'table': 'quotes', 'key': lit('V2'), 'set': {'stock': {'sub': [ref('row.stock'), lit(15)]}}},
            {'table': 'budgets', 'key': lit('P1'), 'set': {'remaining_cents': {'sub': [ref('row.remaining_cents'), lit(440)]}}},
            {'table': 'orders', 'key': ref('args.order_id'), 'create': True, 'set': {
                'cost_cents': lit(440), 'lines': lit([{'quote_id': 'V1', 'quantity': 12}, {'quote_id': 'V2', 'quantity': 15}])}}]})
    source['user']['facts']['approve_purchase'] = not refuse
    source['user']['stages'].append({'id': 'purchase', 'kind': 'authorization', 'tool': 'purchase', 'depends_on': []})
    world = Sandbox(source)
    page = world.call('read_quotes', {}, call_id='quotes_read')
    world.call('read_quotes', {}, call_id='quotes_read_2', cursor=page['next_cursor'])
    world.interact('purchase', question='确认购买方案吗？',
                   proposal={'tool': 'purchase', 'arguments': {'order_id': 'O1'}})
    if not refuse:
        world.call('purchase', {'order_id': 'O1'}, call_id='actual_purchase')
    truth = business_truth(row['acceptance'], row['business'], row['user'],
                           user_request=row['raw_request'], world=world, refuse=refuse)
    assert truth['final_state'] == world.data
    assert truth['answer'] == {'cost_cents': 440, 'overage': 4, 'committed': not refuse}
    assert truth['runtime_evidence']['authorization_index'] == 2
    assert len(truth['runtime_evidence']['submission_attempts']) == (0 if refuse else 1)
