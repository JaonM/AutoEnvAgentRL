"""Branch selection must agree across causal gates, user stages and weights."""
from copy import deepcopy

import pytest

from env_factory.sandbox_runtime import (
    ContractRewardAggregator, ContractRewardGate, ContractUserSimulator,
    DeclarativeMetricEvaluator, EpisodeStore, ExecutionRoutes, SandboxError,
)


def contract():
    fact = {'at_tool': {'name': 'stock', 'expression': {'lookup': {
        'table': 'stock', 'field': 'available', 'where': {'id': 1}}}}}
    condition = {'op': 'gte', 'args': [fact, {'literal': 12}]}
    shortage = {'op': 'lt', 'args': [fact, {'literal': 12}]}
    spec = {
        'training_contract': {'category': 'multi_step_agentic'},
        'tool_contracts': [{'name': name, 'execution': {'operation': operation}}
                           for name, operation in [('order', 'select'), ('stock', 'select'), ('reserve', 'update')]],
        'capability_dag': {'nodes': ['order', 'stock', 'reserve'], 'edges': [
            {'from_tool': 'order', 'to_tool': 'stock', 'result_path': '$.id', 'argument_path': '$.id'},
            {'from_tool': 'stock', 'to_tool': 'reserve', 'result_path': '$.id', 'argument_path': '$.id'}],
            'success_paths': [['order', 'stock', 'reserve']]},
        'goal_contract': {'requires_state_change': True, 'row_predicates': [
            {'table': 'stock', 'where': {'id': 1}, 'values': {}, 'count': 1,
             'value_expressions': {'available': {'if': {'condition': condition,
                 'then': {'op': 'sub', 'args': [fact, {'literal': 12}]}, 'else': fact}}}}],
            'table_primary_keys': {'stock': ['id']}},
        'execution_routes': [
            {'id': 'reserve', 'when': condition, 'required_tools': ['order', 'stock', 'reserve'],
             'required_stages': ['clarify', 'approve'], 'writes': 'required'},
            {'id': 'shortage', 'when': shortage, 'required_tools': ['order', 'stock'],
             'required_stages': ['clarify'], 'writes': 'forbidden'}],
        'process_metric_tools': {'p_order': 'order', 'p_stock': 'stock', 'p_reserve': 'reserve'},
        'requires_user_interaction': True,
    }
    metrics = [{'id': name, 'category': category, 'weight': weight, 'score_range': [0, 1]}
               for name, category, weight in [('p_order', 'process', .05), ('p_stock', 'process', .05),
                                               ('p_reserve', 'process', .1), ('answer', 'outcome', .8)]]
    return spec, metrics


def context(stock=8):
    initial = {'stock': [{'id': 1, 'available': stock}]}
    return {'initial_business_state': initial, 'business_state': deepcopy(initial),
            'reward_state_snapshots': {'stock': deepcopy(initial)},
            'trajectory': {'events': [
                {'event': 'user_protocol', 'payload': {'required_stages': ['clarify', 'approve']}},
                {'event': 'user_turn', 'result': {'interaction_stage': 'clarify', 'fsm_transition_applied': True}},
                {'event': 'tool_call', 'payload': {'tool_name': 'order', 'arguments': {}}, 'result': {'id': 1}},
                {'event': 'tool_call', 'payload': {'tool_name': 'stock', 'arguments': {'id': 1}}, 'result': {'id': 1}}]}}


def reward(ctx, scores=None, spec=None):
    default, metrics = contract()
    gate = ContractRewardGate(spec or default, metrics)
    scores = scores or {'p_order': 1, 'p_stock': 1, 'p_reserve': 0, 'answer': 1}
    return ContractRewardAggregator(metrics).aggregate(gate.apply(scores, ctx))


def test_shortage_skips_write_obligation_but_does_not_award_free_process_credit():
    report = reward(context())
    assert report['reward'] == pytest.approx(1)
    assert report['execution_route'] == 'shortage'
    assert report['inactive_components'] == ['p_reserve']
    assert report['components']['p_reserve'] == 0
    report = reward(context(), {'p_order': 1, 'p_stock': 1, 'p_reserve': 1, 'answer': 0})
    assert report['reward'] == pytest.approx(.1 / .9)


def test_success_requires_write_and_authorization():
    ctx = context(16)
    assert reward(ctx)['reward'] == 0
    ctx['business_state']['stock'][0]['available'] = 4
    ctx['trajectory']['events'].append({'event': 'tool_call', 'payload': {
        'tool_name': 'reserve', 'arguments': {'id': 1}}, 'result': {}})
    scores = {'p_order': 1, 'p_stock': 1, 'p_reserve': 1, 'answer': 1}
    assert reward(ctx, scores)['components']['answer'] == 0
    ctx['trajectory']['events'].append({'event': 'user_turn', 'result': {
        'interaction_stage': 'approve', 'fsm_transition_applied': True}})
    assert reward(ctx, scores)['reward'] == 1


def test_shortage_cannot_write_then_undo_or_skip_observation():
    ctx = context()
    ctx['trajectory']['events'].append({'event': 'tool_call', 'payload': {'tool_name': 'reserve'}, 'result': {}})
    assert reward(ctx)['reward'] == 0
    ctx = context()
    ctx['trajectory']['events'] = [e for e in ctx['trajectory']['events'] if e.get('payload', {}).get('tool_name') != 'order']
    assert reward(ctx)['reward'] == 0
    ctx = context()
    ctx['trajectory']['events'][-1]['payload']['arguments']['id'] = 2
    assert reward(ctx)['reward'] == 0


def test_route_conditions_cannot_follow_final_writes_or_missing_checkpoint():
    ctx = context(16)
    ctx['business_state']['stock'][0]['available'] = 8
    spec, _ = contract()
    assert ExecutionRoutes(spec).select(ctx)['id'] == 'reserve'
    ctx['reward_state_snapshots'] = {}
    assert reward(ctx)['reward'] == 0
    assert reward(ctx)['execution_route'] is None
    for values in ([True, True], [False, False]):
        spec, _ = contract()
        for route, value in zip(spec['execution_routes'], values):
            route['when'] = {'literal': value}
        assert reward(context(), spec=spec)['reward'] == 0
    spec, _ = contract()
    spec['execution_routes'][0]['when'] = {'op': 'gte', 'args': [
        {'lookup': {'table': 'stock', 'field': 'available', 'where': {'id': 1}}}, {'literal': 12}]}
    with pytest.raises(SandboxError, match='never live state'):
        ExecutionRoutes(spec)


def test_external_stock_change_does_not_count_as_agent_write():
    ctx = context(8)
    ctx['initial_business_state'] = {'stock': [{'id': 1, 'available': 16}]}
    ctx['trajectory']['events'].insert(0, {'event': 'external_business_change',
        'payload': {'table': 'stock', 'selector': {'id': 1}, 'values': {'available': 8}},
        'result': {'updated_count': 1}})
    assert reward(ctx)['reward'] == pytest.approx(1)


def test_user_can_finish_shortage_without_irrelevant_approval(tmp_path):
    spec, _ = contract()
    ctx = context(8)
    store = EpisodeStore(tmp_path / 'user.sqlite3')
    stages = [{'id': name, 'kind': kind, 'before_tool': before, 'assistant_contains_all': [name],
               'user_reply': 'ok', 'retry_reply': 'ask again'} for name, kind, before in [
                   ('clarify', 'information_required', 'order'), ('approve', 'execution_confirmation', 'reserve')]]
    script = {'script_id': 's', 'initial_state': 'stage-0', 'interaction_protocol': {'stages': stages}}
    user = ContractUserSimulator(store, profiles=[{'profile_id': 'p'}], scripts=[script],
        execution_routes=ExecutionRoutes(spec), route_context=lambda: ctx, completion_check=lambda: True)
    user.reset()
    state = store.get_state(user.STATE_KEY)
    user._protocol_turn(script, state, [{'role': 'assistant', 'content': 'clarify'}])
    result = user._protocol_turn(script, state, [{'role': 'assistant', 'content': 'Stock is insufficient.'}])
    assert result['should_end'] and result['outcome_category'] == 'goal_satisfied'
    skipped = [e for e in store.replay()['events'] if e['event'] == 'interaction_stage_inactive']
    assert skipped[0]['payload'] == {'stage_id': 'approve', 'execution_route': 'shortage'}
    assert not store.get_state('business_grants', [])
