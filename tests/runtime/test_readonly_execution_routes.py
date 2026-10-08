from copy import deepcopy

import pytest

from env_factory.sandbox_runtime import (
    ContractRewardAggregator, ContractRewardGate, DeclarativeMetricEvaluator,
    ExecutionRoutes, SandboxError,
)


def contract():
    def count(tool):
        return {'tool_result': {'name': tool, 'path': '$.count'}}
    first_found = {'op': 'eq', 'args': [count('identity'), {'literal': 1}]}
    def later(operator):
        return {'op': 'and', 'args': [deepcopy(first_found), {'op': operator,
            'args': [count('details'), {'literal': 0}]}]}
    spec = {'environment_contract': {'mode': 'reference_data'},
        'training_contract': {'category': 'multi_step_agentic'},
        'tool_contracts': [{'name': n, 'execution': {'operation': 'select'}}
                           for n in ('identity', 'details')],
        'capability_dag': {'nodes': ['identity', 'details'], 'edges': [
            {'from_tool': 'identity', 'to_tool': 'details', 'result_path': '$.id', 'argument_path': '$.id'}]},
        'execution_routes': [
            {'id': 'missing', 'when': {'op': 'ne', 'args': [count('identity'), {'literal': 1}]},
             'required_tools': ['identity'], 'required_stages': [], 'writes': 'forbidden'},
            {'id': 'empty', 'when': later('eq'), 'required_tools': ['identity', 'details'],
             'required_stages': [], 'writes': 'forbidden'},
            {'id': 'found', 'when': later('gt'), 'required_tools': ['identity', 'details'],
             'required_stages': [], 'writes': 'forbidden'}],
        'process_metric_tools': {'p_identity': 'identity', 'p_details': 'details'}}
    metrics = [{'id': n, 'category': category, 'weight': weight, 'score_range': [0, 1]}
               for n, category, weight in [('p_identity', 'process', .1),
                   ('p_details', 'process', .1), ('answer', 'outcome', .8)]]
    return spec, metrics


def context(first=0, later=None):
    baseline = {'records': [{'id': 1}]}
    results = {'identity': {'count': first}}
    events = [{'event': 'tool_call', 'payload': {'tool_name': 'identity', 'arguments': {}},
               'result': {'id': 1}}]
    if later is not None:
        results['details'] = {'count': later}
        events.append({'event': 'tool_call', 'payload': {'tool_name': 'details', 'arguments': {'id': 1}},
                       'result': {'count': later}})
    return {'initial_business_state': baseline, 'business_state': deepcopy(baseline),
            'reward_state_snapshots': {'$tool_results': results}, 'trajectory': {'events': events}}


def reward(ctx, scores=None, spec=None):
    default, metrics = contract()
    return ContractRewardAggregator(metrics).aggregate(ContractRewardGate(spec or default, metrics).apply(
        scores or {'p_identity': 1, 'p_details': 1, 'answer': 1}, ctx))


def test_missing_first_record_does_not_require_unreachable_later_evidence():
    result = reward(context())
    assert result['execution_route'] == 'missing' and result['reward'] == pytest.approx(1)
    assert result['inactive_components'] == ['p_details']
    assert result['components']['p_details'] == 0


@pytest.mark.parametrize('first,later,route', [(1, None, None), (1, 0, 'empty'), (1, 2, 'found')])
def test_later_checkpoint_is_required_only_after_first_read_succeeds(first, later, route):
    report = reward(context(first, later))
    assert report['execution_route'] == route
    assert report['reward'] == (pytest.approx(1) if route else 0)


@pytest.mark.parametrize('defect', ['no_call', 'no_checkpoint', 'wrong_argument_score', 'wrong_answer',
                                  'changed_state', 'broken_dependency', 'overlap'])
def test_readonly_routes_reject_missing_or_incorrect_evidence(defect):
    ctx, scores, spec = context(), None, None
    if defect == 'no_call':
        ctx['trajectory']['events'] = []
    elif defect == 'no_checkpoint':
        ctx['reward_state_snapshots'] = {}
    elif defect == 'wrong_argument_score':
        scores = {'p_identity': 0, 'p_details': 1, 'answer': 1}
    elif defect == 'wrong_answer':
        scores = {'p_identity': 1, 'p_details': 1, 'answer': 0}
    elif defect == 'changed_state':
        ctx['business_state']['records'][0]['id'] = 8
    elif defect == 'broken_dependency':
        ctx = context(1, 2)
        ctx['trajectory']['events'][-1]['payload']['arguments']['id'] = 7
    else:
        spec, _ = contract()
        spec['execution_routes'][1]['when'] = deepcopy(spec['execution_routes'][0]['when'])
    assert reward(ctx, scores, spec)['reward'] < .2


def test_readonly_contract_cannot_smuggle_mutating_tools():
    spec, _ = contract()
    spec['tool_contracts'][1]['execution']['operation'] = 'update'
    with pytest.raises(SandboxError, match='readonly routes cannot contain mutating tools'):
        ExecutionRoutes(spec)


def test_checkpoint_must_belong_to_the_selected_routes_required_reads():
    spec, _ = contract()
    spec['execution_routes'][0]['when']['args'][0]['tool_result']['name'] = 'details'
    with pytest.raises(SandboxError, match='required reads of their route'):
        ExecutionRoutes(spec)


def test_stateful_first_read_stop_preserves_goals_and_rejects_writes_then_undo():
    spec, _ = contract()
    spec['environment_contract']['mode'] = 'stateful'
    spec['goal_contract'] = {'row_predicates': [
        {'table': 'records', 'where': {'id': 1}, 'values': {}, 'count': 1}],
        'requires_state_change': True, 'table_primary_keys': {'records': ['id']}}
    spec['tool_contracts'].append({'name': 'update', 'execution': {'operation': 'update'}})
    ctx = context()
    assert reward(ctx, spec=spec)['reward'] == pytest.approx(1)
    ctx['trajectory']['events'].append({'event': 'tool_call',
        'payload': {'tool_name': 'update', 'arguments': {}}, 'result': {}})
    # Same final state cannot disguise an inappropriate successful write.
    assert reward(ctx, spec=spec)['reward'] == 0
    ctx = context()
    ctx['business_state']['records'] = []
    assert reward(ctx, spec=spec)['reward'] == 0


def test_stateful_routes_still_require_typed_goals():
    spec, _ = contract()
    spec['environment_contract']['mode'] = 'stateful'
    with pytest.raises(SandboxError, match='typed business state goals'):
        ExecutionRoutes(spec)


@pytest.mark.parametrize('op,known,expected', [('and', False, False), ('and', True, None),
    ('or', True, True), ('or', False, None)])
def test_unobserved_boolean_operand_is_never_converted_to_false(op, known, expected):
    expr = {'op': op, 'args': [{'literal': known}, {'op': 'eq', 'args': [
        {'tool_result': {'name': 'later', 'path': '$.count'}}, {'literal': 0}]}]}
    assert DeclarativeMetricEvaluator._value_expression(expr, {}) is expected
