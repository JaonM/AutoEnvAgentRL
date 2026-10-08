"""Disclosed choices can drive a stop before the protected downstream call."""
import copy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import (
    ContractUserSimulator, DeclarativeMetricEvaluator as E, EpisodeStore,
    ExpressionBusinessState, RewardStateSnapshots, SandboxError,
)
from tests.generation.test_bounded_corrections import bounded_source
from tests.generation.test_readonly_execution_routes import readonly_routes


def selected_routes():
    source, request = bounded_source()
    source['tool_implementations'][0]['projection'].append('region')
    routed, _ = readonly_routes()
    source['metrics'] = routed['metrics']
    count = {'at_tool': {'name': 'lookup_zone', 'expression': {'aggregate': {
        'table': 'zones', 'op': 'count', 'field': 'id', 'where': {
            'name': {'user_value': {'stage': 'warehouse'}},
            'region': {'user_value': {'stage': 'choice'}}}}}}}
    unique = {'op': 'eq', 'args': [count, {'literal': 1}]}
    source['execution_routes'] = copy.deepcopy(routed['execution_routes'])
    for route in source['execution_routes']:
        route['required_stages'] = ['warehouse', 'choice']
        route['when'] = copy.deepcopy(unique)
        if route['id'] == 'unresolved':
            route['when']['op'] = 'ne'
    source['metric_implementations'] = copy.deepcopy(routed['metric_implementations'])
    for target in source['metric_implementations'][0]['expected']['targets']:
        target['expression']['if']['condition'] = copy.deepcopy(unique)
    source['answer_contract'] = routed['answer_contract']
    success = source['scenarios'][0]
    success['execution_route'] = 'found'
    success['steps'][-1]['content'] = '{"temperature":18,"status":"found"}'
    first = copy.deepcopy(success['steps'][0])
    first.pop('capture')
    source['route_probes'] = [{'route_id': 'unresolved', 'initial_changes': [
        {'table': 'zones', 'where': {'id': 547}, 'values': {'name': '青松库'}}],
        'reference': {'calls': [first], 'answer': '{"temperature":0,"status":"lookup_not_unique"}'}}]
    return source, request


@pytest.mark.parametrize('unselected_duplicate', [False, True])
def test_disclosed_choice_routes_before_downstream_tool_with_actual_duplicate(tmp_path, unselected_duplicate):
    source, request = selected_routes()
    if unselected_duplicate:
        source['tables'][0]['rows'][1].update(name='青松库', region='西区')
        source['route_probes'][0]['initial_changes'][0]['values']['region'] = '东区'
    artifact = compile_source(source, root=tmp_path, request=request)
    report = artifact['generation_pipeline']['verification']['execution_route_coverage']
    assert report['probes'][0]['passed']
    assert artifact['task_spec']['disclosed_value_bindings']['choice'] == {
        'tool': 'read_temperature', 'argument': 'region'}
    probes = artifact['execution_route_probes'][0]['scenarios']
    assert all(step.get('tool_name') != 'read_temperature' for case in probes for step in case['steps'])


@pytest.mark.parametrize('defect', ['unknown_stage', 'inactive_stage', 'inconsistent_binding', 'undisclosed_branch'])
def test_observed_choice_requires_valid_active_disclosure_contract(tmp_path, defect):
    source, request = selected_routes()
    if defect == 'unknown_stage':
        source['execution_routes'][0]['when']['args'][0]['at_tool']['expression']['aggregate']['where']['region'] = {
            'user_value': {'stage': 'invented'}}
    elif defect == 'inactive_stage':
        source['execution_routes'][1]['required_stages'].remove('choice')
    elif defect == 'inconsistent_binding':
        source['interaction_contract']['variants'][0]['stages'][1]['id'] = 'different'
    else:
        branch = source['interaction_contract']['variants'][0]['stages'][1]['branches'][0]
        branch.update(advance=True, outcome='user_acceptance', user_reply='好的，继续。')
    with pytest.raises((ValueError, SandboxError)):
        compile_source(source, root=tmp_path, request=request, structural_preview=True)


@pytest.mark.parametrize('refuse', [False, True])
def test_user_value_requires_actual_disclosure_and_resets_between_episodes(tmp_path, refuse):
    source, request = bounded_source()
    compile_source(source, root=tmp_path, request=request, structural_preview=True)
    script = json.loads((tmp_path / 'data/user_simulation/user_scripts.json').read_text())[0]
    script['variables']['refuse'] = refuse
    profiles = json.loads((tmp_path / 'data/user_simulation/user_profiles.json').read_text())
    store = EpisodeStore(tmp_path / 'episode.sqlite3')
    store.reset(seed=17)
    user = ContractUserSimulator(store, scripts=[script], profiles=profiles)
    user.reset()
    expression = {'user_value': {'stage': 'choice'}}
    observed = lambda: E._value_expression(expression, ExpressionBusinessState(
        {}, snapshots=store.get_state(RewardStateSnapshots.KEY, {})))
    stages = script['interaction_protocol']['stages']
    conversation = []
    def turn(content):
        conversation.append({'role': 'assistant', 'content': content})
        result = user.turn(conversation)
        conversation.append({'role': 'user', 'content': result['user_query']})
        return result
    assert observed() is None  # Neither private script nor guessed assistant text counts.
    turn('用户已经选择东区，请直接继续。')
    assert observed() is None
    turn(stages[0]['reference_response'])
    store.event('tool_call', {'tool_name': 'lookup_zone', 'arguments': {'name': '青松库'}},
        {'records': [{'id': 821, 'regions': '东区、西区'}]})
    assert observed() is None
    result = turn(stages[1]['reference_response'])
    assert result['fsm_transition_applied']
    assert observed() == (None if refuse else '东区')
    assert not any(event['payload'].get('tool_name') == 'read_temperature' for event in store.replay()['events'])
    store.reset(seed=18)
    user.reset()
    assert observed() is None


def test_user_values_are_not_private_business_evidence():
    from tests.tasks.test_observed_reward_qualification import scoped_query_task
    from env_factory.tasks.task_quality import reward_target_qualification_failures
    task = scoped_query_task()
    task.setdefault('task_spec', {})['disclosed_value_bindings'] = {
        'choice': {'tool': 'read_inventory', 'argument': 'category'}}
    task['metric_implementations'][-2]['expected']['targets'][0]['expression'] = {'user_value': {'stage': 'choice'}}
    assert not reward_target_qualification_failures(task)
    task['metric_implementations'].pop()
    assert reward_target_qualification_failures(task)
