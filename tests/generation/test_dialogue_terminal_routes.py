from copy import deepcopy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import SandboxError
from tests.generation.test_interaction_contract import interactive_source
from tests.generation.test_readonly_execution_routes import readonly_routes


def refusal_source():
    source, request = interactive_source()
    routed, _ = readonly_routes()
    request['require_structured_interactions'] = True
    for key in ['metrics', 'metric_implementations', 'execution_routes', 'answer_contract', 'route_probes']:
        source[key] = deepcopy(routed[key])
    source['description']['public_input']['initial_user_message'] += '如果我拒绝提供信息，停止查询并报告user_declined，温度填0。'
    source['scenarios'][0]['execution_route'] = 'found'
    source['scenarios'][0]['steps'][-1]['content'] = '{"temperature":18,"status":"found"}'
    for route in source['execution_routes']:
        route['required_stages'] = ['warehouse', 'choice']
    first = deepcopy(source['scenarios'][0]['steps'][0])
    first.pop('capture')
    source['route_probes'][0]['reference']['calls'] = [first]
    refused = {'user_rejection': {'stage': 'warehouse', 'branch': 'decline'}}
    source['execution_routes'].append({'id': 'declined', 'when': refused,
        'required_tools': [], 'required_stages': ['warehouse'], 'writes': 'forbidden'})
    for variant in source['interaction_contract']['variants']:
        variant['variables'] = {'refuse_warehouse': False}
        variant['stages'][0]['branches'] = [{'id': 'decline', 'when': [
            {'source': 'user_state', 'path': '$.refuse_warehouse', 'operator': 'eq', 'expected': True}],
            'user_reply': '我拒绝提供库区，请停止查询。', 'outcome': 'user_rejection', 'advance': False}]
    for target in source['metric_implementations'][0]['expected']['targets']:
        target['expression'] = {'if': {'condition': deepcopy(refused),
            'then': {'literal': 'user_declined' if target['key'] == 'status' else 0}, 'else': target['expression']}}
    source['route_probes'].append({'route_id': 'declined', 'initial_changes': [],
        'user_variables': {'refuse_warehouse': True},
        'reference': {'calls': [], 'answer': '{"temperature":0,"status":"user_declined"}'}})
    return source, request


def test_real_user_refusal_passes_without_tools_and_skipped_dialogue_fails(tmp_path):
    source, request = refusal_source()
    artifact = compile_source(source, request=request, root=tmp_path)
    probes = artifact['generation_pipeline']['verification']['execution_route_coverage']['probes']
    terminal = next(probe for probe in probes if probe['route_id'] == 'declined')
    assert terminal['passed'] and all(terminal['checks'].values())
    assert len([name for name in terminal['checks'] if name.startswith('reference_route_')]) == 3
    assert len([name for name in terminal['checks'] if name.startswith('skip_interaction_')]) == 3
    assert len([name for name in terminal['checks'] if name.startswith('wrong_answer_')]) == 3
    original = json.loads((tmp_path/'data/user_simulation/user_scripts.json').read_text())
    changed = json.loads((tmp_path/'route_probes/declined/data/user_simulation/user_scripts.json').read_text())
    assert all(not script['variables']['refuse_warehouse'] for script in original)
    assert all(script['variables']['refuse_warehouse'] for script in changed)


@pytest.mark.parametrize('defect', ['assistant_trigger', 'no_preference', 'unknown_preference', 'no_dialogue',
                                  'extra_read', 'advance', 'not_refusal', 'fake_fixture'])
def test_terminal_route_cannot_bypass_user_intent_or_execution_requirements(tmp_path, defect):
    source, request = refusal_source()
    branch = source['interaction_contract']['variants'][0]['stages'][0]['branches'][0]
    route = source['execution_routes'][-1]
    if defect == 'assistant_trigger': branch['when'].append({'source': 'assistant_text', 'path': '$', 'operator': 'contains', 'expected': '停止'})
    elif defect == 'no_preference': branch['when'][0].update(source='assistant_text', path='$', operator='contains', expected='停止')
    elif defect == 'unknown_preference': branch['when'][0]['path'] = '$.invented'
    elif defect == 'no_dialogue': route['required_stages'] = []
    elif defect == 'extra_read': route['required_tools'] = ['read_temperature']
    elif defect == 'advance': branch['advance'] = True
    elif defect == 'not_refusal': branch['outcome'] = 'user_acceptance'
    elif defect == 'fake_fixture': source['route_probes'][-1]['user_variables'] = {'invented': True}
    with pytest.raises((ValueError, SandboxError)) as failure:
        compile_source(source, request=request, root=tmp_path, structural_preview=True)
    assert (failure.value.code == 'EXECUTION_ROUTE_INVALID' if isinstance(failure.value, SandboxError)
            else 'EXECUTION_ROUTE_INVALID' in str(failure.value))


def test_asking_to_stop_does_not_overwrite_private_user_intent(tmp_path):
    from env_factory.sandbox_runtime import ContractUserSimulator, EpisodeStore, ExecutionRoutes, RewardStateSnapshots
    source, request = refusal_source()
    artifact = compile_source(source, request=request, root=tmp_path, structural_preview=True)
    users = artifact['user_simulation_manifest']
    scripts = json.loads((tmp_path/users['root']/users['scripts_file']).read_text())
    profiles = json.loads((tmp_path/users['root']/users['profiles_file']).read_text())
    store = EpisodeStore(tmp_path/'user_intent.sqlite3')
    store.reset(seed=17)
    routes = ExecutionRoutes(artifact['task_spec'])
    context = lambda: {'reward_state_snapshots': store.get_state(RewardStateSnapshots.KEY, {})}
    user = ContractUserSimulator(store, scripts=scripts, profiles=profiles,
                                 execution_routes=routes, route_context=context)
    user.reset()
    active = user.scripts[store.get_state(user.STATE_KEY)['script_id']]
    content = json.loads(active['interaction_protocol']['stages'][0]['reference_response'])
    content['question'] = '请告知目标库区，是否停止？是否取消？'
    answer = user.turn([{'role': 'assistant', 'content': json.dumps(content, ensure_ascii=False)}])
    assert answer['interaction_stage'] == 'warehouse'
    assert answer['interaction_branch'] is None
    assert routes.select(context()) is None
    # A different user intent is fixed before the episode, not by an agent utterance.
    for script in user.scripts.values(): script['variables']['refuse_warehouse'] = True
    store.reset(seed=17)
    user.reset()
    assert routes.select(context()) is None  # undisclosed preference alone is not reward evidence
    answer = user.turn([{'role': 'assistant', 'content': json.dumps(content, ensure_ascii=False)}])
    assert answer['interaction_branch'] == 'decline'
    assert routes.select(context())['id'] == 'declined'
    with pytest.raises(SandboxError) as stopped:
        user.guard_tool('lookup_zone')
    assert stopped.value.code == 'USER_REQUEST_STOPPED'


def test_refusal_can_be_an_actual_user_variant_not_only_a_probe(tmp_path):
    source, request = refusal_source()
    for target in source['metric_implementations'][0]['expected']['targets']:
        if target['key'] == 'temperature':
            target['expression']['if']['else']['if']['then'] = {'from_tool': {
                'name': 'read_temperature', 'field': 'temperature'}}
    source['scenarios'][0]['user_script_indices'] = [0, 1]
    source['scenarios'].append({'scenario_id': 'user_declines', 'kind': 'goal_success',
        'user_script_indices': [2], 'execution_route': 'declined',
        'steps': [{'operation': 'agent_response', 'content': '{"temperature":0,"status":"user_declined"}'}],
        'assertions': []})
    source['interaction_contract']['variants'][2]['variables']['refuse_warehouse'] = True
    source['route_probes'].pop()
    source['route_probes'][0]['user_variables'] = {'refuse_warehouse': False}
    artifact = compile_source(source, request=request, root=tmp_path)
    report = artifact['generation_pipeline']['verification']
    assert report['passed']
    assert report['execution_route_coverage']['references'] == ['declined', 'found']
    refused = next(case for case in artifact['acceptance_contract']['executable_scenarios']
                   if case['scenario_id'] == 'user_declines-script-3')
    assert not any(step.get('operation') == 'tool_call' for step in refused['steps'])
    assert any(step.get('expected_branch') == 'decline' for step in refused['steps'])
    # Qualification must accept a disclosed refusal without inventing the
    # missing private input, but still require the actual refusing user turn.
    from env_factory.generation.artifacts import write_task_artifact
    from env_factory.tasks.task import Task
    from env_factory.tasks.task_quality import _defined_interaction_inputs
    task_path = write_task_artifact(tmp_path, Task(artifact['task'], artifact['environment'],
        artifact['metrics'], task_intent=artifact['task_intent'], artifacts=artifact), 'multi_step_agentic')
    delivered = json.loads(task_path.read_text())
    assert _defined_interaction_inputs(delivered, tmp_path)
    for scenario in delivered['acceptance_contract']['executable_scenarios']:
        if scenario['scenario_id'] == 'user_declines-script-3':
            scenario['steps'] = [s for s in scenario['steps'] if s.get('operation') != 'dialogue_turn']
    assert not _defined_interaction_inputs(delivered, tmp_path)


def test_refusal_after_a_read_keeps_the_real_prefix_and_omits_later_read(tmp_path):
    source, request = refusal_source()
    source['tools'][0]['function']['parameters']['properties'].pop('region')
    source['tools'][0]['function']['parameters']['required'].remove('region')
    source['tool_implementations'][0]['filters'] = [f for f in source['tool_implementations'][0]['filters'] if f['argument'] != 'region']
    source['scenarios'][0]['steps'][0]['arguments'].pop('region')
    source['route_probes'][0]['reference']['calls'][0]['arguments'].pop('region')
    schema = source['tools'][1]['function']['parameters']
    schema['properties']['region'] = {'type': 'string', 'description': '用户确认的区域'}
    schema['required'].append('region')
    source['tool_implementations'][1]['filters'].append({'argument': 'region', 'column': 'region', 'operator': 'eq'})
    source['scenarios'][0]['steps'][1]['arguments']['region'] = '东区'
    source['execution_routes'][1]['required_stages'] = ['warehouse']
    second_refusal = {'user_rejection': {'stage': 'choice', 'branch': 'decline_region'}}
    for variant in source['interaction_contract']['variants']:
        variant['variables']['refuse_region'] = False
        stage = variant['stages'][1]
        stage.update(before_tool='read_temperature', bind_to={'tool': 'read_temperature', 'argument': 'region'}, requires_tools=['lookup_zone'])
        stage['branches'] = [{'id': 'decline_region', 'when': [
            {'source': 'user_state', 'path': '$.refuse_region', 'operator': 'eq', 'expected': True}],
            'user_reply': '我不提供区域，请停止查询。', 'outcome': 'user_rejection', 'advance': False}]
    for target in source['metric_implementations'][0]['expected']['targets']:
        target['expression']['if']['condition'] = {'op': 'or', 'args': [target['expression']['if']['condition'], second_refusal]}
    source['execution_routes'].append({'id': 'declined_region', 'when': second_refusal,
        'required_tools': ['lookup_zone'], 'required_stages': ['warehouse', 'choice'], 'writes': 'forbidden'})
    source['route_probes'].append({'route_id': 'declined_region', 'initial_changes': [],
        'user_variables': {'refuse_region': True}, 'reference': {
            'calls': [deepcopy(source['scenarios'][0]['steps'][0])],
            'answer': '{"temperature":0,"status":"user_declined"}'}})
    artifact = compile_source(source, request=request, root=tmp_path)
    probes = artifact['generation_pipeline']['verification']['execution_route_coverage']['probes']
    report = next(p for p in probes if p['route_id'] == 'declined_region')
    assert report['passed'] and all(report['checks'].values())
    assert len([key for key in report['checks'] if key.startswith('no_tools_')]) == 3
