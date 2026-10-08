import copy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.interaction_contract import compile_interactions
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, ContractUserSimulator, EpisodeStore, SandboxError
from tests.generation.test_interaction_contract import interactive_source


def source_with_reuse():
    source, request = interactive_source()
    request['require_structured_interactions'] = True
    source['metrics'][0]['weight'] = .8
    source['metrics'].append({**copy.deepcopy(source['metrics'][0]), 'id': 'read_with_disclosed_region',
        'category': 'process', 'scope': 'trajectory', 'target_action': 'read_temperature', 'weight': .2})
    tool = source['tools'][1]['function']['parameters']
    tool['properties']['region'] = {'type': 'string', 'description': '沿用用户已确认的区域'}
    tool['required'].append('region')
    source['tool_implementations'][1]['filters'].append({'argument': 'region', 'column': 'region', 'operator': 'eq'})
    source['scenarios'][0]['steps'][1]['arguments']['region'] = '东区'
    for variant in source['interaction_contract']['variants']:
        variant['stages'][1]['reuse_bindings'] = [{'tool': 'read_temperature', 'argument': 'region'}]
    return source, request


def test_one_disclosure_constrains_later_query_and_cannot_be_skipped(tmp_path):
    source, request = source_with_reuse()
    original = copy.deepcopy(source)
    artifacts = compile_source(source, root=tmp_path, request=request)
    assert source == original
    assert artifacts['generation_pipeline']['verification']['passed']
    rule = next(rule for rule in artifacts['metric_implementations']
                if rule.get('expected', {}).get('tool_name') == 'read_temperature')
    for options in rule['expected']['by_user_script'].values():
        assert all(option['disclosed_arguments']['region'] == '东区' for option in options)
    scripts = json.loads((tmp_path / 'data/user_simulation/user_scripts.json').read_text())
    profiles = json.loads((tmp_path / 'data/user_simulation/user_profiles.json').read_text())
    store = EpisodeStore(tmp_path / 'reuse.sqlite3')
    store.reset(seed=17)
    user = ContractUserSimulator(store, scripts=scripts, profiles=profiles)
    user.reset()
    with pytest.raises(SandboxError, match='complete required user interaction'):
        user.guard_tool('read_temperature')
    # Even an identical wrong reference/call cannot overwrite the disclosed fact.
    expected = copy.deepcopy(rule['expected']['by_user_script']['script-1'][0])
    expected['arguments']['region'] = '西区'
    events = [{'event': 'tool_call', 'payload': {'tool_name': 'lookup_zone', 'arguments': {'name': '青松库', 'region': '东区'}},
               'result': {'records': [{'id': 821}]}},
              {'event': 'tool_call', 'payload': {'tool_name': 'read_temperature', 'arguments': {'id': 821, 'region': '西区'}}, 'result': {}}]
    assert not DeclarativeMetricEvaluator._compare(events, 'contains_tool_call', expected)


@pytest.mark.parametrize('defect', ['changed_value', 'earlier_use', 'unknown_argument', 'duplicate', 'approval_reuse'])
def test_invalid_reuse_does_not_create_a_disclosure_shortcut(defect, tmp_path):
    source, request = source_with_reuse()
    stages = source['interaction_contract']['variants'][0]['stages']
    if defect == 'changed_value': source['scenarios'][0]['steps'][1]['arguments']['region'] = '西区'
    elif defect == 'earlier_use': source['scenarios'][0]['steps'][:2] = list(reversed(source['scenarios'][0]['steps'][:2]))
    elif defect == 'unknown_argument': stages[1]['reuse_bindings'][0]['argument'] = 'missing'
    elif defect == 'duplicate': stages[1]['reuse_bindings'] *= 2
    elif defect == 'approval_reuse':
        stages[1]['kind'] = 'execution_confirmation'
        source['tool_implementations'][0]['operation'] = 'update'
    with pytest.raises(ValueError, match='INTERACTION_REUSE_INVALID'):
        compile_interactions(source['interaction_contract'], source['description'], source['tools'], 3,
            implementations=source['tool_implementations'], scenarios=source['scenarios'],
            structured_requests=True, capture_bindings=[])


def test_reused_values_follow_each_script_instead_of_the_first_reference(tmp_path):
    source, request = source_with_reuse()
    schema = source['tools'][1]['function']['parameters']
    schema['properties']['name'] = {'type': 'string', 'description': '此前用户选择的库区'}
    schema['required'].append('name')
    source['tool_implementations'][1]['filters'].append({'argument': 'name', 'column': 'name', 'operator': 'eq'})
    original = source['scenarios'][0]
    original['user_script_indices'] = [0, 2]
    original['steps'][1]['arguments']['name'] = '青松库'
    alternate = copy.deepcopy(original)
    alternate.update(scenario_id='alternate', user_script_indices=[1])
    alternate['steps'][0]['arguments']['name'] = '白杨库'
    alternate['steps'][1]['arguments']['name'] = '白杨库'
    alternate['steps'][-1]['content'] = '温度：23摄氏度。'
    source['scenarios'].append(alternate)
    for index, variant in enumerate(source['interaction_contract']['variants']):
        stage = variant['stages'][0]
        stage['reuse_bindings'] = [{'tool': 'read_temperature', 'argument': 'name'}]
        if index == 1:
            stage.update(private_fact='白杨库', user_reply='请查询白杨库。')
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'tool_result': {'name': 'read_temperature', 'path': '$.records[0].temperature'}}
    result = compile_source(source, root=tmp_path, request=request)
    assert result['generation_pipeline']['verification']['passed']
    rule = next(rule for rule in result['metric_implementations'] if rule.get('expected', {}).get('tool_name') == 'read_temperature')
    assert [rule['expected']['by_user_script'][f'script-{i}'][0]['disclosed_arguments']['name'] for i in range(1, 4)] == ['青松库', '白杨库', '青松库']
