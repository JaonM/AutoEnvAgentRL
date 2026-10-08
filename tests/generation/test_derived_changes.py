from copy import deepcopy

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.delivery_preflight import verify_delivery
from .test_dynamic_state_goal import dynamic_source


def derived_source():
    source, request = dynamic_source()
    expression = {'at_tool': {'name': 'read_temperature', 'expression':
        deepcopy(source['semantic_goal']['row_predicates'][0]['value_expressions']['temperature'])}}
    source['semantic_goal']['row_predicates'][0]['value_expressions']['temperature'] = expression
    tool = source['tools'][-1]['function']['parameters']
    tool['properties'].pop('value')
    tool['required'].remove('value')
    implementation = source['tool_implementations'][-1]
    implementation['changes'] = {}
    implementation['derived_changes'] = {'temperature': deepcopy(expression)}
    for scenario in source['scenarios']:
        for step in scenario['steps']:
            if step.get('tool_name') == 'update_temperature':
                step['arguments'].pop('value', None)
        scenario['steps'].insert(0, {'operation': 'reset', 'body': {'seed': 17}})
        scenario['steps'].append({'operation': 'reward'})
    return source, request


def test_server_derived_write_compiles_replays_and_passes_delivery_gate(tmp_path):
    source, request = derived_source()
    result = compile_source(source, request=request, root=tmp_path)
    assert result['generation_pipeline']['verification']['passed']
    assert verify_delivery(result, tmp_path)['passed']
    implementation = result['tool_implementations'][-1]
    assert 'from_tool' not in str(implementation['derived_changes'])
    assert 'at_tool' in implementation['derived_changes']['temperature']


def test_actual_query_result_binding_survives_generated_service(tmp_path):
    source, request = derived_source()
    expression = {'tool_result': {'name': 'read_temperature', 'path': '$.records[0].configured_temperature'}}
    source['semantic_goal']['row_predicates'][0]['value_expressions']['temperature'] = expression
    source['tool_implementations'][-1]['derived_changes']['temperature'] = deepcopy(expression)
    artifacts = compile_source(source, root=tmp_path, request=request)
    assert artifacts['generation_pipeline']['verification']['passed']
    assert verify_delivery(artifacts, tmp_path)['passed']
