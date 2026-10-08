"""A legitimate empty lookup ends early; successful work still needs its chain."""
from copy import deepcopy

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import SandboxError
from .test_agent_authoring import fixture


def readonly_routes():
    source, request = fixture('multi_step_agentic')
    source['metrics'][0]['weight'] = .8
    for name in ('lookup_zone', 'read_temperature'):
        source['metrics'].append({'id': 'process_' + name, 'category': 'process',
            'type': 'rule-based', 'scope': 'step', 'weight': .1, 'score_range': [0, 1],
            'rubric': 'Read the requested record', 'target_action': name,
            'evaluator': {'kind': 'trajectory_rule', 'source': 'runtime_rule',
                'assertion': 'Read the requested record', 'score_mapping': {'pass': 1, 'fail': 0}}})
    observed_count = {'tool_result': {'name': 'lookup_zone', 'path': '$.count'}}
    unique = {'op': 'eq', 'args': [observed_count, {'literal': 1}]}
    nonunique = {'op': 'ne', 'args': [deepcopy(observed_count), {'literal': 1}]}
    source['execution_routes'] = [
        {'id': 'found', 'when': unique, 'required_tools': ['lookup_zone', 'read_temperature'],
         'required_stages': [], 'writes': 'forbidden'},
        {'id': 'unresolved', 'when': nonunique, 'required_tools': ['lookup_zone'],
         'required_stages': [], 'writes': 'forbidden'}]
    source['metric_implementations'][0].update(operator='value_targets', expected={
        'answer_format': 'json_object', 'targets': [{'key': 'temperature', 'expression': {
            'if': {'condition': unique, 'then': {'tool_result': {
                'name': 'read_temperature', 'path': '$.records[0].temperature'}},
                'else': {'literal': 0}}}}, {'key': 'status', 'expression': {'if': {
                    'condition': unique, 'then': {'literal': 'found'},
                    'else': {'literal': 'lookup_not_unique'}}}}]})
    source['answer_contract'] = {'format': 'json_object', 'schema': {'type': 'object',
        'properties': {'temperature': {'type': 'integer'}, 'status': {'type': 'string'}},
        'required': ['temperature', 'status'], 'additionalProperties': False}}
    success = source['scenarios'][0]
    success['execution_route'] = 'found'
    success['steps'][-1]['content'] = '{"temperature":18,"status":"found"}'
    first = deepcopy(success['steps'][0])
    first.pop('capture')
    source['route_probes'] = [{'route_id': 'unresolved', 'initial_changes': [
        {'table': 'zones', 'where': {'id': 821}, 'values': {'name': '已撤销库区'}}],
        'reference': {'calls': [first], 'answer': '{"temperature":0,"status":"lookup_not_unique"}'}}]
    return source, request


def test_readonly_task_executes_full_chain_and_first_read_stop(tmp_path):
    source, request = readonly_routes()
    artifacts = compile_source(source, request=request, root=tmp_path)
    report = artifacts['generation_pipeline']['verification']['execution_route_coverage']
    assert report['references'] == ['found']
    assert report['probes'][0]['route_id'] == 'unresolved'
    assert report['probes'][0]['passed']
    assert all(report['probes'][0]['checks'].values())


@pytest.mark.parametrize('defect', ['no_multistep_route', 'downstream_only', 'constant_stop'])
def test_early_stop_is_not_a_way_to_relabel_one_tool_tasks(tmp_path, defect):
    source, request = readonly_routes()
    if defect == 'no_multistep_route':
        source['execution_routes'][0]['required_tools'] = ['lookup_zone']
    elif defect == 'downstream_only':
        source['execution_routes'][1]['required_tools'] = ['read_temperature']
        source['execution_routes'][1]['when'] = {'op': 'eq', 'args': [
            {'tool_result': {'name': 'read_temperature', 'path': '$.count'}}, 0]}
    else:
        source['execution_routes'][1]['when'] = {'literal': True}
    with pytest.raises(SandboxError) as error:
        compile_source(source, request=request, root=tmp_path, structural_preview=True)
    assert error.value.code == 'EXECUTION_ROUTE_INVALID'
