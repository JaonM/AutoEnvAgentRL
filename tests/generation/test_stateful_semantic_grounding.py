import json

import pytest

from env_factory.contracts.reward_contract import STATEFUL_GOAL_CRITERION
from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.semantic_reward import compile_semantic_outcome
from .test_agent_authoring import fixture


def stateful_source():
    source, request = fixture('multi_step_agentic')
    source['description']['task_intent'] = 'modify'
    source['description']['public_input']['initial_user_message'] = '将青松库温度从18改为19，其他库区不变，并用 JSON why 说明实际保存结果。'
    source['environment_plan'].update(mode='stateful', requires_persistence=True)
    params = source['tools'][1]['function']['parameters']
    params['properties']['temperature'] = {'type': 'integer', 'description': '目标温度'}
    params['required'].append('temperature')
    source['tool_implementations'][1] = {'tool_name': 'read_temperature', 'operation': 'update',
        'table': 'zones', 'result_field': 'updated_count', 'selector': {'id': 'id'},
        'changes': {'temperature': 'temperature'}}
    source['scenarios'][0]['steps'][1]['arguments']['temperature'] = 19
    source['metric_implementations'][0].update(source='business_state', operator='eq',
        path='$.zones[?(@.name=="青松库")][0].temperature', expected=19)
    source['semantic_goal'] = {'row_predicates': [{'table': 'zones', 'where': {'name': '青松库'},
        'values': {'temperature': 19}, 'count': 1}], 'expected_delta': [{'table': 'zones',
        'where': {'name': '青松库'}, 'field': 'temperature', 'before': 18, 'after': 19}]}
    schema = {'type': 'object', 'properties': {'why': {'type': 'string'}},
              'required': ['why'], 'additionalProperties': False}
    item = {'id': 'receipt', 'rubric': 'Explain the actual saved result.', 'semantic': {
        'fields': ['why'], 'criteria': ['Compare the explanation with the actual saved temperature.'],
        'cases': [{'answer': {'why': text}, 'expected': label} for text, label in [
            ('已保存为19度。', 'pass'), ('修改完成，当前记录为19度。', 'pass'),
            ('仍是18度。', 'fail'), ('修改失败。', 'fail')]]}}
    metric, cases = compile_semantic_outcome(item, schema=schema, weight=.5)
    source['metrics'][0]['weight'] = .5
    source['metrics'].append(metric)
    source['semantic_calibration'] = {'receipt': cases}
    source['answer_contract'] = {'format': 'json_object', 'schema': schema}
    for index, scene in enumerate(source['scenarios']):
        scene['steps'][-1]['content'] = json.dumps(cases[0 if index == 0 else 2]['answer'])
    source = json.loads(json.dumps(source).replace('"read_temperature"', '"update_temperature"'))
    return source, request


def test_stateful_semantic_rule_is_generated_and_public_before_execution(tmp_path):
    source, request = stateful_source()
    original = json.loads(json.dumps(source))
    result = compile_source(source, root=tmp_path, request=request, structural_preview=True)
    metric = next(m for m in result['metrics'] if m['id'] == 'receipt')
    assert metric['criteria'].count(STATEFUL_GOAL_CRITERION) == 1
    assert STATEFUL_GOAL_CRITERION in str(result['public_input'])
    assert {'business_data', 'tool_results', 'final_agent_response'} <= set(metric['evaluation_inputs'])
    assert source == original


def test_authored_required_rule_is_not_duplicated(tmp_path):
    source, request = stateful_source()
    source['metrics'][-1]['criteria'].append(STATEFUL_GOAL_CRITERION)
    result = compile_source(source, root=tmp_path, request=request, structural_preview=True)
    assert next(m for m in result['metrics'] if m['id'] == 'receipt')['criteria'].count(STATEFUL_GOAL_CRITERION) == 1


def test_stateful_semantic_contradiction_still_fails(tmp_path):
    source, request = stateful_source()
    source['metrics'][-1]['criteria'].append('无需核对业务数据')
    with pytest.raises(ValueError, match='contradicts business evidence'):
        compile_source(source, root=tmp_path, request=request, structural_preview=True)
