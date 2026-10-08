"""Partial outcome failures need measured component evidence, even without a requirement link."""
import copy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from tests.generation.test_agent_authoring import fixture


def partial_failure_source():
    source, request = fixture('simple_agentic')
    # Both outcome components must qualify independently for a private-data task.
    source['tables'][0]['columns'].append({'name': 'unit', 'type': 'text', 'nullable': False})
    for row in source['tables'][0]['rows']:
        row['unit'] = '摄氏度'
    source['tool_implementations'][0]['projection'].append('unit')
    source['metrics'][0]['weight'] = .5
    source['metrics'].append({**copy.deepcopy(source['metrics'][0]), 'id': 'unit', 'weight': .5})
    expression = source['metric_implementations'][0]['expected']['targets'][0]['expression']
    source['metric_implementations'][0].update(operator='value_targets', expected={
        'answer_format': 'json_object', 'targets': [{'key': 'temperature', 'expression': expression}]})
    source['metric_implementations'].append({'metric_id': 'unit', 'source': 'final_agent_response',
        'path': '$', 'operator': 'value_targets', 'expected': {'answer_format': 'json_object',
        'targets': [{'key': 'unit', 'expression': {'lookup': {'table': 'zones', 'field': 'unit', 'where': {'name': '青松库'}}}}]}, 'score_mapping': {'pass': 1, 'fail': 0}})
    source['answer_contract'] = {'format': 'json_object', 'schema': {'type': 'object',
        'properties': {'temperature': {'type': 'number', 'description': '温度'},
                       'unit': {'type': 'string', 'description': '温度单位'}},
        'required': ['temperature', 'unit'], 'additionalProperties': False}}
    source['scenarios'][0]['steps'][-1]['content'] = '{"temperature":18,"unit":"摄氏度"}'
    negative = copy.deepcopy(source['scenarios'][0])
    negative.update(scenario_id='wrong_temperature', kind='goal_failure')
    negative['steps'][-1]['content'] = '{"temperature":999,"unit":"摄氏度"}'
    negative['steps'].append({'operation': 'reward', 'step_id': 'score'})
    negative['assertions'] = [{'source': 'step:score', 'path': '$.components.temperature_answer',
                               'operator': 'lte', 'expected': .2}]
    source['scenarios'].append(negative)
    return source, request


def test_unbound_component_negative_retains_correct_other_outcome(tmp_path):
    source, request = partial_failure_source()
    result = compile_source(source, root=tmp_path, request=request)
    assert result['generation_pipeline']['verification']['checks']['authored_failure_1']
    evidence = json.loads((tmp_path / 'component_negative_validation.json').read_text())[0]
    assert .2 < evidence['reward'] < 1
    assert evidence['components'] == {'temperature_answer': 0}


@pytest.mark.parametrize('case', ['wrong_component', 'loose_bound', 'no_assertion', 'early_reward'])
def test_partial_failure_without_valid_final_component_evidence_is_rejected(tmp_path, case):
    source, request = partial_failure_source()
    negative = source['scenarios'][-1]
    if case == 'wrong_component':
        negative['assertions'][0]['path'] = '$.components.unit'
    elif case == 'loose_bound':
        negative['assertions'][0]['expected'] = 1
    elif case == 'no_assertion':
        negative['assertions'] = []
    else:
        # A zero component before the answer cannot certify the final bad answer.
        negative['steps'].insert(0, negative['steps'].pop())
    with pytest.raises(ValueError, match='AGENT_SCENARIO_FAILED|AGENT_PREFLIGHT_FAILED|DETERMINISTIC_PREFLIGHT_FAILED'):
        compile_source(source, root=tmp_path, request=request)


@pytest.mark.parametrize('mutation', ['full_reward', 'inactive', 'missing', 'nan'])
def test_component_evidence_cannot_certify_full_credit_or_unmeasured_outcome(mutation):
    from env_factory.generation.scenario_assertions import measured_component_failure
    source, _ = partial_failure_source()
    body = {'reward': .7, 'components': {'temperature_answer': 0}}
    if mutation == 'full_reward':
        body['reward'] = 1
    elif mutation == 'inactive':
        body['inactive_components'] = ['temperature_answer']
    elif mutation == 'missing':
        body['components'] = {}
    else:
        body['components']['temperature_answer'] = float('nan')
    assert not measured_component_failure(source['scenarios'][-1],
        {'history': [{'operation': 'reward', 'body': body}]}, source['metrics'])['passed']


@pytest.mark.parametrize('operator,expected,value,passed', [
    ('eq', 0, .1, False), ('lt', .2, .2, False), ('lte', .2, .2, True),
    ('eq', 0, 0, True), ('lt', .2, .1, True),
])
def test_component_evidence_enforces_authored_comparator(operator, expected, value, passed):
    from env_factory.generation.scenario_assertions import measured_component_failure
    source, _ = partial_failure_source()
    scenario = source['scenarios'][-1]
    scenario['assertions'][0].update(operator=operator, expected=expected)
    run = {'history': [{'operation': 'reward', 'body': {
        'reward': .6, 'components': {'temperature_answer': value}}}]}
    assert measured_component_failure(scenario, run, source['metrics'])['passed'] is passed


@pytest.mark.parametrize('run', [None, {}, {'history': []}, {'history': [
    {'operation': 'reward', 'body': {'reward': .5, 'components': {'temperature_answer': 0}}},
    {'operation': 'agent_response', 'body': {}}]}])
def test_component_evidence_requires_fresh_terminal_reward(run):
    from env_factory.generation.scenario_assertions import measured_component_failure
    source, _ = partial_failure_source()
    assert not measured_component_failure(source['scenarios'][-1], run, source['metrics'])['passed']
