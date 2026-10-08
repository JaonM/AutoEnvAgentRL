from copy import deepcopy

import pytest

from tests.generation.test_business_conditions import design
from env_factory.generation.reward_bindings import bind_reward_queries
from env_factory.generation.business_conditions import (
    compile_business_conditions, validate_condition_bindings, probe_business_conditions)


def source_with_variants():
    source = design()
    source['scenarios'][0]['scenario_id'] = 'small'
    source['scenarios'][0]['steps'][0]['arguments']['attendance'] = 8
    second = deepcopy(source['scenarios'][0])
    second['scenario_id'] = 'large'
    second['steps'][0]['arguments']['attendance'] = 18
    source['scenarios'].append(second)
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'tool_result': {'name': 'venues', 'path': '$.records[0].name'}}
    source['tables'][0]['rows'] = [{'name': 'Hall', 'supports_persian': True, 'capacity': 30}]
    return source


def test_different_user_operands_get_separate_validated_condition_probes():
    source = source_with_variants()
    conditions = compile_business_conditions(source)
    references = []
    bindings = bind_reward_queries(source, reference_details=references)
    assert 'venues' not in bindings  # no ambiguous from_tool fallback
    validate_condition_bindings(conditions, bindings, source, reference_details=references)
    report = probe_business_conditions(conditions, bindings, source, reference_details=references)
    assert report['verified'] == 4
    capacity = [case for case in report['cases'] if case['condition_id'] == 'capacity']
    assert {(tuple(c['reference_ids']), c['resolved_operand']) for c in capacity} == {
        (('small',), 8), (('large',), 18)}
    assert {c['query_where']['capacity']['gte'] for c in capacity} == {8, 18}


@pytest.mark.parametrize('defect', ['missing_argument', 'ambiguous_calls', 'one_static_reward', 'literal_reward'])
def test_every_reference_must_bind_and_score_the_condition(defect):
    source = source_with_variants()
    if defect == 'missing_argument':
        del source['scenarios'][1]['steps'][0]['arguments']['attendance']
    elif defect == 'ambiguous_calls':
        source['scenarios'][1]['steps'].append(deepcopy(source['scenarios'][1]['steps'][0]))
    elif defect == 'one_static_reward':
        source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
            'lookup': {'table': 'venues', 'field': 'name',
                       'where': {'supports_persian': {'eq': True}, 'capacity': {'gte': 8}}}}
    elif defect == 'literal_reward':
        source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {'literal': 'Hall'}
    conditions = compile_business_conditions(source)
    references = []
    bindings = bind_reward_queries(source, reference_details=references)
    with pytest.raises(ValueError, match='BUSINESS_CONDITION_UNBOUND|BUSINESS_CONDITION_UNSCORED'):
        validate_condition_bindings(conditions, bindings, source, reference_details=references)


def test_each_script_executes_only_its_own_query_condition_counterfactual(tmp_path):
    from env_factory.generation.agent_authoring import compile_source
    from tests.generation.test_interaction_reference_scope import scoped_source
    from tests.generation.test_interaction_contract import interactive_source

    source = scoped_source()
    _, request = interactive_source()
    request['require_structured_interactions'] = True
    source['description']['public_input']['initial_user_message'] += '温度不得超过30摄氏度。'
    schema = source['tools'][1]['function']['parameters']
    schema['properties']['max_temperature'] = {'type': 'integer', 'description': '允许的最高温度'}
    schema['required'].append('max_temperature')
    for case in source['scenarios']:
        for step in case['steps']:
            if step.get('tool_name') == 'read_temperature':
                step['arguments']['max_temperature'] = 30
    source['business_conditions'] = [{
        'id': 'temperature_limit', 'public_requirement': '温度不得超过30摄氏度',
        'tool': 'read_temperature', 'argument': 'max_temperature',
        'column': 'temperature', 'operator': 'lte'}]
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'tool_result': {'name': 'read_temperature', 'path': '$.records[0].temperature'}}

    artifact = compile_source(source, root=tmp_path, request=request)
    execution = artifact['generation_pipeline']['verification']['business_condition_execution']
    assert len(execution) == 3
    assert {case['scenario_id'] for case in execution} == {
        'success-script-1', 'success-script-3', 'alternate_success-script-2'}
    for case in execution:
        assert case['status'] == 'passed'
        assert case['execution']['reward']['reward'] < 1
        reference = 'alternate_success' if case['scenario_id'].endswith('script-2') else 'success'
        assert case['mutation']['reference_ids'] == [reference]
