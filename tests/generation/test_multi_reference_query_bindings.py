import copy

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.reward_bindings import bind_reward_queries
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator
from .test_agent_authoring import fixture


def shared_references():
    source, request = fixture('multi_step_agentic')
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'from_tool': {'name': 'read_temperature', 'field': 'temperature'}}
    alternate = copy.deepcopy(source['scenarios'][0])
    alternate['scenario_id'] = 'alternate'
    # Capture names are local implementation details, not query differences.
    alternate['steps'][0]['capture'] = {'other_zone': '$.records[0].id'}
    alternate['steps'][1]['arguments']['id'] = {'$ref': 'other_zone'}
    source['scenarios'].append(alternate)
    return source, request


def test_shared_queries_compile_and_follow_mutated_business_identity(tmp_path):
    source, request = shared_references()
    result = compile_source(source, root=tmp_path, request=request)
    assert result['task_readiness']['ready']
    assert set(result['generation_pipeline']['reward_query_bindings']) == {
        'lookup_zone', 'read_temperature'}
    rule = result['metric_implementations'][0]
    context = {'business_state': {'zones': [
        {'id': 999, 'name': '青松库', 'temperature': 31},
        {'id': 821, 'name': '旧址', 'temperature': 18}]}}
    evaluator = DeclarativeMetricEvaluator()
    assert evaluator.evaluate(rule, {**context, 'final_agent_response': '温度：31摄氏度。'}) == 1
    assert evaluator.evaluate(rule, {**context, 'final_agent_response': '温度：18摄氏度。'}) == 0


@pytest.mark.parametrize('change', ['upstream_scope', 'missing_call', 'missing_capture',
                                   'forward_capture', 'extra_query', 'different_type'])
@pytest.mark.parametrize('reverse', [False, True])
def test_incompatible_references_cannot_borrow_shared_binding(change, reverse):
    source, _ = shared_references()
    steps = source['scenarios'][-1]['steps']
    if change == 'upstream_scope':
        steps[0]['arguments']['name'] = '白杨库'
    elif change == 'missing_call':
        steps.pop(1)
    elif change == 'missing_capture':
        steps[0].pop('capture')
        steps[1]['arguments']['id'] = {'$ref': 'zone_id'}
    elif change == 'forward_capture':
        steps[0], steps[1] = steps[1], steps[0]
    elif change == 'extra_query':
        extra = copy.deepcopy(steps[1])
        extra['arguments']['id'] = 547
        steps.insert(2, extra)
    elif change == 'different_type':
        # Python dict equality would incorrectly consider True equal to 1.
        source['scenarios'][0]['steps'][1]['arguments']['id'] = 1
        steps[1]['arguments']['id'] = True
    if reverse:
        source['scenarios'].reverse()
    with pytest.raises(ValueError, match='one unambiguous select call: read_temperature'):
        bind_reward_queries(source)


def test_unused_divergent_query_does_not_poison_shared_upstream_binding():
    source, _ = shared_references()
    source['scenarios'][-1]['steps'][1]['arguments']['id'] = 547
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'from_tool': {'name': 'lookup_zone', 'field': 'id'}}
    bindings = bind_reward_queries(source)
    assert set(bindings) == {'lookup_zone'}


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('defect', [None, 'required_call', 'unresolved_capture', 'different_query', 'unknown_route'])
def test_refusal_may_omit_only_an_inapplicable_shared_query(reverse, defect):
    source, _ = shared_references()
    alternate = source['scenarios'][-1]
    alternate['execution_route'] = 'declined'
    source['execution_routes'] = [{'id': 'declined',
        'when': {'user_rejection': {'stage': 'choice', 'branch': 'decline'}},
        'required_tools': ['lookup_zone']}]
    if defect in {'unresolved_capture', 'different_query'}:
        if defect == 'unresolved_capture': alternate['steps'][0].pop('capture')
        else: alternate['steps'][1]['arguments']['id'] = 547
    else:
        alternate['steps'].pop(1)
    if defect == 'required_call': source['execution_routes'][0]['required_tools'].append('read_temperature')
    if defect == 'unknown_route': alternate['execution_route'] = 'invented'
    if reverse: source['scenarios'].reverse()
    if defect:
        with pytest.raises(ValueError, match='one unambiguous select call: read_temperature'):
            bind_reward_queries(source)
    else:
        assert set(bind_reward_queries(source)) == {'lookup_zone', 'read_temperature'}
