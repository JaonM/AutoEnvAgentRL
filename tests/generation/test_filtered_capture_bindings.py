from copy import deepcopy

import pytest

from env_factory.generation.reward_bindings import bind_reward_queries
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator as Evaluator
from tests.generation.test_multi_reference_query_bindings import shared_references


def design():
    source, _ = shared_references()
    source['scenarios'] = [s for s in source['scenarios'] if s.get('kind') == 'goal_success']
    for scenario in source['scenarios']:
        scenario['steps'][0]['capture'] = {'chosen': '$.records[?(@.id==250)].id'}
        scenario['steps'][1]['arguments']['id'] = {'$ref': 'chosen'}
    return source


def test_filtered_capture_retains_producer_scope_and_rejects_ambiguous_identity():
    source = design()
    bindings = bind_reward_queries(source)
    expression = bindings['read_temperature']['where']['id']['eq']
    state = {'zones': [{'id': 250, 'name': '青松库'}, {'id': 820, 'name': '青松库'}]}
    assert Evaluator._value_expression(expression, state) == 250
    assert Evaluator._value_expression(expression, {'zones': [{'id': 250, 'name': '其他库'}]}) is None
    state['zones'].append(deepcopy(state['zones'][0]))
    assert Evaluator._value_expression(expression, state) is None


def test_filtered_capture_respects_projection_aliases():
    source = design()
    producer = source['tool_implementations'][0]
    producer['projection_aliases'] = {'public_id': 'id'}
    for scenario in source['scenarios']:
        scenario['steps'][0]['capture'] = {'chosen': '$.records[?(@.public_id==250)].public_id'}
    binding = bind_reward_queries(source)['read_temperature']['where']['id']['eq']['lookup']
    assert binding['field'] == 'id' and binding['where']['id'] == {'eq': 250}


@pytest.mark.parametrize('path', ['$.records[*].id', '$.records[1].id',
    '$.records[?(@.unexposed==250)].id', '$.records[?(@.id==250)].unexposed'])
def test_unsupported_or_hidden_capture_cannot_supply_shared_reward_query(path):
    source = design()
    source['tool_implementations'][0]['projection'] = ['id', 'name']
    for scenario in source['scenarios']:
        scenario['steps'][0]['capture'] = {'chosen': path}
    with pytest.raises(ValueError, match='one unambiguous select call'):
        bind_reward_queries(source)
