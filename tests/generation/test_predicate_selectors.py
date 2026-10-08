from copy import deepcopy

import pytest

from env_factory.generation.predicate_selectors import validate_predicate_selectors
from env_factory.sandbox_runtime import BusinessGoalEvaluator


def test_structural_preview_rejects_realistic_dynamic_selector_before_execution(tmp_path):
    from env_factory.generation.agent_authoring import compile_source
    from tests.generation.test_dynamic_state_goal import dynamic_source
    source, request = dynamic_source()
    source['semantic_goal']['row_predicates'][0]['where'] = {
        'id': {'from_tool': {'name': 'read_temperature', 'field': 'id'}}}
    with pytest.raises(ValueError, match=r'STATE_SELECTOR_EXPRESSION:.*where.id'):
        compile_source(source, request=request, root=tmp_path, structural_preview=True)


def test_dynamic_selector_reports_all_reward_paths_without_mutating_source():
    predicate = {'table': 'stock', 'where': {'id': {'at_tool': {
        'name': 'read_stock', 'expression': {'from_tool': {'name': 'read_stock', 'field': 'id'}}}}},
        'values': {'reserved': 20}, 'count': 1}
    source = {'semantic_goal': {'row_predicates': [predicate]},
              'metric_implementations': [{'operator': 'state_predicates', 'expected': [predicate]}]}
    before = deepcopy(source)
    with pytest.raises(ValueError) as error:
        validate_predicate_selectors(source)
    assert 'semantic_goal.row_predicates[0].where.id' in str(error.value)
    assert 'metric_implementations[0].expected[0].where.id' in str(error.value)
    assert 'value_expressions' in str(error.value)
    assert source == before


def test_supported_dynamic_target_keeps_real_business_reward():
    state = {'stock': [{'id': 7, 'name': 'target', 'reserved': 20}]}
    predicate = {'table': 'stock', 'where': {'name': 'target'}, 'values': {'reserved': 20},
                 'value_expressions': {'id': {'lookup': {
                     'table': 'stock', 'field': 'id', 'where': {'name': 'target'}}}}, 'count': 1}
    validate_predicate_selectors({'semantic_goal': {'row_predicates': [predicate]}})
    assert BusinessGoalEvaluator.evaluate([predicate], state)
    state['stock'][0]['reserved'] = 19
    assert not BusinessGoalEvaluator.evaluate([predicate], state)


def test_literal_json_selector_is_not_blanket_rejected():
    validate_predicate_selectors({'semantic_goal': {'row_predicates': [
        {'where': {'metadata': {'zone': 'north'}, 'enabled': True, 'label': None}}]}})
