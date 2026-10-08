import copy

import pytest

from env_factory.generation.business_contract import expand_business_contract


def source():
    return {'business_contract': {'version': '1.0', 'state_targets': {
        'stock': {'table': 'stock', 'where': {'sku': 'A'}, 'values': {'reserved': 20}, 'count': 1}},
        'outcomes': [{'id': 'reserved', 'rubric': 'Correct reservation', 'targets': ['stock']}]}}


def test_goal_and_reward_derive_from_one_target_without_mutating_source():
    original = source()
    before = copy.deepcopy(original)
    expanded = expand_business_contract(original)
    assert original == before
    goal = expanded['semantic_goal']['row_predicates'][0]
    reward = expanded['outcomes'][0]['rule']['expected'][0]
    assert goal == reward == original['business_contract']['state_targets']['stock']
    goal['values']['reserved'] = 999
    assert reward['values']['reserved'] == 20
    original['business_contract']['state_targets']['stock']['values']['reserved'] = 24
    newer = expand_business_contract(original)
    assert newer['semantic_goal']['row_predicates'][0]['values']['reserved'] == 24
    assert newer['outcomes'][0]['rule']['expected'][0]['values']['reserved'] == 24


@pytest.mark.parametrize('field', ['semantic_goal', 'outcomes'])
def test_duplicate_handwritten_definitions_rejected(field):
    value = source()
    value[field] = {}
    with pytest.raises(ValueError, match='BUSINESS_CONTRACT_DUPLICATE'):
        expand_business_contract(value)


def test_unscored_or_unknown_targets_rejected():
    value = source()
    value['business_contract']['outcomes'][0]['targets'] = ['missing']
    with pytest.raises(ValueError, match='unknown'):
        expand_business_contract(value)
    value = source()
    value['business_contract']['state_targets']['other'] = {'table': 'other'}
    with pytest.raises(ValueError, match='BUSINESS_CONTRACT_UNSCORED'):
        expand_business_contract(value)


def test_operation_changes_and_static_argument_constraints_share_target():
    value = source()
    value['business_tools'] = [{'name': 'reserve', 'parameters': {'type': 'object',
        'properties': {'sku': {'type': 'string'}, 'amount': {'type': 'integer'}},
        'required': ['sku', 'amount']}}]
    value['business_contract']['operations'] = [{'tool': 'reserve', 'target': 'stock',
        'selector': {'sku': 'sku'}, 'bindings': {'reserved': 'amount'},
        'policy': {'guards': [{'id': 'capacity', 'table': 'stock',
            'selector': {'sku': {'argument': 'sku'}}, 'conditions': {'available': {'gte': 20}}}]}}]
    result = expand_business_contract(value)
    tool = result['business_tools'][0]
    assert tool['implementation']['changes'] == {'amount': 'reserved'}
    assert tool['parameters']['properties']['amount']['enum'] == [20]
    assert result['business_lifecycle']['tools']['reserve']['require_idempotency']
    assert len(result['business_lifecycle']['tools']['reserve']['guards']) == 2
    value['business_contract']['state_targets']['stock']['values']['available'] = 0
    with pytest.raises(ValueError, match='bind every target field'):
        expand_business_contract(value)


@pytest.mark.parametrize('report', [
    {'id': '', 'rubric': 'Explain', 'semantic': {}},
    {'id': 'reserved', 'rubric': 'Explain', 'semantic': {}},
    {'id': 'explain', 'rubric': '', 'semantic': {}},
    {'id': 'explain', 'rubric': 'Explain', 'semantic': {}, 'rule': {}},
    {'id': 'explain', 'rubric': 'Explain', 'rule': []},
    {'id': 'explain', 'rubric': 'Explain', 'rule': {
        'source': 'business_state', 'operator': 'state_predicates'}},
])
def test_report_outcomes_cannot_bypass_shared_state_contract(report):
    value = source()
    value['business_contract']['report_outcomes'] = [report]
    with pytest.raises(ValueError, match='BUSINESS_CONTRACT_INVALID'):
        expand_business_contract(value)


def test_semantic_report_and_calibration_preserved():
    value = source()
    report = {'id': 'explain', 'rubric': 'Explain the decision', 'semantic': {
        'fields': ['explanation'], 'criteria': ['Explain capacity'],
        'cases': [{'answer': {'explanation': 'Enough capacity'}, 'expected': 'pass'}]}}
    value['business_contract']['report_outcomes'] = [report]
    expanded = expand_business_contract(value)
    assert expanded['outcomes'][-1] == report
    assert expanded['outcomes'][-1] is not report


def test_derived_operation_uses_target_definition_without_client_value():
    value = source()
    value['business_tools'] = [{'name': 'reserve', 'parameters': {'type': 'object',
        'properties': {'sku': {'type': 'string'}}, 'required': ['sku']}}]
    value['business_contract']['operations'] = [{'tool': 'reserve', 'target': 'stock',
        'selector': {'sku': 'sku'}, 'bindings': {}, 'derive': ['reserved']}]
    expanded = expand_business_contract(value)
    implementation = expanded['business_tools'][0]['implementation']
    assert implementation['changes'] == {}
    assert implementation['derived_changes'] == {'reserved': 20}
    value['business_contract']['state_targets']['stock']['values']['reserved'] = 24
    expanded = expand_business_contract(value)
    assert expanded['business_tools'][0]['implementation']['derived_changes'] == {'reserved': 24}
    assert expanded['outcomes'][0]['rule']['expected'][0]['values']['reserved'] == 24


def test_write_by_primary_key_preserves_business_identity_guard():
    value = source()
    value['tables'] = [{'table_name': 'stock', 'columns': [
        {'name': 'id'}, {'name': 'sku'}, {'name': 'reserved'}]}]
    value['business_tools'] = [{'name': 'reserve', 'parameters': {
        'properties': {'record_id': {'type': 'integer'}}, 'required': ['record_id']}}]
    value['business_contract']['operations'] = [{'tool': 'reserve', 'target': 'stock',
        'selector': {'record_id': 'id'}, 'bindings': {}, 'derive': ['reserved']}]
    result = expand_business_contract(value)
    guard = result['business_lifecycle']['tools']['reserve']['guards'][0]
    assert guard['selector'] == {'id': {'argument': 'record_id'}}
    assert guard['conditions'] == {'sku': {'eq': 'A'}}
    assert result['business_tools'][0]['implementation']['selector'] == {'record_id': 'id'}
    value['business_contract']['operations'][0]['selector']['record_id'] = 'unknown'
    with pytest.raises(ValueError, match='declared fields'):
        expand_business_contract(value)


def test_dynamic_identity_is_shared_without_becoming_a_write():
    value = source()
    target = value['business_contract']['state_targets']['stock']
    target['where'] = {}
    expression = {'at_tool': {'name': 'read_order', 'expression': {'lookup': {
        'table': 'orders', 'where': {'id': 'order'}, 'field': 'sku'}}}}
    target['identity'] = {'sku': expression}
    value['business_tools'] = [{'name': 'reserve', 'parameters': {'type': 'object',
        'properties': {'sku': {'type': 'string'}}, 'required': ['sku']}}]
    value['business_contract']['operations'] = [{'tool': 'reserve', 'target': 'stock',
        'selector': {'sku': 'sku'}, 'bindings': {}, 'derive': ['reserved']}]
    expanded = expand_business_contract(value)
    predicate = expanded['semantic_goal']['row_predicates'][0]
    assert predicate['value_expressions']['sku'] == expression
    assert expanded['outcomes'][0]['rule']['expected'][0] == predicate
    assert expanded['business_tools'][0]['implementation']['derived_changes'] == {'reserved': 20}
    guard = expanded['business_lifecycle']['tools']['reserve']['guards'][0]
    assert guard['conditions']['sku']['eq'] == {'expression': expression}
    target['identity']['sku'] = 'A'
    with pytest.raises(ValueError, match='observed at_tool'):
        expand_business_contract(value)


def test_shared_fact_expansion_and_cycles():
    value = source()
    contract = value['business_contract']
    contract['facts'] = {'approved_quantity': {'literal': 20}}
    target = contract['state_targets']['stock']
    target.pop('values')
    target['value_expressions'] = {'reserved': {'fact': 'approved_quantity'}}
    result = expand_business_contract(value)
    assert result['semantic_goal']['row_predicates'][0]['value_expressions']['reserved'] == {'literal': 20}
    contract['facts']['approved_quantity'] = {'fact': 'approved_quantity'}
    with pytest.raises(ValueError, match='cyclic fact'):
        expand_business_contract(value)
