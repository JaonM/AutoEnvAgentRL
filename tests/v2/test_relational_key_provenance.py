from copy import deepcopy

import pytest

from env_factory_v2.contract import validate_rule_links
from env_factory_v2.expressions import ContractError, evaluate, validate_expr


TABLES = {
    'projects': {'primary_key': 'id', 'schema': {'properties': {'id': {}, 'active': {}}}},
    'expenses': {'primary_key': 'id', 'schema': {'properties': {
        'id': {}, 'project_id': {}, 'amount': {}}}, 'foreign_keys': [
        {'column': 'project_id', 'target_table': 'projects', 'target_column': 'id'}]},
}
RULES = {'active': {'predicate': {'eq': [{'ref': 'row.active'}, {'literal': True}]}}}


def selected_keys(column='id', predicate=None):
    return {'map': {'items': {'filter': {'items': {'ref': 'data.projects'},
        'value': predicate or {'rule': {'id': 'active', 'arguments': {}}}}},
        'value': {'ref': f'row.{column}'}}}


def membership(keys=None, column='project_id'):
    return {'let': {'bindings': {'keys': keys or selected_keys()}, 'value': {'filter': {
        'items': {'ref': 'data.expenses'},
        'value': {'in': [{'ref': f'row.{column}'}, {'ref': 'vars.keys'}]}}}}}


def validate(expr):
    validate_expr(expr, roots={'data', 'initial'})
    validate_rule_links(expr, RULES, shared_filters=True, tables=TABLES)


def test_membership_tracks_declared_relationship_and_public_selection():
    expr = membership()
    validate(expr)
    context = {'_rules': RULES, 'data': {
        'projects': [{'id': 'P1', 'active': True}, {'id': 'P2', 'active': False}],
        'expenses': [{'id': 'P1', 'project_id': 'P2', 'amount': 999},
                     {'id': 'E1', 'project_id': 'P1', 'amount': 7}]}}
    assert [row['id'] for row in evaluate(expr, context)] == ['E1']
    for rows in context['data'].values():
        rows.reverse()
    assert [row['id'] for row in evaluate(expr, context)] == ['E1']
    context['data']['expenses'][0]['project_id'] = 'P2'
    assert evaluate(expr, context) == []


@pytest.mark.parametrize('expr', [
    membership(column='id'),
    membership(selected_keys('active')),
    membership({'literal': ['P1']}),
    membership({'map': {'items': {'ref': 'data.projects'}, 'value': {'ref': 'row.id'}}}),
    membership(selected_keys(predicate={'or': [
        {'rule': {'id': 'active', 'arguments': {}}}, {'literal': True}]})),
    membership(selected_keys(predicate={'and': [
        {'rule': {'id': 'active', 'arguments': {}}},
        {'eq': [{'ref': 'row.id'}, {'literal': 'P1'}]}]})),
])
def test_unproven_keys_or_invented_eligibility_are_rejected(expr):
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate(expr)


def test_reverse_foreign_key_collection_is_valid():
    rules = {'selected_expense': {'predicate': {'gt': [{'ref': 'row.amount'}, {'literal': 5}]}}}
    expr = {'let': {'bindings': {'keys': {'map': {'items': {'filter': {
        'items': {'ref': 'data.expenses'}, 'value': {'rule': {
            'id': 'selected_expense', 'arguments': {}}}}}, 'value': {'ref': 'row.project_id'}}}},
        'value': {'filter': {'items': {'ref': 'data.projects'},
            'value': {'in': [{'ref': 'row.id'}, {'ref': 'vars.keys'}]}}}}}
    validate_expr(expr, roots={'data'})
    validate_rule_links(expr, rules, shared_filters=True, tables=TABLES)


def test_outer_scalar_capture_supports_join_and_rejects_literal_shadow():
    expr = {'map': {'items': {'ref': 'data.projects'}, 'value': {'let': {
        'bindings': {'key': {'ref': 'row.id'}}, 'value': {'filter': {
            'items': {'ref': 'data.expenses'},
            'value': {'eq': [{'ref': 'row.project_id'}, {'ref': 'vars.key'}]}}}}}}}
    validate(expr)
    context = {'data': {'projects': [{'id': 'P1'}, {'id': 'P2'}], 'expenses': [
        {'id': 'E1', 'project_id': 'P2', 'amount': 11}]}}
    assert evaluate(expr, context) == [[], context['data']['expenses']]
    shadowed = deepcopy(expr)
    shadowed['map']['value']['let']['value'] = {'let': {
        'bindings': {'key': {'literal': 'P1'}}, 'value': expr['map']['value']['let']['value']}}
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate(shadowed)


def test_removed_relationship_does_not_pass_from_matching_column_names():
    tables = deepcopy(TABLES)
    tables['expenses']['foreign_keys'] = []
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate_rule_links(membership(), RULES, shared_filters=True, tables=tables)
