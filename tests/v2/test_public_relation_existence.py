from copy import deepcopy

import pytest

from env_factory_v2.contracts.contract import validate_rule_links
from env_factory_v2.contracts.expressions import ContractError, evaluate, validate_expr
from test_relational_key_provenance import TABLES


RULES = {'active': {'predicate': {'eq': [{'ref': 'row.active'}, {'literal': True}]}}}


def expression(threshold=0):
    return {'filter': {'items': {'ref': 'data.expenses'}, 'value': {'let': {
        'bindings': {'project': {'ref': 'row.project_id'}}, 'value': {'gt': [
            {'count': [{'filter': {'items': {'ref': 'data.projects'}, 'value': {'and': [
                {'rule': {'id': 'active', 'arguments': {}}},
                {'eq': [{'ref': 'row.id'}, {'ref': 'vars.project'}]}]}}}]},
            {'literal': threshold}]}}}}}


def validate(expr, tables=TABLES):
    validate_expr(expr, roots={'data'})
    validate_rule_links(expr, RULES, shared_filters=True, tables=tables)


def test_public_relation_existence_uses_current_keys_and_rule():
    expr = expression()
    validate(expr)
    context = {'_rules': RULES, 'data': {
        'projects': [{'id': 'P1', 'active': True}, {'id': 'P2', 'active': False}],
        'expenses': [{'id': 'P1', 'project_id': 'P2', 'amount': 999},
                     {'id': 'E1', 'project_id': 'P1', 'amount': 7}]}}
    assert [r['id'] for r in evaluate(expr, context)] == ['E1']
    for rows in context['data'].values():
        rows.reverse()
    assert [r['id'] for r in evaluate(expr, context)] == ['E1']
    context['data']['projects'][1]['active'] = False
    assert evaluate(expr, context) == []


def test_existence_cannot_invent_cardinality_requirement():
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate(expression(1))


def test_existence_cannot_hide_inline_business_filter():
    expr = expression()
    predicate = expr['filter']['value']['let']['value']['gt'][0]['count'][0]['filter']['value']
    predicate['and'].append({'gt': [{'ref': 'row.amount'}, {'literal': 10}]})
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate(expr)


def test_existence_requires_declared_relationship():
    tables = deepcopy(TABLES)
    tables['expenses']['foreign_keys'] = []
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate(expression(), tables)


def test_sibling_foreign_keys_join_only_with_common_declared_target():
    tables = deepcopy(TABLES)
    tables['approvals'] = {'primary_key': 'id', 'schema': {'properties': {
        'id': {}, 'project_id': {}, 'active': {}}}, 'foreign_keys': [
        {'column': 'project_id', 'target_table': 'projects', 'target_column': 'id'}]}
    expr = expression()
    inner = expr['filter']['value']['let']['value']['gt'][0]['count'][0]['filter']
    inner['items'] = {'ref': 'data.approvals'}
    inner['value']['and'][1]['eq'][0] = {'ref': 'row.project_id'}
    validate(expr, tables)
    nullable = deepcopy(tables)
    nullable['approvals']['schema']['properties']['project_id']['nullable'] = True
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate(expr, nullable)
    tables['approvals']['foreign_keys'] = []
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate(expr, tables)
