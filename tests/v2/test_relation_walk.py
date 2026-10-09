import copy

import pytest

from env_factory_v2.expressions import ContractError, evaluate, validate_expr
from env_factory_v2.contract import validate_rule_links


def walk(limit=20):
    return {'walk': {'items': {'ref': 'data.nodes'}, 'start': {'literal': 'A'},
                     'key': 'id', 'next': 'next', 'max_nodes': limit}}


def rows():
    return [{'id': 'C', 'next': None, 'cost': 3}, {'id': 'A', 'next': 'B', 'cost': 1},
            {'id': 'B', 'next': 'C', 'cost': 2}]


def test_walk_derives_order_and_cost_from_actual_links_not_table_positions():
    expression = walk()
    validate_expr(expression, roots={'data'})
    validate_rule_links(expression, {}, shared_filters=True)
    data = rows()
    original = copy.deepcopy(data)
    context = {'data': {'nodes': data}}
    ids = {'map': {'items': expression, 'value': {'ref': 'row.id'}}}
    cost = {'sum': [{'map': {'items': expression, 'value': {'ref': 'row.cost'}}}]}
    assert evaluate(ids, context) == ['A', 'B', 'C']
    assert evaluate(cost, context) == 6
    assert data == original
    data[1]['next'] = 'C'
    assert evaluate(ids, context) == ['A', 'C']
    assert evaluate(cost, context) == 4


@pytest.mark.parametrize('change,message', [
    (lambda data: data[0].update(next='A'), 'cycle'),
    (lambda data: data[1].update(next='missing'), 'missing key'),
    (lambda data: data.append(dict(data[0])), 'duplicate'),
    (lambda data: data[0].pop('next'), 'columns')])
def test_invalid_relations_fail_explicitly(change, message):
    data = rows()
    change(data)
    with pytest.raises(ContractError, match=message):
        evaluate(walk(), {'data': {'nodes': data}})


def test_bound_is_enforced_and_null_terminates():
    with pytest.raises(ContractError, match='exceeds max_nodes'):
        evaluate(walk(2), {'data': {'nodes': rows()}})
    expression = walk()
    expression['walk']['start'] = {'literal': None}
    assert evaluate(expression, {'data': {'nodes': rows()}}) == []


def test_numeric_identity_matches_contract_equality_without_boolean_collision():
    expression = walk()
    expression['walk']['start'] = {'literal': 1.0}
    data = [{'id': True, 'next': None}, {'id': 1, 'next': True}]
    assert evaluate(expression, {'data': {'nodes': data}}) == data[::-1]
    data.append({'id': 1.0, 'next': None})
    with pytest.raises(ContractError, match='duplicate'):
        evaluate(expression, {'data': {'nodes': data}})


@pytest.mark.parametrize('bound', [0, 10001, True, 2.5])
def test_compile_rejects_invalid_bounds(bound):
    with pytest.raises(ContractError, match='max_nodes'):
        validate_expr(walk(bound), roots={'data'})
