from itertools import product

import pytest

from env_factory_v2.contracts.contract import free_row_fields, validate_rule_links
from env_factory_v2.contracts.expressions import ContractError, evaluate, validate_expr


def interval(stop, maximum=10000):
    return {'range': {'start': {'literal': 0}, 'stop': stop, 'max_items': maximum}}


def combinations():
    return {'flat_map': {'items': interval({'ref': 'data.a'}), 'value': {'let': {
        'bindings': {'a': {'ref': 'row'}}, 'value': {'map': {
            'items': interval({'ref': 'data.b'}), 'value': {'list': [
                {'ref': 'vars.a'}, {'ref': 'row'}]}}}}}}}


def test_cartesian_candidates_follow_dynamic_stock_and_preserve_scope():
    expr = combinations()
    validate_expr(expr, roots={'data'})
    validate_rule_links(expr, {}, shared_filters=True)
    assert free_row_fields(expr) == set()
    for a, b in [(5, 4), (7, 5), (0, 9)]:
        context = {'data': {'a': a, 'b': b}, 'row': 'outside'}
        assert evaluate(expr, context) == [list(v) for v in product(range(a), range(b))]
        assert context['row'] == 'outside' and 'vars' not in context


def test_expanded_stock_exposes_new_lower_cost_plan():
    expr = combinations()
    def best(a):
        choices = evaluate(expr, {'data': {'a': a, 'b': 4}})
        feasible = [(x * 100 + y * 110, x * 6 + y * 5 - 30)
                    for x, y in choices if x * 6 + y * 5 >= 30]
        return min(feasible)
    assert best(5)[0] == 620
    assert best(6)[0] == 500


def test_range_empty_and_overflow_never_truncate():
    assert evaluate(interval({'literal': -5}), {}) == []
    with pytest.raises(ContractError, match='exceeds max_items'):
        evaluate(interval({'literal': 4}, maximum=3), {})
    with pytest.raises(ContractError, match='10000 output'):
        evaluate(combinations(), {'data': {'a': 101, 'b': 100}})


@pytest.mark.parametrize('endpoint', [True, 3.0, '3'])
def test_range_rejects_noninteger_endpoints(endpoint):
    with pytest.raises(ContractError, match='endpoints must be integers'):
        evaluate(interval({'literal': endpoint}), {})


def test_flat_map_rejects_scalar_and_hidden_business_filter():
    with pytest.raises(ContractError, match='must return a list'):
        evaluate({'flat_map': {'items': {'literal': [1]}, 'value': {'ref': 'row'}}}, {})
    expr = {'flat_map': {'items': {'literal': [1]}, 'value': {'filter': {
        'items': {'literal': [1]}, 'value': {'literal': True}}}}}
    validate_expr(expr, roots=set())
    with pytest.raises(ContractError, match='invoke named public rules'):
        validate_rule_links(expr, {}, shared_filters=True)


@pytest.mark.parametrize('maximum', [True, 0, 10001])
def test_range_budget_is_bounded_literal(maximum):
    with pytest.raises(ContractError, match='max_items'):
        validate_expr(interval({'literal': 1}, maximum), roots=set())
