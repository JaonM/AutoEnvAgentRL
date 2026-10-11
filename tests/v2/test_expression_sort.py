import pytest

from env_factory_v2.contracts.contract import contains_rule, expression_children
from env_factory_v2.contracts.expressions import ContractError, evaluate, validate_expr


def expression(rows, orders=None):
    return {'sort': {'items': {'literal': rows}, 'order_by': orders or [{'field': 'sku', 'direction': 'asc'}]}}


def test_sort_uses_current_string_identity_and_preserves_null_last_and_ties():
    rows = [{'sku': 'Z', 'quantity': 2}, {'sku': None, 'quantity': 9},
            {'sku': 'A', 'quantity': 4}, {'sku': 'A', 'quantity': 3}]
    expr = expression(rows)
    validate_expr(expr, roots=set())
    assert evaluate(expr, {}) == [rows[2], rows[3], rows[0], rows[1]]
    assert evaluate(expression(rows, [{'field': 'sku', 'direction': 'desc'}]), {}) == [rows[0], rows[2], rows[3], rows[1]]
    rows[0]['sku'] = '0'
    assert evaluate(expression(rows), {})[0]['sku'] == '0'


def test_multikey_numeric_sort_and_public_rule_provenance():
    rows = [{'sku': 'A', 'quantity': 1.0}, {'sku': 'B', 'quantity': 2}, {'sku': 'A', 'quantity': 3}]
    expr = expression(rows, [{'field': 'sku', 'direction': 'asc'}, {'field': 'quantity', 'direction': 'desc'}])
    assert evaluate(expr, {}) == [rows[2], rows[0], rows[1]]
    expr['sort']['items'] = {'filter': {'items': {'ref': 'data.items'}, 'value': {'rule': {'id': 'eligible', 'arguments': {}}}}}
    validate_expr(expr, roots={'data'})
    assert contains_rule(expr)
    assert expression_children(expr) == [expr['sort']['items']]


@pytest.mark.parametrize('rows', [[{'sku': True}, {'sku': 1}], [{'sku': '1'}, {'sku': 1}],
                                 [{'sku': []}], [{}], [{'sku': float('inf')}], [{}] * 10001])
def test_invalid_sort_does_not_coerce_or_drop_rows(rows):
    with pytest.raises(ContractError):
        evaluate(expression(rows), {})
