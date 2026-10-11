import pytest

from env_factory_v2.contracts.contract import compile_contract, free_row_fields
from env_factory_v2.examples import transaction, ref, lit
from env_factory_v2.contracts.expressions import ContractError


def test_collection_rows_do_not_become_operation_row_requirements():
    expr = {'and': [{'ge': [ref('row.balance'), lit(0)]}, {'eq': [{'count': [{'filter': {
        'items': ref('data.accounts'), 'value': {'eq': [ref('row.non_target_field'), lit(1)]}}}]}, lit(0)]}]}
    assert free_row_fields(expr) == {'balance'}
    assert free_row_fields({'literal': {'row.missing': 1}}) == set()


def test_cross_table_transaction_rejects_rule_missing_target_row_fields():
    source = transaction()
    table = source['tables'][0]
    table['schema']['properties']['status'] = {'type': 'string'}
    table['schema']['required'].append('status')
    for row in table['rows']:
        row['status'] = 'open'
    source['tables'].append({'id': 'other', 'primary_key': 'id', 'schema': {
        'type': 'object', 'properties': {'id': {'type': 'string'}, 'balance': {'type': 'integer'}},
        'required': ['id', 'balance'], 'additionalProperties': False}, 'rows': [{'id': 'X', 'balance': 0}]})
    source['rules'].append({'id': 'open', 'public': 'Target open.', 'predicate': {'eq': [ref('row.status'), lit('open')]}})
    source['tools'][1]['rule_ids'].append('open')
    source['tools'][1]['operations'].append({'table': 'other', 'key': lit('X'), 'set': {'balance': lit(0)}})
    with pytest.raises(ContractError, match='transaction rule open references row fields absent from table other: status'):
        compile_contract(source)
