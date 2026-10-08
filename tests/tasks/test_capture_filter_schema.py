from copy import deepcopy

import pytest

from env_factory.contracts.tool_chain_contract import captured_schema, schema_conflict
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, validate_json_schema


SCHEMA = {'type': 'object', 'properties': {'records': {'type': 'array', 'items': {
    'type': 'object', 'properties': {'code': {'type': 'string'}, 'price': {'type': 'number'}},
    'required': ['code', 'price'], 'additionalProperties': False}}}}
PATH = '$.records[?(@.code=="selected")].price'


@pytest.mark.parametrize('rows,expected', [
    ([{'code': 'other', 'price': 5}, {'code': 'selected', 'price': 9}], 9),
    ([{'code': 'selected', 'price': 9}, {'code': 'other', 'price': 5}], 9),
    ([], []),
    ([{'code': 'selected', 'price': 9}, {'code': 'selected', 'price': 12}], [9, 12]),
])
def test_filter_capture_schema_matches_runtime_cardinality(rows, expected):
    schema = captured_schema(SCHEMA, PATH)
    assert schema is not None
    result = DeclarativeMetricEvaluator._resolve({'records': rows}, PATH)
    assert result == expected
    validate_json_schema(schema, result, 'filtered_capture')


def test_filter_index_capture_preserves_field_type():
    path = '$.records[?(@.code=="selected")][0].price'
    assert captured_schema(SCHEMA, path) == {'type': 'number'}
    assert schema_conflict(captured_schema(SCHEMA, path), {'type': 'string'})


def test_filter_projection_rejects_incompatible_downstream_type():
    schema = captured_schema(SCHEMA, PATH)
    assert schema_conflict(schema, {'type': 'number'}) is None
    assert schema_conflict(schema, {'type': 'array', 'items': {'type': 'number'}}) is None
    assert schema_conflict(schema, {'type': 'boolean'}) is not None


@pytest.mark.parametrize('path', [
    '$.records[?(@.missing=="selected")].price',
    '$.records[?(@.code==3)].price',
    '$.records[?(@.code=="selected")].missing',
    '$.records[?(@.code>"selected")].price',
    '$.records[0].code[?(@.code=="selected")]',
])
def test_invalid_filter_capture_still_rejected(path):
    assert captured_schema(SCHEMA, path) is None


def test_capture_schema_resolution_preserves_source_schema():
    original = deepcopy(SCHEMA)
    captured_schema(SCHEMA, PATH)
    assert SCHEMA == original
