import pytest

from env_factory_v2.contracts.contract import ContractError, validate_schema, validate_value


def test_nested_array_lengths_are_enforced_without_changing_values():
    schema = {'type': 'array', 'minItems': 0, 'maxItems': 3,
              'items': {'type': 'array', 'minItems': 2, 'maxItems': 2,
                        'items': {'type': ['string', 'integer']}}}
    validate_schema(schema)
    for value in ([], [['Q1', 2]], [['Q1', 2], ['Q2', 3]]):
        validate_value(value, schema)
    for value in ([['Q1']], [['Q1', 2, 3]], [['Q1', 2]] * 4):
        with pytest.raises(ContractError):
            validate_value(value, schema)


@pytest.mark.parametrize('bounds', [{'minItems': True}, {'maxItems': False},
    {'minItems': -1}, {'maxItems': 10001}, {'minItems': 3, 'maxItems': 2},
    {'maxItems': 2.0}, {'minItems': None}])
def test_invalid_array_bounds_fail_at_schema_validation(bounds):
    with pytest.raises(ContractError):
        validate_schema({'type': 'array', 'items': {'type': 'integer'}, **bounds})


@pytest.mark.parametrize('kind', ['integer', 'string', ['integer', 'string']])
def test_non_arrays_cannot_claim_item_bounds(kind):
    with pytest.raises(ContractError, match='require array type'):
        validate_schema({'type': kind, 'maxItems': 2})


def test_zero_length_array_and_defaults_keep_empty_results_valid():
    schema = {'type': 'array', 'maxItems': 0, 'items': {'type': 'integer'}}
    validate_schema(schema)
    validate_value([], schema)
    with pytest.raises(ContractError, match='exceeds maxItems'):
        validate_value([1], schema)
    validate_value([], {'type': 'array', 'items': {'type': 'integer'}})
