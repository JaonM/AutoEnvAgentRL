import pytest
from env_factory_v2.contract import validate_schema, validate_value
from env_factory_v2.expressions import ContractError

SCHEMA = {'type': 'array', 'items': {'type':'array', 'items':{'type':'string'}}}
PLANS = {'type':'array','items': {'type':'array', 'items': {
    'type':'array','items': {'type':['string','integer']}}}}


def test_structured_procurement_plans_keep_original_values():
    validate_schema(PLANS)
    plans = [[['Q1', 2], ['Q2', 3]], [['Q3', 1]]]
    validate_value(plans, PLANS)
    assert plans == [[['Q1',2],['Q2',3]],[['Q3',1]]]


@pytest.mark.parametrize('value', [True, 2.0, None, {}, []])
def test_integer_string_union_does_not_admit_other_types(value):
    with pytest.raises(ContractError, match='expected one of'):
        validate_value(value, {'type':['string','integer']})


@pytest.mark.parametrize('kind', [[], ['integer','integer'], ['integer',[]], ['object','string'],
    ['array'], ['null'], {}, 1])
def test_malformed_type_declarations_are_contract_errors_not_host_exceptions(kind):
    with pytest.raises(ContractError): validate_schema({'type':kind})


def test_union_preserves_enum_nullable_and_finite_number_checks():
    schema = {'type':['integer','string'],'enum':[1,'one'], 'nullable':True}
    validate_value(None,schema)
    validate_value(1,schema)
    validate_value('one',schema)
    for bad in [True,2,'two']:
        with pytest.raises(ContractError): validate_value(bad,schema)
    with pytest.raises(ContractError):
        validate_value(float('inf'), {'type':['number','string']})


def test_scalar_union_cannot_hide_structural_fields():
    with pytest.raises(ContractError,match='structural'):
        validate_schema({'type':['string','integer'],'items':{'type':'integer'}})


def test_invalid_nested_plan_reports_contract_error():
    with pytest.raises(ContractError):
        validate_value([[['Q1',True]]],PLANS,'purchase_plans')
