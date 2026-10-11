import pytest

from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.examples import simple
from env_factory_v2.contracts.expressions import ContractError


def test_mutated_positive_reports_missing_baseline_even_with_valid_negative():
    source = simple()
    source['scenarios'][0]['mutations'] = [{'table': 'items', 'key': 'I1', 'column': 'display', 'value': 'changed'}]
    with pytest.raises(ContractError, match='baseline success scenario required'):
        compile_contract(source)


def test_negative_probe_does_not_replace_negative_scenario():
    source = simple()
    source['scenarios'] = [source['scenarios'][0]]
    with pytest.raises(ContractError, match='negative scenario required'):
        compile_contract(source)


def test_scenario_field_error_identifies_case_and_missing_and_extra_fields():
    source = simple()
    scenario = source['scenarios'][0]
    missing = next(iter(scenario['answer']))
    scenario['answer'].pop(missing)
    scenario['answer']['unexpected_field'] = {'literal': 0}
    with pytest.raises(ContractError) as error:
        compile_contract(source)
    message = str(error.value)
    assert f"scenario={scenario['id']}" in message
    assert f"missing=['{missing}']" in message
    assert "unexpected=['unexpected_field']" in message
