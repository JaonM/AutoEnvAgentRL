import pytest

from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import simple
from env_factory_v2.expressions import ContractError


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
