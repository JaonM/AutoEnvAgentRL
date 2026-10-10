import pytest

from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import direct, retrieval
from env_factory_v2.expressions import ContractError
from env_factory_v2.verification import run_scenario


def test_simple_goal_can_read_normalized_tables_without_category_change():
    source = retrieval(rows=70)
    source['category'] = 'simple_agentic'
    compiled = compile_contract(source)
    assert compiled['source']['category'] == 'simple_agentic'
    assert len(compiled['source']['tools']) == 2
    for scenario in compiled['source']['scenarios']:
        result = run_scenario(compiled, scenario)
        assert result['outcome']['success'] == scenario['expect_success']


def test_simple_task_still_requires_real_business_tool():
    source = direct()
    source['category'] = 'simple_agentic'
    with pytest.raises(ContractError, match='requires a business tool'):
        compile_contract(source)


def test_direct_category_cannot_hide_stateful_reads():
    source = retrieval(rows=3)
    source['category'] = 'direct_response'
    with pytest.raises(ContractError, match='must be stateless'):
        compile_contract(source)
