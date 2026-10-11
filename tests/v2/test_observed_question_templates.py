import copy
import pytest
from env_factory_v2.contracts.contract import compile_contract, validate_question
from env_factory_v2.examples import retrieval
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.validation.verification import run_scenario
from env_factory_v2.contracts.expressions import ContractError


def source():
    s=retrieval()
    s['scenarios'][0]['steps'][2]['question']={'parts':['已查询品牌：',
        {'$result':'brand_result','path':'items'},'。需要哪个尺码？']}
    return s


def test_reference_trace_contains_actual_rendered_observation():
    s=source()
    compile_contract(s)
    result=run_scenario(s,s['scenarios'][0])
    assert '已查询品牌：' in result['trace'][2]['question']
    assert isinstance(result['trace'][2]['question'],str)


@pytest.mark.parametrize('part', [{'ref':'data.brands'}, {'$user':'brand','path':'value'},
    {'$result':'x','path':'items','code':'eval'},42])
def test_private_or_executable_question_parts_rejected(part):
    with pytest.raises(ContractError):validate_question({'parts':[part]})


def test_unknown_observation_and_length_limits_rejected():
    w=Sandbox(retrieval())
    with pytest.raises(ContractError,match='unknown observed result'):
        w.interact('brand',question={'parts':[{'$result':'missing','path':'items'}]})
    with pytest.raises(ContractError):validate_question({'parts':['x']*33})
    with pytest.raises(ContractError):validate_question('x'*8001)


def test_display_tracks_actual_query_result_after_public_id_changes():
    s=source()
    original=run_scenario(s,s['scenarios'][0])['trace'][2]['question']
    for table in s['tables']:
        for row in table['rows']:
            for key in list(row):
                if row[key] == 'B1':row[key]='BrandUpdated'
    compile_contract(s)
    updated=run_scenario(s,s['scenarios'][0])['trace'][2]['question']
    assert updated != original
    assert 'BrandUpdated' in updated
