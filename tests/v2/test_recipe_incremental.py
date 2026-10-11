import copy
from unittest.mock import patch
from env_factory_v2.examples import direct
from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.validation.verification import preflight_recipes,run_scenario


def test_fixture_only_change_reexecutes_changed_case_only():
    source=direct();cache={}
    a=preflight_recipes(compile_contract(source),cache=cache)
    assert a['execution_reuse']['reused']==0
    source['scenarios'][1]['answer']['temperature']={'literal':999}
    with patch('env_factory_v2.validation.verification.run_scenario',wraps=run_scenario) as run:
        b=preflight_recipes(compile_contract(source),cache=cache)
    assert run.call_count==1 and b['execution_reuse']['reused']==1
    assert b['passed']


def test_runtime_change_invalidates_every_case_and_preserves_failed_evidence():
    source=direct();cache={};preflight_recipes(compile_contract(source),cache=cache)
    source['answers'][0]['expected']={'literal':22}
    with patch('env_factory_v2.validation.verification.run_scenario',wraps=run_scenario) as run:
        result=preflight_recipes(compile_contract(source),cache=cache)
    assert run.call_count==2 and not result['passed']
    again=preflight_recipes(compile_contract(source),cache=cache)
    assert again['checks']==result['checks'] and not again['passed']
