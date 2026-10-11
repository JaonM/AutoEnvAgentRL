import copy
from unittest.mock import patch

import pytest

from env_factory_v2.contracts.contract import ContractError, compile_contract
from env_factory_v2.examples import transaction
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.validation.verification import expected_result, preflight_recipes, run_scenario


@pytest.mark.parametrize('corruption', ['data', 'consent', 'version'])
def test_error_text_cannot_hide_failed_transaction_corruption(corruption):
    compiled = compile_contract(transaction())
    case = next(c for c in compiled['source']['scenarios'] if c['id'] == 'no_authorization')
    original = Sandbox.call
    def faulty(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except ContractError:
            if corruption == 'data':
                self.data['accounts'][0]['balance'] += 1
            elif corruption == 'consent':
                self.user.states['approve'] = 'consent.consumed'
            else:
                self.version += 1
            raise
    with patch.object(Sandbox, 'call', faulty):
        run = run_scenario(compiled, case)
        preflight = preflight_recipes(compiled)
    assert case['expect_error'] in run['error']
    assert not expected_result(case, run)
    assert not run['failure_integrity']['passed']
    assert not preflight['passed']


def test_failed_second_transaction_preserves_first_commit_and_due_events():
    source = transaction()
    case = copy.deepcopy(source['scenarios'][0])
    case['id'] = 'second_write_without_new_consent'
    second = copy.deepcopy(case['steps'][-1])
    second['id'] = 'second_transfer'
    case['steps'].append(second)
    case.update(expect_success=False, expect_error='authorization')
    source['scenarios'].append(case)
    source['events'] = [{'at_seconds': 4, 'table': 'accounts', 'key': 'A', 'set': {'balance': 77}}]
    source['time']['seconds_per_action'] = 1
    compiled = compile_contract(source)
    run = run_scenario(compiled, next(c for c in compiled['source']['scenarios'] if c['id'] == case['id']))
    assert expected_result(case, run)
    assert run['failure_integrity']['passed']
    assert {r['id']: r['balance'] for r in run['final_state']['accounts']} == {'A': 77, 'B': 30}
