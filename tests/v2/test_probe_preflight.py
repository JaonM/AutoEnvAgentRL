import copy

from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import simple
from env_factory_v2.verification import preflight_recipes


def test_preflight_executes_probe_recipes_without_grading_or_changing_source():
    source = simple()
    source['probes'][-1]['answer'] = {'ids': {'ref': 'evidence.missing.items'}}
    before = copy.deepcopy(source)
    report = preflight_recipes(compile_contract(source))
    checks = {item['id']: item for item in report['checks']}
    assert not report['passed']
    assert checks['correct']['passed']
    assert checks['changed_size']['passed']
    assert not checks['stale_size_answer']['passed']
    assert checks['stale_size_answer']['kind'] == 'probe'
    assert checks['stale_size_answer']['error']
    assert source == before


def test_preflight_rejects_stale_probe_expected_error_even_if_execution_succeeds():
    source = simple()
    source['probes'][-1]['expect_error'] = 'transaction business precondition failed'
    report = preflight_recipes(compile_contract(source))
    check = next(item for item in report['checks'] if item['id'] == 'stale_size_answer')
    assert not check['passed']
    assert check['error'] is None
