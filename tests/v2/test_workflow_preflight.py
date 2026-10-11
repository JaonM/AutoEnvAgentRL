import copy

from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.examples import transaction, lit, ref
from env_factory_v2.validation.verification import preflight_recipes, run_scenario


def refusal_contract():
    source = transaction()
    source['invariants'] = []
    source['goals'] = [{'id': 'preserved', 'rule_ids': [source['rules'][-1]['id']],
        'predicate': {'eq': [ref('data.accounts'), ref('initial.accounts')]}}]
    source['success'] = [{'id': 'revoked', 'public': 'Unchanged balances after revocation.',
        'when': {'eq': [ref('user_state.regions.approve'), lit('consent.revoked')]},
        'required_tools': ['accounts'], 'required_stages': ['approve']}]
    scenario = copy.deepcopy(source['scenarios'][0])
    scenario['id'] = 'correct_revocation'
    scenario['steps'] = scenario['steps'][:2] + [{'kind': 'revoke', 'stage': 'approve'}]
    negative = copy.deepcopy(scenario)
    negative.update(id='unfinished', expect_success=False)
    negative['steps'] = negative['steps'][:-1]
    source['scenarios'] = [scenario, negative]
    source['probes'] = []
    return source


def test_positive_revocation_with_impossible_completion_is_caught_before_judgment():
    compiled = compile_contract(refusal_contract())
    preflight = preflight_recipes(compiled)
    assert not preflight['passed']
    check = preflight['checks'][0]
    assert check['error'] is None
    assert check['deterministic_grade']['process'] == [{'id': 'revoked', 'active': True, 'passed': False}]
    assert run_scenario(compiled, compiled['source']['scenarios'][0])['outcome']['reward'] == 0


def test_correct_refusal_preflight_matches_final_process_without_semantic_grade():
    source = refusal_contract()
    source['success'][0]['required_stages'] = []
    compiled = compile_contract(source)
    preflight = preflight_recipes(compiled)
    assert preflight['passed']
    run = run_scenario(compiled, compiled['source']['scenarios'][0])
    assert run['outcome']['success']
    assert preflight['checks'][0]['deterministic_grade']['process'] == run['outcome']['process']


def test_negative_workflow_failure_remains_a_valid_counterexample():
    source = refusal_contract()
    source['success'][0]['required_stages'] = []
    compiled = compile_contract(source)
    assert preflight_recipes(compiled)['passed']
    assert not run_scenario(compiled, compiled['source']['scenarios'][1])['outcome']['success']
