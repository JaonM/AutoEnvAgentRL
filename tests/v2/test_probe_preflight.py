import copy

from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.examples import simple, direct
from env_factory_v2.validation.verification import preflight_recipes


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


def test_negative_probe_that_actually_gets_full_credit_is_rejected_before_model_review():
    source = simple()
    source['probes'][-1]['answer'] = source['scenarios'][0]['answer']
    report = preflight_recipes(compile_contract(source))
    check = next(item for item in report['checks'] if item['id'] == 'stale_size_answer')
    assert not check['passed']
    assert check['error'] is None
    assert check['deterministic_grade']['reward'] == 1
    assert check['deterministic_grade']['success']


def explained_direct():
    source = direct()
    source['answers'][0]['weight'] = 0.8
    source['explanations'] = [{'id': 'reason', 'meaning': '解释实际温度', 'weight': 0.2,
        'criteria': [{'id': 'fact', 'rule_id': 'fixed_fact', 'obligation': '说明实际温度。'}],
        'context_fields': ['temperature']}]
    for scenario in source['scenarios']:
        scenario['answer']['reason'] = {'literal': '温度依据实际规则。'}
    return source


def test_explanation_does_not_hide_wrong_structured_positive_recipe():
    source = explained_direct()
    positive = next(s for s in source['scenarios'] if s['expect_success'])
    positive['answer']['temperature'] = {'literal': 19}
    report = preflight_recipes(compile_contract(source))
    record = next(c for c in report['checks'] if c['id'] == positive['id'])
    assert not record['passed']
    assert not record['structured_components']['temperature']['passed']
    assert 'semantic_evidence' not in record and 'deterministic_grade' not in record


def test_invalid_reward_target_is_distinct_from_incorrect_submitted_fact():
    source = explained_direct()
    source['answers'][0]['expected'] = {'ref': 'data.missing'}
    report = preflight_recipes(compile_contract(source))
    record = next(c for c in report['checks'] if c['id'] == 'correct')
    assert not record['passed']
    component = record['structured_components']['temperature']
    assert component['target_error'] and 'unresolved path' in component['error']
