from copy import deepcopy
from unittest.mock import Mock

import pytest

from env_factory_v2.challenge import challenge
from env_factory_v2.contract import ContractError, compile_contract
from env_factory_v2.demand import layered_demand
from env_factory_v2.examples import direct
from env_factory_v2.requirement_review import obligations, validate_review


def fixture():
    source = direct()
    source['demand'] = layered_demand(source['task'], '温度为18。', document_id='temperature.json')
    compiled = compile_contract(source)
    requirements = obligations(compiled)
    review = {'business_consistent': True, 'reason': 'fixed temperature', 'cases': [
        {'id': 'yes', 'rule_ids': ['fixed_fact'], 'scenario_id': 'correct', 'reason': 'correct',
         'expect_success': True, 'mutations_json': '[]', 'answer_json': '{}'},
        {'id': 'no', 'rule_ids': ['fixed_fact'], 'scenario_id': 'wrong', 'reason': 'wrong',
         'expect_success': False, 'mutations_json': '[]', 'answer_json': '{}'}],
        'requirement_reviews': [{'requirement_id': r['id'], 'consistent': True,
            'reason': 'Expected temperature is the publicly declared value.',
            'implementation_paths': ['/answers/0/expected'], 'case_ids': ['yes', 'no']}
            for r in requirements]}
    return compiled, requirements, review


def test_every_public_clause_links_to_existing_behavior_and_executed_counterexamples():
    compiled, requirements, review = fixture()
    agent = Mock()
    agent.ask.return_value = (review, {'native': True})
    result = challenge(compiled, {'goal': compiled['source']['task']}, agent)
    assert result['passed']
    assert result['requirement_review']['passed']
    assert len(result['records']) == 2
    assert all(r['passed'] for r in result['records'])
    assert agent.ask.call_count == 1
    assert agent.ask.call_args.args[1]['public_requirements'] == requirements


@pytest.mark.parametrize('defect', ['omitted', 'duplicate', 'unknown_case', 'nonexistent_path',
                                  'private_facts', 'text_only', 'negative_index'])
def test_coverage_cannot_be_claimed_by_invalid_evidence(defect):
    compiled, requirements, value = fixture()
    item = value['requirement_reviews'][0]
    if defect == 'omitted':
        value['requirement_reviews'].pop()
    elif defect == 'duplicate':
        value['requirement_reviews'][1] = deepcopy(item)
    elif defect == 'unknown_case':
        item['case_ids'] = ['not-executed']
    else:
        item['implementation_paths'] = [{
            'nonexistent_path': '/answers/0/nonexistent', 'private_facts': '/user/facts/temperature',
            'text_only': '/demand/task_spec/requirements', 'negative_index': '/answers/-1/expected'}[defect]]
    with pytest.raises(ContractError):
        validate_review(compiled, requirements, value)


def test_passed_cases_cannot_override_failed_clause_interpretation():
    compiled, _, review = fixture()
    review['requirement_reviews'][0]['consistent'] = False
    agent = Mock()
    agent.ask.return_value = (review, {})
    result = challenge(compiled, {}, agent)
    assert not result['passed']
    assert result['status'] == 'inconclusive'
    assert all(r['passed'] for r in result['records'])


def test_case_label_audit_cannot_override_failed_requirement_review():
    compiled, _, review = fixture()
    review['business_consistent'] = False
    review['requirement_reviews'][0]['consistent'] = False
    audit = {'business_consistent': True, 'reason': 'case labels are correct',
             'cases': [{key: case[key] for key in ('id', 'expect_success', 'reason')}
                       for case in review['cases']]}
    agent = Mock()
    agent.ask.side_effect = [(review, {}), (audit, {})]
    result = challenge(compiled, {}, agent)
    assert result['business_consistent']
    assert 'review_audit' in result
    assert not result['passed']
    assert result['status'] == 'inconclusive'
