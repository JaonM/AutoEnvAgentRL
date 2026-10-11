from copy import deepcopy
from unittest.mock import Mock

import pytest

from env_factory_v2.challenge import challenge
from env_factory_v2.contract import ContractError, compile_contract
from env_factory_v2.demand import layered_demand
from env_factory_v2.examples import direct
from env_factory_v2.requirement_review import obligations, validate_review, implementation_at, mandatory_obligations


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
    review['obligation_reviews'] = [{'path': item['path'], 'consistent': True,
        'reason': 'The user explicitly requests this temperature field.',
        'requirement_ids': [requirements[0]['id']], 'case_ids': ['yes', 'no']}
        for item in mandatory_obligations(compiled['source'])]
    return compiled, requirements, review


def test_grouped_obligation_evidence_expands_to_identical_per_path_review():
    compiled, requirements, review = fixture()
    original = validate_review(compiled, requirements, review)
    grouped = deepcopy(review)
    item = grouped['obligation_reviews'][0]
    grouped['obligation_reviews'] = [{**{k:v for k,v in item.items() if k != 'path'},
        'paths': [i['path'] for i in grouped['obligation_reviews']]}]
    assert validate_review(compiled, requirements, grouped) == original


@pytest.mark.parametrize('mutation', ['empty', 'duplicate', 'missing', 'unknown', 'mixed'])
def test_grouping_cannot_bypass_complete_unique_obligation_review(mutation):
    compiled, requirements, review = fixture()
    item = review['obligation_reviews'][0]
    group = {**{k:v for k,v in item.items() if k != 'path'},
             'paths': [i['path'] for i in review['obligation_reviews']]}
    review['obligation_reviews'] = [group]
    if mutation == 'empty': group['paths'] = []
    elif mutation == 'duplicate': group['paths'].append(group['paths'][0])
    elif mutation == 'missing': group['paths'].pop()
    elif mutation == 'unknown': group['paths'][0] = '/unknown'
    else: group['path'] = group['paths'][0]
    with pytest.raises(ContractError):
        validate_review(compiled, requirements, review)


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


def test_failed_requirement_review_does_not_trigger_useless_case_label_audit():
    compiled, _, review = fixture()
    review['business_consistent'] = False
    review['requirement_reviews'][0]['consistent'] = False
    audit = {'business_consistent': True, 'reason': 'case labels are correct',
             'cases': [{key: case[key] for key in ('id', 'expect_success', 'reason')}
                       for case in review['cases']]}
    agent = Mock()
    agent.ask.side_effect = [(review, {}), (audit, {})]
    result = challenge(compiled, {}, agent)
    assert not result['business_consistent']
    assert 'review_audit' not in result
    assert agent.ask.call_count == 1
    assert not result['passed']
    assert result['status'] == 'inconclusive'


def test_declared_integrity_constraints_are_evidence_but_fixture_rows_are_not():
    source = {'tables': [{'schema': {'type': 'object'}, 'primary_key': 'id',
                         'foreign_keys': [], 'data': [{'id': 'P1'}]}]}
    assert implementation_at(source, '/tables/0/primary_key') == 'id'
    assert implementation_at(source, '/tables/0/foreign_keys') == []
    with pytest.raises(ContractError):
        implementation_at(source, '/tables/0/data')


def test_scope_classification_preserves_case_gate_and_cannot_exclude_user_request():
    compiled, requirements, review = fixture()
    policy = next(item for item in review['requirement_reviews']
                  if item['requirement_id'].startswith('business_policy_'))
    policy['applicability'] = 'outside_request'
    assert validate_review(compiled, requirements, review)['passed']
    policy['case_ids'] = []
    with pytest.raises(ContractError, match='actual independent executable cases'):
        validate_review(compiled, requirements, review)
    policy['case_ids'] = ['yes']
    review['requirement_reviews'][0]['applicability'] = 'outside_request'
    with pytest.raises(ContractError, match='user request obligations cannot be excluded'):
        validate_review(compiled, requirements, review)


@pytest.mark.parametrize('defect', ['omitted', 'unknown_path', 'uncited', 'unknown_requirement', 'unknown_case'])
def test_reverse_review_cannot_claim_unsupported_coverage(defect):
    compiled, requirements, review = fixture()
    item = review['obligation_reviews'][0]
    if defect == 'omitted':
        review['obligation_reviews'] = []
    elif defect == 'unknown_path':
        item['path'] = '/private/facts'
    elif defect == 'uncited':
        item['requirement_ids'] = []
    elif defect == 'unknown_requirement':
        item['requirement_ids'] = ['invented']
    else:
        item['case_ids'] = ['invented']
    with pytest.raises(ContractError):
        validate_review(compiled, requirements, review)


def test_unsupported_output_rejection_survives_passing_forward_review_and_cases():
    compiled, requirements, review = fixture()
    item = review['obligation_reviews'][0]
    item.update(consistent=False, requirement_ids=[], reason='Unit test: unsupported mandatory output.')
    agent = Mock()
    agent.ask.return_value = (review, {})
    result = challenge(compiled, {}, agent)
    assert all(record['passed'] for record in result['records'])
    assert all(item['consistent'] for item in result['requirement_review']['reviews'])
    assert not result['passed']
    assert not result['requirement_review']['passed']
    assert agent.ask.call_count == 1


def test_reverse_catalog_includes_nested_output_and_observation_columns():
    source = direct()
    source['answers'][0]['schema'] = {'type': 'array', 'items': {'type': 'object',
        'properties': {'requested': {'type': 'number'}, 'unasked/metadata': {'type': 'string'}}}}
    source['success'] = [{'id': 'done', 'public': 'Finish', 'when': {'literal': True},
        'required_tools': ['read'], 'required_stages': [],
        'required_observations': [{'tools': ['read'], 'columns': ['requested', 'unasked']}]}]
    paths = {item['path'] for item in mandatory_obligations(source)}
    assert '/answers/0/schema/items/properties/unasked~1metadata' in paths
    assert '/success/0/required_observations/0/columns/1' in paths
    assert '/success/0/required_tools/0' in paths


@pytest.mark.parametrize('original_input', [{'goal': 'Report the original value.'},
                                    {'user_request': 'Report the original value.'},
                                    {'raw_request': 'Report the original value.'},
                                    'Report the original value.'])
def test_plain_contract_reviews_original_request_not_generated_task(original_input):
    compiled = compile_contract(direct())
    requirements = obligations(compiled, original_input)
    assert [r['text'] for r in requirements] == ['Report the original value.']
    assert all(r['source'] == 'user_request' for r in requirements)
    assert all('fixed_fact' not in r['text'] for r in requirements)


def test_plain_contract_cannot_skip_bidirectional_review():
    compiled = compile_contract(direct())
    value = {'business_consistent': True, 'reason': 'Missing review must not pass', 'cases': [
        {'id': 'yes', 'rule_ids': ['fixed_fact'], 'scenario_id': 'correct', 'reason': 'correct',
         'expect_success': True, 'mutations_json': '[]', 'answer_json': '{}'},
        {'id': 'no', 'rule_ids': ['fixed_fact'], 'scenario_id': 'wrong', 'reason': 'wrong',
         'expect_success': False, 'mutations_json': '[]', 'answer_json': '{}'}]}
    agent = Mock(); agent.ask.return_value = (value, {})
    with pytest.raises(RuntimeError, match='every public clause'):
        challenge(compiled, {'goal': direct()['task']}, agent)
    payload = agent.ask.call_args.args[1]
    assert payload['public_requirements'] and payload['mandatory_obligation_catalog']
