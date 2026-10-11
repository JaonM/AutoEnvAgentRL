import copy
import json
from unittest.mock import Mock

from env_factory_v2.challenge import challenge
from env_factory_v2.communication_coverage import communication_coverage
from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import simple, lit


def reviewed():
    source = simple()
    source['answers'][0]['weight'] = 0.8
    source['explanations'] = [{'id': 'reason', 'meaning': 'Explain relevant branch', 'weight': 0.2,
        'context_fields': ['ids'], 'criteria': [{'id': 'explain', 'rule_id': 'size_matches',
            'obligation': 'Explain the results for this branch.',
            'when': {'eq': [{'ref': 'user.size'}, lit('M')]}}]}]
    for scenario in source['scenarios']:
        scenario['answer']['reason'] = lit('The returned records match the requested size.')
    source['probes'][2]['answer']['reason'] = lit('The returned records match the requested size.')
    compiled = compile_contract(source)
    cases = [{'id': identity, 'scenario_id': scenario, 'rule_ids': ['size_matches'],
              'reason': 'Unit test only', 'mutations_json': '[]', 'answer_json': '{}',
              'user_overrides_json': json.dumps({'size': size}), 'steps_json': 'null',
              'expect_success': positive}
             for identity, scenario, size, positive in [('active', 'correct', 'M', True),
               ('inactive', 'correct', 'L', True), ('wrong', 'wrong', 'M', False)]]
    agent = Mock()
    agent.ask.return_value = ({'business_consistent': True, 'reason': 'Unit test only', 'cases': cases}, {})
    judgment = {'criteria': {'explain': True}, 'provenance': {'unit_test_only': True}}
    agent.judge.return_value = judgment
    agent.independent_judge.return_value = judgment
    return compiled, challenge(compiled, {'goal': source['task']}, agent)


def test_coverage_comes_from_both_executions_not_case_labels():
    compiled, report = reviewed()
    assert report['passed']
    entry = report['communication_coverage']['conditions'][0]
    assert entry['status'] == 'both_observed'
    assert entry['applicable_case_ids'] == ['active', 'wrong']
    assert entry['inapplicable_case_ids'] == ['inactive']
    assert entry['unobserved_case_ids'] == []
    records = copy.deepcopy(report['records'])
    records[0]['case']['expect_success'] = False
    assert communication_coverage(compiled, records) == report['communication_coverage']


def test_errored_and_historical_runs_do_not_prove_a_branch():
    compiled, report = reviewed()
    records = copy.deepcopy(report['records'][:2])
    records[0]['first']['error'] = 'failed before evaluation'
    del records[1]['repeat']['outcome']['communication_applicability']
    entry = communication_coverage(compiled, records)['conditions'][0]
    assert entry['status'] == 'not_observed'
    assert entry['unobserved_case_ids'] == ['active', 'inactive']


def test_one_observed_branch_is_not_an_automatic_business_rejection():
    compiled, report = reviewed()
    coverage = communication_coverage(compiled, report['records'][:1])
    assert coverage['conditions'][0]['status'] == 'applicable_only'
    assert 'passed' not in coverage
    assert 'reachability' in coverage['policy']


def test_disagreeing_replay_cannot_count_as_both_branches():
    compiled, report = reviewed()
    records = copy.deepcopy(report['records'][:1])
    records[0]['repeat']['outcome']['communication_applicability']['reason']['explain'] = False
    entry = communication_coverage(compiled, records)['conditions'][0]
    assert entry['status'] == 'replay_disagreement'
    assert entry['replay_disagreement_case_ids'] == ['active']
    assert not entry['applicable_case_ids'] and not entry['inapplicable_case_ids']
