import copy

import pytest

from env_factory.generation.requirement_coverage import validate_requirement_coverage


def source():
    return {'description': {'public_input': {'initial_user_message': 'Stay within budget'}},
            'metrics': [{'id': 'budget', 'category': 'outcome'}, {'id': 'read', 'category': 'process'}],
            'tools': [{'function': {'name': 'quote'}}],
            'scenarios': [{'scenario_id': 'success', 'kind': 'goal_success'},
                          {'scenario_id': 'over_budget', 'kind': 'goal_failure'}],
            'business_requirements': [{'id': 'cost', 'statement': 'Do not exceed the budget',
                'input_refs': ['/description/public_input/initial_user_message'], 'tools': ['quote'],
                'outcome_ids': ['budget'], 'positive_scenarios': ['success'],
                'negative_scenarios': ['over_budget']}]}


def test_coverage_binds_real_inputs_outcomes_and_both_scenario_polarities():
    document = source()
    report = validate_requirement_coverage(document, required=True)
    assert report['status'] == 'structurally_valid'
    assert report['paired_execution_verified'] is False
    assert report['semantic_completeness_verified'] is False
    document['business_requirements'][0]['statement'] = 'changed'
    assert report['requirements'][0]['statement'] != 'changed'


@pytest.mark.parametrize('field,value', [
    ('input_refs', ['/description/private/missing']), ('outcome_ids', ['read']),
    ('tools', ['fabricated_tool']), ('negative_scenarios', ['success']),
    ('positive_scenarios', []), ('statement', ''),
])
def test_invalid_or_unverified_links_are_rejected(field, value):
    document = source()
    document['business_requirements'][0][field] = value
    with pytest.raises(ValueError):
        validate_requirement_coverage(document, required=True)


def test_uncovered_outcome_and_duplicate_obligation_are_rejected():
    document = source()
    document['metrics'].append({'id': 'deadline', 'category': 'outcome'})
    with pytest.raises(ValueError, match='uncovered outcomes'):
        validate_requirement_coverage(document)
    document = source()
    document['business_requirements'].append(copy.deepcopy(document['business_requirements'][0]))
    with pytest.raises(ValueError, match='unique'):
        validate_requirement_coverage(document)


def test_legacy_source_is_explicitly_unverified():
    assert validate_requirement_coverage({})['status'] == 'not_declared'
    with pytest.raises(ValueError, match='nonempty'):
        validate_requirement_coverage({}, required=True)


def test_bound_negative_allows_partial_reward_but_wrong_binding_fails(tmp_path):
    from tests.generation.test_agent_authoring import fixture
    from env_factory.generation.agent_authoring import compile_source, verify_execution
    document, request = fixture('simple_agentic')
    document['tables'][0]['columns'].append({'name': 'unit', 'type': 'text', 'nullable': False})
    for row in document['tables'][0]['rows']:
        row['unit'] = '摄氏度'
    document['tool_implementations'][0]['projection'].append('unit')
    document['metrics'][0]['weight'] = .5
    document['metrics'].append({**copy.deepcopy(document['metrics'][0]), 'id': 'unit', 'weight': .5})
    expression = document['metric_implementations'][0]['expected']['targets'][0]['expression']
    document['metric_implementations'][0].update(operator='value_targets', expected={
        'answer_format': 'json_object', 'targets': [{'key': 'temperature', 'expression': expression}]})
    document['metric_implementations'].append({'metric_id': 'unit', 'source': 'final_agent_response',
        'path': '$', 'operator': 'value_targets', 'expected': {'answer_format': 'json_object',
        'targets': [{'key': 'unit', 'expression': {'lookup': {'table': 'zones', 'field': 'unit', 'where': {'name': '青松库'}}}}]}, 'score_mapping': {'pass': 1, 'fail': 0}})
    document['answer_contract'] = {'format': 'json_object', 'schema': {'type': 'object',
        'properties': {'temperature': {'type': 'number', 'description': '温度'},
                       'unit': {'type': 'string', 'description': '温度单位'}},
        'required': ['temperature', 'unit'], 'additionalProperties': False}}
    document['scenarios'][0]['steps'][-1]['content'] = '{"temperature":18,"unit":"摄氏度"}'
    negative = copy.deepcopy(document['scenarios'][0])
    negative.update(scenario_id='wrong_temperature', kind='goal_failure')
    negative['steps'][-1]['content'] = '{"temperature":999,"unit":"摄氏度"}'
    document['scenarios'].append(negative)
    document['business_requirements'] = [{'id': key, 'statement': key,
        'input_refs': ['/description/public_input/initial_user_message'], 'tools': ['read_temperature'],
        'outcome_ids': [metric], 'positive_scenarios': ['success'], 'negative_scenarios': [case]}
        for key, metric, case in [('temperature', 'temperature_answer', 'wrong_temperature'), ('unit', 'unit', 'failure')]]
    artifacts = compile_source(document, root=tmp_path, request=request)
    assert artifacts['generation_pipeline']['verification']['checks']['authored_failure_1']
    # Rebinding the same partial failure to a component it did not violate must
    # fail, including nested route verification that skips coverage reporting.
    artifacts['requirement_coverage']['requirements'][0]['outcome_ids'] = ['unit']
    with pytest.raises(ValueError, match='REQUIREMENT_REWARD_MISMATCH'):
        verify_execution(artifacts, tmp_path)
    with pytest.raises(ValueError, match='AGENT_PREFLIGHT_FAILED:.*authored_failure_1'):
        verify_execution(artifacts, tmp_path, verify_route_probes=False)


def test_reward_pairs_measure_component_errors_not_just_total_reward():
    from env_factory.generation.requirement_coverage import score_requirement_pairs
    coverage = validate_requirement_coverage(source())
    metrics = [{'id': 'budget', 'score_range': [0, 1]}]
    observations = {
        'success': {'kind': 'completed', 'reward': {'reward': 1, 'components': {'budget': 1}}},
        'over_budget': {'kind': 'completed', 'reward': {'reward': 0, 'components': {'budget': 1}}},
    }
    report = score_requirement_pairs(coverage, metrics, observations)
    assert not report['passed']
    assert report['false_positive']['error_rate'] == 1
    assert report['false_negative']['error_rate'] == 0
    observations['over_budget']['reward']['components']['budget'] = 0
    assert score_requirement_pairs(coverage, metrics, observations)['passed']
    observations['success']['reward']['components']['budget'] = 0
    assert score_requirement_pairs(coverage, metrics, observations)['false_negative']['error_rate'] == 1


def test_missing_or_nonfinite_evidence_is_inconclusive_not_a_correct_rejection():
    from env_factory.generation.requirement_coverage import score_requirement_pairs
    coverage = validate_requirement_coverage(source())
    metrics = [{'id': 'budget', 'score_range': [0, 1]}]
    for evidence in ({}, {'kind': 'client_rejection'},
                     {'kind': 'completed', 'reward': {'components': {'budget': float('nan')}}}):
        report = score_requirement_pairs(coverage, metrics, {'over_budget': evidence})
        assert report['passed'] is False
        assert report['false_positive']['error_rate'] is None
        assert report['false_positive']['inconclusive'] == 1


def test_interaction_expansion_binds_exact_scenario_ids():
    from env_factory.generation.requirement_coverage import bind_scenarios
    coverage = validate_requirement_coverage(source())
    scenarios = [{'scenario_id': 'success-script-1'}, {'scenario_id': 'success-script-2'},
                 {'scenario_id': 'over_budget'}, {'scenario_id': 'success-unrelated'}]
    result = bind_scenarios(coverage, scenarios, ['script-1', 'script-2'])
    assert result['requirements'][0]['positive_scenarios'] == ['success-script-1', 'success-script-2']
    with pytest.raises(ValueError, match='disappeared'):
        bind_scenarios(coverage, [{'scenario_id': 'success-unrelated'}])


def test_compiler_runs_and_exports_requirement_reward_evidence(tmp_path):
    import json
    from tests.generation.test_agent_authoring import fixture
    from env_factory.generation.agent_authoring import compile_source
    document, request = fixture('multi_step_agentic')
    document['business_requirements'] = [{
        'id': 'temperature', 'statement': 'Return the actual queried temperature',
        'input_refs': ['/description/public_input/initial_user_message'],
        'tools': [t['function']['name'] for t in document['tools']],
        'outcome_ids': ['temperature_answer'], 'positive_scenarios': ['success'],
        'negative_scenarios': ['failure'],
    }]
    document['reward_pair_probes'] = [{'id': 'wrong_lookup_then_correct', 'reference_id': 'success',
        'kind': 'readonly_recovery', 'before_tool': 'lookup_zone',
        'calls': [{'tool_name': 'lookup_zone', 'arguments': {'name': '白杨库'}}]}]
    request['require_business_requirements'] = True
    artifact = compile_source(document, root=tmp_path, request=request)
    evidence = artifact['requirement_coverage']
    assert artifact['reward_pair_validation']['passed'] is True
    assert artifact['reward_pair_validation']['measured'] == 2
    assert evidence['paired_execution_verified'] is True
    assert evidence['semantic_completeness_verified'] is False
    report = json.loads((tmp_path / 'requirement_reward_validation.json').read_text())
    assert report['false_negative']['measured'] == 1
    assert report['false_positive']['measured'] == 1
    assert report['false_positive']['errors'] == 0


def test_requirement_semantics_require_complete_independent_review():
    from env_factory.generation.semantic_review import CHECKS, resolve_requirement_review, SemanticReviewUnavailable
    artifact = {'requirement_coverage': validate_requirement_coverage(source())}
    with pytest.raises(SemanticReviewUnavailable):
        resolve_requirement_review(artifact, {'status': 'pass'})
    assert artifact['requirement_coverage']['semantic_completeness_verified'] is False
    review = {'status': 'pass', 'checks': {name: {'passed': True, 'evidence': 'Checked against public obligations'}
                                         for name in CHECKS}, 'findings': []}
    resolve_requirement_review(artifact, review)
    assert artifact['requirement_coverage']['semantic_completeness_verified'] is True
    assert artifact['requirement_coverage']['paired_execution_verified'] is False
