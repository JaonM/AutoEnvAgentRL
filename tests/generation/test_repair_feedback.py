import json
from env_factory.generation.repair_feedback import repair_feedback


def setup(tmp_path):
    root=tmp_path;attempt=root/'code_agent_attempts/attempt-1';(attempt/'semantic_review').mkdir(parents=True)
    (root/'authoring').mkdir()
    (root/'authoring/source.json').write_text('{"description":{"task":"Preserve explanations"}}')
    return root,attempt


def test_ablation_feedback_identifies_distinct_omissions_and_retained_reward(tmp_path):
    root, attempt = setup(tmp_path)
    evidence = [{'check': f'skip_0_{index}', 'scenario_id': 'shortage',
                 'omitted_step_indexes': [index], 'omitted_steps': [{'tool_name': 'read_status'}],
                 'actual_reward': 1.0, 'maximum_allowed_reward': .2,
                 'message': 'Removing this request retained reward.'} for index in (3, 4)]
    packet = repair_feedback(ValueError('AGENT_PREFLIGHT_FAILED: ' + json.dumps({
        'failed_checks': ['skip_0_3', 'skip_0_4'], 'ablations': evidence})),
        root=root, attempt_root=attempt)
    assert [i['observed_failure'] for i in packet['issues']] == evidence
    assert len({i['issue_id'] for i in packet['issues']}) == 2
    assert all(i['evidence_level'] == 'execution_observation' for i in packet['issues'])


def test_condition_feedback_retains_inconclusive_state_and_events(tmp_path):
    root, attempt = setup(tmp_path)
    finding = {'scenario_id': 'main', 'status': 'needs_authored_case',
        'reason': 'query-time predicate cannot be isolated',
        'mutation': {'condition_id': 'current'},
        'execution': {'condition_observations': [{'eligible_count': 1,
            'external_changes': [{'event_id': 'activate'}]}]}}
    packet = repair_feedback(ValueError('AGENT_PREFLIGHT_FAILED: ' + json.dumps({
        'failed_checks': ['business_condition_0_current'], 'business_conditions': [finding]})),
        root=root, attempt_root=attempt)
    issue, = packet['issues']
    assert issue['observed_failure'] == finding
    assert issue['claim'] == finding['reason']
    assert 'Preserve task dynamics' in issue['suggested_change']


def test_review_claim_contains_resolved_evidence_without_becoming_proof(tmp_path):
    root,attempt=setup(tmp_path)
    bundle={'compiled_tool_implementations':[{'filters':[{'argument':'state','column':'state','operator':'eq'}]}]}
    (attempt/'semantic_review/candidate.json').write_text(json.dumps(bundle))
    findings=[{'code':'missing_filter','reason':'Filter seems absent','repair':'Verify state filter',
        'source_paths':['compiled_tool_implementations[0].filters','source.unknown']}]
    packet=repair_feedback(ValueError('SOURCE_SEMANTIC_REVIEW_FAILED: '+json.dumps(findings)),root=root,attempt_root=attempt)
    issue=packet['issues'][0]
    assert issue['evidence_level']=='model_review_claim'
    assert issue['locations'][0]['value']==bundle['compiled_tool_implementations'][0]['filters']
    assert issue['locations'][1]['resolved'] is False
    assert 'grants readiness' in packet['acceptance_policy']
    assert packet['source_sha256']


def test_execution_failure_retains_actual_expected_and_unresolved_lookup(tmp_path):
    root,attempt=setup(tmp_path)
    failure={'scenario_id':'success','code':'SCENARIO_ASSERTION_FAILED','message':'assertion failed',
        'details':{'actual':.6,'assertion':{'operator':'gte','expected':1}},
        'diagnostic':{'answer_targets':[{'field':'dish','expected':None,'unresolved_lookups':[{'reason':'lookup_not_unique','matched_row_count':2}]}]}}
    packet=repair_feedback(ValueError('AGENT_SCENARIO_FAILED: '+json.dumps(failure)),root=root,attempt_root=attempt)
    issue=packet['issues'][0]
    assert issue['evidence_level']=='execution_observation'
    assert issue['observed_failure']==failure['details']
    assert issue['answer_targets']==failure['diagnostic']['answer_targets']
    assert 'original assertion' in issue['closure']


def test_calibration_label_is_disagreement_not_proven_bad_example(tmp_path):
    root,attempt=setup(tmp_path)
    report={'cases':[{'metric_id':'why','index':0,'expected_label':'pass','actual_score':0,'passed':False},
                     {'metric_id':'why','index':1,'passed':True}]}
    packet=repair_feedback(ValueError('SEMANTIC_CALIBRATION_FAILED: '+json.dumps(report)),root=root,attempt_root=attempt)
    assert len(packet['issues'])==1
    assert packet['issues'][0]['evidence_level']=='observed_judge_disagreement'
    assert 'do not relabel' in packet['issues'][0]['suggested_change']


def test_execution_feedback_retains_gate_and_preservation_evidence(tmp_path):
    root, attempt = setup(tmp_path)
    diagnostic = {'gate_diagnostics': {'dependency_path_passed': False},
        'metric_state_predicates': [{'metric_id': 'reserved', 'expected_values': {'quantity': 24}}],
        'preservation_violations': [{'table': 'reservations', 'identity': {'id': 'R1'},
                                   'unexpected_fields': ['fee']}]}
    packet = repair_feedback(ValueError('AGENT_SCENARIO_FAILED: ' + json.dumps({
        'scenario_id': 'reference', 'diagnostic': diagnostic})), root=root, attempt_root=attempt)
    for key, value in diagnostic.items():
        assert packet['issues'][0][key] == value


def test_execution_feedback_keeps_semantic_failure_when_rule_components_pass(tmp_path):
    root, attempt = setup(tmp_path)
    diagnostic = {'gated_components': {'facts': 1},
        'last_evaluated_components': {'facts': 1, 'explanation': 0},
        'model_evidence': [{'event': 'evaluator_call', 'payload': {
            'metric_id': 'explanation', 'context_hash': 'exact-context', 'used_fallback': False},
            'result': {'label': 'fail'}}]}
    issue = repair_feedback(ValueError('AGENT_SCENARIO_FAILED: ' + json.dumps({
        'scenario_id': 'refused', 'diagnostic': diagnostic})),
        root=root, attempt_root=attempt)['issues'][0]
    assert issue['components'] == {'facts': 1}
    assert issue['last_evaluated_components'] == diagnostic['last_evaluated_components']
    assert issue['model_evidence'] == diagnostic['model_evidence']


def test_calibration_feedback_preserves_failed_answer_and_reference_scope(tmp_path):
    root, attempt = setup(tmp_path)
    case = {'metric_id': 'why', 'index': 9, 'expected_label': 'pass', 'actual_score': 0,
        'passed': False, 'reference_id': 'refused-script-3',
        'failed_case_evidence': {'answer': {'analysis': 'No inspection after refusal.'},
            'semantic_fields': ['analysis'], 'criteria': ['Explain the observed stop.']}}
    issue = repair_feedback(ValueError('SEMANTIC_CALIBRATION_FAILED: ' + json.dumps({
        'cases': [case]})), root=root, attempt_root=attempt)['issues'][0]
    assert issue['reference_id'] == case['reference_id']
    assert issue['failed_case_evidence'] == case['failed_case_evidence']


def test_goal_reward_conflicts_preserve_both_values_and_distinct_paths(tmp_path):
    root, attempt = setup(tmp_path)
    conflicts = [{'metric_id': 'reservation', 'field': field, 'reward_value': 24, 'goal_value': 20,
                  'reward_path': f'metric_implementations[0].expected[0].values.{field}',
                  'goal_path': f'semantic_goal.row_predicates[0].values.{field}'}
                 for field in ('reserved', 'quantity')]
    packet = repair_feedback(ValueError('REWARD_GOAL_CONFLICT: ' + json.dumps(conflicts)),
                             root=root, attempt_root=attempt)
    assert [issue['observed_failure'] for issue in packet['issues']] == conflicts
    assert len({issue['issue_id'] for issue in packet['issues']}) == 2
    assert all('business_contract' in issue['suggested_change'] for issue in packet['issues'])


def test_invalid_json_has_safe_fallback_and_stable_issue_id(tmp_path):
    root,attempt=setup(tmp_path);(root/'authoring/source.json').write_text('{bad')
    args={'root':root,'attempt_root':attempt}
    a=repair_feedback(ValueError('invalid source'),**args)
    b=repair_feedback(ValueError('invalid source'),**args)
    assert a['issues'][0]['issue_id']==b['issues'][0]['issue_id']
    assert a['issues'][0]['evidence_level']=='validator_diagnostic'


def test_related_privacy_surfaces_are_preserved_without_declaring_all_hits_defects(tmp_path):
    root,attempt=setup(tmp_path)
    obligations=[{'surface':'description.public_input.semantic_answer_criteria[0].rubric','private_fact':'PrivateChoice',
                  'status':'requires_independent_privacy_review'}]
    (attempt/'semantic_review/candidate.json').write_text(json.dumps({'privacy_review_obligations':obligations}))
    finding={'code':'private_choice_leaked','source_paths':['source.criteria'],'reason':'Criteria reveals choice','repair':'Remove revealed selection'}
    packet=repair_feedback(ValueError('SOURCE_SEMANTIC_REVIEW_FAILED: '+json.dumps([finding])),root=root,attempt_root=attempt)
    assert packet['related_review_obligations']['items']==obligations
    assert packet['related_review_obligations']['evidence_level']=='compiler_lexical_findings_require_semantic_review'
    assert 'Do not assume' in packet['related_review_obligations']['policy']
    assert 'legitimate unselected option lists' in packet['issues'][0]['validate_before_edit'][-1]


def test_wildcard_review_path_exposes_each_actual_script_value(tmp_path):
    from env_factory.generation.repair_feedback import _location
    bundle={'source':{'variants':[{'stages':[{'value':'A'},{'value':'X'}]},
                                   {'stages':[{'value':'B'},{'value':'X'}]}]}}
    location=_location(bundle,'source.variants[*].stages[1].value')
    assert location['resolved'] is True
    assert location['value']==['X','X']
    assert _location(bundle,'source.variants[*].missing')['resolved'] is False
    assert _location(bundle,'source.variants[100].stages')['resolved'] is False
