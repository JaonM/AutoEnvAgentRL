import copy

from env_factory.generation.interaction_diversity import interaction_diversity


def inputs():
    scripts = [{'script_id': f'script-{i}', 'interaction_protocol': {'stages': [{
        'id': 'choice', 'kind': 'information_required', 'before_tool': 'lookup',
        'bind_to': {'tool': 'lookup', 'argument': 'region'}, 'private_fact': 'east',
        'user_reply': 'east' if i == 1 else 'Please use east',
        'reference_response': 'Where?' if i == 1 else 'Which region?'}]}} for i in (1, 2)]
    cases = [{'scenario_id': f'reference-script-{i}', 'kind': 'goal_success', 'steps': [{
        'operation': 'tool_call', 'tool_name': 'lookup', 'arguments': {'region': 'east'}}]}
        for i in (1, 2)]
    return scripts, cases


def test_wording_changes_are_not_behavioral_diversity():
    scripts, cases = inputs()
    report = interaction_diversity(scripts, cases)
    assert report['wording_only_variants']
    assert report['distinct_decisions'] == report['distinct_tool_call_sequences'] == 1
    assert 'not_rollout_coverage' in report['scope']


def test_correction_differs_in_behavior_even_when_final_choice_is_same():
    scripts, cases = inputs()
    scripts[1]['interaction_protocol']['stages'][0].update(kind='correction', previous_fact='west')
    report = interaction_diversity(scripts, cases)
    assert not report['wording_only_variants']
    assert report['distinct_behaviors'] == 2
    assert report['same_terminal_decisions']
    assert report['distinct_tool_call_sequences'] == 1


def test_alternate_choice_needs_separate_reference_evidence():
    scripts, cases = inputs()
    scripts[1]['interaction_protocol']['stages'][0]['private_fact'] = 'west'
    report = interaction_diversity(scripts, cases)
    assert report['distinct_decisions'] == 2
    assert report['distinct_tool_call_sequences'] == 1
    changed = copy.deepcopy(cases)
    changed[1]['steps'][0]['arguments']['region'] = 'west'
    assert interaction_diversity(scripts, changed)['distinct_tool_call_sequences'] == 2
    report = interaction_diversity(scripts, cases[:1])
    assert report['variants_without_reference'] == ['script-2']


def test_new_generation_policy_rejects_wording_only_scripts(tmp_path):
    import pytest
    from tests.generation.test_interaction_contract import interactive_source
    from env_factory.generation.agent_authoring import compile_source
    source, request = interactive_source()
    base = source['interaction_contract']['variants'][0]
    source['interaction_contract']['variants'] = [copy.deepcopy(base) for _ in range(3)]
    for index, variant in enumerate(source['interaction_contract']['variants']):
        variant['stages'][0]['user_reply'] += '。' * index
    request['require_behavioral_diversity'] = True
    with pytest.raises(ValueError, match='INTERACTION_BEHAVIOR_DIVERSITY'):
        compile_source(source, root=tmp_path, request=request)


def test_genuine_correction_variants_meet_new_policy(tmp_path):
    from tests.generation.test_interaction_contract import interactive_source
    from env_factory.generation.agent_authoring import compile_source
    source, request = interactive_source()
    request['require_behavioral_diversity'] = True
    artifact = compile_source(source, root=tmp_path, request=request)
    report = artifact['task_spec']['interaction_complexity']['behavioral_diversity']
    assert report['distinct_behaviors'] == 3
    assert report['same_terminal_decisions']


def test_untriggered_branches_do_not_satisfy_executed_diversity(tmp_path):
    import pytest
    from tests.generation.test_interaction_contract import interactive_source
    from env_factory.generation.agent_authoring import compile_source
    source, request = interactive_source()
    base = source['interaction_contract']['variants'][0]
    variants = [copy.deepcopy(base) for _ in range(3)]
    for index, variant in enumerate(variants):
        variant['stages'][0]['branches'] = [{
            'id': 'unused', 'when': [{'source': 'assistant_text', 'path': '$',
                'operator': 'contains', 'expected': f'never-trigger-{index}'}],
            'user_reply': '请重新提问。', 'advance': False, 'outcome': 'information_required'}]
    source['interaction_contract']['variants'] = variants
    request['require_behavioral_diversity'] = True
    with pytest.raises(ValueError, match='successful references execute only one user behavior'):
        compile_source(source, root=tmp_path, request=request)


def test_executed_nonadvancing_branch_counts_without_stage_id():
    from env_factory.generation.interaction_diversity import executed_behavior_signature
    scripts, _ = inputs()
    stage = scripts[0]['interaction_protocol']['stages'][0]
    stage['branches'] = [{'id': 'retry', 'advance': False, 'outcome': 'user_rejection'}]
    events = [{'event': 'user_protocol', 'payload': {'script_id': 'script-1'}},
              {'event': 'user_turn', 'result': {'fsm_transition_applied': True,
               'fsm_state_before': 'stage-0', 'interaction_branch': 'retry'}}]
    original = executed_behavior_signature(events, scripts)
    renamed = copy.deepcopy(scripts)
    renamed[0]['interaction_protocol']['stages'][0]['branches'][0]['outcome'] = 'user_correction'
    assert original != executed_behavior_signature(events, renamed)
    events[1]['result']['interaction_stage'] = 'choice'
    assert original == executed_behavior_signature(events, scripts)
