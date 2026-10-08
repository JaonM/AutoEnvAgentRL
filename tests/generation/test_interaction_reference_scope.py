import copy

import pytest

from env_factory.generation.interaction_contract import compile_interactions, interleave_scenarios
from tests.generation.test_interaction_contract import interactive_source


def scoped_source():
    source, _ = interactive_source()
    original = source['scenarios'][0]
    original['user_script_indices'] = [0, 2]
    alternate = copy.deepcopy(original)
    alternate['scenario_id'] = 'alternate_success'
    alternate['user_script_indices'] = [1]
    alternate['steps'][0]['arguments']['name'] = '白杨库'
    alternate['steps'][-1]['content'] = '温度：23摄氏度。'
    source['scenarios'].append(alternate)
    stage = source['interaction_contract']['variants'][1]['stages'][0]
    stage['private_fact'] = '白杨库'
    stage['user_reply'] = '请查询白杨库。'
    return source


def compile_scripts(source):
    return compile_interactions(source['interaction_contract'], source['description'], source['tools'], 3,
        implementations=source['tool_implementations'], scenarios=source['scenarios'], structured_requests=True)


def test_different_user_choices_select_corresponding_reference_only():
    source = scoped_source()
    scripts = compile_scripts(source)
    expanded = interleave_scenarios(source['scenarios'], scripts)
    successes = [case for case in expanded if case['kind'] == 'goal_success']
    assert [case['scenario_id'] for case in successes] == [
        'success-script-1', 'success-script-3', 'alternate_success-script-2']
    for case in successes:
        call = next(step for step in case['steps'] if step.get('tool_name') == 'lookup_zone')
        assert call['arguments']['name'] == ('白杨库' if case['scenario_id'].endswith('script-2') else '青松库')
        assert 'user_script_indices' not in case


@pytest.mark.parametrize('indices', [[], [True], [-1], [3], [0, 0], '1'])
def test_invalid_scope_is_rejected_before_execution(indices):
    source = scoped_source()
    source['scenarios'][0]['user_script_indices'] = indices
    with pytest.raises(ValueError, match='INTERACTION_REFERENCE_SCOPE'):
        compile_scripts(source)


def test_reference_for_another_script_cannot_validate_private_binding():
    source = scoped_source()
    source['scenarios'][-1]['steps'][0]['arguments']['name'] = '青松库'
    with pytest.raises(ValueError, match='does not match executable reference'):
        compile_scripts(source)


def test_every_script_requires_its_own_reachable_success():
    source = scoped_source()
    source['scenarios'].pop()
    with pytest.raises(ValueError, match='variant 1 has no successful reference'):
        compile_scripts(source)


def test_distinct_choices_execute_and_reward_their_own_answers(tmp_path):
    import json
    from env_factory.generation.agent_authoring import compile_source
    source = scoped_source()
    _, request = interactive_source()
    request.update(require_structured_interactions=True, require_business_requirements=True)
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'tool_result': {'name': 'read_temperature', 'path': '$.records[0].temperature'}}
    positives = [case for case in source['scenarios'] if case['kind'] == 'goal_success']
    negative_ids = []
    for case in positives:
        wrong = copy.deepcopy(case)
        wrong.update(scenario_id=case['scenario_id'] + '_wrong_answer', kind='goal_failure', derive_interactions=True)
        wrong['steps'][-1]['content'] = '温度：23摄氏度。' if case['scenario_id'] == 'success' else '温度：18摄氏度。'
        source['scenarios'].append(wrong)
        negative_ids.append(wrong['scenario_id'])
    source['business_requirements'] = [{'id': 'selected_temperature',
        'statement': 'Report the actual temperature for the warehouse selected by this user',
        'input_refs': ['/description/public_input/initial_user_message'],
        'tools': ['lookup_zone', 'read_temperature'], 'outcome_ids': ['temperature_answer'],
        'positive_scenarios': [case['scenario_id'] for case in positives], 'negative_scenarios': negative_ids}]
    artifact = compile_source(source, root=tmp_path, request=request)
    diversity = artifact['task_spec']['interaction_complexity']['behavioral_diversity']
    assert diversity['distinct_decisions'] == diversity['distinct_tool_call_sequences'] == 2
    report = json.loads((tmp_path / 'requirement_reward_validation.json').read_text())
    assert report['passed']
    assert report['false_negative']['measured'] == report['false_positive']['measured'] == 3
    assert report['false_negative']['errors'] == report['false_positive']['errors'] == 0


def test_compact_alternate_reference_and_scoped_negative_execute(tmp_path):
    from env_factory.generation.agent_authoring import compile_source
    full = scoped_source()
    _, request = interactive_source()
    request['require_structured_interactions'] = True
    compact = {key: copy.deepcopy(full[key]) for key in (
        'version', 'description', 'environment_plan', 'tables', 'interaction_contract')}
    compact['business_tools'] = [{'name': tool['function']['name'],
        'description': tool['function']['description'], 'parameters': tool['function']['parameters'],
        'implementation': implementation} for tool, implementation in zip(full['tools'], full['tool_implementations'])]
    rule = copy.deepcopy(full['metric_implementations'][0])
    rule['expected']['targets'][0]['expression'] = {'tool_result': {
        'name': 'read_temperature', 'path': '$.records[0].temperature'}}
    compact['outcomes'] = [{'id': 'temperature_answer', 'rubric': 'Report the chosen temperature', 'rule': rule}]
    def reference(case):
        return {'calls': [step for step in case['steps'] if step['operation'] == 'tool_call'],
                'answer': case['steps'][-1]['content'], 'user_script_indices': case['user_script_indices']}
    compact['reference'] = reference(full['scenarios'][0])
    compact['reference_variants'] = [dict(reference(full['scenarios'][-1]), id='alternate')]
    compact['negative_scenarios'] = [{'scenario_id': 'wrong_record', 'kind': 'goal_failure',
        'from_reference': True, 'reference_id': 'alternate',
        'argument_mutations': [{'tool_name': 'read_temperature', 'arguments': {'id': 821}}],
        'assertions': [{'source': 'step:reward', 'path': '$.reward', 'operator': 'lte', 'expected': 0}]}]
    artifact = compile_source(compact, root=tmp_path, request=request)
    ids = [case['scenario_id'] for case in artifact['acceptance_contract']['executable_scenarios']]
    assert 'alternate-script-2' in ids
    assert 'alternate-script-1' not in ids
    assert 'wrong_record--alternate-script-2' in ids
    assert 'wrong_record--alternate-script-1' not in ids


def test_process_rule_selects_only_the_active_script():
    from env_factory.sandbox_runtime import DeclarativeMetricEvaluator
    expected = {'tool_name': 'lookup', 'arguments': {'region': 'east'}, 'captures': [],
        'by_user_script': {
            'script-1': [{'tool_name': 'lookup', 'arguments': {'region': 'east'}, 'captures': []}],
            'script-2': [{'tool_name': 'lookup', 'arguments': {'region': 'west'}, 'captures': []}]}}
    calls = [{'event': 'tool_call', 'payload': {'tool_name': 'lookup', 'arguments': {'region': 'west'}}, 'result': {}}]
    evaluate = lambda events: DeclarativeMetricEvaluator._compare(events, 'contains_tool_call', expected)
    assert not evaluate(calls)
    assert not evaluate([{'event': 'user_protocol', 'payload': {'script_id': 'script-1'}}, *calls])
    assert evaluate([{'event': 'user_protocol', 'payload': {'script_id': 'script-2'}}, *calls])
