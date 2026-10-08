import copy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import ContractUserSimulator, EpisodeStore, SandboxError
from tests.generation.test_interaction_contract import interactive_source


def bounded_source():
    source, request = interactive_source()
    request['require_structured_interactions'] = True
    first, second = source['tools']
    del first['function']['parameters']['properties']['region']
    first['function']['parameters']['required'].remove('region')
    second['function']['parameters']['properties']['region'] = {'type': 'string', 'description': '用户最终区域'}
    second['function']['parameters']['required'].append('region')
    source['tool_implementations'][0]['filters'] = [{'argument': 'name', 'column': 'name', 'operator': 'eq'}]
    source['tool_implementations'][0]['projection'].append('regions')
    source['tool_implementations'][1]['filters'].append({'argument': 'region', 'column': 'region', 'operator': 'eq'})
    source['tables'][0]['columns'].append({'name': 'regions', 'type': 'text', 'nullable': False})
    for row in source['tables'][0]['rows']:
        row['regions'] = '东区、西区'
    calls = source['scenarios'][0]['steps']
    calls[0]['arguments'].pop('region')
    calls[1]['arguments']['region'] = '东区'
    for variant in source['interaction_contract']['variants']:
        stage = variant['stages'][1]
        stage.update(before_tool='read_temperature', bind_to={'tool': 'read_temperature', 'argument': 'region'},
            requires_tools=['lookup_zone'], reference_response='已查到可选区域东区、西区。请确认最终区域。',
            options_from={'tool': 'lookup_zone', 'path': '$.records[0].regions', 'separator': '、'})
        stage['branches'] = [{'id': 'decline', 'when': [
            {'source': 'user_state', 'path': '$.refuse', 'operator': 'eq', 'expected': True}],
            'user_reply': '我不愿更改区域，请停止。', 'outcome': 'user_rejection', 'advance': False}]
        variant['variables'] = {'refuse': False}
    return source, request


def test_bounded_choices_corrections_and_updates_compile_and_execute(tmp_path):
    source, request = bounded_source()
    artifact = compile_source(source, root=tmp_path, request=request)
    assert artifact['generation_pipeline']['verification']['passed']
    scripts = json.loads((tmp_path/'data/user_simulation/user_scripts.json').read_text())
    assert {s['interaction_protocol']['stages'][1]['kind'] for s in scripts} == {
        'option_selection', 'correction', 'constraint_update'}
    for script in scripts:
        envelope = json.loads(script['interaction_protocol']['stages'][1]['reference_response'])
        assert envelope['request']['options'] == ['东区', '西区']


def test_correction_cannot_drop_an_observed_domain_shared_by_other_variants(tmp_path):
    source, request = bounded_source()
    source['interaction_contract']['variants'][2]['stages'][1].pop('options_from')
    with pytest.raises(ValueError, match='INTERACTION_OPTION_DOMAIN.*read_temperature.*region'):
        compile_source(source, root=tmp_path, request=request, structural_preview=True)


@pytest.mark.parametrize('variant_index', [1, 2])
@pytest.mark.parametrize('refuse, choices, expected_applied', [
    (False, ['东区', '南区'], True),
    (False, ['西区', '南区'], False),
    (True, ['西区', '南区'], True),
])
def test_actual_catalog_controls_corrections_without_blocking_real_refusal(
        tmp_path, variant_index, refuse, choices, expected_applied):
    source, request = bounded_source()
    compile_source(source, root=tmp_path, request=request, structural_preview=True)
    scripts = json.loads((tmp_path/'data/user_simulation/user_scripts.json').read_text())
    profiles = json.loads((tmp_path/'data/user_simulation/user_profiles.json').read_text())
    script = copy.deepcopy(scripts[variant_index])
    script['variables']['refuse'] = refuse
    store = EpisodeStore(tmp_path/'episode.sqlite3')
    store.reset(seed=17)
    user = ContractUserSimulator(store, scripts=[script], profiles=profiles)
    user.reset()
    stages = script['interaction_protocol']['stages']
    conversation = []
    def turn(content):
        conversation.append({'role': 'assistant', 'content': content})
        result = user.turn(conversation)
        conversation.append({'role': 'user', 'content': result['user_query']})
        return result
    first = turn(stages[0]['reference_response'])
    assert first['fsm_transition_applied']
    store.event('tool_call', {'tool_name': 'lookup_zone', 'arguments': {'name': '青松库'}},
                {'records': [{'id': 821, 'regions': '、'.join(choices)}]})
    envelope = json.loads(stages[1]['reference_response'])
    # The old fixture menu must never authorize a corrected value.
    stale = turn(json.dumps(envelope, ensure_ascii=False))
    assert not stale['fsm_transition_applied']
    envelope['request']['options'] = choices
    envelope['question'] = '实际可选区域为' + '、'.join(choices) + '，请给出最终选择。'
    actual = turn(json.dumps(envelope, ensure_ascii=False))
    assert actual['fsm_transition_applied'] is expected_applied
    if refuse:
        assert actual['interaction_branch'] == 'decline'
        assert actual['outcome_category'] == 'user_rejection'
        assert actual['interaction_stage'] is None
    elif expected_applied:
        assert actual['outcome_category'] == 'user_correction'
        user.guard_tool('read_temperature')
    if refuse or not expected_applied:
        with pytest.raises(SandboxError, match='required user interaction'):
            user.guard_tool('read_temperature')
