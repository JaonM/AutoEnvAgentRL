import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import AcceptanceScenarioRunner
from tests.generation.test_bounded_corrections import bounded_source


def test_reference_templates_execute_against_changed_observed_menu(tmp_path):
    source, request = bounded_source()
    for row in source['tables'][0]['rows']:
        row['regions'] = '东区、南区'
    for variant in source['interaction_contract']['variants']:
        variant['stages'][1]['reference_response'] = '实际查询结果：{{observed_result}}。可选区域：{{observed_options}}。请给出最终区域。'
    artifact = compile_source(source, root=tmp_path, request=request)
    assert artifact['generation_pipeline']['verification']['passed']
    dialogues = [step for scenario in artifact['acceptance_contract']['executable_scenarios']
                 for step in scenario['steps'] if step.get('observed_options_from')]
    assert dialogues
    assert all(step['observed_options_from']['tool'] == 'lookup_zone' for step in dialogues)


def test_reference_template_has_no_access_to_private_choice_or_future_result():
    content = json.dumps({'question': '实际记录{{observed_result}}，可选{{observed_options}}，请选择。',
                          'request': {'intent': 'request_input', 'tool': 'downstream',
                                      'arguments': ['choice'], 'options': ['old', 'hidden']}})
    binding = {'tool': 'catalog', 'path': '$.records[*].id'}
    events = [{'event': 'tool_call', 'payload': {'tool_name': 'other'}, 'result': {'records': [{'id': 'secret'}]}},
              {'event': 'tool_call', 'payload': {'tool_name': 'catalog'}, 'result': {'records': [{'id': 'a'}, {'id': 'b'}]}}]
    rendered = json.loads(AcceptanceScenarioRunner.render_observed_dialogue(content, binding, events))
    assert rendered['request']['options'] == ['a', 'b']
    assert all(text not in rendered['question'] for text in ['old', 'hidden', 'secret'])
    absent = json.loads(AcceptanceScenarioRunner.render_observed_dialogue(content, binding, []))
    assert absent['request']['options'] == []  # Actual simulator must reject; never invent options.


@pytest.mark.parametrize('template', ['请选{{unknown}}', '记录{{observed_result}}'])
def test_incomplete_or_unknown_menu_template_is_rejected(tmp_path, template):
    source, request = bounded_source()
    source['interaction_contract']['variants'][0]['stages'][1]['reference_response'] = template
    with pytest.raises(ValueError, match='INTERACTION_REFERENCE_TEMPLATE'):
        compile_source(source, root=tmp_path, request=request, structural_preview=True)
