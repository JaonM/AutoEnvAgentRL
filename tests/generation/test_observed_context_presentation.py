"""An internally read catalog must be presented before a separate topic choice."""
import copy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import AcceptanceScenarioRunner, interaction_request_matches, observed_interaction_options
from tests.generation.test_bounded_corrections import bounded_source


def context_source():
    source, request = bounded_source()
    for variant in source['interaction_contract']['variants']:
        stage = variant['stages'][1]
        stage.pop('options_from')  # Fixed choices still require an observed business list.
        stage['context_from'] = {'tool': 'lookup_zone', 'path': '$.records'}
        stage['reference_response'] = '实际库区清单：{{observed_context}}。请确认最终区域。'
    return source, request


def test_context_presentation_compiles_and_executes_with_fixed_choices(tmp_path):
    source, request = context_source()
    artifact = compile_source(source, root=tmp_path, request=request)
    assert artifact['generation_pipeline']['verification']['passed']
    assert any(item['name'] == 'user_interaction_context' for item in artifact['public_input']['materials'])
    assert any(step.get('observed_context_from') for scenario in artifact['acceptance_contract']['executable_scenarios']
               for step in scenario['steps'])


@pytest.mark.parametrize('defect', ['missing', 'partial', 'stale', 'wrong_type'])
def test_claiming_to_have_read_records_cannot_replace_presenting_them(defect):
    stage = {'kind': 'option_selection', 'before_tool': 'research', 'bind_to': {'argument': 'topic'},
        'options': ['pilling', 'static'], 'context_from': {'tool': 'eligible', 'path': '$.records'}}
    records = [{'id': 'A', 'label': 'sweater one'}, {'id': 'B', 'label': 'sweater two'}]
    events = [{'event': 'tool_call', 'payload': {'tool_name': 'eligible'}, 'result': {'records': records}}]
    resolved = observed_interaction_options(stage, events)
    content = {'question': 'I have read the list. Which topic?', 'request': {
        'intent': 'request_input', 'tool': 'research', 'arguments': ['topic'], 'options': stage['options']}}
    if defect == 'partial':
        content['request']['context'] = records[:1]
    elif defect == 'stale':
        content['request']['context'] = [{'id': 'old', 'label': 'removed'}]
    elif defect == 'wrong_type':
        content['request']['context'] = json.dumps(records)
    assert not interaction_request_matches(resolved, json.dumps(content), 'structured_v1')
    content['request']['context'] = copy.deepcopy(records)
    assert interaction_request_matches(resolved, json.dumps(content), 'structured_v1')
    assert observed_interaction_options(stage, []) is None


def test_reference_context_uses_actual_observed_records_including_empty_result():
    binding = {'tool': 'eligible', 'path': '$.records'}
    content = json.dumps({'question': '清单 {{observed_context}}；请选择主题。',
        'request': {'intent': 'request_input', 'tool': 'research', 'arguments': ['topic'], 'options': ['pilling', 'static']}})
    for rows in ([], [{'id': 'new', 'label': 'newly eligible'}]):
        events = [{'event': 'tool_call', 'payload': {'tool_name': 'eligible'}, 'result': {'records': rows}}]
        result = json.loads(AcceptanceScenarioRunner.render_observed_context(content, binding, events))
        assert result['request']['context'] == rows
        assert result['request']['options'] == ['pilling', 'static']
        assert json.dumps(rows, ensure_ascii=False) in result['question']


def test_combined_reference_templates_do_not_reinterpret_returned_strings():
    binding = {'tool': 'eligible', 'path': '$.records[*].id'}
    context = {'tool': 'eligible', 'path': '$.records'}
    rows = [{'id': '{{observed_context}}', 'label': '{{observed_options}}'}, {'id': 'B', 'label': 'B'}]
    content = json.dumps({'question': '{{observed_result}} {{observed_options}} {{observed_context}}',
        'request': {'intent': 'request_input', 'tool': 'research', 'arguments': ['topic']}})
    events = [{'event': 'tool_call', 'payload': {'tool_name': 'eligible'}, 'result': {'records': rows}}]
    result = json.loads(AcceptanceScenarioRunner.render_observed_dialogue(content, binding, events, context_binding=context))
    assert result['request']['context'] == rows
    assert result['request']['options'] == ['{{observed_context}}', 'B']
    assert result['question'].count('{{observed_context}}') == 3
    assert result['question'].count('{{observed_options}}') == 2


@pytest.mark.parametrize('defect', ['unobserved_producer', 'missing_template'])
def test_context_contract_rejects_unexecutable_presentation(tmp_path, defect):
    source, request = context_source()
    stage = source['interaction_contract']['variants'][0]['stages'][1]
    if defect == 'unobserved_producer':
        stage['context_from']['tool'] = 'read_temperature'
    else:
        stage['reference_response'] = '我已读完清单。请选择区域。'
    with pytest.raises(ValueError, match='context_from'):
        compile_source(source, root=tmp_path, request=request, structural_preview=True)
