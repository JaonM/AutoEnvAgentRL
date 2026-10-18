import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
from prepare_public_mix import signature, validate
from public_data import toolace, glaive, when2call, with_original_system
from sft_train import render_episode, strip_notices
import random


def example():
    return {'tools': [{'type': 'function', 'function': {'name': 'lookup', 'parameters': {
        'type': 'object', 'properties': {'id': {'type': 'string'}}, 'required': ['id']}}}],
        'messages': [{'role': 'user', 'content': 'Look up X'}, {'role': 'assistant', 'content': '',
          'tool_calls': [{'id': 'a', 'function': {'name': 'lookup', 'arguments': '{"id":"X"}'}}]},
          {'role': 'tool', 'tool_call_id': 'a', 'name': 'lookup', 'content': '{"ok":true}'},
          {'role': 'assistant', 'content': 'Found it.'}]}


def test_public_examples_require_real_tool_result_pairing():
    row = example()
    validate(row)
    row['messages'].pop(2)
    with pytest.raises(ValueError, match='missing tool results'):
        validate(row)


def test_invalid_arguments_are_rejected():
    import jsonschema
    row = example()
    row['messages'][1]['tool_calls'][0]['function']['arguments'] = '{}'
    with pytest.raises(jsonschema.ValidationError):
        validate(row)


def test_duplicates_ignore_call_ids_and_argument_json_spacing():
    row, other = example(), copy.deepcopy(example())
    other['messages'][1]['tool_calls'][0]['id'] = 'changed'
    other['messages'][2]['tool_call_id'] = 'changed'
    other['messages'][1]['tool_calls'][0]['function']['arguments'] = json.dumps({'id': 'X'})
    assert signature(row) == signature(other)


@pytest.mark.parametrize('source', ['toolace', 'glaive'])
def test_original_system_survives_conversion_and_rendering(tmp_path, source):
    from transformers import AutoTokenizer
    spec = {'name': 'lookup', 'description': 'Lookup', 'parameters': {
        'type': 'object', 'properties': {'id': {'type': 'string'}}, 'required': ['id']}}
    prefix = '  Preserve this exact policy.\n(System notice: part of ORIGINAL instructions)\n'
    if source == 'toolace':
        system = prefix + json.dumps([spec]) + '\nOriginal call format instructions.  '
        row = {'system': system, 'conversations': [
            {'from': 'user', 'value': 'Find X'}, {'from': 'assistant', 'value': '[lookup(id="X")]'},
            {'from': 'tool', 'value': '[{"results": "Found X"}]'},
            {'from': 'assistant', 'value': 'Found it.'}]}
        convert = toolace
    else:
        system = prefix + json.dumps(spec) + '\nOriginal call format instructions.  '
        row = {'system': system, 'chat': 'USER: Find X ASSISTANT: <functioncall> '
               '{"name":"lookup","arguments":{"id":"X"}} FUNCTION RESPONSE: Found X '
               'ASSISTANT: Found it.<|endoftext|>'}
        convert = glaive
    path = tmp_path / 'raw.json'
    path.write_text(json.dumps([row]))
    converted = convert(path, random.Random(0), 1)[0]
    assert converted['messages'][0] == {'role': 'system', 'content': system}
    validate(converted)
    tokenizer = AutoTokenizer.from_pretrained('models/Qwen3-0.6B', local_files_only=True)
    ids, mask = render_episode(tokenizer, converted, True)[0]
    text = tokenizer.decode(ids)
    assert text.count(system) == 1
    assert '# Tools' in text
    supervised = tokenizer.decode([token for token, train in zip(ids, mask) if train])
    assert 'Preserve this exact policy' not in supervised
    assert 'lookup' in supervised


def test_when2call_preserves_existing_system_without_duplication(tmp_path):
    system = '  Existing instructions\n'
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': 'Do this'},
                {'role': 'assistant', 'content': 'Which object?'}]
    path = tmp_path / 'raw.jsonl'
    path.write_text(json.dumps({'system': system, 'messages': messages, 'tools': []}) + '\n')
    assert when2call(path, random.Random(0), 4)[0]['messages'] == messages
    assert with_original_system({}, messages) == messages


def test_notice_stripping_preserves_system_instructions():
    text = '(System notice: keep this policy)'
    result = strip_notices([{'role': 'system', 'content': text},
                            {'role': 'user', 'content': 'Hello ' + text}])
    assert result[0]['content'] == text
    assert result[1]['content'] == 'Hello'
