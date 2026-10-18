import json
from unittest.mock import Mock

from env_factory.core.extract import Extractor, conversation_of


def test_context_includes_opening_and_closing_reply_but_no_private_tool_data():
    trace = [{'kind': 'tool', 'result': {'secret': 'hidden'}},
             {'kind': 'chat', 'message': 'Change it?', 'reply': 'No, keep it.',
              'decision': False, 'proposal': {'internal': 'hidden'}}]
    assert conversation_of(trace, 'Please review the change.') == [
        {'role': 'user', 'content': 'Please review the change.'},
        {'role': 'assistant', 'content': 'Change it?'},
        {'role': 'user', 'content': 'No, keep it.'}]


def test_user_numbers_and_identifiers_cannot_be_credited_as_agent_report():
    client = Mock()
    client.chat.return_value = {'content': json.dumps({'amount': 200, 'receipt': 'R-12345'})}
    extractor = Extractor(client)
    schema = {'fields': {'amount': {'type': 'number', 'nullable': True},
                         'receipt': {'type': 'string', 'nullable': True}}}
    result = extractor.extract(schema, ['I will investigate.'], dialogue=[
        {'role': 'user', 'content': 'I received $200 with receipt R-12345.'},
        {'role': 'tool', 'content': 'PRIVATE_TOOL_MARKER'},
        {'role': 'assistant', 'content': 'I will investigate.'}])
    assert result == {'amount': None, 'receipt': None}
    prompt = client.chat.call_args.args[0][0]['content']
    assert 'R-12345' in prompt
    assert 'PRIVATE_TOOL_MARKER' not in prompt


def test_context_changes_invalidate_extraction_cache_and_empty_speech_stays_empty():
    client = Mock()
    client.chat.return_value = {'content': '{"status": "refused"}'}
    extractor = Extractor(client)
    schema = {'fields': {'status': {'type': 'string', 'enum': ['refused']}}}
    for text in ['No, keep it.', 'I have not decided.', 'I have not decided.']:
        extractor.extract(schema, ['No change was made.'], dialogue=[{'role': 'user', 'content': text}])
    assert client.chat.call_count == 2
    assert extractor.extract(schema, [], dialogue=[{'role': 'user', 'content': 'It is done.'}]) is None
    assert client.chat.call_count == 2
