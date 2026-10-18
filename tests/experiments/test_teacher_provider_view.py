import copy
import importlib.util
from pathlib import Path

from env_factory.certify.policy_gate import provider_messages


def module(name):
    path = Path(__file__).resolve().parents[2] / 'experiments' / (name + '.py')
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_provider_reasoning_survives_tool_and_text_turns_without_training_contamination():
    collector = module('collect_teacher')
    messages = [{'role': 'system', 'content': 'Amounts use crown cents.'},
                {'role': 'user', 'content': 'Please look up my balance.'},
                {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 't', 'type': 'function',
                    'function': {'name': 'balance', 'arguments': '{}'}}]},
                {'role': 'tool', 'tool_call_id': 't', 'content': '12'},
                {'role': 'assistant', 'content': 'Your balance is 12 crown cents.'}]
    original = copy.deepcopy(messages)
    view = provider_messages(collector.teacher_view(messages, True, english=True), {2: 'tool reasoning', 4: ''})
    assert view[2]['reasoning_content'] == 'tool reasoning'
    assert view[4]['reasoning_content'] == ''
    assert all('reasoning_content' not in m for m in messages)
    assert messages == original
    assert view[2]['tool_calls'] == original[2]['tool_calls']
    assert 'say amounts in dollars' not in view[0]['content']


def test_explicit_minor_units_are_not_classified_as_internal_leaks():
    collector, reviewer = module('collect_teacher'), module('teacher_review')
    messages = [{'role': 'assistant', 'content': 'The fee is 25 crown cents.'}]
    assert collector.leaks(messages) == 0
    assert 'leak' not in reviewer.flags({'messages': messages, 'language': 'en'})
    assert collector.leaks([{'role': 'assistant', 'content': 'The internal amount_cents is 25.'}]) == 1
