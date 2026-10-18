import copy

import pytest

from env_factory.production.teacher import ConstrainedTeacher, TeacherProtocolError


class Client:
    model = 'scripted'
    def __init__(self, replies):
        self.replies, self.views = iter(replies), []
    def chat(self, messages, tools=None, **parameters):
        self.views.append(copy.deepcopy(messages))
        return next(self.replies)


def test_bad_draft_is_never_returned_for_execution_or_added_to_student_history():
    call = {'id': 'c', 'function': {'name': 'lookup', 'arguments': '{}'}}
    bad = {'content': 'I will check.', 'tool_calls': [call]}
    good = {'content': '', 'tool_calls': [call], 'reasoning_content': 'accepted reasoning'}
    client = Client([bad, good])
    teacher = ConstrainedTeacher(client, {'allow_text_with_tool_calls': False}, attempts=2)
    messages = [{'role': 'system', 'content': 'Public policy'}, {'role': 'user', 'content': 'Check my account.'}]
    before = copy.deepcopy(messages)
    assert teacher.chat(messages) == good
    assert messages == before and all(m['role'] != 'assistant' for m in client.views[1])
    assert teacher.report()['rejected_drafts'] == 1
    assert teacher.report()['rejections'][0]['reply'] == bad


def test_exhausted_draft_budget_is_generation_failure_without_fake_success():
    client = Client([{'content': 'Check', 'tool_calls': [{}]}] * 3)
    teacher = ConstrainedTeacher(client, {'allow_text_with_tool_calls': False}, attempts=2)
    with pytest.raises(TeacherProtocolError):
        teacher.chat([{'role': 'system', 'content': 'Public policy'}])
    assert len(client.views) == 2 and teacher.report()['rejected_drafts'] == 2


def test_unconstrained_teacher_passes_through_without_extra_sampling():
    reply = {'content': 'Allowed', 'tool_calls': [{}, {}]}
    teacher = ConstrainedTeacher(Client([reply]), {})
    assert teacher.chat([{'role': 'system', 'content': 'Policy'}]) == reply
    assert teacher.report()['drafts'] == 1
