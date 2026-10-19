"""Private intent must reach the simulator/auditor, never the student view."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

from env_factory.core.disclosure import Auditor
from env_factory.serving.server import Service
from env_factory.serving.user_sim import UserSimulator


def test_audit_receives_opening_and_refusal_and_keeps_quote_validation():
    client = Mock()
    client.chat.return_value = {'content': json.dumps({'faults': [
        {'kind': 'contradicted', 'fact': 'approve_change', 'quote': 'Yes, change it'},
        {'kind': 'invented', 'quote': 'a quote that never occurred'},
    ]})}
    auditor = Auditor(client)
    fake = SimpleNamespace(request={'user_request': 'We are three passengers.',
                                    'user_script': {'goals': ['change booking']}},
                           auditor=lambda: auditor)
    record = {'facts': {'approve_change': False}, 'episode': SimpleNamespace(trace=[
        {'kind': 'chat', 'message': 'Change it?', 'reply': 'Yes, change it'}])}
    result = Service.audit_simulator(fake, record)
    prompt = client.chat.call_args.args[0][0]['content']
    assert 'We are three passengers.' in prompt
    assert '"approve_change": false' in prompt
    assert not result['valid']
    assert len(result['simulator_audit']['faults']) == 1
    # Opening is context, not a fabricated quotation from a later user reply.
    client.chat.return_value = {'content': json.dumps({'faults': [
        {'kind': 'invented', 'quote': 'We are three passengers.'}]})}
    assert Auditor(client).audit(record['facts'], {}, record['episode'].trace,
                                 opening=fake.request['user_request'])['valid']


def test_auditor_cache_is_scoped_to_opening_and_approval_facts():
    client = Mock()
    client.chat.return_value = {'content': '{"faults": []}'}
    auditor = Auditor(client)
    trace = [{'kind': 'chat', 'message': 'Confirm?', 'reply': 'No thanks'}]
    for opening, approval in [('A', False), ('B', False), ('B', True), ('B', True)]:
        auditor.audit({'approve_change': approval}, {}, trace, opening=opening)
    assert client.chat.call_count == 3


def test_scripted_questions_and_change_of_choice_are_delivered_in_both_languages():
    script = {'inquisitive': ['What is the fee?'],
              'change_of_mind': {'from': 'first option', 'to': 'second option', 'after': 'a fee applies'}}
    for language in ('en', 'zh'):
        user = UserSimulator(Mock(), {'id': 'test', 'language': language, 'user_request': 'Help me',
                                     'user_script': script},
                             {'approve_first': False, 'approve_second': True},
                             {'approve_first', 'approve_second'}, plain=True)
        prompt = user.messages[0]['content']
        for value in ['What is the fee?', 'first option', 'second option', 'a fee applies']:
            assert value in prompt
        assert ('first: no' in prompt and 'second: yes' in prompt) if language == 'en' else (
            'first: 不同意' in prompt and 'second: 同意' in prompt)


def test_unavailable_auditor_is_unknown_and_can_be_retried():
    from env_factory.core.llm import ProviderError
    calls = []
    def audit(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise ProviderError('temporary provider failure')
        return {'valid': True, 'faults': []}
    fake = SimpleNamespace(request={'user_request': 'x'}, auditor=lambda: SimpleNamespace(audit=audit))
    record = {'facts': {}, 'episode': SimpleNamespace(trace=[])}
    unknown = Service.audit_simulator(fake, record)
    assert unknown['valid'] is False and unknown['audit_status'] == 'unknown'
    retried = Service.audit_simulator(fake, record)
    assert retried['valid'] is True and retried['audit_status'] == 'valid'
    assert len(calls) == 2
