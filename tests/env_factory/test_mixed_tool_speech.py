"""Mixed assistant turns deliver text and calls without creating retroactive consent."""
import pytest

from env_factory.certify import policy_gate
from rl.environments.sandbox import SandboxEpisode


def message():
    return {'role': 'assistant', 'content': 'The address will be checked.', 'tool_calls': [
        {'id': 'c1', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}},
        {'id': 'c2', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}]}


def test_training_adapter_delivers_mixed_speech_after_tool_results():
    env = SandboxEpisode.__new__(SandboxEpisode)
    env.terminated, env.call_ids, env.messages = False, set(), []
    env.user_mode, env.result_style = 'natural', 'envelope'
    actions = []
    env._dispatch = lambda name, params: actions.append(('tool', name)) or {'ok': True, 'result': {}}
    env._step = lambda action: actions.append((action['type'], action.get('message'))) or {
        'ok': True, 'result': {'reply': 'Thanks.'}}
    env._after_chat = lambda result: None
    env.step(message())
    assert [a[0] for a in actions] == ['tool', 'tool', 'chat']
    assert actions[-1][1] == message()['content']
    assert [m['role'] for m in env.messages] == ['assistant', 'tool', 'tool', 'user']


def test_policy_player_delivers_mixed_speech_and_retains_batch(monkeypatch):
    actions = []
    def post(opener, url, body):
        if url.endswith('/reset'):
            return {'episode': 'e', 'instructions': {'tools': [{'name': 'lookup'}]}}
        if url.endswith('/step'):
            actions.append(body['action'])
            return {'ok': True, 'result': {'reply': 'Thanks.'}}
        if url.endswith('/reward'):
            return {'success': True, 'reward': 1}
        return {}
    monkeypatch.setattr(policy_gate, '_post', post)
    monkeypatch.setattr(policy_gate.agent_view, 'tools', lambda *a, **k: [])
    monkeypatch.setattr(policy_gate.agent_view, 'messages', lambda *a, **k: [])
    monkeypatch.setattr(policy_gate.agent_view, 'speaks', lambda instructions: True)
    monkeypatch.setattr(policy_gate.agent_view, 'language', lambda instructions: 'en')
    class Client:
        def chat(self, *args):
            return message()
    result = policy_gate.play('http://test', Client(), 'original', 'simple', max_turns=1)
    assert [a['type'] for a in actions] == ['tool', 'tool', 'chat']
    assert actions[-1]['message'] == message()['content']
    assert len(result['messages'][0]['tool_calls']) == 2
    assert result['messages'][-1] == {'role': 'user', 'content': 'Thanks.'}


@pytest.mark.parametrize('mixed', [False, True])
def test_policy_player_preserves_final_user_reply_when_closed(monkeypatch, mixed):
    def post(opener, url, body):
        if url.endswith('/reset'):
            return {'episode': 'e', 'instructions': {'tools': [{'name': 'lookup'}]}}
        if url.endswith('/step'):
            return {'ok': True, 'result': {'reply': 'Yes, that was the correct identifier.', 'ended': 'closed'}}
        if url.endswith('/reward'):
            return {'success': True, 'reward': 1}
        return {}
    monkeypatch.setattr(policy_gate, '_post', post)
    monkeypatch.setattr(policy_gate.agent_view, 'tools', lambda *a, **k: [])
    monkeypatch.setattr(policy_gate.agent_view, 'messages', lambda *a, **k: [])
    monkeypatch.setattr(policy_gate.agent_view, 'speaks', lambda _: True)
    monkeypatch.setattr(policy_gate.agent_view, 'language', lambda _: 'en')
    class Client:
        def chat(self, *args):
            return message() if mixed else {'content': 'The result is ready.'}
    result = policy_gate.play('http://test', Client(), 'original', 'simple', max_turns=1)
    assert result['messages'][-1] == {'role': 'user', 'content': 'Yes, that was the correct identifier.'}
