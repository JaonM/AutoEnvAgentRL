import json
import pytest

from rl.environments.local import SandboxEpisode
from rl.environments.remote import RemoteSandboxEpisode
from rl.environments.user_interaction import interaction_tool


@pytest.mark.parametrize('runtime', [SandboxEpisode, RemoteSandboxEpisode])
def test_native_question_uses_existing_endpoints_and_hides_internal_state(runtime):
    env = runtime.__new__(runtime)
    env.reward, env.terminated = None, False
    env.messages, env.conversation, env.names = [], [], {'lookup'}
    env.user_interaction_tool = interaction_tool(
        {'materials': [{'name': 'user_interaction_format'}]},
        [{'function': {'name': 'lookup'}}])
    calls = []
    def request(method, path, body=None):
        calls.append((path, body))
        return 200, ({'user_query': 'next month', 'should_end': False,
                      'fsm_script_id': 'private-script', 'hidden': 'secret'}
                     if path == '/v1/user_simulator' else {})
    env.request = request
    action = {'role': 'assistant', 'tool_calls': [{'type': 'function', 'id': 'q1',
        'function': {'name': 'request_user_input', 'arguments': {
            'question': 'Which month?', 'business_tool': 'lookup', 'requested_fields': ['month']}}}]}
    assert env.step(action) == (0, False)
    assert [path for path, _ in calls] == ['/v1/agent_response', '/v1/user_simulator']
    assert json.loads(calls[0][1]['content']) == {
        'question': 'Which month?', 'request': {
            'intent': 'request_input', 'tool': 'lookup', 'arguments': ['month']}}
    visible = json.loads(env.messages[-1]['content'])['tool_result']
    assert visible == {'user_query': 'next month', 'should_end': False}
    assert env.reward is None  # reward stays episode-end only
    calls.clear()
    action['tool_calls'][0]['id'] = 'q2'
    action['tool_calls'][0]['function']['arguments']['requested_fields'] = {'month': 'guessed'}
    env.step(action)
    assert calls == []
    assert json.loads(env.messages[-1]['content'])['status'] == 400


def test_native_tool_only_for_declared_public_protocol_and_no_business_name_shadowing():
    tools = [{'function': {'name': 'request_user_input'}}]
    assert interaction_tool({}, tools) is None
    native = interaction_tool({'materials': [{'name': 'user_interaction_format'}]}, tools)
    assert native['function']['name'] == '_request_user_input'
    assert native['function']['parameters']['properties']['business_tool']['enum'] == ['request_user_input']


def test_public_interface_translation_preserves_task_and_original_materials():
    from rl.environments.user_interaction import policy_public_input
    public = {'initial_user_message': 'Find my booking', 'materials': [
        {'name': 'user_interaction_format', 'content': 'old JSON interface'},
        {'name': 'business_rules', 'content': 'Must get permission'}]}
    native = interaction_tool(public, [{'function': {'name': 'lookup'}}])
    translated = policy_public_input(public, native)
    assert translated['initial_user_message'] == public['initial_user_message']
    assert translated['materials'][1] == public['materials'][1]
    assert 'call the request_user_input tool' in translated['materials'][0]['content']
    assert public['materials'][0]['content'] == 'old JSON interface'


@pytest.mark.parametrize('runtime', [SandboxEpisode, RemoteSandboxEpisode])
def test_native_tool_survives_snapshot_restore(tmp_path, runtime):
    import sqlite3
    from types import SimpleNamespace
    env = runtime.__new__(runtime)
    path = tmp_path / 'episode.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE example (value TEXT)')
    env.app = SimpleNamespace(episode_store=SimpleNamespace(db_path=path))
    env.identity = 'sandbox-identity'
    env._call = lambda *args, **kwargs: (200, {'database': 'opaque-remote-snapshot'})
    env.task = {'public_input': {'materials': [{'name': 'user_interaction_format'}]}}
    env.tools = [{'function': {'name': 'lookup'}}]
    env.names = {'lookup'}
    native = interaction_tool(env.task['public_input'], env.tools)
    env.user_interaction_tool = native
    env.tools.append(native)
    env.messages, env.conversation, env.trace = [], [], []
    env.reward, env.terminated = None, False
    state = env.snapshot()
    del env.user_interaction_tool
    env.restore(state)
    assert env.user_interaction_tool == native
    # Pre-adapter snapshots must not retain another episode's native tool.
    state.pop('user_interaction_tool')
    env.user_interaction_tool = None
    env.restore(state)
    assert env.user_interaction_tool == native
    state['tools'] = [state['tools'][0]]
    env.restore(state)
    assert env.user_interaction_tool is None
