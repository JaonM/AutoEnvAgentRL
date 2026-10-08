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


def test_native_proposal_and_observed_context_reach_real_user_consent_and_write(tmp_path, monkeypatch):
    from env_factory.sandbox_runtime import ApprovalTerms, ContractToolRegistry, ContractUserSimulator, SandboxApplication
    from tests.runtime.test_proposal_approval import proposal_booking
    store, data, stage, lifecycle, book, script, _ = proposal_booking(tmp_path)
    stage['context_from'] = {'tool': 'quote', 'path': '$.records'}
    stage['requires_tools'] = ['quote']
    user = ContractUserSimulator(store, profiles=[{'profile_id': 'p'}], scripts=[script])
    user.completion_check = lambda: False
    quote = {'type': 'function', 'function': {'name': 'quote', 'parameters': {
        'type': 'object', 'properties': {}, 'additionalProperties': False}}}
    records = [{'id': 1, 'fee': 950}]
    registry = ContractToolRegistry([quote, book], {'quote': lambda args: {'records': records},
        'book': lambda args: {'updated': data.update('bookings', {'id': args['id']}, {'daily_fee': args['fee']})}},
        event_recorder=store.event, lifecycle=lifecycle)
    def reset(_):
        data.reset()
        user.reset()
    app = SandboxApplication(episode_store=store, tool_registry=registry, observation=lambda: {},
        reward=lambda: {}, user_turn=user.turn, reset_hook=reset)
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'native-proposal-test')
    env = SandboxEpisode.__new__(SandboxEpisode)
    env.app, env.headers = app, {'Authorization': 'Bearer native-proposal-test'}
    env.task = {'task': '请确认报价后预订。', 'public_input': {'materials': [
        {'name': 'user_interaction_format', 'content': 'structured_v1'}]}}
    env.reset(17)
    def call(name, params):
        env.step({'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {
            'name': name, 'arguments': params}}]})
        return json.loads(env.messages[-1]['content'])
    assert call('quote', {})['status'] == 200
    consent = {'question': '请确认当前950报价。', 'business_tool': 'book', 'requested_fields': [],
               'intent': 'confirm_execution', 'context': records, 'proposal': {'id': 1, 'fee': 950}}
    # A stale or partial context must still fail at the actual user protocol.
    call(env.user_interaction_tool['function']['name'], {**consent, 'context': []})
    assert store.get_state('business_grants', []) == []
    result = call(env.user_interaction_tool['function']['name'], consent)
    assert result['status'] == 200
    assert '950' in result['tool_result']['user_query']
    assert store.get_state(ApprovalTerms.KEY)['book']['arguments'] == {'id': 1, 'fee': 950}
    assert call('book', {'id': 1, 'fee': 1000})['status'] == 403
    assert data.table('bookings')[0]['daily_fee'] == 0
    assert call('book', {'id': 1, 'fee': 950})['status'] == 200
    assert data.table('bookings')[0]['daily_fee'] == 950


@pytest.mark.parametrize('value', [None, [], {}, {'$ref': 'hidden'}, float('nan')])
def test_native_proposals_require_actual_scalar_values(value):
    from env_factory.sandbox_runtime import SandboxError
    from rl.environments.user_interaction import interaction_content
    native = interaction_tool({'materials': [{'name': 'user_interaction_format'}]}, [{'function': {'name': 'book'}}])
    with pytest.raises(SandboxError):
        interaction_content({'question': 'Approve?', 'business_tool': 'book', 'requested_fields': [],
                             'proposal': {'price': value}}, native['function']['parameters'])
