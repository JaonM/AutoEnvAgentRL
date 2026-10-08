import copy
import json
import threading
import uuid
from urllib.request import ProxyHandler, build_opener

import pytest

from env_factory.generation.difficulty_profiles import requirements, transaction_covered
from env_factory.sandbox_runtime import ApprovalTerms, SandboxApplication
from rl.environments.local import SandboxEpisode
from rl.environments.remote import RemoteSandboxEpisode
from rl.environments.services.sandbox_service import SessionService, make_server
from rl.environments.tool_transport import retry_tool
from tests.runtime.test_transaction_witness import runtime


def action(name, arguments, call_id=None):
    call = {'type': 'function', 'function': {'name': name, 'arguments': arguments}}
    if call_id is not None:
        call['id'] = call_id
    return {'role': 'assistant', 'tool_calls': [call]}


@pytest.fixture(params=[(cls, phase) for cls in [SandboxEpisode, RemoteSandboxEpisode]
                        for phase in ['before', 'after_commit']])
def episode(request, tmp_path, monkeypatch):
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'retry-test')
    cls, phase = request.param
    applications = []
    def factory(db_path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        store, data, registry, observe = runtime(db_path.parent, fault_phase=phase)
        approved = copy.deepcopy(store.get_state(ApprovalTerms.KEY))
        def reset(_):
            data.reset()
            store.set_state('business_grants', list(registry.lifecycle.rules))
            store.set_state(ApprovalTerms.KEY, approved)
        app = SandboxApplication(episode_store=store, tool_registry=registry, observation=lambda: {},
            reward=lambda: {'reward': float(transaction_covered(requirements('complex_write_transactions'), observe()))},
            business_snapshot=data._all, reset_hook=reset,
            user_turn=lambda messages: {'user_query': '已完成', 'should_end': True})
        applications.append((app, data, observe))
        return app
    env = cls.__new__(cls)
    env.task = {'task': '完成授权的跨表登记，故障后补偿库存，再重试登记。',
                'business_lifecycle': {'dynamics': {'faults': [
                    {'tool': 'place', 'phase': phase, 'status': 503 if phase == 'before' else 504}]}}}
    env.headers = {'Authorization': 'Bearer retry-test', 'X-RL-Session': uuid.uuid4().hex}
    server = None
    if cls is SandboxEpisode:
        env.app = factory(tmp_path / 'episode.sqlite3')
    else:
        service = SessionService(factory, tmp_path / 'sessions', 'retry-test-identity', 'retry-test')
        server = make_server(service, '127.0.0.1', 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env.identity = 'retry-test-identity'
        env.url, env.timeout, env.opener = f'http://127.0.0.1:{server.server_port}', 5, build_opener(ProxyHandler({}))
    try:
        env.reset(17)
        yield env, applications
    finally:
        if server:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()


def call(env, name, arguments, call_id=None):
    env.step(action(name, arguments, call_id))
    return json.loads(env.messages[-1]['content'])


def test_actual_failure_snapshot_and_repeated_native_retry_only_write_once(episode):
    env, applications = episode
    assert 'retry_tool_call' not in env.names
    reserve = call(env, 'reserve', {'value': 1}, 'reserve-native')
    assert reserve['status'] == 200
    lost = call(env, 'place', {'value': 1}, 'place-native')
    phase = env.task['business_lifecycle']['dynamics']['faults'][0]['phase']
    assert lost['status'] == (503 if phase == 'before' else 504)
    handle = lost['retry_request_id']
    saved = env.snapshot()
    env.retry_requests.clear()
    env.retry_tool = None
    env.restore(saved)
    assert env.retry_requests[handle]['idempotency_key'] == 'place-native'
    call(env, 'release', {'value': 0}, 'release-native')
    name = env.retry_tool['function']['name']
    for call_id in ['retry-native-1', 'retry-native-2']:
        retried = call(env, name, {'request_id': handle}, call_id)
        assert retried == {'status': 200, 'tool_result': {'count': 2},
                           'retry_request_id': handle, 'retried_tool': 'place'}
    env.step('已完成登记，恢复了库存。')
    assert env.finish()['final_reward'] == 1
    app, data, observe = applications[-1]
    assert observe()['idempotent_replay_tools'] == ['place']
    events = app.episode_store.replay()['events']
    assert len([e for e in events if e['event'] == 'tool_call' and e['payload']['tool_name'] == 'place']) == 1
    assert len([e for e in events if e['event'] == 'tool_call_replay' and e['payload']['tool_name'] == 'place']) == (1 if phase == 'before' else 2)
    assert data.table('orders') == [{'id': 1, 'value': 1}, {'id': 2, 'value': 1}]
    assert 'Idempotency-Key' not in env.headers


def test_retry_cannot_change_arguments_or_access_a_previous_episode(episode):
    env, _ = episode
    response = call(env, 'reserve', {'value': 1}, 'original')
    handle, retry_name = response['retry_request_id'], env.retry_tool['function']['name']
    count = len(env.trace)
    for arguments in [{'request_id': handle, 'arguments': {'value': 99}},
                      {'request_id': []}, {}, {'request_id': ''}]:
        assert call(env, retry_name, arguments)['status'] == 400
    assert call(env, retry_name, {'request_id': 'unknown'})['status'] == 404
    assert len(env.trace) == count
    env.reset(18)
    call(env, 'reserve', {'value': 1}, 'original')
    count = len(env.trace)
    assert call(env, retry_name, {'request_id': handle})['status'] == 404
    assert len(env.trace) == count


def test_native_ids_still_unique_and_time_calls_are_not_retryable(episode):
    env, _ = episode
    call(env, 'reserve', {'value': 1}, 'same-native-id')
    count = len(env.trace)
    result = call(env, env.retry_tool['function']['name'],
                  {'request_id': next(iter(env.retry_requests))}, 'same-native-id')
    assert 'unique' in result['protocol_error'] and len(env.trace) == count
    assert 'retry_request_id' not in call(env, env.time_tool['function']['name'], {})


def test_retry_tool_does_not_shadow_authored_names():
    assert retry_tool([{'function': {'name': name}} for name in ['retry_tool_call', '_retry_tool_call']])['function']['name'] == '__retry_tool_call'


def test_budget_admission_uses_same_local_and_http_task_contract(episode):
    from rl.rollout.budgets import admit, execution_budget, TaskBudgetError
    from tests.rl.test_task_budgets import budget, CountingPolicy, config, SECRET
    env, _ = episode
    value = budget()
    env.task['task_spec'] = {'difficulty_requirements': {'profile': 'complex_write_transactions'},
                             'execution_budget': {**value['contract'], 'profile': 'complex_write_transactions'}}
    env.task['artifacts'] = {'generation_pipeline': {'verification': {'policy_budget_references': value['references']}}}
    before = copy.deepcopy(env.messages)
    report = admit(CountingPolicy(), env.messages, {'tools': env.tools}, config(), execution_budget(env))
    assert report['profile'] == 'complex_write_transactions'
    assert report['reference_serial_actions'] == 3
    assert SECRET not in json.dumps([report, env.messages, env.trace])
    with pytest.raises(TaskBudgetError, match='CONTEXT_BUDGET_INCOMPATIBLE'):
        admit(CountingPolicy(), env.messages, {'tools': env.tools}, {**config(), 'max_context': 10}, execution_budget(env))
    assert env.messages == before
