import copy

import pytest

from env_factory.sandbox_runtime import BusinessLifecycle, SandboxApplication, SandboxError, VirtualClock, validate_json_schema
from .test_business_lifecycle import setup


def application(tmp_path, dynamics=None):
    store, data, registry = setup(tmp_path, dynamics=dynamics)
    config = copy.deepcopy(registry.lifecycle.config)
    config['clock'] = {'origin_datetime': '2026-12-31T23:59:55+08:00'}
    registry.lifecycle = BusinessLifecycle(config, data, store, ['submit'])
    app = SandboxApplication(episode_store=store, tool_registry=registry, observation=lambda: {},
        reward=lambda: {}, user_turn=lambda messages: {}, reset_hook=data.reset)
    return app, store, data


def test_virtual_time_reads_are_pure_and_faults_retries_and_reset_keep_clock_semantics(tmp_path, monkeypatch):
    app, store, data = application(tmp_path, {'tick_seconds': 10, 'faults': [
        {'id': 'busy', 'tool': 'submit', 'call_index': 1, 'phase': 'before', 'status': 503},
        {'id': 'lost', 'tool': 'submit', 'call_index': 2, 'phase': 'after_commit', 'status': 504}],
        'events': [{'id': 'ready', 'at_tick': 2, 'changes': [
            {'table': 'orders', 'selector': {'id': 1}, 'values': {'status': 'draft'}}]}]})
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'clock-test')
    auth = {'Authorization': 'Bearer clock-test'}
    app.handle('POST', '/v1/reset', {'episode_id': 'first', 'seed': 17}, auth)
    def clock():
        before, rows = store.replay(), data._all()
        status, result, _ = app.handle('GET', '/v1/time')
        assert status == 200
        validate_json_schema(VirtualClock.RESPONSE_SCHEMA, result)
        assert store.replay() == before and data._all() == rows
        return result
    initial = clock()
    assert initial == {'origin_datetime': '2026-12-31T15:59:55+00:00',
        'virtual_datetime': '2026-12-31T15:59:55+00:00', 'tick': 0,
        'elapsed_seconds': 0, 'tick_seconds': 10, 'advances_on_tool_attempt': True}
    assert clock() == initial
    assert store.get_state('scenario_dynamics') is None
    store.set_state('business_grants', ['purchase'])
    headers = {'Idempotency-Key': 'order-1'}
    assert app.handle('POST', '/v1/tools/submit', {'id': 1}, headers)[0] == 503
    assert clock()['elapsed_seconds'] == 10
    assert app.handle('POST', '/v1/tools/submit', {'id': 1}, headers)[0] == 504
    assert clock()['virtual_datetime'] == '2026-12-31T16:00:15+00:00'
    assert app.handle('POST', '/v1/tools/submit', {'id': 1}, headers)[0] == 200
    assert clock()['tick'] == 2  # Committed retry cannot advance twice.
    assert app.handle('POST', '/v1/tools/submit', {'id': 1}, {'Idempotency-Key': 'new'})[0] == 409
    assert clock()['tick'] == 3  # Validated failed business attempt consumes a tick.
    assert app.handle('POST', '/v1/tools/submit', {'id': 'invalid'}, {})[0] == 400
    assert clock()['tick'] == 3
    assert store.get_state('scenario_dynamics')['fired'] == ['ready']
    from rl.environments.services.sandbox_service import snapshot_database, restore_database
    saved = snapshot_database(store.db_path)
    state = store.get_state('scenario_dynamics')
    store.set_state('scenario_dynamics', {**state, 'tick': 100})
    assert clock()['elapsed_seconds'] == 1000
    restore_database(store.db_path, saved)
    assert clock()['tick'] == 3
    app.handle('POST', '/v1/reset', {'episode_id': 'second', 'seed': 17}, auth)
    assert clock() == initial
    assert app.handle('GET', '/v1/time', headers={'X-Episode-ID': 'first'})[1]['tick'] == 3
    assert clock() == initial
    app.handle('POST', '/v1/reset', {'episode_id': 'first', 'seed': 17}, auth)
    assert clock() == initial


def test_configuring_origin_alone_does_not_enable_dynamics(tmp_path):
    app, store, _ = application(tmp_path)
    store.set_state('business_grants', ['purchase'])
    before = app.handle('GET', '/v1/time')[1]
    assert app.handle('POST', '/v1/tools/submit', {'id': 1}, {'Idempotency-Key': 'a'})[0] == 200
    assert app.handle('GET', '/v1/time')[1] == before
    assert before['advances_on_tool_attempt'] is False and before['tick_seconds'] == 0


@pytest.mark.parametrize('origin', ['2026-01-01', '2026-01-01T00:00:00', 'not-a-date', 42,
                                   '2026-01-01T00:00:00.123+00:00'])
def test_invalid_origins_do_not_fall_back_to_host_time(origin):
    with pytest.raises(SandboxError, match='origin_datetime'):
        VirtualClock({'origin_datetime': origin}, None)


def test_runtime_contract_requires_public_shared_time_endpoint():
    from env_factory.generation.contract_compiler import ContractCompiler
    from env_factory.contracts.runtime_contract import missing_system_endpoints
    interface = ContractCompiler._build_runtime_interface([], {})
    endpoint = next(e for e in interface['endpoints'] if e['name'] == 'time')
    assert endpoint['response_schema'] == VirtualClock.RESPONSE_SCHEMA
    assert not missing_system_endpoints(interface)
    interface['endpoints'].remove(endpoint)
    assert ('time', 'GET', '/v1/time') in missing_system_endpoints(interface)
