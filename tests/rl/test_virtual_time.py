import json
import threading
from urllib.request import build_opener, ProxyHandler
from wsgiref.simple_server import make_server

import pytest

from env_factory.sandbox_runtime import ContractToolRegistry, EpisodeStore, SandboxApplication
from rl.environments.local import SandboxEpisode
from rl.environments.remote import RemoteSandboxEpisode
from rl.environments.virtual_time import clock_tool, format_current_time
from rl.errors import InfrastructureError


@pytest.mark.parametrize('runtime', [SandboxEpisode, RemoteSandboxEpisode])
def test_local_and_http_reset_and_native_tool_use_sandbox_time(tmp_path, monkeypatch, runtime):
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'clock-rl')
    app = SandboxApplication(episode_store=EpisodeStore(tmp_path / 'episode.sqlite3'),
        tool_registry=ContractToolRegistry([], {}), observation=lambda: {}, reward=lambda: {},
        user_turn=lambda messages: {'user_query': 'done', 'should_end': True})
    env = runtime.__new__(runtime)
    env.task, env.headers, env.app = {'task': '读取时间'}, {'Authorization': 'Bearer clock-rl'}, app
    server = make_server('127.0.0.1', 0, app) if runtime is RemoteSandboxEpisode else None
    if server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        env.url, env.timeout, env.opener = f'http://127.0.0.1:{server.server_port}', 5, build_opener(ProxyHandler({}))
    try:
        env.reset(17)
        assert '2026-01-01 08:00:00' in env.messages[0]['content']
        assert env.names == set()  # System time is not a business dependency.
        replay = app.episode_store.replay()
        def action(arguments):
            return {'role': 'assistant', 'tool_calls': [{'type': 'function', 'function': {
                'name': env.time_tool['function']['name'], 'arguments': arguments}}]}
        for _ in range(2):
            assert env.step(action({})) == (0., False)
            result = json.loads(env.messages[-1]['content'])
            assert result['status'] == 200
            assert result['tool_result']['virtual_datetime'] == '2026-01-01T00:00:00+00:00'
        assert app.episode_store.replay() == replay
        count = len(env.trace)
        env.step(action({'advance': 10}))
        assert len(env.trace) == count and json.loads(env.messages[-1]['content'])['status'] == 400
        assert app.episode_store.replay() == replay
        if runtime is SandboxEpisode:
            saved = env.snapshot()
            env.time_tool = None
            env.restore(saved)
            assert env.time_tool == saved['time_tool']
    finally:
        if server:
            server.shutdown()
            worker.join(timeout=5)
            server.server_close()


def test_time_tool_names_do_not_shadow_business_or_interaction_tools():
    tools = [{'function': {'name': name}} for name in ['get_current_time', '_get_current_time']]
    assert clock_tool(tools)['function']['name'] == '__get_current_time'
    assert format_current_time({'virtual_datetime': '2026-12-31T16:00:05+00:00'}) == '2027-01-01 00:00:05'
    for bad in [None, 5, 'tomorrow', '2026-01-01T12:00:00']:
        with pytest.raises(InfrastructureError):
            format_current_time({'virtual_datetime': bad})
