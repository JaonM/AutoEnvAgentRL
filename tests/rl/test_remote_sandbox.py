import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace
import pytest

from rl.kubernetes import resources
from rl.remote_environment import RemoteSandboxEpisode
from rl.sandbox_service import SessionService, artifact_identity, make_server


class App:
    def __init__(self, db_path):
        self.episode_store = SimpleNamespace(db_path=db_path)
        with sqlite3.connect(db_path) as db:
            db.execute('pragma journal_mode=WAL')
            db.execute('create table if not exists state (value integer)')
    def handle(self, method, path, body, headers):
        with sqlite3.connect(self.episode_store.db_path) as db:
            if path == '/v1/reset':
                db.execute('delete from state')
                db.execute('insert into state values (0)')
                result = {'reward_mode': body['reward_mode']}
            elif path == '/v1/tools':
                result = {'tools': [{'function': {'name': 'increment'}}]}
            elif path == '/v1/tools/increment':
                db.execute('update state set value=value+1')
                result = {'ok': True}
            elif path == '/v1/reward':
                result = {'reward': db.execute('select value from state').fetchone()[0] / 2}
            elif path == '/v1/replay':
                result = {'events': []}
            else:
                result = {}
        return 200, result, {}


@pytest.fixture
def remote(tmp_path, monkeypatch):
    root = tmp_path / 'sandbox'
    root.mkdir()
    (root / 'task.json').write_text(json.dumps({'task': 'test'}))
    # If the remote adapter imports local task code, this test must fail.
    (root / 'app.py').write_text('raise AssertionError("must run remotely")')
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
    (root / 'status.json').write_text(json.dumps({'training_ready': True, 'artifact_hashes': hashes}))
    (root / 'pipeline_result.json').write_text('{"training_ready":true}')
    identity = artifact_identity(hashes)
    service = SessionService(App, tmp_path / 'sessions', identity, 'test-secret')
    server = make_server(service, '127.0.0.1', 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    manifest = tmp_path / 'services.json'
    manifest.write_text(json.dumps({'services': {identity: {'url': f'http://127.0.0.1:{server.server_port}'}},
                                    'startup_timeout': .2, 'request_timeout': 1}))
    monkeypatch.setenv('RL_SANDBOX_SERVICES', str(manifest))
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'test-secret')
    yield root, service, manifest
    server.shutdown()
    server.server_close()
    thread.join()


def test_http_episode_isolation_and_snapshot_recovery(remote):
    root, service, _ = remote
    first, second = RemoteSandboxEpisode(root), RemoteSandboxEpisode(root)
    try:
        first.reset(1)
        second.reset(2)
        action = '{"kind":"tool","name":"increment","arguments":{}}'
        assert first.step(action) == (0.0, False)
        assert not any(item['path'] == '/v1/reward' for item in first.trace)
        saved = first.snapshot()
        first.step(action)
        assert first.finish()['final_reward'] == 1
        assert first.finish()['final_reward'] == 1
        assert sum(item['path'] == '/v1/reward' for item in first.trace) == 1
        assert second.finish()['final_reward'] == 0
        # Simulate losing all in-memory sessions when a Pod is replaced.
        with service.lock:
            for key in list(service.sessions):
                service._remove(key)
        restored = RemoteSandboxEpisode(root)
        try:
            restored.restore(saved)
            assert restored.reward is None
            restored.step(action)
            assert restored.finish()['final_reward'] == 1
        finally:
            restored.close()
    finally:
        first.close()
        second.close()
    assert not service.sessions


def test_remote_identity_and_auth_fail_closed(remote, monkeypatch):
    root, service, _ = remote
    service.identity = 'wrong'
    with pytest.raises(ValueError, match='identity'):
        RemoteSandboxEpisode(root)
    service.identity = artifact_identity(json.loads((root / 'status.json').read_text())['artifact_hashes'])
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'incorrect')
    episode = RemoteSandboxEpisode(root)
    with pytest.raises(RuntimeError, match='401'):
        episode.reset(1)
    episode.close()


def test_kubernetes_manifest_uses_docker_service_and_secret(remote):
    root, _, _ = remote
    task = SimpleNamespace(id='task-1', sandbox=str(root))
    image = 'registry/sandbox@sha256:' + 'a' * 64
    document, services = resources([task], {'task-1': image}, namespace='default', secret='sandbox-env', prefix='test')
    deployment = next(item for item in document['items'] if item['kind'] == 'Deployment')
    pod = deployment['spec']['template']['spec']
    container = pod['containers'][0]
    assert container['image'] == image
    assert container['envFrom'] == [{'secretRef': {'name': 'sandbox-env'}}]
    assert container['readinessProbe']['httpGet']['path'] == '/health'
    assert pod['automountServiceAccountToken'] is False
    assert next(item for item in document['items'] if item['kind'] == 'Service')['spec']['type'] == 'NodePort'
    assert 'test-secret' not in json.dumps(document)
    assert len(services) == 1
    with pytest.raises(ValueError, match='immutable'):
        resources([task], {'task-1': 'sandbox:latest'}, namespace='default', secret='env', prefix='test')


def test_start_submits_resources_without_waiting_for_pods(tmp_path, monkeypatch):
    from rl import kubernetes
    root = tmp_path / 'sandbox'
    root.mkdir()
    (root / 'status.json').write_text(json.dumps({'artifact_hashes': {'app.py': 'a' * 64}}))
    monkeypatch.setattr(kubernetes, 'load_tasks', lambda **kwargs: [SimpleNamespace(id='task-1', sandbox=str(root))])
    image_file = tmp_path / 'images.json'
    image_file.write_text(json.dumps({'task-1': 'registry/task@sha256:' + 'a' * 64}))
    output = tmp_path / 'deploy'
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({'spec': {'ports': [{'nodePort': 30123}]}}))
    monkeypatch.setattr(kubernetes.subprocess, 'run', run)
    monkeypatch.setattr('sys.argv', ['kubernetes', 'start', '--sandbox', str(root), '--images', str(image_file),
                                    '--host', '10.0.0.1', '--output', str(output)])
    kubernetes.main()
    assert len(calls) == 2
    assert calls[0][1] == 'apply'
    assert calls[1][1:3] == ['get', 'service']
    services = json.loads((output / 'services.json').read_text())['services']
    assert next(iter(services.values()))['url'] == 'http://10.0.0.1:30123'
