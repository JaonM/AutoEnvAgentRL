import json
import time
import fcntl
from pathlib import Path
from types import SimpleNamespace
import pytest

from rl.docker_manager import DockerEngine, DockerManager
from rl.remote_environment import docker_task_ready
from rl.task_queue import TaskQueue


class Engine:
    def __init__(self):
        self.started, self.removed, self.ready = [], [], set()
    def start(self, identity, entry):
        self.started.append(identity)
        return {'container_id': identity, 'url': 'http://127.0.0.1:1234'}
    def healthy(self, entry):
        return entry['identity'] in self.ready
    def remove(self, cid, identity):
        self.removed.append(cid)


def tasks(tmp_path, count=2):
    result = []
    for index in range(count):
        root = tmp_path / str(index)
        root.mkdir()
        (root / 'task.json').write_text(json.dumps({'task': str(index)}))
        result.append(SimpleNamespace(id=str(index), sandbox=str(root)))
    return result


def settle(manager):
    for future in manager.futures.values():
        future.result(timeout=1)
    manager.tick()


def test_startup_nonblocking_readiness_and_owned_cleanup(tmp_path):
    engine = Engine()
    manager = DockerManager(tasks(tmp_path), tmp_path / 'manager', images={'0': 'image0', '1': 'image1'}, engine=engine)
    try:
        manager.tick()
        settle(manager)
        assert len(engine.started) == 2
        assert all(entry['state'] == 'warming' for entry in manager.entries.values())
        engine.ready.add(engine.started[1])
        manager.tick()
        manifest = manager.root / 'services.json'
        assert not docker_task_ready(manifest, '0')
        assert docker_task_ready(manifest, '1')
        assert manager.entries[engine.started[0]]['state'] == 'warming'
    finally:
        manager.close()
    assert set(engine.removed) == set(engine.started)
    assert json.loads((manager.root / 'services.json').read_text())['manager_state'] == 'stopped'


def test_active_limit_leases_pressure_eviction_and_reuse(tmp_path):
    engine = Engine()
    manager = DockerManager(tasks(tmp_path), tmp_path / 'manager', images={'0': 'a', '1': 'b'},
                            max_active=1, engine=engine)
    first, second = list(manager.entries)
    lease = None
    try:
        manager.tick(); settle(manager)
        engine.ready.add(first)
        manager.tick()
        # Repeated claims reuse a healthy container.
        manager.tick()
        assert engine.started == [first]
        manager.entries[first]['ready_at'] -= 5
        lease = (manager.root / 'leases' / first).open('a')
        fcntl.flock(lease, fcntl.LOCK_SH)
        manager.demand(second)
        manager.tick()
        assert not engine.removed
        assert engine.started == [first]
        lease.close(); lease = None
        manager.tick(); settle(manager)
        assert engine.removed == [first]
        assert engine.started == [first, second]
        assert sum(e['state'] in ('starting', 'warming', 'ready') for e in manager.entries.values()) == 1
    finally:
        if lease: lease.close()
        manager.close()


def test_readiness_timeout_and_failed_start_are_explicit(tmp_path):
    engine = Engine()
    manager = DockerManager(tasks(tmp_path, 1), tmp_path / 'manager', images={'0': 'a'}, startup_timeout=1, engine=engine)
    try:
        manager.tick(); settle(manager)
        entry = next(iter(manager.entries.values()))
        entry['started'] -= 2
        manager.tick()
        assert entry['state'] == 'failed'
        with pytest.raises(RuntimeError, match='failed to start'):
            docker_task_ready(manager.root / 'services.json', '0')
        assert engine.removed
    finally:
        manager.close()


def test_ready_queue_skips_slow_sandbox_without_leasing_it(tmp_path):
    jobs = [{'dataset_index': i, 'dataset_epoch': 0, 'seed': i, 'task_id': str(i)} for i in range(2)]
    queue = TaskQueue(tmp_path, jobs, 2)
    with queue.claim(ready=lambda job: job['task_id'] == '1') as job:
        assert job['task_id'] == '1'
        with queue.claim(ready=lambda job: job['task_id'] == '0') as other:
            assert other['task_id'] == '0'


def test_docker_command_is_local_limited_and_contains_no_secret(tmp_path, monkeypatch):
    engine = DockerEngine(tmp_path, 'owner')
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'DO-NOT-LOG')
    commands = []
    def run(args):
        commands.append(args)
        if args[:2] == ['image', 'inspect']: return 'sha256:' + 'a' * 64
        if args[0] == 'port': return '127.0.0.1:32456'
        return 'container-id'
    monkeypatch.setattr(engine, 'run', run)
    result = engine.start('identity', {'image': 'local-image'})
    command = commands[1]
    assert command[command.index('--publish') + 1] == '127.0.0.1::8000'
    assert '--memory' in command and '--cpus' in command and '--pids-limit' in command
    assert 'DO-NOT-LOG' not in json.dumps(commands)
    assert 'SANDBOX_TRAINER_API_KEY' in command
    assert result['url'] == 'http://127.0.0.1:32456'


def test_only_owned_container_can_be_removed(tmp_path, monkeypatch):
    engine = DockerEngine(tmp_path, 'owner')
    monkeypatch.setattr(engine, 'run', lambda args: 'another-owner')
    with pytest.raises(ValueError, match='another manager'):
        engine.remove('container-id', 'identity')


def test_training_owns_manager_and_cleans_up_on_failure(tmp_path, monkeypatch):
    from rl import train as trainer
    import rl.tasks
    calls = []
    class Manager:
        def __init__(self, *args, **kwargs): calls.append('create')
        def start(self): calls.append('start'); return str(tmp_path / 'services.json')
        def close(self): calls.append('close')
    monkeypatch.setattr('rl.docker_manager.DockerManager', Manager)
    monkeypatch.setattr(rl.tasks, 'load_tasks', lambda **kwargs: [])
    def fail(config):
        calls.append('train')
        assert config.sandbox_services.endswith('services.json')
        raise RuntimeError('test interruption')
    monkeypatch.setattr(trainer, '_train_with_status', fail)
    config = trainer.Config(sandbox='sandbox', output=str(tmp_path), sandbox_backend='docker')
    with pytest.raises(RuntimeError, match='test interruption'):
        trainer.train(config)
    assert calls == ['create', 'start', 'train', 'close']


def test_idle_container_eviction_does_not_restart_without_demand(tmp_path):
    engine = Engine()
    manager = DockerManager(tasks(tmp_path, 1), tmp_path / 'manager', images={'0': 'a'}, engine=engine, idle_timeout=1)
    try:
        manager.tick(); settle(manager)
        identity = engine.started[0]
        engine.ready.add(identity); manager.tick()
        manager.entries[identity]['last_used'] -= 2
        manager.tick(); manager.tick()
        assert engine.removed == [identity]
        assert engine.started == [identity]
    finally:
        manager.close()


def test_stale_manager_requires_explicit_owned_cleanup(tmp_path):
    engine = Engine()
    task_list = tasks(tmp_path, 1)
    root = tmp_path / 'manager'
    root.mkdir()
    (root / 'services.json').write_text(json.dumps({'backend': 'docker', 'manager_state': 'running', 'owner': 'old'}))
    with pytest.raises(ValueError, match='stop first'):
        DockerManager(task_list, root, images={'0': 'a'}, engine=engine)
    assert not engine.started


def test_standalone_stop_recovers_orphans_by_owner_label(tmp_path, monkeypatch):
    import rl.docker_manager as module
    manifest = {'backend': 'docker', 'owner': 'our-owner', 'manager_state': 'running',
                'services': {'id': {'state': 'ready', 'url': 'http://127.0.0.1:1234'}}}
    (tmp_path / 'services.json').write_text(json.dumps(manifest))
    calls = []
    monkeypatch.setattr(module.DockerEngine, 'run', lambda self, args: calls.append(args) or 'owned-container')
    monkeypatch.setattr(module.DockerEngine, 'remove', lambda self, cid, identity: calls.append(('remove', self.owner, cid)))
    monkeypatch.setattr('sys.argv', ['docker_manager', 'stop', '--output', str(tmp_path)])
    module.main()
    assert calls[0] == ['ps', '-aq', '--filter', 'label=envfactory.manager=our-owner']
    assert calls[1] == ('remove', 'our-owner', 'owned-container')
    result = json.loads((tmp_path / 'services.json').read_text())
    assert result['manager_state'] == 'stopped'
    assert 'url' not in result['services']['id']
