"""Bounded single-host Docker service lifecycle, asynchronous startup and leases."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.request import build_opener, ProxyHandler
import uuid

from .checkpoint import atomic_json
from .remote_environment import container_hashes
from .sandbox_service import artifact_identity
from .tasks import load_tasks


class DockerEngine:
    def __init__(self, root, owner, *, cpus='1', memory='1g'):
        self.root, self.owner, self.cpus, self.memory = Path(root), owner, cpus, memory

    def run(self, args, *, include_stderr=False, **kwargs):
        result = subprocess.run(['docker', *args], check=True, capture_output=True, text=True,
                              timeout=120, **kwargs)
        return (result.stdout + (result.stderr if include_stderr else '')).strip()

    def start(self, identity, entry):
        # Pin tags to a local immutable image ID before starting a container.
        image = self.run(['image', 'inspect', entry['image'], '--format', '{{.Id}}'])
        name = f'ef-rl-{self.owner[:12]}-{identity[:12]}'
        args = ['run', '-d', '--name', name, '--label', f'envfactory.manager={self.owner}',
                '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
                '--user', '10001:10001', '--cpus', self.cpus, '--memory', self.memory,
                '--pids-limit', '128', '--tmpfs', '/tmp:rw,nosuid,size=256m',
                '--tmpfs', '/app/.runtime:rw,nosuid,size=128m,uid=10001,gid=10001,mode=0700',
                '--publish', '127.0.0.1::8000',
                '--mount', f'type=bind,src={self.root / "runtime" / identity},dst=/opt/envfactory,readonly',
                '--env', 'RL_ARTIFACT_HASHES_FILE=/opt/envfactory/hashes.json',
                '--env', 'PYTHONUNBUFFERED=1']
        # Pass names only; credentials stay out of argv, manifests and manager logs.
        for name_env in os.environ:
            if name_env == 'SANDBOX_TRAINER_API_KEY' or name_env.startswith(('SANDBOX_LLM_', 'LLM_')):
                args += ['--env', name_env]
        args += ['--entrypoint', 'python3', image, '/opt/envfactory/serve.py']
        try:
            cid = self.run(args)
            port = self.run(['port', cid, '8000/tcp']).rsplit(':', 1)[1]
            return {'container_id': cid, 'url': 'http://127.0.0.1:' + port, 'image_id': image}
        except Exception:
            self.remove(name, identity)
            raise

    def healthy(self, entry):
        try:
            with build_opener(ProxyHandler({})).open(entry['url'] + '/health', timeout=.5) as response:
                return response.status == 200 and json.load(response).get('artifact_identity') == entry['identity']
        except (OSError, ValueError):
            return False

    def remove(self, container, identity):
        try:
            label = self.run(['inspect', container, '--format', '{{index .Config.Labels "envfactory.manager"}}'])
            if label != self.owner:
                raise ValueError('refusing to remove a container owned by another manager')
        except subprocess.CalledProcessError:
            return
        try:
            logs = self.run(['logs', '--tail', '2000', container], include_stderr=True)
            (self.root / f'{identity}.container.log').write_text(logs)
        finally:
            self.run(['rm', '-f', container])


class DockerManager:
    def __init__(self, tasks, root, *, images=None, max_active=4, startup_workers=2,
                 idle_timeout=300, startup_timeout=600, cpus='1', memory='1g', engine=None):
        if any(not math.isfinite(value) or value <= 0 for value in (max_active, startup_workers, idle_timeout, startup_timeout)):
            raise ValueError('manager limits and timeouts must be positive')
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (self.root / 'manager.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError('Docker manager already running for this output')
        previous = self.root / 'services.json'
        if previous.exists():
            old = json.loads(previous.read_text())
            if old.get('backend') == 'docker' and old.get('manager_state') != 'stopped':
                self.lock.close()
                raise ValueError('previous manager did not stop cleanly; run sandbox_services.sh stop first')
        self.owner = uuid.uuid4().hex
        self.max_active, self.startup_workers = max_active, startup_workers
        self.idle_timeout, self.startup_timeout = idle_timeout, startup_timeout
        self.engine = engine or DockerEngine(self.root, self.owner, cpus=cpus, memory=memory)
        self.stop_event, self.thread = threading.Event(), None
        self.entries, self.futures = {}, {}
        self.pool = ThreadPoolExecutor(max_workers=startup_workers)
        self.error = None
        self.closed = False
        (self.root / 'leases').mkdir(exist_ok=True)
        (self.root / 'requests').mkdir(exist_ok=True)
        self.stop_file = self.root / 'stop'
        self.stop_file.unlink(missing_ok=True)
        images = images or {}
        try:
            for task in tasks:
                sandbox = Path(task.sandbox)
                hashes = container_hashes(sandbox)
                identity = artifact_identity(hashes)
                image = images.get(task.id)
                if not image:
                    metadata = json.loads((sandbox / 'docker_image_metadata.json').read_text())
                    image = metadata['image_id']
                directory = self.root / 'runtime' / identity
                directory.mkdir(parents=True, exist_ok=True)
                directory.chmod(0o755)
                (directory / 'serve.py').write_text(Path(__file__).with_name('sandbox_service.py').read_text())
                atomic_json(directory / 'hashes.json', hashes)
                self.entries[identity] = {'identity': identity, 'task_id': task.id, 'image': image,
                                          'state': 'queued', 'last_used': time.monotonic()}
            # Prewarm only a bounded window; the scheduler requests other tasks on demand.
            for identity in list(self.entries)[:max_active]:
                self.demand(identity)
            self.publish('starting')
        except BaseException:
            self.pool.shutdown(wait=False, cancel_futures=True)
            self.lock.close()
            raise

    def demand(self, identity):
        (self.root / 'requests' / identity).touch(exist_ok=True)

    def publish(self, state='running'):
        atomic_json(self.root / 'services.json', {'backend': 'docker', 'owner': self.owner,
                    'manager_state': state, 'manager_pid': os.getpid(), 'updated_at': time.time(),
                    'lease_dir': str(self.root / 'leases'), 'request_dir': str(self.root / 'requests'),
                    'services': self.entries, 'startup_timeout': self.startup_timeout, 'request_timeout': 120})

    def log(self, event, **fields):
        with (self.root / 'manager.jsonl').open('a') as stream:
            stream.write(json.dumps({'at': time.time(), 'event': event, **fields}) + '\n')

    def unleased(self, identity):
        lock = (self.root / 'leases' / identity).open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock
        except BlockingIOError:
            lock.close()
            return None

    def evict(self, identity, entry):
        lock = self.unleased(identity)
        if lock is None:
            entry['last_used'] = time.monotonic()
            return False
        try:
            self.engine.remove(entry['container_id'], identity)
            entry.pop('container_id', None)
            entry.pop('url', None)
            entry['state'] = 'queued'
            self.log('evicted', task=entry['task_id'])
            self.publish()
            return True
        finally:
            lock.close()

    def tick(self):
        now = time.monotonic()
        for identity, future in list(self.futures.items()):
            if not future.done():
                continue
            entry = self.entries[identity]
            try:
                entry.update(future.result(), state='warming', started=now)
                self.log('container_started', task=entry['task_id'])
            except Exception as error:
                entry.update(state='failed', error_type=type(error).__name__)
                detail = getattr(error, 'stderr', '') or str(error)
                self.log('start_failed', task=entry['task_id'], error_type=type(error).__name__, detail=detail[-4000:])
            del self.futures[identity]
        for identity, entry in self.entries.items():
            if entry['state'] == 'ready' and now - entry.get('health_checked', now) > 5:
                entry['health_checked'] = now
                if not self.engine.healthy(entry):
                    entry.update(state='warming', started=now)
                    self.log('unhealthy', task=entry['task_id'])
            if entry['state'] == 'warming':
                if self.engine.healthy(entry):
                    entry.update(state='ready', last_used=now, ready_at=now, health_checked=now)
                    (self.root / 'requests' / identity).unlink(missing_ok=True)
                    self.log('ready', task=entry['task_id'])
                elif now - entry['started'] > self.startup_timeout:
                    self.engine.remove(entry['container_id'], identity)
                    entry.pop('url', None)
                    entry.pop('container_id', None)
                    entry.update(state='failed', error_type='ReadinessTimeout')
                    self.log('readiness_failed', task=entry['task_id'])
        requested = [identity for identity in self.entries
                     if (self.root / 'requests' / identity).exists() and self.entries[identity]['state'] == 'queued']
        active = sum(e['state'] in ('starting', 'warming', 'ready') for e in self.entries.values())
        for identity, entry in self.entries.items():
            if entry['state'] != 'ready':
                continue
            lock = self.unleased(identity)
            if lock is None:
                entry['last_used'] = now
                continue
            lock.close()
            pressure = bool(requested) and active >= self.max_active and now - entry['ready_at'] > 2
            if pressure or now - entry['last_used'] > self.idle_timeout:
                if self.evict(identity, entry):
                    active -= 1
        for identity in requested:
            if active >= self.max_active or len(self.futures) >= self.startup_workers:
                break
            entry = self.entries[identity]
            entry['state'] = 'starting'
            (self.root / 'requests' / identity).unlink(missing_ok=True)
            self.futures[identity] = self.pool.submit(self.engine.start, identity, dict(entry))
            active += 1
        self.publish()

    def _watch(self):
        try:
            while not self.stop_event.is_set() and not self.stop_file.exists():
                self.tick()
                self.stop_event.wait(.25)
        except Exception as error:
            self.error = error
            self.log('manager_failed', error_type=type(error).__name__)
            self.publish('failed')

    def start(self):
        self.thread = threading.Thread(target=self._watch, name='docker-sandbox-manager', daemon=True)
        self.thread.start()
        return str(self.root / 'services.json')

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop_event.set()
        if self.thread:
            self.thread.join()
        self.pool.shutdown(wait=True)
        for identity, future in self.futures.items():
            try:
                self.entries[identity].update(future.result())
            except Exception:
                pass
        errors = []
        for identity, entry in self.entries.items():
            if entry.get('container_id'):
                try:
                    self.engine.remove(entry['container_id'], identity)
                except Exception as error:
                    errors.append(type(error).__name__)
            entry['state'] = 'stopped'
            entry.pop('url', None)
        self.publish('cleanup_failed' if errors else 'stopped')
        self.lock.close()
        if errors:
            raise RuntimeError('Docker cleanup failed; inspect manager logs and owned container labels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status', 'stop', '_serve'))
    parser.add_argument('--sandbox', default='')
    parser.add_argument('--tasks', default='')
    parser.add_argument('--images', default='', help='optional JSON task-ID to local image mapping; default qualified image metadata')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-active', type=int, default=4)
    parser.add_argument('--startup-workers', type=int, default=2)
    parser.add_argument('--idle-timeout', type=float, default=300)
    parser.add_argument('--startup-timeout', type=float, default=600)
    parser.add_argument('--cpus', default='1')
    parser.add_argument('--memory', default='1g')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.action == 'status':
        print((args.output / 'services.json').read_text())
        return
    if args.action == 'stop':
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / 'manager.lock').open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                (args.output / 'stop').touch()
                print('Stop requested; manager will save logs and remove its containers.')
                return
            path = args.output / 'services.json'
            if not path.exists():
                print('No manager state found.')
                return
            manifest = json.loads(path.read_text())
            if manifest.get('backend') != 'docker':
                raise ValueError('not a Docker manager output')
            engine = DockerEngine(args.output, manifest['owner'])
            # Includes containers whose IDs had not yet been published when the manager died.
            ids = engine.run(['ps', '-aq', '--filter', f'label=envfactory.manager={manifest["owner"]}']).splitlines()
            for cid in ids:
                engine.remove(cid, cid)
            for entry in manifest['services'].values():
                entry.update(state='stopped')
                entry.pop('url', None)
            manifest.update(manager_state='stopped', updated_at=time.time())
            atomic_json(path, manifest)
            print('Reclaimed containers owned by the stopped manager.')
        return
    if args.action == 'start':
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / 'manager.log').open('a') as log:
            process = subprocess.Popen([sys.executable, '-m', 'rl.docker_manager', '_serve', *sys.argv[2:]],
                                       stdout=log, stderr=log, start_new_session=True)
        # Wait only for configuration handoff, never for docker run or readiness.
        deadline = time.monotonic() + 30
        path = args.output / 'services.json'
        while True:
            if process.poll() is not None:
                raise RuntimeError('manager failed during initialization; see manager.log')
            if path.exists():
                state = json.loads(path.read_text())
                if state.get('manager_pid') == process.pid:
                    break
            if time.monotonic() >= deadline:
                process.terminate()
                process.wait(timeout=10)
                raise TimeoutError('manager configuration handoff timed out; see manager.log')
            time.sleep(.05)
        print(json.dumps({'submitted': True, 'pid': process.pid, 'services': str(path),
                          'log': str(args.output / 'manager.log')}))
        return
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
    if not os.environ.get('SANDBOX_TRAINER_API_KEY'):
        raise ValueError('set SANDBOX_TRAINER_API_KEY for standalone Docker services and training')
    tasks = load_tasks(sandbox=args.sandbox, manifest=args.tasks)
    manager = DockerManager(tasks, args.output, images=json.loads(Path(args.images).read_text()) if args.images else None,
                            max_active=args.max_active, startup_workers=args.startup_workers,
                            idle_timeout=args.idle_timeout, startup_timeout=args.startup_timeout,
                            cpus=args.cpus, memory=args.memory)
    signal.signal(signal.SIGTERM, lambda *_: manager.stop_event.set())
    signal.signal(signal.SIGINT, lambda *_: manager.stop_event.set())
    try:
        manager.start()
        manager.thread.join()
    finally:
        manager.close()


if __name__ == '__main__':
    main()
