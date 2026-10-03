"""Rollout over Docker/Kubernetes HTTP services; no task code executes locally."""
import fcntl
import hashlib
import copy
import json
import math
import os
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler
import uuid

from .environment import SandboxEpisode, verify_training_sandbox
from .errors import InfrastructureError
from .sandbox_service import artifact_identity


def container_hashes(root):
    """Bind executable/data inputs, excluding reports produced after image build."""
    root = Path(root)
    files = set(root.glob('*.py'))
    files.update(root / name for name in ('task.json', 'BUILD_CONTRACT.json', 'tools.json')
                 if (root / name).is_file())
    files.update(path for path in (root / 'data').rglob('*') if path.is_file())
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


class RemoteSandboxEpisode(SandboxEpisode):
    def __init__(self, root):
        self.root = verify_training_sandbox(root)
        self.task = json.loads((self.root / 'task.json').read_text())
        hashes = container_hashes(self.root)
        self.identity = artifact_identity(hashes)
        self.manifest_path = Path(os.environ['RL_SANDBOX_SERVICES'])
        manifest = json.loads(self.manifest_path.read_text())
        self.headers = {'Authorization': 'Bearer ' + os.environ['SANDBOX_TRAINER_API_KEY'],
                        'Content-Type': 'application/json', 'X-RL-Session': uuid.uuid4().hex}
        self.timeout = float(manifest.get('request_timeout', 120))
        startup_timeout = float(manifest.get('startup_timeout', 600))
        if any(not math.isfinite(value) or value <= 0 for value in (self.timeout, startup_timeout)):
            raise ValueError('service timeouts must be positive and finite')
        self.opener = build_opener(ProxyHandler({}))
        self.trace, self._closed, self.lease, self.url = [], False, None, ''
        try:
            if manifest.get('backend') == 'docker':
                self.lease = (Path(manifest['lease_dir']) / self.identity).open('a')
                fcntl.flock(self.lease, fcntl.LOCK_SH)
            deadline = time.monotonic() + startup_timeout
            while True:
                manifest = json.loads(self.manifest_path.read_text())
                entry = manifest['services'][self.identity]
                if manifest.get('backend') == 'docker':
                    if manifest.get('manager_state') in ('stopped', 'failed', 'cleanup_failed'):
                        raise InfrastructureError('Docker sandbox manager is not running')
                    if entry['state'] == 'failed':
                        raise RuntimeError('Docker sandbox startup failed; see manager and container logs')
                    if entry['state'] != 'ready':
                        (Path(manifest['request_dir']) / self.identity).touch(exist_ok=True)
                self.url = entry.get('url', '').rstrip('/')
                if self.url:
                    if not self.url.startswith(('http://', 'https://')):
                        raise ValueError('sandbox endpoint must be HTTP(S)')
                    try:
                        status, health = self._call('GET', '/health', timeout=min(3, max(.01, deadline-time.monotonic())))
                        if status == 200:
                            if health.get('artifact_identity') != self.identity:
                                raise ValueError('remote sandbox artifact identity mismatch')
                            break
                    except InfrastructureError:
                        pass
                if time.monotonic() >= deadline:
                    raise InfrastructureError('sandbox readiness deadline exceeded')
                time.sleep(.2)
        except BaseException:
            self.close()
            raise

    def _call(self, method, path, body=None, *, timeout=None):
        request = Request(self.url + path, method=method, headers=self.headers,
                          data=json.dumps(body).encode() if body is not None else None)
        try:
            with self.opener.open(request, timeout=timeout or self.timeout) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            try:
                value = json.loads(error.read())
            except ValueError:
                value = {'error': 'non-JSON error response'}
            return error.code, value
        except (URLError, TimeoutError, OSError) as error:
            # Do not retry mutations: an HTTP timeout may occur after commit.
            raise InfrastructureError('sandbox HTTP transport failed') from error

    def request(self, method, path, body=None):
        started = time.monotonic()
        status, value = self._call(method, path, body)
        self.trace.append({'path': path, 'request': copy.deepcopy(body), 'status': status,
                           'response': copy.deepcopy(value), 'seconds': time.monotonic()-started})
        from .tool_transport import is_authored_tool_fault, is_business_conflict
        if ((status >= 500 and not is_authored_tool_fault(self.task, path, status, value))
                or (status == 409 and not is_business_conflict(path, status, value))):
            raise InfrastructureError(f'sandbox service unavailable: {path} ({status})')
        if status >= 400 and not path.startswith('/v1/tools/'):
            raise RuntimeError(f'sandbox service rejected request: {path} ({status})')
        return status, value

    def snapshot(self):
        status, payload = self._call('GET', '/_rl/snapshot')
        if status != 200:
            raise InfrastructureError('remote snapshot failed')
        return {'database': payload['database'], 'artifact_identity': self.identity,
                **{name: copy.deepcopy(getattr(self, name)) for name in
                   ('messages', 'conversation', 'reward', 'terminated', 'trace', 'tools')}, 'names': sorted(self.names)}

    def restore(self, state):
        if state['artifact_identity'] != self.identity:
            raise ValueError('snapshot sandbox identity mismatch')
        status, _ = self._call('POST', '/_rl/restore', {'database': state['database']})
        if status != 200:
            raise InfrastructureError('remote restore failed')
        for name in ('messages', 'conversation', 'reward', 'terminated', 'trace', 'tools'):
            setattr(self, name, copy.deepcopy(state[name]))
        self.names = set(state['names'])

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self.url:
                self._call('POST', '/_rl/close', {}, timeout=2)
        except InfrastructureError:
            pass  # Server TTL reclaims abandoned sessions after worker death.
        finally:
            if self.lease is not None:
                self.lease.close()
                self.lease = None


def docker_task_ready(manifest_path, task_id):
    """Cheap scheduler check; never wait for a container while holding a task lease."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('backend') != 'docker':
        return True
    if manifest.get('manager_state') in ('failed', 'stopped', 'cleanup_failed') or time.time() - manifest['updated_at'] > 180:
        raise InfrastructureError('Docker sandbox manager stopped or heartbeat expired')
    for identity, entry in manifest['services'].items():
        if entry['task_id'] != task_id:
            continue
        if entry['state'] == 'failed':
            raise RuntimeError(f'Docker sandbox {task_id} failed to start; inspect manager logs')
        if entry['state'] == 'ready':
            return True
        (Path(manifest['request_dir']) / identity).touch(exist_ok=True)
        return False
    raise ValueError(f'no Docker service configured for task {task_id}')
