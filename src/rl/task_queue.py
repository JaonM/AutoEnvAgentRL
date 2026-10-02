"""Shared local task leases and durable complete groups (POSIX file locks)."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

from .checkpoint import atomic_json


class TaskQueue:
    def __init__(self, root, jobs, capacity):
        self.root, self.jobs, self.capacity = Path(root), jobs, capacity
        self.root.mkdir(parents=True, exist_ok=True)

    def commit_consumed(self, indices, attempts=None):
        atomic_json(self.root / 'consumed.json', {'consumed':sorted(indices),
                    'attempts':self.state().get('attempts', {}) if attempts is None else attempts})

    def state(self):
        path = self.root / 'consumed.json'
        return json.loads(path.read_text()) if path.exists() else {}

    def attempt(self, index):
        return self.state().get('attempts', {}).get(str(index), 0)

    def consumed(self):
        return set(self.state().get('consumed', []))

    @contextmanager
    def claim(self, ready=None):
        """A process death releases its lease; another worker resumes the same job."""
        claimed, handle = None, None
        with (self.root / 'scheduler.lock').open('a') as scheduler:
            fcntl.flock(scheduler, fcntl.LOCK_EX)
            state = self.state()
            consumed = set(state.get('consumed', []))
            remaining = [j for j in self.jobs if j['dataset_index'] not in consumed]
            # Keep epoch coverage strict, but remove all batch boundaries within an epoch.
            epoch = min((j['dataset_epoch'] for j in remaining), default=None)
            outstanding, available = 0, []
            for job in remaining:
                if job['dataset_epoch'] != epoch:
                    continue
                index = job['dataset_index']
                attempt = state.get('attempts', {}).get(str(index), 0)
                job = {**job, 'rollout_attempt':attempt, 'seed':job['seed'] + attempt * 1000003}
                if self.result_path(index, attempt).exists():
                    outstanding += 1
                    continue
                lock = (self.root / f'job-{index}.lock').open('a')
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    outstanding += 1
                    lock.close()
                    continue
                try:
                    eligible = ready is None or ready(job)
                except BaseException:
                    lock.close()
                    for _, held in available:
                        held.close()
                    raise
                if not available and eligible:
                    available.append((job, lock))
                else:
                    lock.close()
            if available and outstanding < self.capacity:
                claimed, handle = available.pop(0)
            for _, lock in available:
                lock.close()
        try:
            yield claimed
        finally:
            if handle is not None:
                handle.close()

    def result_path(self, index, attempt=0):
        return self.root / f'result-{index:08d}-attempt-{attempt}.json'

    def publish(self, group):
        atomic_json(self.result_path(group['dataset_index'], group.get('rollout_attempt', 0)), group)

    def ready(self, excluded):
        groups = []
        attempts = self.state().get('attempts', {})
        for path in self.root.glob('result-*.json'):
            if int(path.stem.split('-')[1]) not in excluded:
                group = json.loads(path.read_text())
                if group.get('rollout_attempt', 0) == attempts.get(str(group['dataset_index']), 0):
                    groups.append(group)
        return sorted(groups, key=lambda group: group['finished_at'])
