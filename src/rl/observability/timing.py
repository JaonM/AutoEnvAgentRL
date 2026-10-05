"""Measured local stage durations; concurrent worker seconds are never wall time."""
from contextlib import contextmanager
import time


class StageTimes:
    def __init__(self, state=None, *, mlx=None):
        self.seconds = dict((state or {}).get('seconds', {}))
        self.counts = dict((state or {}).get('counts', {}))
        self.memory = dict((state or {}).get('memory', {}))
        self.mlx, self.depth = mlx, 0
        if mlx is not None:
            self.memory['initialization'] = {'peak_bytes': max(self.memory.get('initialization', {}).get('peak_bytes', 0), mlx.get_peak_memory()),
                                             'active_after_bytes': mlx.get_active_memory()}

    def add(self, name, seconds, count=1):
        self.seconds[name] = self.seconds.get(name, 0.) + max(0., seconds)
        self.counts[name] = self.counts.get(name, 0) + count

    @contextmanager
    def measure(self, name):
        outer = self.mlx is not None and self.depth == 0
        if outer:
            self.mlx.synchronize()
            before = self.mlx.get_active_memory()
            self.mlx.reset_peak_memory()
        self.depth += 1
        started = time.perf_counter()
        try:
            yield
        finally:
            self.depth -= 1
            if outer:
                self.mlx.synchronize()
                self.memory[name] = {'peak_bytes': max(self.memory.get(name, {}).get('peak_bytes', 0), self.mlx.get_peak_memory()),
                                     'active_before_bytes': before, 'active_after_bytes': self.mlx.get_active_memory()}
            self.add(name, time.perf_counter() - started)

    def state(self):
        return {'seconds': dict(self.seconds), 'counts': dict(self.counts), 'memory': dict(self.memory),
                'semantics': 'actor stages are elapsed seconds; rollout sums are worker-seconds and may overlap'}

    def ingest_rollout(self, group):
        self.add('rollout_group_wall', group.get('finished_at', 0) - group.get('started_at', 0))
        self.add('rollout_queue_dwell', time.time() - group.get('finished_at', time.time()))
        for name, value in group.get('timings', {}).items():
            self.add('rollout_' + name, value)
        for episode in group['episodes']:
            for action in episode['actions']:
                self.add('rollout_generation_compute', action.get('generation_compute_seconds', 0.))
                self.add('rollout_prefill', action.get('generation_prefill_seconds', 0.))
                self.add('rollout_decode', action.get('generation_decode_seconds', 0.))
                self.add('rollout_behavior_verification', action.get('verification_seconds', 0.))
            for entry in episode.get('trace', []):
                path = entry['path']
                kind = ('user_simulator' if path == '/v1/user_simulator' else
                        'tool' if path.startswith('/v1/tools/') else
                        'reward' if path == '/v1/reward' else 'other_environment')
                self.add('rollout_' + kind, entry['seconds'])


def rollout_training_overlap(records):
    """Use durable metric intervals, including groups later pruned or rejected."""
    rollouts, updates = [], []
    for row in records:
        start, end = row.get('started_at'), row.get('finished_at')
        if start is None or end is None or end <= start:
            continue
        if row.get('event') == 'rollout_received':
            rollouts.append((start, end))
        elif row.get('event') == 'actor_optimizer_step':
            updates.append((start, end))
    rollouts.sort()
    updates.sort()
    i = j = 0
    while i < len(rollouts) and j < len(updates):
        a, b = rollouts[i], updates[j]
        if a[0] < b[1] and b[0] < a[1]:
            return True
        if a[1] <= b[0]:
            i += 1
        else:
            j += 1
    return False
