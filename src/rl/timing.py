"""Measured local stage durations; concurrent worker seconds are never wall time."""
from contextlib import contextmanager
import time


class StageTimes:
    def __init__(self, state=None):
        self.seconds = dict((state or {}).get('seconds', {}))
        self.counts = dict((state or {}).get('counts', {}))

    def add(self, name, seconds, count=1):
        self.seconds[name] = self.seconds.get(name, 0.) + max(0., seconds)
        self.counts[name] = self.counts.get(name, 0) + count

    @contextmanager
    def measure(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - started)

    def state(self):
        return {'seconds': dict(self.seconds), 'counts': dict(self.counts),
                'semantics': 'actor stages are elapsed seconds; rollout sums are worker-seconds and may overlap'}

    def ingest_rollout(self, group):
        self.add('rollout_group_wall', group.get('finished_at', 0) - group.get('started_at', 0))
        self.add('rollout_queue_dwell', time.time() - group.get('finished_at', time.time()))
        for name, value in group.get('timings', {}).items():
            self.add('rollout_' + name, value)
        for episode in group['episodes']:
            for action in episode['actions']:
                self.add('rollout_generation_compute', action.get('generation_compute_seconds', 0.))
                self.add('rollout_behavior_verification', action.get('verification_seconds', 0.))
            for entry in episode.get('trace', []):
                path = entry['path']
                kind = ('user_simulator' if path == '/v1/user_simulator' else
                        'tool' if path.startswith('/v1/tools/') else
                        'reward' if path == '/v1/reward' else 'other_environment')
                self.add('rollout_' + kind, entry['seconds'])
