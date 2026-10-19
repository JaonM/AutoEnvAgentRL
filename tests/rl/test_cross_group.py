import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from rl.rollout.scheduler import EnvironmentPool
from rl.rollout.multiplex import sample_groups


class Runtime:
    def __init__(self, root):
        self.root = root
        self.steps = 0
        self.messages = []
        self.terminated = False
    def reset(self, seed):
        self.steps = 0
        self.seed = seed
        self.terminated = False
        self.messages = [{'role': 'user', 'content': str(seed)}]
    def step(self, action):
        # One group has long environment calls; the other must use the free slots.
        time.sleep(.06 if self.seed == 1 else .005)
        self.steps += 1
        self.terminated = self.steps >= 2
        self.messages.append({'role': 'assistant', 'content': action})
        with (Path(self.root) / 'events').open('a') as stream:
            stream.write(f'{self.seed}:{self.steps}\n')
        return 0., self.terminated
    def snapshot(self): return dict(vars(self))
    def restore(self, snapshot): self.__dict__.update(snapshot)
    def finish(self): return {'final_reward': float(self.seed), 'terminated': self.terminated}
    def close(self): pass


class Decoder:
    def __init__(self): self.requests, self.batches = {}, []
    def add(self, key, *args, **kwargs): self.requests[key] = 12
    def tick(self):
        time.sleep(.005)
        self.batches.append(list(self.requests))
        for key in self.requests:
            self.requests[key] -= 1
        result = {key: {'prompt': [1], 'tokens': [2], 'old_logp': [-.1], 'text': 'action'} for key in self.requests if self.requests[key] == 0}
        for key in result:
            del self.requests[key]
        return result


class Policy:
    def __init__(self): self.decoder = Decoder()
    def batch_decoder(self): return self.decoder


def specs(tmp_path, *, tree=False):
    result = []
    for index in range(2):
        root = tmp_path / str(index)
        root.mkdir(exist_ok=True)
        continuation = root / 'journal'
        continuation.mkdir(exist_ok=True)
        result.append({'key': index, 'sandbox': str(root), 'seed': index + 1, 'continuation': continuation,
                       'config': {'rollout_group': 3, 'max_steps': 3, 'max_tokens': 8, 'max_context': 64,
                                  'rollout_timeout': 5, 'environment_factory': f'{__name__}:Runtime',
                                  'rollout_mode': 'tree' if tree else 'chain', 'tree_forks': 1, 'tree_branches': 2}})
    return result


@pytest.mark.parametrize('tree', [False, True])
def test_cross_group_slots_decode_and_reward_isolation(tmp_path, tree):
    groups = specs(tmp_path, tree=tree)
    policy = Policy()
    pool = EnvironmentPool(None, 4)
    try:
        completed = dict(sample_groups(policy, pool, groups))
    finally:
        pool.close()
    assert any(len({key[0] for key in batch}) == 2 for batch in policy.decoder.batches)
    for key, episodes in completed.items():
        assert len(episodes) == (5 if tree else 3)
        assert all(e['seed'] == e['final_reward'] == key + 1 for e in episodes)
        assert (tmp_path / str(key) / 'events').read_text().count('\n') == (8 if tree else 6)
    a, b = completed[0], completed[1]
    assert min(e['started_at'] for e in b) < max(e['finished_at'] for e in a)
    assert max(len(batch) for batch in policy.decoder.batches) <= 4


def test_cross_group_resume_keeps_committed_actions_and_slot_ownership(tmp_path):
    groups = specs(tmp_path, tree=True)
    def interrupt(stage):
        if stage.endswith('-step'):
            raise InterruptedError('lost reply')
    pool = EnvironmentPool(None, 4)
    try:
        with pytest.raises(InterruptedError):
            dict(sample_groups(Policy(), pool, groups, notify=interrupt))
    finally:
        pool.close()
    pool = EnvironmentPool(None, 4)
    try:
        completed = dict(sample_groups(Policy(), pool, groups))
    finally:
        pool.close()
    assert all(len(episodes) == 5 for episodes in completed.values())
    for key in completed:
        assert (tmp_path / str(key) / 'events').read_text().count('\n') == 8
    # Completed journals do not invoke the environment again.
    assert dict(sample_groups(Policy(), SimpleNamespace(workers=[]), groups)) == completed


def test_action_random_keys_are_stable_and_distinct():
    from rl.rollout.scheduler import action_seed
    keys = {action_seed({'_sampling_seed': 123}, 42, episode, step)
            for episode in range(6) for step in range(4)}
    assert len(keys) == 24
    assert action_seed({'_sampling_seed': 123}, 42, 2, 1) == action_seed({'_sampling_seed': 123}, 42, 2, 1)
    assert keys.isdisjoint({action_seed({'_sampling_seed': 124}, 42, i, j) for i in range(6) for j in range(4)})


class SlowScoreRuntime(Runtime):
    def finish(self):
        time.sleep(.15)
        return {**super().finish(), 'frozen_steps': self.steps}


def test_async_scoring_uses_frozen_state_and_releases_slots(tmp_path):
    groups = specs(tmp_path)
    for spec in groups:
        spec['config'].update(scoring_workers=1, scoring_queue_size=2,
                              environment_factory=f'{__name__}:SlowScoreRuntime')
    pool = EnvironmentPool(None, 1)
    try:
        completed = dict(sample_groups(Policy(), pool, groups))
    finally:
        pool.close()
    episodes = [e for group in completed.values() for e in group]
    assert all(e['final_reward'] == e['seed'] and e['frozen_steps'] == 2 for e in episodes)
    scores = [x for e in episodes for x in e['environment_intervals'] if x['operation'] == 'score']
    steps = [x for e in episodes for x in e['environment_intervals'] if x['operation'] == 'step']
    assert any(s['started_at'] < t['finished_at'] and t['started_at'] < s['finished_at'] for s in scores for t in steps)
    assert len(list(tmp_path.glob('*/journal/score-*.json'))) == 6


def test_async_score_reply_recovery_does_not_repeat_actions_or_change_reward(tmp_path):
    groups = specs(tmp_path, tree=True)
    def interrupt(event, **fields):
        if event == 'finish':
            raise InterruptedError('score completed before group journal commit')
    for spec in groups:
        spec['config'].update(scoring_workers=1, scoring_queue_size=2)
        spec['observe'] = interrupt
    pool = EnvironmentPool(None, 2)
    try:
        with pytest.raises(InterruptedError):
            dict(sample_groups(Policy(), pool, groups))
    finally:
        pool.close()
    for spec in groups:
        spec.pop('observe')
    pool = EnvironmentPool(None, 2)
    try:
        completed = dict(sample_groups(Policy(), pool, groups))
    finally:
        pool.close()
    assert all(len(e) == 5 for e in completed.values())
    assert all(e['final_reward'] == e['seed'] for episodes in completed.values() for e in episodes)
    assert all((tmp_path / str(k) / 'events').read_text().count('\n') == 8 for k in completed)


def test_independently_interrupted_groups_can_resume_on_shared_slots(tmp_path):
    groups = specs(tmp_path)
    def interrupt(stage):
        if stage.endswith('-step'):
            raise InterruptedError('separate worker interruption')
    # Both groups previously owned slot zero, on different worker processes.
    for spec in groups:
        pool = EnvironmentPool(None, 1)
        try:
            with pytest.raises(InterruptedError):
                dict(sample_groups(Policy(), pool, [spec], notify=interrupt))
        finally:
            pool.close()
    pool = EnvironmentPool(None, 2)
    try:
        result = dict(sample_groups(Policy(), pool, groups))
    finally:
        pool.close()
    assert all(len(e) == 3 for e in result.values())
    assert all((tmp_path / str(k) / 'events').read_text().count('\n') == 6 for k in result)
