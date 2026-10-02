import json
import sqlite3
from types import SimpleNamespace
import pytest
from rl.parallel_rollout import EnvironmentPool, parallel_rollouts
from rl.environment import SandboxEpisode


class ResumableRuntime:
    def __init__(self, sandbox):
        self.root = sandbox
        self.messages = []
        self.terminated = False
        self.steps = 0
    def reset(self, seed):
        self.steps = 0
        self.terminated = False
        self.messages = [{'role':'user', 'content':str(seed)}]
    def step(self, text):
        from pathlib import Path
        with (Path(self.root) / 'executed.txt').open('a') as stream:
            stream.write('step\n')
        self.steps += 1
        self.terminated = self.steps == 2
        self.messages.append({'role':'assistant', 'content':text})
        return .5, self.terminated
    def snapshot(self): return dict(self.__dict__)
    def restore(self, state): self.__dict__.update(state)
    def finish(self): return {'final_reward':self.steps / 2, 'terminated':self.terminated}
    def close(self): pass


class Policy:
    def iter_sample(self, messages, max_tokens, max_context):
        yield None
        return {'prompt':[1], 'tokens':[2], 'old_logp':[-.1], 'text':'action'}


def test_lost_step_reply_resumes_environment_without_repeating_committed_action(tmp_path):
    continuation = tmp_path / 'continuation'
    continuation.mkdir()
    config = {'rollout_group':1, 'max_steps':3, 'max_tokens':8, 'max_context':64,
              'algorithm':'grpo', 'rollout_timeout':5}
    pool = EnvironmentPool(str(tmp_path), 1, factory=f'{__name__}:ResumableRuntime')
    def interrupt(stage):
        if stage == 'rollout-0-step':
            raise InterruptedError('fault after durable environment reply before learner acknowledgement')
    try:
        with pytest.raises(InterruptedError):
            parallel_rollouts(Policy(), pool, config, 42, notify=interrupt, continuation=continuation)
    finally:
        pool.close()
    pool = EnvironmentPool(str(tmp_path), 1, factory=f'{__name__}:ResumableRuntime')
    try:
        episodes = parallel_rollouts(Policy(), pool, config, 42, continuation=continuation)
    finally:
        pool.close()
    assert len(episodes[0]['actions']) == 2
    assert episodes[0]['final_reward'] == 1
    assert (tmp_path / 'executed.txt').read_text().splitlines() == ['step', 'step']
    assert json.loads((continuation / 'progress.json').read_text())['completed'] == episodes
    # Fully completed groups require no runtime or policy invocation on recovery.
    assert parallel_rollouts(None, SimpleNamespace(workers=[]), config, 42, continuation=continuation) == episodes


def test_sandbox_snapshot_restores_sqlite_and_conversation(tmp_path):
    path = tmp_path / 'episode.sqlite3'
    with sqlite3.connect(path) as db:
        db.execute('CREATE TABLE state (value TEXT)')
        db.execute("INSERT INTO state VALUES ('before')")
    episode = SandboxEpisode.__new__(SandboxEpisode)
    episode.app = SimpleNamespace(episode_store=SimpleNamespace(db_path=str(path)))
    episode.messages = [{'role':'user', 'content':'before'}]
    episode.conversation = list(episode.messages)
    episode.reward, episode.terminated, episode.trace, episode.names = .5, False, [], {'tool'}
    saved = episode.snapshot()
    with sqlite3.connect(path) as db:
        db.execute("UPDATE state SET value='after'")
    episode.messages = []
    episode.restore(saved)
    with sqlite3.connect(path) as db:
        assert db.execute('SELECT value FROM state').fetchone()[0] == 'before'
    assert episode.messages[0]['content'] == 'before'
    assert episode.names == {'tool'} and episode.reward == .5


def test_real_rollout_workers_share_queue_and_publish_complete_groups(tmp_path):
    import os
    model = os.environ.get('RL_TEST_MODEL')
    if not model:
        pytest.skip('set RL_TEST_MODEL for local Metal integration')
    import multiprocessing as mp
    from dataclasses import asdict
    from rl.model import Policy as RealPolicy
    from rl.train import Config
    from rl.tasks import TaskSpec
    from rl.dataset import DatasetSchedule, DatasetResults
    from rl.task_queue import TaskQueue
    from rl.process_control import StopSignal, PolicyVersion
    from rl.runtime import RolloutPool
    tasks = [TaskSpec(str(i), str(tmp_path), identity='test-runtime') for i in range(3)]
    schedule = DatasetSchedule(tasks, epochs=1, batch_size=1, seed=42)
    task_queue = TaskQueue(tmp_path / 'queue', schedule.jobs, 4)
    policy = RealPolicy(model)
    policy.save_snapshot(tmp_path / 'snapshots', 0)
    context = mp.get_context('spawn')
    results = context.Queue(2)
    stop = StopSignal(context)
    version = PolicyVersion(tmp_path / 'version.json', 0)
    config = {**asdict(Config(sandbox='unused', output=str(tmp_path), model=model,
                  rollout_workers=2, rollout_group=2, max_tokens=4, max_steps=2)),
              '_tasks':[asdict(task) for task in tasks], '_dataset_jobs':schedule.jobs,
              '_task_queue':str(task_queue.root), '_queue_capacity':4,
              'environment_factory':f'{__name__}:ResumableRuntime'}
    positions = {}
    pool = RolloutPool(context, config, version, results, stop, positions)
    loader = DatasetResults(schedule, results, pool, positions, rollout_workers=2,
                            timeout=5, validate=lambda _:None, task_queue=task_queue)
    try:
        pool.start_all()
        groups = []
        for mini in loader.ready_count(1, 3, 1):
            groups.extend(mini)
            task_queue.commit_consumed(loader.completed)
    finally:
        pool.close()
    assert {group['dataset_index'] for group in groups} == {0, 1, 2}
    assert all(group['shared_task'] and len(group['episodes']) == 2 for group in groups)
    assert all(len(episode['actions']) == 2 for group in groups for episode in group['episodes'])
    assert len(task_queue.ready(set())) == 3
