"""Environment failures are excluded without manufacturing policy rewards."""
import json
import queue
from types import SimpleNamespace

import pytest

from rl.data.task_queue import TaskQueue
from rl.data.dataset import DatasetResults, DatasetSchedule
from rl.data.tasks import TaskSpec
from rl.trainer.optimization import validate_actor_group


def test_quarantine_survives_restart_and_skips_only_affected_sandbox(tmp_path):
    tasks = [TaskSpec('broken', '/broken', identity='bad-hash'),
             TaskSpec('healthy', '/healthy', identity='good-hash')]
    schedule = DatasetSchedule(tasks, epochs=2, batch_size=2, seed=1)
    task_queue = TaskQueue(tmp_path, schedule.jobs, 4)
    task_queue.quarantine('broken', 'bad-hash', 'non-finite reward')
    task_queue = TaskQueue(tmp_path, schedule.jobs, 4)
    claimed = []
    consumed = set()
    session = SimpleNamespace(tasks=tasks, policy_identity='policy')
    for epoch in (1, 2):
        with task_queue.claim() as job:
            assert job['task_id'] == 'healthy'
            claimed.append(job['dataset_index'])
            consumed.add(job['dataset_index'])
        quarantines = task_queue.ready(consumed)
        assert len(quarantines) == 1
        group = quarantines[0]
        assert group['dataset_epoch'] == epoch
        validate_actor_group(session, group)
        assert group['episodes'] == []
        assert 'final_reward' not in group
        loader = DatasetResults(schedule, queue.Queue(), None, {}, rollout_workers=1,
                                timeout=1, validate=lambda g: validate_actor_group(session, g),
                                task_queue=task_queue)
        loader._accept(group)
        assert loader._consume(group['dataset_index'])['quarantined']
        consumed.add(group['dataset_index'])
        task_queue.commit_consumed(consumed)
    assert len(claimed) == 2
    with task_queue.claim() as job:
        assert job is None


def test_quarantine_identity_and_empty_episodes_are_required():
    task = TaskSpec('broken', '/broken', identity='hash')
    session = SimpleNamespace(tasks=[task], policy_identity='policy')
    valid = dict(quarantined=True, task_id='broken', task_identity='hash',
                 episodes=[], reason='environment schema error')
    validate_actor_group(session, valid)
    for change in ({'task_identity': 'other'}, {'episodes': [{'final_reward': 0}]}, {'reason': ''}):
        with pytest.raises(ValueError, match='quarantine'):
            validate_actor_group(session, {**valid, **change})


def failing_environment(connection, root, parent_pid):
    while True:
        operation, _ = connection.recv()
        if operation == 'close':
            break
        connection.send({'error': 'reward response violates schema', 'retryable': False})
    connection.close()


def test_environment_rpc_failure_is_distinct_from_policy_failure(tmp_path):
    from rl.errors import EnvironmentProtocolError
    from rl.rollout.scheduler import EnvironmentPool, parallel_rollouts
    pool = EnvironmentPool(str(tmp_path), 1, target=failing_environment)
    try:
        with pytest.raises(EnvironmentProtocolError, match='reward response'):
            parallel_rollouts(object(), pool, {
                'rollout_group': 1, 'max_steps': 2, 'max_tokens': 8,
                'max_context': 32, 'algorithm': 'grpo', 'rollout_timeout': 10,
            }, 42)
    finally:
        pool.close()


def test_trainer_never_optimizes_or_replays_quarantined_group(tmp_path, monkeypatch):
    from contextlib import nullcontext
    from rl.trainer.loop import TrainingSession
    from rl.checkpointing import retention
    monkeypatch.setattr(retention, 'prune_live_artifacts', lambda *args, **kwargs: None)
    job = {'dataset_index': 0, 'dataset_epoch': 1, 'task_id': 'broken'}
    group = {**job, 'quarantined': True, 'reason': 'invalid reward', 'episodes': []}
    session = TrainingSession.__new__(TrainingSession)
    session.config = SimpleNamespace(resume=True, max_groups=0, mini_batch_size=1,
        optimization_passes=1, replay_batches_per_batch=0, rollout_group=2,
        checkpoint_keep=1, group_artifacts_keep=1)
    session.state = SimpleNamespace(dataset_cursor=0, inflight={}, groups=0,
        optimizer_step=0, updates=0, committed_version=0, consumed_jobs=set(), skipped_jobs={})
    session.output = tmp_path
    session.schedule = SimpleNamespace(batches=[[job]], jobs=[job])
    session.pool = SimpleNamespace(start_all=lambda: None, close=lambda: None)
    session.loader = SimpleNamespace(ready_count=lambda *args: iter([[group]]))
    session.times = SimpleNamespace(measure=lambda *args: nullcontext())
    session.prepared_records = lambda **kwargs: []
    session.commit = lambda **kwargs: None
    session.publish_policy = lambda: None
    session.evaluate_periodically = lambda **kwargs: None
    session.finalize = lambda: 'done'
    session.mx = SimpleNamespace(clear_cache=lambda: None)
    session.task_queue = None
    emitted, metrics = [], []
    session.monitor = SimpleNamespace(runtime=SimpleNamespace(
        emit=lambda *args, **kwargs: emitted.append((args, kwargs))))
    class Journal:
        append = staticmethod(metrics.append)
        export = staticmethod(lambda: None)
        def __iter__(self):
            return iter(metrics)
    session.metrics = Journal()
    session.optimizer_metrics = SimpleNamespace(export=lambda: None)
    def forbidden(*args, **kwargs):
        pytest.fail('quarantined data reached training')
    session.prepare = forbidden
    session.optimize = forbidden
    session.replay = SimpleNamespace(add=forbidden, sample=forbidden)
    assert session.run() == 'done'
    assert session.state.consumed_jobs == {0}
    assert session.state.skipped_jobs == {'0': 'environment_quarantined'}
    assert next(m for m in metrics if m['event'] == 'dataset_batch_completed')['fresh_rollouts'] == 0
    summary = json.loads((tmp_path / 'learning_signals.json').read_text())
    assert summary['totals']['quarantined_jobs'] == 1
    assert summary['totals']['reward_success_rate'] is None
    assert emitted[0][0] == ('task/quarantined',)


def test_external_judge_fallback_is_retryable_not_permanent_quarantine():
    from rl.environments.local import SandboxEpisode
    from rl.errors import InfrastructureError
    episode = SandboxEpisode.__new__(SandboxEpisode)
    episode.reward = None
    episode.terminated = True
    episode.trace = []
    def request(method, path):
        if path == '/v1/reward':
            return 200, {'reward': 0}
        return 200, {'events': [{'event': 'evaluator_call', 'payload': {
            'used_fallback': True, 'judgment_obtained': False}}]}
    episode.request = request
    with pytest.raises(InfrastructureError, match='used fallback'):
        episode.finish()


def test_invalid_numeric_reward_remains_non_retryable_protocol_failure():
    from rl.environments.local import SandboxEpisode
    episode = SandboxEpisode.__new__(SandboxEpisode)
    episode.reward = None
    episode.request = lambda *args: (200, {'reward': float('nan')})
    with pytest.raises(ValueError, match='non-finite'):
        episode.finish()
