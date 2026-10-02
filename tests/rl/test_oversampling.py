import queue
import pytest
from rl.train import Config
from rl.task_queue import TaskQueue
from rl.dataset import DatasetSchedule, DatasetResults
from rl.tasks import TaskSpec


def test_oversampling_defaults_and_bounds():
    config = Config(sandbox='unused', output='unused', batch_size=4)
    config.validate()
    assert config.sampling_pool_size == 8
    config.over_sampling_batch_size = 3
    with pytest.raises(ValueError, match='at least batch_size'):
        config.validate()
    config.over_sampling_batch_size = 8
    config.rollout_max_attempts = 0
    with pytest.raises(ValueError, match='rollout_max_attempts'):
        config.validate()


def test_rejected_attempt_requeues_same_sandbox_with_new_seed_and_ignores_stale_results(tmp_path):
    schedule = DatasetSchedule([TaskSpec('a', '/a')], epochs=1, batch_size=1, seed=42)
    tasks = TaskQueue(tmp_path, schedule.jobs, 2)
    with tasks.claim() as first:
        tasks.publish({**first, 'shared_task':True, 'rollout_worker':0, 'finished_at':1,
                       'episodes':[{'seed':first['seed']}]})
    tasks.commit_consumed([], {'0':1})  # Atomic checkpoint-authorized rejection/retry.
    with tasks.claim() as second:
        assert second['task_id'] == first['task_id']
        assert second['rollout_attempt'] == 1
        assert second['seed'] != first['seed']
        tasks.publish({**second, 'shared_task':True, 'rollout_worker':1, 'finished_at':2,
                       'episodes':[{'seed':second['seed']}]})
    assert [group['rollout_attempt'] for group in tasks.ready(set())] == [1]
    class Pool:
        def raise_if_failed(self): pass
    loader = DatasetResults(schedule, queue.Queue(), Pool(), {}, rollout_workers=2,
                            timeout=1, validate=lambda _:None, task_queue=tasks)
    loader._accept({**first, 'shared_task':True, 'episodes':[{'seed':first['seed']}]})
    assert not loader.pending
    assert next(loader.ready_count(1, 1, 1))[0]['rollout_attempt'] == 1
    assert loader.completed == {0}


@pytest.mark.parametrize('exhaust', [False, True])
def test_real_training_refill_recovery_and_exhaustion(tmp_path, monkeypatch, exhaust):
    import importlib
    import os
    from pathlib import Path
    model = os.environ.get('RL_TEST_MODEL')
    if not model:
        pytest.skip('set RL_TEST_MODEL for local Metal integration')
    import mlx.core as mx
    import rl.tasks
    import rl.runtime
    import rl.evaluation
    from rl.model import Policy
    from rl.checkpoint import read_checkpoint
    training = importlib.import_module('rl.train')
    tasks = [TaskSpec(str(i), str(tmp_path / str(i)), identity=f'test-{i}') for i in range(2)]
    tasks.append(TaskSpec('eval', str(tmp_path / 'eval'), split='eval', identity='test-eval'))
    monkeypatch.setattr(rl.tasks, 'load_tasks', lambda **kw:tasks)
    class Pool:
        def __init__(self, ctx, config, version, results, stop, positions, **kw):
            self.config, self.results = config, results
            self.version = version.value
            self.sent = set()
        def start_all(self):
            self.policy = Policy(model, tuning='lora')
            self.policy.restore(Path(self.config['output']) / f'snapshots/policy-{self.version:06d}.safetensors')
        def raise_if_failed(self):
            queue_state = TaskQueue(self.config['_task_queue'], self.config['_dataset_jobs'], 4)
            for job in self.config['_dataset_jobs']:
                index = job['dataset_index']
                attempt = queue_state.attempt(index)
                if index in queue_state.consumed() or (index, attempt) in self.sent:
                    continue
                self.sent.add((index, attempt))
                seed = job['seed'] + attempt * 1000003
                decoder = self.policy.batch_decoder()
                for row in range(2):
                    decoder.add(row, [{'role':'user', 'content':'Invent a creative name for a new imaginary animal.'}], 8, 4096)
                sampled = {}
                while decoder.requests:
                    sampled.update(decoder.tick())
                rewards = [0., 0.] if index == 1 and (attempt == 0 or exhaust) else [0., 1.]
                episodes = []
                for row, reward in enumerate(rewards):
                    action = sampled[row]
                    action.update(reward=reward, terminated=True)
                    episodes.append({'task_id':job['task_id'], 'seed':seed, 'final_reward':reward,
                                     'terminated':True, 'bootstrap':0., 'actions':[action]})
                group = {**job, 'seed':seed, 'rollout_attempt':attempt, 'shared_task':True,
                         'rollout_worker':0, 'group_index':index, 'schema_version':2,
                         'task_identity':next(t.identity for t in tasks if t.id == job['task_id']),
                         'policy_identity':self.config['_policy_identity'], 'policy_version':self.version,
                         'temperature':1., 'episodes':episodes, 'finished_at':float(index + attempt * 10)}
                self.results.put(group)
        def close(self):
            self.results.close();self.results.join_thread()
        def checkpoint_state(self): return {}
    monkeypatch.setattr(rl.runtime, 'RolloutPool', Pool)
    monkeypatch.setattr(rl.evaluation, 'evaluate', lambda *args:{
        'independent_eval':False, 'by_task':{}, 'by_split':{}, 'mean_reward':0., 'episodes':[]})
    config = Config(tasks='test-only', output=str(tmp_path / 'run'), model=model, tuning='lora',
                    epochs=1, batch_size=2, mini_batch_size=2, rollout_group=2,
                    rollout_max_attempts=2, target_kl=0., max_policy_lag=10, queue_size=8)
    original = training.save_checkpoint
    def interrupt_after_refill_commit(root, policy, optimizer, state):
        original(root, policy, optimizer, state)
        if state.get('rollout_attempts') == {'1':1} and state['optimizer_step'] == 0:
            raise InterruptedError('after refill checkpoint')
    monkeypatch.setattr(training, 'save_checkpoint', interrupt_after_refill_commit)
    with pytest.raises(InterruptedError, match='refill checkpoint'):
        training.train(config)
    _, state = read_checkpoint(Path(config.output) / 'checkpoints')
    assert state['consumed_jobs'] == [0]
    assert not state['inflight_batch']['records'][0]['trained']
    monkeypatch.setattr(training, 'save_checkpoint', original)
    config.resume = True
    if exhaust:
        with pytest.raises(RuntimeError, match='oversampling exhausted 2 attempts'):
            training.train(config)
    else:
        report = training.train(config)
        assert report['completed'] and report['accepted_groups'] == 2
        assert report['sampling_attempts'] == 3 and report['optimizer_steps'] == 1
        _, state = read_checkpoint(Path(config.output) / 'checkpoints')
        assert state['consumed_jobs'] == [0, 1]
        assert state['rollout_attempts'] == {'1':1}
