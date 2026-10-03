"""Real Metal optimizer/checkpoint tests with controlled, local rollout rewards."""
import json
import os
from pathlib import Path
import pytest

MODEL = os.environ.get('RL_TEST_MODEL')
pytestmark = pytest.mark.skipif(not MODEL, reason='set RL_TEST_MODEL for local Metal integration')


@pytest.mark.parametrize('algorithm', ['ppo', 'grpo'])
def test_training_minibatches_and_critic_checkpoint(tmp_path, monkeypatch, algorithm, *, tuning='lora', model_path=None, thinking_mode='auto', qat_scope='projections', config_overrides=None, deferred_failure=False):
    model_path = model_path or MODEL
    import mlx.core as mx
    from rl.model import Policy
    from rl.model_options import policy_options
    from rl.train import Config, train
    from rl.tasks import TaskSpec
    from rl.checkpoint import read_checkpoint, restore_checkpoint
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    import rl.tasks
    import rl.runtime
    import rl.evaluation

    tasks = [TaskSpec(str(i), str(tmp_path / str(i)), identity=f'synthetic-test-{i}') for i in range(2)]
    tasks.append(TaskSpec('eval', str(tmp_path / 'eval'), split='eval', identity='synthetic-eval'))
    monkeypatch.setattr(rl.tasks, 'load_tasks', lambda **kw: tasks)
    captured = []

    class LocalPool:
        def __init__(self, ctx, config, version, results, stop, positions, **kw):
            self.config, self.results, self.positions = config, results, positions
            self.version = version.value

        def start_all(self):
            config = self.config
            actor = Policy(model_path, tuning=tuning, layers=config['layers'], rank=config['rank'], bits=config['bits'], critic=algorithm == 'ppo', **policy_options(config, inference=True))
            actor.restore(Path(config['output']) / f'snapshots/policy-{self.version:06d}.safetensors')
            start = self.positions.get('0', {}).get('next_group', 0)
            for index, job in enumerate(config['_dataset_jobs'][start:], start):
                task = next(task for task in tasks if task.id == job['task_id'])
                episodes = []
                decoder = actor.batch_decoder()
                for row in range(2):
                    decoder.add(row, [{'role': 'user', 'content': f'Invent a short name and description for a fictional animal from planet {index}.'}], 16, 4096)
                samples = {}
                while decoder.requests:
                    samples.update(decoder.tick())
                for row, reward in enumerate((0., 1.)):
                    sample = samples[row]
                    sample.update(reward=reward, terminated=True)
                    episodes.append({'task_id': task.id, 'seed': job['seed'], 'final_reward': reward,
                        'terminated': True, 'bootstrap': 0., 'actions': [sample]})
                group = {**job, 'schema_version': 2, 'rollout_worker': 0, 'group_index': index,
                    'task_id': task.id, 'task_identity': task.identity, 'policy_version': self.version,
                    'rollout_rng_after': [key.tolist() for key in mx.random.state],
                    'policy_identity': config['_policy_identity'], 'temperature': 1.,
                    'episodes': episodes}
                captured.append(group)
                self.results.put(group)

        def raise_if_failed(self): pass
        def close(self):
            self.results.close()
            self.results.join_thread()
        def record_position(self, actor, position): self.positions[str(actor)] = position
        def checkpoint_state(self): return {'rollout_positions': self.positions}

    monkeypatch.setattr(rl.runtime, 'RolloutPool', LocalPool)
    # Evaluation deliberately stubbed: this test measures optimization, not task ability.
    monkeypatch.setattr(rl.evaluation, 'evaluate', lambda *args: {
        'independent_eval': False, 'by_task': {}, 'by_split': {}, 'mean_reward': 0., 'episodes': []})
    config = Config(tasks=str(tmp_path / 'manifest.json'), output=str(tmp_path / algorithm), model=model_path,
                    algorithm=algorithm, tuning=tuning, thinking_mode=thinking_mode, qat_scope=qat_scope, rollout_group=2, batch_size=2, mini_batch_size=1, epochs=2,
                    queue_size=8, max_policy_lag=10, target_kl=0., max_tokens=4,
                    eval_interval_steps=2, group_artifacts_keep=2, replay_capacity=1, checkpoint_keep=2)
    for key, value in (config_overrides or {}).items():
        setattr(config, key, value)
    report = train(config)
    assert report['completed'] and report['optimizer_steps'] == 4
    assert report['critic_enabled'] == (algorithm == 'ppo')
    assert report['max_physical_batch_observed'] >= 2
    assert report['timings']['seconds']['actor_forward_backward'] > 0
    assert report['role_names']['actor'] == 'policy trainer'
    assert report['epochs_completed'] == 2 and report['fresh_groups'] == 4
    assert all(sorted(g['task_id'] for g in captured if g['dataset_epoch'] == epoch) == ['0', '1']
               for epoch in (1, 2))
    directory, state = read_checkpoint(Path(config.output) / 'checkpoints')
    tensors = mx.load(str(directory / 'policy.safetensors'))
    critic = {k: v for k, v in tensors.items() if k.startswith('critic.')}
    if algorithm == 'ppo':
        assert critic and any(bool(mx.any(v != 0)) for v in critic.values())
        assert all('old_values' in e['actions'][0] for g in captured for e in g['episodes'])
    else:
        assert not critic
        assert all('old_values' not in e['actions'][0] for g in captured for e in g['episodes'])
    records = json.loads((Path(config.output) / 'optimizer_metrics.json').read_text())
    assert [(r['epoch'], r['mini_batch']) for r in records] == [(1, 1), (1, 2), (2, 1), (2, 2)]
    assert all(r['mini_batch_sandboxes'] == 1 and r['mini_batch_rollouts'] == 2 for r in records)
    assert state['dataset_cursor'] == 2
    assert all(r['value_loss'] == 0 for r in records) if algorithm == 'grpo' else any(r['value_loss'] > 0 for r in records)
    restored = Policy(model_path, tuning=tuning, layers=config.layers, rank=config.rank, bits=config.bits, critic=algorithm == 'ppo', **policy_options(config))
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    optimizer.init(restored.trainable_parameters())
    saved = restore_checkpoint(Path(config.output) / 'checkpoints', restored, optimizer)
    assert saved['optimizer_step'] == 4
    actual = dict(tree_flatten(restored.trainable_parameters()))
    assert actual.keys() == tensors.keys()
    assert all(bool(mx.all(actual[k] == tensors[k])) for k in tensors)
    assert int(optimizer.state['step']) == 4
    # Parameter layouts intentionally reject loading PPO checkpoints into GRPO and vice versa.
    wrong = Policy(model_path, tuning=tuning, layers=config.layers, rank=config.rank, bits=config.bits, critic=algorithm != 'ppo', **policy_options(config))
    with pytest.raises(ValueError, match='parameters do not match'):
        restore_checkpoint(Path(config.output) / 'checkpoints', wrong, optimizer)

    del restored, wrong, optimizer, tensors, actual, critic
    import gc
    gc.collect()
    mx.clear_cache()
    config.resume = True
    config.epochs = 3
    if algorithm == 'grpo' and deferred_failure:
        from rl.actor import ActorTrainer
        original_step = ActorTrainer.step
        def interrupt_uncommitted(self, *args, **kwargs):
            result = original_step(self, *args, **kwargs)
            if int(self.optimizer.state['step']) == 5:
                raise RuntimeError('injected uncommitted optimizer interruption')
            return result
        monkeypatch.setattr(ActorTrainer, 'step', interrupt_uncommitted)
        with pytest.raises(RuntimeError, match='uncommitted optimizer interruption'):
            train(config)
        _, partial = read_checkpoint(Path(config.output) / 'checkpoints')
        assert partial['optimizer_step'] == 4 and partial['dataset_cursor'] == 2
        monkeypatch.setattr(ActorTrainer, 'step', original_step)
        gc.collect()
        mx.clear_cache()
    elif algorithm == 'grpo':
        import rl.train as trainer_module
        original_save = trainer_module.save_checkpoint
        def fail_after_durable_mini_batch(root, policy, optimizer, state):
            value = original_save(root, policy, optimizer, state)
            if state['optimizer_step'] == 5:
                raise RuntimeError('injected interruption after mini-batch commit')
            return value
        monkeypatch.setattr(trainer_module, 'save_checkpoint', fail_after_durable_mini_batch)
        with pytest.raises(RuntimeError, match='injected interruption'):
            train(config)
        _, partial = read_checkpoint(Path(config.output) / 'checkpoints')
        assert partial['optimizer_step'] == 5 and partial['dataset_cursor'] == 2
        assert len(partial['inflight_batch']['records']) == 1
        monkeypatch.setattr(trainer_module, 'save_checkpoint', original_save)
    resumed_report = train(config)
    assert resumed_report['optimizer_steps'] == 6
    assert resumed_report['fresh_groups'] == 6 and resumed_report['epochs_completed'] == 3
    committed = [json.loads(path.read_text()) for path in Path(config.output).glob('group-*.json')]
    assert sorted(g['task_id'] for g in committed if g['dataset_epoch'] == 3) == ['0', '1']
    final_steps = json.loads((Path(config.output) / 'optimizer_metrics.json').read_text())
    assert [r['optimizer_step'] for r in final_steps] == list(range(1, 7))
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    dashboard = EventAccumulator(str(Path(config.output) / 'tensorboard/train')).Reload()
    assert [point.step for point in dashboard.Scalars('train/loss')] == list(range(1, 7))
    entropy = dashboard.Scalars('train/policy_entropy')
    assert [point.step for point in entropy] == list(range(1, 7))
    assert all(0 < point.value < 20 for point in entropy)
    assert [point.value for point in entropy] == pytest.approx([r['policy_entropy'] for r in final_steps])
    groups_dashboard = EventAccumulator(str(Path(config.output) / 'tensorboard/rollout_groups')).Reload()
    assert len(groups_dashboard.Scalars('rollout/reward_mean')) == 6
    assert all(point.value == .5 for point in groups_dashboard.Scalars('rollout/reward_mean'))
    assert 'eval/after/reward_mean' in dashboard.Tags()['scalars']
    assert [point.step for point in dashboard.Scalars('eval/periodic/reward_mean')] == [0, 2, 4, 6]
    _, latest_state = read_checkpoint(Path(config.output) / 'checkpoints')
    assert 'metrics' not in latest_state and 'optimizer_metrics' not in latest_state
    assert latest_state['metric_journals']['optimizer_metrics']['count'] == 6
    assert latest_state['eval_state']['last_step'] == 6
    assert len(latest_state['updated_jobs']) == 6
    assert (Path(config.output) / 'best_policy.json').exists()
    assert len(list(Path(config.output).glob('group-*.json'))) < 6
