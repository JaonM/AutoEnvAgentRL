"""Actor assignment follows the finite dataset schedule after recovery."""
from dataclasses import asdict
from types import SimpleNamespace
import threading
import pytest


def test_worker_claims_remaining_shared_jobs_and_keeps_group_seed(tmp_path, monkeypatch):
    pytest.importorskip('mlx.core', exc_type=ImportError)
    import rl.rollout_worker as actor
    import rl.model as model
    from rl.dataset import DatasetSchedule
    from rl.tasks import TaskSpec
    from rl.checkpoint import atomic_json
    from rl.train import Config

    tasks = [TaskSpec(str(i), '/' + str(i)) for i in range(3)]
    plan = DatasetSchedule(tasks, epochs=2, batch_size=2, seed=42)
    stop = threading.Event()
    received, restored, environments = [], [], []
    limit = tmp_path / 'limit.json'
    atomic_json(limit, {'version': len(plan.jobs)})

    class Policy:
        def __init__(self, *args, **kwargs): pass
        def restore(self, path): restored.append(path.name)

    class Environment:
        def __init__(self, path, workers, **kwargs):
            self.closed = False
            environments.append(self)
        def close(self): self.closed = True

    from rl.task_queue import TaskQueue
    task_queue = TaskQueue(tmp_path / 'queue', plan.jobs, 4)
    task_queue.commit_consumed([0, 1, 2])

    class Results:
        def put_nowait(self, group):
            received.append(group)
            task_queue.commit_consumed([0, 1, 2] + [g['dataset_index'] for g in received])
            if group['dataset_index'] == 5:
                stop.set()

    monkeypatch.setattr(model, 'Policy', Policy)
    monkeypatch.setattr(actor, 'EnvironmentPool', Environment)
    monkeypatch.setattr(actor, 'parallel_rollouts', lambda policy, env, config, seed, **kwargs:
                        [{'seed': seed, 'actions': [{'prompt': [1], 'tokens': [2], 'old_logp': [-.5]}]} for _ in range(config['rollout_group'])])
    config = {**asdict(Config(sandbox='unused', output=str(tmp_path), rollout_workers=2, rollout_group=3)),
              '_tasks': [asdict(task) for task in tasks], '_dataset_jobs': plan.jobs,
              '_task_queue': str(task_queue.root), '_queue_capacity': 4}
    actor.rollout_worker_main(1, config, SimpleNamespace(value=7), Results(), stop)
    assert [g['dataset_index'] for g in received] == [3, 4, 5]
    assert restored == ['policy-000007.safetensors']
    assert all(len(g['episodes']) == 3 and all(e['seed'] == plan.jobs[g['dataset_index']]['seed']
                                               for e in g['episodes']) for g in received)
    assert all(env.closed for env in environments)
