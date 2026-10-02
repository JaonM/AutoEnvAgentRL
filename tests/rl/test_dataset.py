import queue
from collections import Counter
import pytest
from rl.dataset import DatasetSchedule, DatasetResults
from rl.tasks import TaskSpec


def tasks():
    return [TaskSpec(str(i), '/' + str(i)) for i in range(5)] + [TaskSpec('eval', '/eval', split='eval')]


def test_exact_coverage_each_epoch_tail_batches_and_actor_sharding():
    plan = DatasetSchedule(tasks(), epochs=3, batch_size=2, seed=42)
    assert [len(batch) for batch in plan.batches] == [2, 2, 1] * 3
    for epoch in range(1, 4):
        assert Counter(job['task_id'] for job in plan.jobs if job['dataset_epoch'] == epoch) == Counter(str(i) for i in range(5))
    assert plan.jobs == DatasetSchedule(tasks(), epochs=3, batch_size=2, seed=42).jobs
    extended = DatasetSchedule(tasks(), epochs=4, batch_size=2, seed=42)
    assert extended.jobs[:15] == plan.jobs
    assert plan.prefetch_limit(0) == 4
    assert plan.prefetch_limit(2) == 7
    assignments = [list(range(actor, len(plan.jobs), 3)) for actor in range(3)]
    assert sorted(sum(assignments, [])) == list(range(15))


class Pool:
    def __init__(self): self.positions = {}
    def raise_if_failed(self): pass
    def record_position(self, actor, position): self.positions[str(actor)] = position


def result(job):
    index = job['dataset_index']
    return {**job, 'rollout_worker': index % 2, 'group_index': index // 2,
            'rollout_rng_after': [index], 'episodes': [{'seed': job['seed']}]}


def test_out_of_order_results_resume_cursor_and_recovery_duplicates():
    plan = DatasetSchedule(tasks(), epochs=1, batch_size=2, seed=42)
    incoming, pool = queue.Queue(), Pool()
    for index in (1, 0, 1, 3, 2, 4):
        incoming.put(result(plan.jobs[index]))
    loader = DatasetResults(plan, incoming, pool, pool.positions, rollout_workers=2, timeout=1, validate=lambda _: None)
    assert [loader.take(job)['dataset_index'] for job in plan.jobs] == list(range(5))
    assert pool.positions == {'0': {'next_group': 3, 'rng': [4]}, '1': {'next_group': 2, 'rng': [3]}}
    resumed = DatasetResults(plan, incoming, pool, pool.positions, rollout_workers=2, timeout=1,
                             validate=lambda _: None, next_index=4)
    incoming.put(result(plan.jobs[3]))  # Old in-flight result cannot count again.
    incoming.put(result(plan.jobs[4]))
    assert resumed.take(plan.jobs[4])['dataset_index'] == 4


def test_wrong_sandbox_cannot_count_toward_epoch():
    plan = DatasetSchedule(tasks(), epochs=1, batch_size=2, seed=42)
    incoming, pool = queue.Queue(), Pool()
    incoming.put({**result(plan.jobs[0]), 'task_id': 'wrong'})
    loader = DatasetResults(plan, incoming, pool, pool.positions, rollout_workers=2, timeout=1, validate=lambda _: None)
    with pytest.raises(ValueError, match='scheduled sandbox'):
        loader.take(plan.jobs[0])


def test_ready_minibatch_trains_without_waiting_for_first_sandbox():
    plan = DatasetSchedule(tasks(), epochs=1, batch_size=2, seed=42)
    incoming, pool = queue.Queue(), Pool()
    incoming.put(result(plan.jobs[1]))  # Dataset index 0 is still running.
    loader = DatasetResults(plan, incoming, pool, pool.positions, rollout_workers=2, timeout=1, validate=lambda _: None)
    iterator = loader.ready_minibatches(plan.batches[0], 1)
    assert next(iterator)[0]['dataset_index'] == 1
    assert pool.positions == {'1': {'next_group': 1, 'rng': [1]}}
    # Resume a checkpoint after the ready mini-batch: index 1 cannot train twice.
    resumed = DatasetResults(plan, incoming, pool, pool.positions, rollout_workers=2, timeout=1,
                             validate=lambda _: None, completed=[1])
    incoming.put(result(plan.jobs[1]))
    incoming.put(result(plan.jobs[0]))
    assert [g['dataset_index'] for part in resumed.ready_minibatches(plan.batches[0], 1) for g in part] == [0]
