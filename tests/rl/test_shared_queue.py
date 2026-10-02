import multiprocessing as mp
import queue
import pytest
from rl.task_queue import TaskQueue
from rl.dataset import DatasetSchedule, DatasetResults
from rl.tasks import TaskSpec


def hold_claim(root, jobs, connection):
    with TaskQueue(root, jobs, 4).claim() as job:
        connection.send(job['dataset_index'])
        connection.recv()


def test_dead_worker_lease_is_reclaimed_and_busy_worker_does_not_block(tmp_path):
    plan = DatasetSchedule([TaskSpec(str(i), '/' + str(i)) for i in range(4)], epochs=1, batch_size=2, seed=1)
    tasks = TaskQueue(tmp_path, plan.jobs, 4)
    context = mp.get_context('spawn')
    parent, child = context.Pipe()
    worker = context.Process(target=hold_claim, args=(str(tmp_path), plan.jobs, child))
    worker.start()
    try:
        assert parent.poll(5) and parent.recv() == 0
        with tasks.claim() as job:
            assert job['dataset_index'] == 1
            tasks.publish({**job, 'finished_at':1})
        with tasks.claim() as job:
            assert job['dataset_index'] == 2  # Crosses the original batch boundary.
        worker.terminate()
        worker.join(5)
        with tasks.claim() as job:
            assert job['dataset_index'] == 0
        assert tasks.ready(set())[0]['dataset_index'] == 1
        tasks.commit_consumed([1])
        assert tasks.ready(tasks.consumed()) == []
    finally:
        if worker.is_alive(): worker.terminate()
        worker.join(5)
        parent.close();child.close()


def test_global_completion_order_crosses_batch_without_losing_epoch_coverage(tmp_path):
    plan = DatasetSchedule([TaskSpec(str(i), '/' + str(i)) for i in range(4)], epochs=1, batch_size=2, seed=1)
    tasks = TaskQueue(tmp_path, plan.jobs, 4)
    class Pool:
        def raise_if_failed(self): pass
    loader = DatasetResults(plan, queue.Queue(), Pool(), {}, rollout_workers=2, timeout=1,
                            validate=lambda _:None, task_queue=tasks)
    for index in (2, 1):
        job = plan.jobs[index]
        tasks.publish({**job, 'shared_task':True, 'rollout_worker':0,
                       'finished_at':3-index, 'episodes':[{'seed':job['seed']}]})
    first = list(loader.ready_count(1, 2, 1))
    assert [g[0]['dataset_index'] for g in first] == [2, 1]
    tasks.commit_consumed(loader.completed)
    resumed = DatasetResults(plan, queue.Queue(), Pool(), {}, rollout_workers=2, timeout=1,
                             validate=lambda _:None, task_queue=tasks, completed=loader.completed)
    for index in (3, 0):
        job = plan.jobs[index]
        tasks.publish({**job, 'shared_task':True, 'rollout_worker':1,
                       'finished_at':10-index, 'episodes':[{'seed':job['seed']}]})
    assert [g['dataset_index'] for part in resumed.ready_count(1, 2, 2) for g in part] == [3, 0]
    assert resumed.completed == {0, 1, 2, 3}


def test_queue_capacity_and_epoch_admission_follow_committed_consumption(tmp_path):
    plan = DatasetSchedule([TaskSpec(str(i), '/' + str(i)) for i in range(2)], epochs=2, batch_size=1, seed=1)
    tasks = TaskQueue(tmp_path, plan.jobs, 1)
    with tasks.claim() as job:
        assert job['dataset_index'] == 0
        tasks.publish({**job, 'finished_at':1})
    with tasks.claim() as job:
        assert job is None  # Ready but unconsumed result holds back admission.
    tasks.commit_consumed([0])
    with tasks.claim() as job:
        assert job['dataset_index'] == 1
        tasks.publish({**job, 'finished_at':2})
    with tasks.claim() as job:
        assert job is None
    tasks.commit_consumed([0, 1])
    with tasks.claim() as job:
        assert job['dataset_epoch'] == 2
