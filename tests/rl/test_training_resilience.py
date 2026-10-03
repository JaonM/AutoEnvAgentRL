import json
import queue
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl.admission import rejection_action, zero_variance_kind
from rl.checkpoint import atomic_json
from rl.dataset import DatasetResults, DatasetSchedule
from rl.journal import MetricJournal
from rl.tasks import TaskSpec
from rl.timing import rollout_training_overlap


def test_result_deadline_ignores_duplicate_notifications():
    jobs = DatasetSchedule([TaskSpec('a', '/a')], epochs=1, batch_size=1, seed=1)
    pool = SimpleNamespace(raise_if_failed=lambda: None, diagnostics=lambda: {'worker': 'alive, waiting'})
    loader = DatasetResults(jobs, queue.Queue(), pool, {}, rollout_workers=1,
                            timeout=.005, progress_timeout=.02, validate=lambda _: None)
    with pytest.raises(TimeoutError, match='waiting=\\[0\\].*alive, waiting'):
        next(loader.ready_count(1, 1, 1))


def test_group_deadline_fires_even_with_fresh_heartbeat(tmp_path):
    from rl.runtime import RolloutPool
    heartbeat = tmp_path / 'rollout_workers/worker-0.json'
    atomic_json(heartbeat, {'at': time.time(), 'stage': 'batched-policy-decode',
                            'group_started_at': time.time() - 100})
    pool = RolloutPool(None, {'output': str(tmp_path), 'rollout_timeout': 10, 'group_timeout': 50},
                       None, None, None, {})
    pool.processes = {0: SimpleNamespace(is_alive=lambda: True)}
    restarts = []
    pool.restart = lambda index, reason: restarts.append(reason)
    pool.check()
    assert restarts == ['group_deadline_exceeded']


def test_model_loading_uses_separate_deadline(tmp_path):
    from rl.runtime import RolloutPool
    atomic_json(tmp_path / 'rollout_workers/worker-0.json',
                {'at': time.time() - 20, 'stage': 'loading_model'})
    pool = RolloutPool(None, {'output': str(tmp_path), 'rollout_timeout': 10, 'model_load_timeout': 50},
                       None, None, None, {})
    pool.processes = {0: SimpleNamespace(is_alive=lambda: True)}
    pool.restart = lambda *args: pytest.fail('model is within its load deadline')
    pool.check()


def test_admission_policies_do_not_call_solved_tasks_training_failures():
    assert zero_variance_kind([{'terminated': True, 'final_reward': 1.}] * 2) == 'all_success'
    assert zero_variance_kind([{'terminated': False, 'final_reward': 0.}] * 2) == 'all_failure'
    assert zero_variance_kind([{'terminated': True, 'final_reward': .5}] * 2) == 'equal_reward'
    for policy, expected in [('retry_skip', 'skip'), ('retry_fail', 'fail'), ('skip', 'skip'), ('fail', 'fail')]:
        assert rejection_action('zero_reward_variance', policy, 2, 3) == expected
    assert rejection_action('zero_reward_variance', 'retry_skip', 0, 3) == 'retry'
    assert rejection_action('importance_drift', 'retry_skip', 2, 3) == 'fail'
    assert rejection_action('context_limit_empty', 'retry_skip', 0, 3) == 'skip'


def test_metric_journal_recovers_committed_prefix_and_preserves_tail(tmp_path):
    log = MetricJournal(tmp_path, 'metrics')
    log.append({'step': 1, 'text': '中文'})
    committed = log.cursor()
    log.append({'step': 2, 'loss': 123})
    log.close()
    with (tmp_path / 'metrics.jsonl').open('ab') as stream:
        stream.write(b'{"torn":')
    resumed = MetricJournal(tmp_path, 'metrics', cursor=committed)
    assert list(resumed) == [{'step': 1, 'text': '中文'}]
    tail = next((tmp_path / 'recovery').glob('*.jsonl')).read_bytes()
    assert b'123' in tail and tail.endswith(b'{"torn":')
    resumed.append({'step': 2, 'loss': 456})
    resumed.export()
    assert json.loads((tmp_path / 'metrics.json').read_text())[-1]['loss'] == 456
    resumed.close()
    content = (tmp_path / 'metrics.jsonl').read_bytes().replace(b'"step": 1', b'"step": 9')
    (tmp_path / 'metrics.jsonl').write_bytes(content)
    with pytest.raises(ValueError, match='checksum'):
        MetricJournal(tmp_path, 'metrics', cursor=committed)


def test_overlap_uses_intervals_not_obsolete_artifact_directory():
    records = [{'event': 'rollout_received', 'started_at': 1, 'finished_at': 4},
               {'event': 'actor_optimizer_step', 'started_at': 3, 'finished_at': 5}]
    assert rollout_training_overlap(records)
    records[1]['started_at'] = 4
    assert not rollout_training_overlap(records)
    assert not rollout_training_overlap([{'event': 'rollout_received', 'started_at': None}])


def test_live_pruning_keeps_leases_replay_continuations_and_checkpoint_inputs(tmp_path):
    from rl.process_control import pin_policy
    from rl.retention import prune_live_artifacts
    from rl.task_queue import TaskQueue
    snapshots = tmp_path / 'snapshots'
    snapshots.mkdir()
    for version in range(7):
        (snapshots / f'policy-{version:06d}.safetensors').write_bytes(b'weights')
    groups = []
    for index in range(1, 7):
        path = tmp_path / f'group-{index:04d}.json'
        path.write_text('{}')
        groups.append(path)
    state = {'updates': 5, 'consumed_jobs': [0], 'inflight_batch': {'records': [{'path': str(groups[0])}]},
             'replay': {'entries': [{'path': str(groups[1])}]}}
    atomic_json(tmp_path / 'checkpoints/step-1/state.json', state)
    tasks = TaskQueue(tmp_path / 'task_queue', [], 2)
    atomic_json(tasks.result_path(0), {})
    atomic_json(tasks.root / 'continuation-0-attempt-0/group.json', {'policy_version': 1})
    atomic_json(tasks.root / 'continuation-1-attempt-0/group.json', {'policy_version': 2})
    with pin_policy(snapshots, lambda: 3):
        prune_live_artifacts(tmp_path, tasks, 6, keep=1, group_keep=1)
        assert (snapshots / 'policy-000003.safetensors').exists()
    assert not tasks.result_path(0).exists()
    assert not (tasks.root / 'continuation-0-attempt-0').exists()
    assert (tasks.root / 'continuation-1-attempt-0').exists()
    assert {p.name for p in tmp_path.glob('group-*.json')} == {groups[i].name for i in (0, 1, 5)}
    assert {int(p.stem.split('-')[1]) for p in snapshots.glob('*.safetensors')} == {0, 2, 3, 5, 6}
    prune_live_artifacts(tmp_path, tasks, 6, keep=1, group_keep=1)
    assert not (snapshots / 'policy-000003.safetensors').exists()


def test_periodic_evaluation_uses_heldout_and_best_pointer_is_committed(tmp_path):
    from rl.evaluation import PeriodicEvaluation
    tasks = [TaskSpec('train', '/train'), TaskSpec('eval', '/eval', split='eval')]
    config = SimpleNamespace(eval_interval_steps=2)
    observed = []
    monitor = SimpleNamespace(evaluation=lambda *args: observed.append(args))
    class Policy:
        def save_snapshot(self, root, version):
            root.mkdir(exist_ok=True)
            path = root / f'policy-{version:06d}.safetensors'
            path.write_bytes(str(version).encode())
            return str(path)
    def evaluate(policy, selected, cfg):
        assert [task.id for task in selected] == ['eval']
        return {'mean_reward': .5, 'episodes': []}
    evaluator = PeriodicEvaluation(tmp_path, tasks, config, monitor)
    assert evaluator.run(Policy(), 0, evaluator=evaluate)
    assert not (tmp_path / 'best_policy.json').exists()
    evaluator.publish_best()
    saved = dict(evaluator.state)
    assert not evaluator.run(Policy(), 1, evaluator=evaluate)
    assert evaluator.run(Policy(), 2, evaluator=evaluate)
    assert evaluator.state['best']['optimizer_step'] == 0  # A tie does not replace best.
    resumed = PeriodicEvaluation(tmp_path, tasks, config, monitor, saved)
    assert resumed.state['last_step'] == 0
    assert json.loads((tmp_path / 'best_policy.json').read_text())['optimizer_step'] == 0
    assert [item[2] for item in observed] == [0, 2]


def test_live_plot_reader_prefers_journal_and_ignores_partial_tail(tmp_path):
    from rl.plot_metrics import load_metric_rows
    (tmp_path / 'metrics.json').write_text('[{"step": 99}]')
    (tmp_path / 'metrics.jsonl').write_bytes(b'{"step": 1}\n{"step":')
    assert load_metric_rows(tmp_path, 'metrics') == [{'step': 1}]
