import hashlib
import json
from pathlib import Path
import threading
from concurrent.futures import CancelledError

import pytest

from env_factory import task_pipeline as pipeline


ROOT = Path(__file__).resolve().parents[2]


def config(**values):
    return dict(task_paths=[], generate_count=4, generation_hops=2,
        code_agent_timeout=600, route_attempts=3, training_mix='multi_step_agentic=1',
        seed=17, generation_timeout=60, generation_workers=3, build_workers=2, **values)


def commit(root, number, *, valid=True):
    directory = root / 'artifacts' / f'artifact-{number}' / 'task'
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'task.json').write_text('{}')
    pipeline.write_json(directory / 'sample_manifest.json', {'status': 'completed',
        'task_sha256': hashlib.sha256(b'{}' if valid else b'incomplete').hexdigest()})
    return directory / 'task.json'


def test_bounded_parallel_builds_commit_order_and_failure_accounting(tmp_path, monkeypatch):
    first_two = threading.Barrier(2, timeout=5)
    both_started = threading.Event()
    guard = threading.Lock()
    active = maximum = 0
    built = []

    def generate(command, *_):
        assert command[command.index('--max-workers') + 1] == '3'
        # artifact-1 is still incomplete while later tasks are ready.
        commit(tmp_path, 1, valid=False)
        commit(tmp_path, 2)
        commit(tmp_path, 3)
        assert both_started.wait(5), 'builds did not overlap generation'
        commit(tmp_path, 1)
        directory = tmp_path / 'artifacts/artifact-4/task'
        directory.mkdir(parents=True)
        pipeline.write_json(directory / 'failure.json', {'message': 'author rejected task',
                                                       'failure_class': 'GEN_SEMANTIC'})
        pipeline.write_json(directory / 'sample_manifest.json', {'status': 'failed'})
        return {'exit_code': 1}

    def build(project, task, output, settings):
        nonlocal active, maximum
        number = int(output.parent.name.split('-')[-1])
        with guard:
            active += 1
            maximum = max(maximum, active)
            built.append(number)
        try:
            if number in (2, 3):
                first_two.wait()
                both_started.set()
            assert settings['route_attempts'] == 3
            output.mkdir()
            return {'passed': number != 3, 'training_ready': number != 3}
        finally:
            with guard:
                active -= 1

    monkeypatch.setattr(pipeline, 'run_process', generate)
    monkeypatch.setattr(pipeline, 'build_one', build)
    assert pipeline.run_pipeline(ROOT, tmp_path, config()) == 1
    assert maximum == 2
    assert sorted(built) == [1, 2, 3]
    events = [json.loads(line) for line in (tmp_path / 'pipeline_events.jsonl').read_text().splitlines()]
    assert events[-1]['stage'] == 'pipeline_finished'
    assert events[-1]['built'] == 3
    assert len([e for e in events if e['stage'] == 'sandbox_finished']) == 3
    for number in (1, 2, 3):
        result = json.loads((tmp_path / f'artifacts/artifact-{number}/sandbox/pipeline_result.json').read_text())
        assert result['training_ready'] is (number != 3)
    assert json.loads((tmp_path / 'artifacts/artifact-4/task/pipeline_result.json').read_text())['training_ready'] is False


def test_worker_exception_cancels_later_launches_and_does_not_start_queued_tasks(tmp_path, monkeypatch):
    rendezvous = threading.Barrier(2, timeout=5)
    built = []
    cancelled_launch = threading.Event()
    real_run = pipeline.run_process
    def generate(*_):
        for number in range(1, 5):
            commit(tmp_path, number)
        return {'exit_code': 0}
    def build(project, task, output, settings):
        number = int(output.parent.name.split('-')[-1])
        built.append(number)
        rendezvous.wait()
        if number == 1:
            raise SystemExit('fatal worker failure')
        assert pipeline.PROCESS_CONTEXT.cancelled.wait(5)
        with pytest.raises(CancelledError):
            real_run(['must-never-launch'], tmp_path, tmp_path / 'cancelled.log', 1)
        cancelled_launch.set()
        return {'passed': False}
    monkeypatch.setattr(pipeline, 'run_process', generate)
    monkeypatch.setattr(pipeline, 'build_one', build)
    with pytest.raises(SystemExit, match='fatal worker failure'):
        pipeline.run_pipeline(ROOT, tmp_path, config())
    assert sorted(built) == [1, 2]
    assert cancelled_launch.is_set()


def test_one_worker_exception_preserves_other_tasks_and_full_denominator(tmp_path, monkeypatch):
    def generate(*_):
        for number in range(1, 5):
            commit(tmp_path, number)
        return {'exit_code': 0}
    built = []
    def build(project, task, output, settings):
        built.append(output.parent.name)
        if output.parent.name == 'artifact-2':
            raise ValueError('local worker error')
        return {'passed': True, 'training_ready': True}
    monkeypatch.setattr(pipeline, 'run_process', generate)
    monkeypatch.setattr(pipeline, 'build_one', build)
    assert pipeline.run_pipeline(ROOT, tmp_path, config()) == 2
    assert len(built) == 4
    result = json.loads((tmp_path / 'artifacts/artifact-2/sandbox/pipeline_result.json').read_text())
    assert result['failure_code'] == 'INFRA'
    assert result['training_ready'] is False


@pytest.mark.parametrize('flag', ['--generation-workers', '--build-workers'])
def test_worker_counts_reject_nonpositive_before_running(flag, monkeypatch):
    monkeypatch.setattr(pipeline, 'run_pipeline', lambda *_: pytest.fail('invalid count launched pipeline'))
    with pytest.raises(SystemExit) as failure:
        pipeline.main([flag, '0'])
    assert failure.value.code == 2


def test_default_serial_and_explicit_parallel_cli(monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, 'run_pipeline', lambda project, root, settings: seen.append(settings) or 0)
    assert pipeline.main([]) == 0
    assert (seen[-1]['generation_workers'], seen[-1]['build_workers']) == (1, 1)
    assert pipeline.main(['--generation-workers', '3', '--build-workers', '2']) == 0
    assert (seen[-1]['generation_workers'], seen[-1]['build_workers']) == (3, 2)
