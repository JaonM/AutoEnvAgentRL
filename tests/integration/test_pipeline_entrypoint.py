"""Exercise the public shell entry point without starting paid work."""
from pathlib import Path
import shlex
import subprocess

from env_factory.generation.task_generator import TaskGenerator

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'scripts/run_pipeline.sh'


def command(*arguments):
    result = subprocess.run(['bash', str(ENTRY), '--dry-run', *arguments],
                            cwd='/tmp', text=True, capture_output=True, check=True)
    return shlex.split(result.stdout)


def test_default_command_selects_complete_code_agent_pipeline():
    args = command()
    for key, value in {'--generation-backend': 'code_agent', '--generate-count': '1', '--layout': 'daily',
                       '--max-rounds': '1', '--sandbox-runtime': 'docker',
                       '--validation': 'live', '--certification-profile': 'pilot'}.items():
        assert args[args.index(key) + 1] == value
    assert TaskGenerator(None, None).generation_backend == 'code_agent'


def test_override_arguments_remain_single_values_and_take_precedence():
    args = command('--output', '/tmp/a batch', '--code-agent-model', 'test-model',
                   '--language', 'en')
    assert args[-6:] == ['--output', '/tmp/a batch', '--code-agent-model', 'test-model',
                         '--language', 'en']


def test_existing_tasks_do_not_conflict_with_generation_default():
    for flags in [('--task-ids', '1,2'), ('--task-ids=1,2',)]:
        assert '--generate-count' not in command(*flags)


def test_help_needs_no_model_or_runtime_configuration():
    result = subprocess.run(['bash', str(ENTRY), '--help'], cwd='/tmp',
                            text=True, capture_output=True, check=True)
    assert 'training_ready' in result.stdout
    assert '--code-agent-model' in result.stdout
    assert '--language' in result.stdout
    assert '--experiment-seed' not in result.stdout


def test_daily_generation_appends_task_and_builds_directly(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts import loop_experiment as loop
    old = tmp_path / 'task/task-1'
    old.mkdir(parents=True)
    (old / 'task.json').write_text('{}')
    config = {'task_paths': [], 'generate_count': 1, 'generation_backend': 'code_agent',
              'generation_hops': 2, 'code_agent_timeout': 600, 'route_attempts': 3,
              'training_mix': 'multi_step_agentic=1', 'experiment_seed': 1, 'generation_timeout': 60}
    config.update(code_agent_model='test-model', language='en')
    def generate(args, *_):
        assert args[args.index('--code-agent-model') + 1] == 'test-model'
        assert args[args.index('--language') + 1] == 'en'
        assert args[args.index('--output') + 1] == str(tmp_path)
        new = tmp_path / 'task/task-2'
        new.mkdir()
        data = b'{}'
        (new / 'task.json').write_bytes(data)
        (new / 'sample_manifest.json').write_text(json.dumps({'status': 'completed',
            'task_sha256': hashlib.sha256(data).hexdigest(), 'training_category': 'multi_step_agentic'}))
        return {'exit_code': 0}
    built = []
    def build(project, task, output, settings):
        built.append((task, output))
        output.mkdir(parents=True)
        return {'passed': True, 'training_ready': True}
    monkeypatch.setattr(loop, 'run_process', generate)
    monkeypatch.setattr(loop, 'build_one', build)
    assert loop.run_daily(ROOT, tmp_path, config) == 0
    assert built == [(tmp_path / 'task/task-2/task.json', tmp_path / 'sandbox/task-2')]
    assert (old / 'task.json').read_text() == '{}'
    assert not list(tmp_path.glob('round-*'))
    assert (tmp_path / 'sandbox/task-2/pipeline_result.json').exists()
    config.update(generate_count=0, task_paths=[str(tmp_path / 'task/task-2/task.json')])
    # Explicit recovery bypasses generation and keeps the stable sandbox path.
    monkeypatch.setattr(loop, 'run_process', lambda *_: (_ for _ in ()).throw(AssertionError('unexpected generation')))
    monkeypatch.setattr(loop, 'build_one', lambda project, task, output, settings: {'passed': False, 'training_ready': False})
    assert loop.run_daily(ROOT, tmp_path, config) == 1


def test_daily_generation_failure_never_builds_previous_task(tmp_path, monkeypatch):
    import json
    from scripts import loop_experiment as loop
    old = tmp_path / 'task/task-1'
    old.mkdir(parents=True)
    (old / 'task.json').write_text('{}')
    def fail_generation(*_):
        new = tmp_path / 'task/task-2'
        new.mkdir()
        (new / 'sample_manifest.json').write_text(json.dumps({'status': 'failed'}))
        (new / 'generation_defect.json').write_text(json.dumps({'failure_class': 'GEN_SEMANTIC'}))
        return {'exit_code': 1}
    monkeypatch.setattr(loop, 'run_process', fail_generation)
    monkeypatch.setattr(loop, 'build_one', lambda *_: (_ for _ in ()).throw(AssertionError('must not build')))
    config = {'task_paths': [], 'generate_count': 1, 'generation_backend': 'code_agent',
              'generation_hops': 2, 'code_agent_timeout': 600, 'route_attempts': 3,
              'training_mix': 'multi_step_agentic=1', 'experiment_seed': 1, 'generation_timeout': 60}
    assert loop.run_daily(ROOT, tmp_path, config) in (1, 2)
    assert not (tmp_path / 'sandbox').exists()
    assert json.loads((tmp_path / 'task/task-2/pipeline_result.json').read_text())['training_ready'] is False


def test_daily_cli_rejects_experiment_and_invalid_arguments():
    for flags in [('--experiment-seed', '1'), ('--generation-backend', 'spec'),
                  ('--max-rounds', '2'), ('--engine-help',), ('--language',),
                  ('--code-agent-model=',), ('--unknown', 'value')]:
        result = subprocess.run(['bash', str(ENTRY), '--dry-run', *flags],
                                text=True, capture_output=True)
        assert result.returncode == 2
        assert not result.stdout


def test_equals_syntax_preserves_model_and_language():
    args = command('--code-agent-model=test-model', '--language=en')
    assert args[-4:] == ['--code-agent-model', 'test-model', '--language', 'en']


def test_agent_selection_and_required_model():
    for agent in ('claude', 'opencode'):
        args = command('--code-agent', agent, '--code-agent-model', 'provider/model')
        assert args[-4:] == ['--code-agent', agent, '--code-agent-model', 'provider/model']
        result = subprocess.run(['bash', str(ENTRY), '--dry-run', '--code-agent', agent],
                                text=True, capture_output=True)
        assert result.returncode == 2
    result = subprocess.run(['bash', str(ENTRY), '--dry-run', '--code-agent', 'unknown'],
                            text=True, capture_output=True)
    assert result.returncode == 2


def test_count_cli_validation():
    for flags in [('--count', '5'), ('--count=5',)]:
        args = command(*flags)
        assert args[args.index('--generate-count') + 1] == '5'
    for flags in [('--count', '0'), ('--count', '-1'), ('--count', '1.5'),
                  ('--count', 'abc'), ('--count', '2', '--task-ids', '1')]:
        result = subprocess.run(['bash', str(ENTRY), '--dry-run', *flags],
                                text=True, capture_output=True)
        assert result.returncode == 2


def test_daily_multiple_tasks_builds_valid_deliveries_despite_partial_failure(tmp_path, monkeypatch):
    import hashlib
    import json
    from scripts import loop_experiment as loop
    config = dict(task_paths=[], generate_count=3, generation_backend='code_agent',
                  generation_hops=2, code_agent_timeout=600, route_attempts=3,
                  training_mix='multi_step_agentic=1', experiment_seed=1, generation_timeout=60)
    def generate(args, *_):
        assert args[args.index('--count') + 1] == '3'
        for index in range(1, 4):
            directory = tmp_path / 'task' / f'task-{index}'
            directory.mkdir(parents=True)
            data = b'{}'
            (directory / 'task.json').write_bytes(data)
            manifest = {'status': 'completed', 'task_sha256': hashlib.sha256(data).hexdigest(),
                        'training_category': 'multi_step_agentic'}
            if index == 2:
                manifest = {'status': 'failed', 'failure_class': 'generation', 'detail': 'invalid task'}
            (directory / 'sample_manifest.json').write_text(json.dumps(manifest))
        return {'exit_code': 1}
    built = []
    def build(project, task, output, settings):
        built.append(output)
        output.mkdir(parents=True)
        return {'passed': task.parent.name == 'task-3', 'training_ready': False}
    monkeypatch.setattr(loop, 'run_process', generate)
    monkeypatch.setattr(loop, 'build_one', build)
    assert loop.run_daily(ROOT, tmp_path, config) in (1, 2)
    events = [json.loads(line) for line in (tmp_path / 'pipeline_events.jsonl').read_text().splitlines()]
    assert events[0]['stage'] == 'pipeline_started'
    assert events[-1]['stage'] == 'pipeline_finished'
    assert len({event['run_id'] for event in events}) == 1
    assert len([event for event in events if event['stage'] == 'sandbox_finished']) == 2
    assert any(event['stage'] == 'task_generation_failed' for event in events)
    assert built == [tmp_path / 'sandbox/task-1', tmp_path / 'sandbox/task-3']
    assert (tmp_path / 'task/task-2/pipeline_result.json').exists()
    assert (tmp_path / 'sandbox/task-3/pipeline_result.json').exists()
    assert not list(tmp_path.glob('round-*'))


def test_pipeline_transcript_preserves_failure_and_dry_run_has_no_logs(tmp_path):
    import shutil
    project = tmp_path / 'project'
    (project / 'scripts').mkdir(parents=True)
    entry = project / 'scripts/run_pipeline.sh'
    shutil.copyfile(ENTRY, entry)
    runner = project / '.venv/bin/python'
    runner.parent.mkdir(parents=True)
    runner.write_text('#!/bin/bash\necho task-progress\necho task-error >&2\nexit 7\n')
    runner.chmod(0o755)
    output = tmp_path / 'output with spaces'
    args = ['bash', str(entry), '--output', str(output)]
    dry = subprocess.run([*args, '--dry-run'], text=True, capture_output=True)
    assert dry.returncode == 0
    assert not output.exists()
    for _ in range(2):
        run = subprocess.run(args, text=True, capture_output=True)
        assert run.returncode == 7
        assert 'task-progress' in run.stdout
        assert 'task-error' in run.stdout
    logs = list((output / 'logs').glob('pipeline-*'))
    assert len(logs) == 2
    for log in logs:
        text = log.read_text()
        assert 'task-progress' in text and 'task-error' in text
        assert '退出码：7' in text
