"""Exercise the public shell entry point without starting paid work."""
from pathlib import Path
import shlex
import subprocess

from env_factory.generation.task_generator import TaskGenerator

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'scripts/run_pipeline_legacy.sh'


def command(*arguments):
    result = subprocess.run(['bash', str(ENTRY), '--dry-run', *arguments],
                            cwd='/tmp', text=True, capture_output=True, check=True)
    return shlex.split(result.stdout)


def test_default_command_selects_complete_code_agent_pipeline():
    args = command()
    assert args[1:3] == ['-m', 'env_factory.task_pipeline']
    assert args[args.index('--count') + 1] == '1'
    assert '--generation-backend' not in args
    assert '--layout' not in args
    assert '--max-rounds' not in args


def test_override_arguments_remain_single_values_and_take_precedence():
    args = command('--output', '/tmp/a batch', '--code-agent-model', 'test-model',
                   '--language', 'en')
    assert args[-6:] == ['--output', '/tmp/a batch', '--code-agent-model', 'test-model',
                         '--language', 'en']


def test_retired_task_ids_are_rejected():
    for flags in [('--task-ids', '1,2'), ('--task-ids=1,2',)]:
        result = subprocess.run(['bash', str(ENTRY), '--dry-run', *flags], capture_output=True)
        assert result.returncode == 2


def test_retired_realism_profile_is_rejected():
    result = subprocess.run(['bash', str(ENTRY), '--dry-run', '--realism-profile', 'recovery'],
                            text=True, capture_output=True)
    assert result.returncode == 2


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
    from env_factory import task_pipeline as loop
    old = tmp_path / 'artifacts/artifact-1/task'
    old.mkdir(parents=True)
    (old / 'task.json').write_text('{}')
    config = {'task_paths': [], 'generate_count': 1, 'generation_backend': 'code_agent',
              'generation_hops': 2, 'code_agent_timeout': 600, 'route_attempts': 3,
              'training_mix': 'multi_step_agentic=1', 'seed': 1, 'generation_timeout': 60}
    config.update(code_agent_model='test-model', language='en', training_category='multi_step_agentic')
    def generate(args, *_):
        assert args[args.index('--code-agent-model') + 1] == 'test-model'
        assert args[args.index('--language') + 1] == 'en'
        assert args[args.index('--training-category') + 1] == 'multi_step_agentic'
        assert args[args.index('--output') + 1] == str(tmp_path)
        new = tmp_path / 'artifacts/artifact-2/task'
        new.mkdir(parents=True)
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
    assert loop.run_pipeline(ROOT, tmp_path, config) == 0
    assert built == [(tmp_path / 'artifacts/artifact-2/task/task.json', tmp_path / 'artifacts/artifact-2/sandbox')]
    assert (old / 'task.json').read_text() == '{}'
    assert not list(tmp_path.glob('round-*'))
    assert (tmp_path / 'artifacts/artifact-2/sandbox/pipeline_result.json').exists()
    config.update(generate_count=0, task_paths=[str(tmp_path / 'artifacts/artifact-2/task/task.json')])
    # Explicit recovery bypasses generation and keeps the stable sandbox path.
    monkeypatch.setattr(loop, 'run_process', lambda *_: (_ for _ in ()).throw(AssertionError('unexpected generation')))
    monkeypatch.setattr(loop, 'build_one', lambda project, task, output, settings: {'passed': False, 'training_ready': False})
    assert loop.run_pipeline(ROOT, tmp_path, config) == 1


def test_daily_generation_failure_never_builds_previous_task(tmp_path, monkeypatch):
    import json
    from env_factory import task_pipeline as loop
    old = tmp_path / 'artifacts/artifact-1/task'
    old.mkdir(parents=True)
    (old / 'task.json').write_text('{}')
    def fail_generation(*_):
        new = tmp_path / 'artifacts/artifact-2/task'
        new.mkdir(parents=True)
        (new / 'sample_manifest.json').write_text(json.dumps({'status': 'failed'}))
        (new / 'generation_defect.json').write_text(json.dumps({'failure_class': 'GEN_SEMANTIC'}))
        return {'exit_code': 1}
    monkeypatch.setattr(loop, 'run_process', fail_generation)
    monkeypatch.setattr(loop, 'build_one', lambda *_: (_ for _ in ()).throw(AssertionError('must not build')))
    config = {'task_paths': [], 'generate_count': 1, 'generation_backend': 'code_agent',
              'generation_hops': 2, 'code_agent_timeout': 600, 'route_attempts': 3,
              'training_mix': 'multi_step_agentic=1', 'seed': 1, 'generation_timeout': 60}
    assert loop.run_pipeline(ROOT, tmp_path, config) in (1, 2)
    assert not (tmp_path / 'sandbox').exists()
    assert json.loads((tmp_path / 'artifacts/artifact-2/task/pipeline_result.json').read_text())['training_ready'] is False


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
        assert args[args.index('--count') + 1] == '5'
    for flags in [('--count', '0'), ('--count', '-1'), ('--count', '1.5'),
                  ('--count', 'abc'), ('--count', '2', '--task-ids', '1')]:
        result = subprocess.run(['bash', str(ENTRY), '--dry-run', *flags],
                                text=True, capture_output=True)
        assert result.returncode == 2


def test_daily_multiple_tasks_builds_valid_deliveries_despite_partial_failure(tmp_path, monkeypatch):
    import hashlib
    import json
    from env_factory import task_pipeline as loop
    config = dict(task_paths=[], generate_count=3, generation_backend='code_agent',
                  task_type="QA,Research", generation_environment_mode="stateful", generation_hops=2, user_script_count=4, code_agent_timeout=600, route_attempts=3,
                  training_mix='multi_step_agentic=1', seed=1, generation_timeout=60)
    def generate(args, *_):
        assert args[args.index('--count') + 1] == '3'
        assert args[args.index('--hops') + 1] == str(config['generation_hops'])
        assert args[args.index('--user-script-count') + 1] == '4'
        assert args[args.index('--task-type') + 1] == 'QA,Research'
        assert args[args.index('--environment-mode') + 1] == 'stateful'
        for index in range(1, 4):
            directory = tmp_path / 'artifacts' / f'artifact-{index}' / 'task'
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
        return {'passed': task.parent.parent.name == 'artifact-3', 'training_ready': False}
    monkeypatch.setattr(loop, 'run_process', generate)
    monkeypatch.setattr(loop, 'build_one', build)
    assert loop.run_pipeline(ROOT, tmp_path, config) in (1, 2)
    events = [json.loads(line) for line in (tmp_path / 'pipeline_events.jsonl').read_text().splitlines()]
    assert events[0]['stage'] == 'pipeline_started'
    assert events[-1]['stage'] == 'pipeline_finished'
    assert len({event['run_id'] for event in events}) == 1
    assert len([event for event in events if event['stage'] == 'sandbox_finished']) == 2
    assert any(event['stage'] == 'task_generation_failed' for event in events)
    assert built == [tmp_path / 'artifacts/artifact-1/sandbox', tmp_path / 'artifacts/artifact-3/sandbox']
    assert (tmp_path / 'artifacts/artifact-2/task/pipeline_result.json').exists()
    assert (tmp_path / 'artifacts/artifact-3/sandbox/pipeline_result.json').exists()
    assert not list(tmp_path.glob('round-*'))


def test_pipeline_transcript_preserves_failure_and_dry_run_has_no_logs(tmp_path):
    import shutil
    project = tmp_path / 'project'
    (project / 'scripts').mkdir(parents=True)
    entry = project / 'scripts/run_pipeline_legacy.sh'
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


def test_sol_defaults_and_independent_sandbox_override():
    from env_factory import task_pipeline as loop
    from env_factory.generation.code_agent import MODEL
    args = command()
    assert MODEL == 'gpt-6.1-sol'
    assert TaskGenerator(None).code_agent_model == 'gpt-6.1-sol'
    assert loop.sandbox_model({}) == 'gpt-6.1-sol'
    assert loop.sandbox_model({'code_agent_model': 'task-model'}) == 'gpt-6.1-sol'
    assert loop.sandbox_model({'sandbox_model': 'build-model'}) == 'build-model'
    assert loop.sandbox_model({'code_agent': 'claude', 'code_agent_model': 'claude-model'}) == 'claude-model'
    assert command('--sandbox-model=build-model')[-2:] == ['--sandbox-model', 'build-model']


def test_python_entrypoint_owns_pipeline_defaults(tmp_path, monkeypatch):
    from env_factory import task_pipeline as pipeline
    captured = {}
    def run(project, root, config):
        captured.update(config)
        assert root == tmp_path / 'output'
        return 0
    monkeypatch.setattr(pipeline, 'run_pipeline', run)
    monkeypatch.setattr(pipeline.signal, 'signal', lambda *_: None)
    assert pipeline.main(['--project', str(tmp_path)]) == 0
    assert captured['generation_hops'] == 2
    assert captured['user_script_count'] == 3
    assert captured['task_type'] is None
    assert captured['generation_environment_mode'] is None
    assert captured['generate_count'] == 1
    assert captured['code_agent_model'] == 'gpt-6.1-sol'
    assert captured['sandbox_model'] == 'gpt-6.1-sol'
    assert captured['sandbox_runtime'] == 'docker'
    assert 'validation' not in captured
    assert 'rollout_episodes' not in captured
    assert captured['generation_timeout'] == 0
    assert not {'layout', 'max_rounds', 'holdout_count', 'generation_backend'} & captured.keys()
    assert pipeline.main(['--project', str(tmp_path), '--code-agent', 'claude',
                          '--code-agent-model', 'claude-model', '--sandbox-model', 'review-model',
                          '--count', '5', '--generation-hops', '4', '--user-script-count', '5', '--task-type', 'QA, Research,QA', '--environment-mode', 'stateful']) == 0
    assert captured['generation_hops'] == 4
    assert captured['user_script_count'] == 5
    assert captured['task_type'] == 'QA,Research'
    assert captured['generate_count'] == 5
    assert captured['code_agent_model'] == 'claude-model'
    assert captured['sandbox_model'] == 'review-model'
    assert captured['generation_environment_mode'] == 'stateful'
    assert 'generation_intent' not in captured


def test_python_entrypoint_rejects_retired_modes_before_work(monkeypatch):
    import pytest
    from env_factory import task_pipeline as pipeline
    monkeypatch.setattr(pipeline, 'run_pipeline', lambda *_: pytest.fail('must reject before execution'))
    for flags in [('--layout', 'experiment'), ('--generation-backend', 'legacy'),
                  ('--generation-backend', 'spec'), ('--max-rounds', '1'),
                  ('--holdout-count', '30'), ('--count', '0'),
                  ('--generation-hops', '-1'), ('--generation-hops', '21'),
                  ('--generation-hops', '2.5'),
                  ('--user-script-count', '0'), ('--user-script-count', '-1'),
                  ('--user-script-count', '1.5'),
                  ('--task-type', 'unknown'), ('--task-type', ''),
                  ('--task-type', 'QA,'), ('--environment-mode', 'unknown'),
                  ('--count', '1', '--task-ids', '1'), ('--code-agent-timeout', 'nan')]:
        with pytest.raises(SystemExit) as exc:
            pipeline.main(list(flags))
        assert exc.value.code == 2



def test_generation_hops_shell_forwarding():
    for flags in [('--generation-hops', '4'), ('--generation-hops=4',)]:
        args = command(*flags)
        assert args[args.index('--generation-hops') + 1] == '4'


def test_user_script_count_shell_forwarding():
    for flags in [('--user-script-count', '5'), ('--user-script-count=5',)]:
        args = command(*flags)
        assert args[args.index('--user-script-count') + 1] == '5'


def test_task_type_shell_forwarding():
    for flags in [("--task-type", "QA"), ("--task-type=QA,Research",)]:
        args = command(*flags)
        assert args[args.index("--task-type") + 1] == flags[-1].split("=")[-1]


def test_environment_mode_shell_forwarding():
    for mode in ('stateless', 'reference_data', 'stateful', 'external_capability'):
        for flags in [('--environment-mode', mode), (f'--environment-mode={mode}',)]:
            args = command(*flags)
            assert args[args.index('--environment-mode') + 1] == mode


def test_build_starts_before_generation_finishes_and_waits_for_commit(tmp_path, monkeypatch):
    import hashlib
    import json
    import threading
    from env_factory import task_pipeline as pipeline
    first_built = threading.Event()
    generation_finished = threading.Event()
    config = dict(task_paths=[], generate_count=2, generation_hops=2,
                  code_agent_timeout=600, route_attempts=3,
                  training_mix='multi_step_agentic=1', seed=1, generation_timeout=60)
    def generate(*_):
        for number in (1, 2):
            directory = tmp_path / 'artifacts' / f'artifact-{number}' / 'task'
            directory.mkdir(parents=True)
            (directory / 'task.json').write_text('{}')
            manifest = {'status': 'reserved'}
            (directory / 'sample_manifest.json').write_text(json.dumps(manifest))
        first = tmp_path / 'artifacts/artifact-1/task/sample_manifest.json'
        first.write_text(json.dumps({'status': 'completed',
                                    'task_sha256': hashlib.sha256(b'{}').hexdigest()}))
        assert first_built.wait(5), 'builder waited for the full generation process'
        second = tmp_path / 'artifacts/artifact-2/task/sample_manifest.json'
        second.write_text(json.dumps({'status': 'completed',
                                     'task_sha256': hashlib.sha256(b'{}').hexdigest()}))
        generation_finished.set()
        return {'exit_code': 0}
    built = []
    def build(project, task, output, settings):
        built.append(output.parent.name)
        if output.parent.name == 'artifact-1':
            assert not generation_finished.is_set()
            first_built.set()
        else:
            assert generation_finished.is_set()
        output.mkdir()
        return {'passed': True, 'training_ready': True}
    monkeypatch.setattr(pipeline, 'run_process', generate)
    monkeypatch.setattr(pipeline, 'build_one', build)
    assert pipeline.run_pipeline(ROOT, tmp_path, config) == 0
    assert built == ['artifact-1', 'artifact-2']


def test_multistep_cohort_cli_preserves_category_and_seed(tmp_path, monkeypatch):
    from env_factory import task_pipeline
    args = command('--training-category', 'multi_step_agentic', '--seed', '20261007')
    def run(project, root, config):
        assert config['training_category'] == 'multi_step_agentic'
        assert config['seed'] == 20261007
        return 0
    monkeypatch.setattr(task_pipeline, 'run_pipeline', run)
    assert task_pipeline.main(args[3:] + ['--project', str(tmp_path)]) == 0
