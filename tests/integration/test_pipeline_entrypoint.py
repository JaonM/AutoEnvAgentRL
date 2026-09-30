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
    for key, value in {'--generation-backend': 'code_agent', '--generate-count': '5',
                       '--max-rounds': '1', '--sandbox-runtime': 'docker',
                       '--validation': 'live', '--certification-profile': 'pilot'}.items():
        assert args[args.index(key) + 1] == value
    assert TaskGenerator(None, None).generation_backend == 'code_agent'


def test_override_arguments_remain_single_values_and_take_precedence():
    args = command('--output', '/tmp/a batch', '--generate-count', '2',
                   '--generation-backend', 'spec')
    assert args[-6:] == ['--output', '/tmp/a batch', '--generate-count', '2',
                         '--generation-backend', 'spec']


def test_existing_tasks_do_not_conflict_with_generation_default():
    for flags in [('--task-ids', '1,2'), ('--task-ids=1,2',)]:
        assert '--generate-count' not in command(*flags)


def test_help_needs_no_model_or_runtime_configuration():
    result = subprocess.run(['bash', str(ENTRY), '--help'], cwd='/tmp',
                            text=True, capture_output=True, check=True)
    assert 'training_ready' in result.stdout
    assert '--engine-help' in result.stdout
