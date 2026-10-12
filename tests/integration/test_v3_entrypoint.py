"""The V3 shell entry point, without starting paid work."""
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / 'scripts/run_pipeline.sh'


def run(*arguments):
    return subprocess.run(['bash', str(ENTRY), *arguments], cwd='/tmp', text=True, capture_output=True)


def command(*arguments):
    result = run('--dry-run', *arguments)
    assert result.returncode == 0, result.stderr
    return shlex.split(result.stdout)


def value(args, flag):
    return args[args.index(flag) + 1]


def test_defaults_are_the_trainable_standard():
    args = command()
    assert args[1:4] == ['-m', 'env_factory_v3', 'run']
    assert (value(args, '--count'), value(args, '--domains'), value(args, '--split')) == ('1', '1', 'dev')
    assert '--docker' in args and value(args, '--policy-k') == '4'
    assert value(args, '--output').startswith('output/v3_run_')


def test_overrides_stay_single_values():
    args = command('--count=30', '--output', '/tmp/a batch', '--domains-file', 'd.json', '--policy-k', '0',
                   '--no-docker', '--rubric-share', '0.2')
    assert value(args, '--output') == '/tmp/a batch'
    assert (value(args, '--count'), value(args, '--domains')) == ('30', '12')
    assert '--split' not in args and '--docker' not in args
    assert args.count('--policy-k') == 1 and value(args, '--policy-k') == '0'
    assert value(args, '--rubric-share') == '0.2'


def test_invalid_arguments_exit_2_without_a_command():
    for flags in [('--count', '0'), ('--count', 'x'), ('--domains', '-1'), ('--model',), ('--model=',),
                  ('--unknown', 'v'), ('--code-agent', 'codex')]:
        result = run('--dry-run', *flags)
        assert result.returncode == 2 and not result.stdout, flags


def test_help_needs_no_configuration():
    result = run('--help')
    assert result.returncode == 0
    assert 'tasks.json' in result.stdout and '--rubric-share' in result.stdout and 'legacy' not in result.stdout
