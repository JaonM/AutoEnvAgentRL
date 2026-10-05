"""Compatibility and source-identity guards for the package refactor."""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('legacy,canonical', [
    ('train', 'trainer.runner'),
    ('model', 'backends.mlx.policy'),
    ('actor', 'backends.mlx.actor'),
    ('runtime', 'rollout.manager'),
    ('rollout_worker', 'rollout.worker'),
    ('environment', 'environments.local'),
    ('remote_environment', 'environments.remote'),
    ('system_prompt', 'environments.system_prompt'),
    ('checkpoint', 'checkpointing.checkpoint'),
    ('evaluation', 'trainer.evaluation'),
    ('convert_cuda', 'export.hf'),
])
def test_legacy_imports_share_the_canonical_module(legacy, canonical):
    old = importlib.import_module('rl.' + legacy)
    new = importlib.import_module('rl.' + canonical)
    assert old is new  # Includes module-level patching and multiprocessing lookup.


def test_nested_source_changes_cannot_escape_resume_identity(tmp_path):
    from rl.checkpointing.provenance import source_identity
    (tmp_path / 'train.py').write_text('entry = 1\n')
    (tmp_path / 'trainer').mkdir()
    nested = tmp_path / 'trainer' / 'train.py'
    nested.write_text('loop = 1\n')
    before = source_identity(tmp_path)
    assert set(before) == {'train.py', 'trainer/train.py'}
    nested.write_text('loop = 2\n')
    after = source_identity(tmp_path)
    assert before['train.py'] == after['train.py']
    assert before['trainer/train.py'] != after['trainer/train.py']


def test_config_data_environment_and_export_do_not_import_mlx():
    script = '''
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'mlx', 'mlx_lm'}:
        raise AssertionError('MLX dependency leaked: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from rl.config import Config
from rl.trainer.cli import parse_args
from rl.data.contracts import TrainingSegment
from rl.environments.local import SandboxEpisode
from rl.export.hf import convert
from rl.export.verify import main
args = parse_args(['--tasks', 'tasks.json', '--output', 'unused', '--tuning', 'full'])
assert args.sandbox_backend == 'docker' and args.tuning == 'full'
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)


def test_service_payload_remains_standalone_after_move():
    from rl.environments.services import docker_manager, kubernetes, sandbox_service
    for module in (docker_manager, kubernetes):
        payload = Path(module.__file__).with_name('sandbox_service.py')
        assert payload.samefile(sandbox_service.__file__)
        # The image receives this source as serve.py without installing rl.
        assert 'from rl.' not in payload.read_text()
