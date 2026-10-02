"""Bound checkpoint storage without deleting the committed pointer or baseline."""
import json
from pathlib import Path
import shutil


def prune_checkpoints(root, keep=3):
    if keep < 1:
        raise ValueError('keep must be positive')
    root = Path(root)
    latest = json.loads((root / 'latest.json').read_text())['directory']
    candidates = []
    for path in root.glob('step-*'):
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            state = json.loads((path / 'state.json').read_text())
            candidates.append((state['optimizer_step'], path.name, path))
        except (OSError, ValueError, KeyError):
            continue  # Preserve unknown/incomplete artifacts for diagnosis.
    ordered = sorted(candidates, reverse=True)
    protected = {latest}
    for _, name, _ in ordered:
        if len(protected) >= keep:
            break
        protected.add(name)
    removed = []
    for _, name, path in ordered:
        if name not in protected:
            shutil.rmtree(path)
            removed.append(name)
    return removed


def prune_snapshots_after_shutdown(root, current, keep=3, *, protected_versions=()):
    """Caller must stop all rollout_workers first; version 0 is the frozen reference."""
    if keep < 1:
        raise ValueError('keep must be positive')
    root = Path(root)
    removed = []
    for path in root.glob('policy-*.safetensors'):
        try:
            version = int(path.stem.removeprefix('policy-'))
        except ValueError:
            continue
        if version and version not in protected_versions and version < current - keep + 1 and not path.is_symlink():
            path.unlink()
            removed.append(path.name)
    return removed
