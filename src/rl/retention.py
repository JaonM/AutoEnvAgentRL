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


def prune_live_artifacts(output, task_queue, current, *, keep=3, group_keep=256):
    """Protect every retained checkpoint, replay entry, continuation and live reader."""
    import fcntl
    output = Path(output)
    states = [json.loads(path.read_text()) for path in (output / 'checkpoints').glob('step-*/state.json')]
    if not states:
        return []
    safe = set.intersection(*(set(state.get('consumed_jobs', [])) for state in states))
    protected_paths = {Path(record['path']).resolve() for state in states
                       for record in state.get('inflight_batch', {}).get('records', [])}
    protected_paths.update(Path(entry['path']).resolve() for state in states
                           for entry in state.get('replay', {}).get('entries', []))
    removed = []
    for index in safe:
        with (task_queue.root / f'job-{index}.lock').open('a') as lease:
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            for path in task_queue.root.glob(f'result-{index:08d}-attempt-*.json'):
                path.unlink()
                removed.append(str(path))
            for path in task_queue.root.glob(f'continuation-{index}-attempt-*'):
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                    removed.append(str(path))
    protected_best = {Path(state['eval_state']['best']['path']).resolve() for state in states
                      if state.get('eval_state', {}).get('best')}
    for path in (output / 'best_policy').glob('policy-*.safetensors'):
        if path.resolve() not in protected_best and not path.is_symlink():
            path.unlink()
            removed.append(str(path))
    protected_eval_steps = {0, *(state.get('eval_state', {}).get('last_step', -1) for state in states)}
    protected_eval_steps.update(state['eval_state']['best']['optimizer_step'] for state in states
                                if state.get('eval_state', {}).get('best'))
    evaluations = sorted((output / 'evaluations').glob('step-*.json'))
    for path in evaluations[:-group_keep]:
        if int(path.stem.split('-')[1]) not in protected_eval_steps and not path.is_symlink():
            path.unlink()
            removed.append(str(path))
    groups = sorted(output.glob('group-*.json'), key=lambda path: int(path.stem.split('-')[1]))
    for path in groups[:-group_keep]:
        if path.resolve() not in protected_paths and not path.is_symlink():
            path.unlink()
            removed.append(str(path))
    root = output / 'snapshots'
    with (root / '.retention.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        protected = {0, current, *(state['updates'] for state in states)}
        version_file = output / 'policy_version.json'
        if version_file.exists():
            protected.add(json.loads(version_file.read_text())['version'])
        protected.update(json.loads(path.read_text())['policy_version']
                         for path in task_queue.root.glob('continuation-*/group.json'))
        for path in root.glob('policy-*.safetensors'):
            version = int(path.stem.split('-')[1])
            if version in protected or version >= current - keep + 1 or path.is_symlink():
                continue
            with path.with_suffix('.lock').open('a') as lease:
                try:
                    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                path.unlink()
                companion = root / 'inference' / path.name
                if companion.exists():
                    companion.unlink()
                    removed.append(str(companion))
                removed.append(str(path))
    return removed
