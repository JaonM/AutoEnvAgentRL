"""Task manifests of certified sandbox bundles (env_factory) and reproducible group scheduling."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path



@dataclass(frozen=True)
class TaskSpec:
    id: str
    sandbox: str
    split: str = 'train'
    weight: float = 1.
    identity: str = ''


def load_tasks(*, sandbox='', manifest=''):
    if bool(sandbox) == bool(manifest):
        raise ValueError('provide exactly one of sandbox or task manifest')
    if manifest:
        source = Path(manifest).resolve()
        value = json.loads(source.read_text())
        entries = value.get('tasks') if isinstance(value, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ValueError('task manifest requires a nonempty tasks list')
        base = source.parent
    else:
        entries = [{'id': 'task-1', 'sandbox': str(Path(sandbox).resolve())}]
        base = Path.cwd()
    tasks, identifiers, identities = [], set(), {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {'id', 'sandbox', 'split', 'weight'}:
            raise ValueError('invalid task manifest entry')
        identifier, split = entry.get('id'), entry.get('split', 'train')
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError('task IDs must be unique nonempty strings')
        if split not in {'train', 'eval'}:
            raise ValueError('task split must be train or eval')
        weight = float(entry.get('weight', 1.))
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('task weight must be finite and positive')
        candidate = (base / entry['sandbox']).resolve()
        from rl.environments.sandbox import bundle_identity, is_sandbox_bundle
        if not is_sandbox_bundle(candidate):
            raise ValueError(f'not a certified sandbox bundle: {candidate}')
        root, identity = candidate, bundle_identity(candidate)
        if identity in identities:
            raise ValueError(f'duplicate sandbox identity: {identifier} and {identities[identity]}')
        identifiers.add(identifier)
        identities[identity] = identifier
        tasks.append(TaskSpec(identifier, str(root), split, weight, identity))
    if not any(t.split == 'train' for t in tasks):
        raise ValueError('at least one training task required')
    return tasks


def tasks_state(tasks):
    return [asdict(task) for task in tasks]
