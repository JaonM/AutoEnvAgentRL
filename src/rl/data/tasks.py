"""Qualified multi-sandbox task manifests and reproducible group scheduling."""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from rl.environments.local import verify_training_sandbox
from rl.checkpointing.checkpoint import sha256


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
    tasks, identifiers, identities, v2_texts = [], set(), {}, []
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
        if (candidate / 'contract.json').is_file() and (candidate / 'delivery.json').is_file():
            from env_factory_v2.delivery.training import verify_training_bundle
            certificate = verify_training_bundle(candidate)
            root, identity = candidate, certificate['identity']
            public = json.loads((candidate / 'public.json').read_text())
            v2_texts.append({'item_id': identifier, 'task': public['task'], 'split': split})
        else:
            root = verify_training_sandbox(candidate)
            status = json.loads((root / 'status.json').read_text())
            identity = hashlib.sha256(json.dumps({'artifacts':status['artifact_hashes'],
                'task_sha256':sha256(root / 'task.json')},sort_keys=True).encode()).hexdigest()
        if identity in identities:
            raise ValueError(f'duplicate sandbox identity: {identifier} and {identities[identity]}')
        identifiers.add(identifier)
        identities[identity] = identifier
        tasks.append(TaskSpec(identifier, str(root), split, weight, identity))
    if not any(t.split == 'train' for t in tasks):
        raise ValueError('at least one training task required')
    if v2_texts:
        from env_factory.tasks.task_similarity import task_family_ids
        families = task_family_ids(v2_texts)
        splits = {}
        for row in v2_texts:
            family = families[row['item_id']]
            if family in splits and splits[family] != row['split']:
                raise ValueError('V2 near-duplicate task family crosses train/eval split')
            splits[family] = row['split']
    return tasks


def tasks_state(tasks):
    return [asdict(task) for task in tasks]
