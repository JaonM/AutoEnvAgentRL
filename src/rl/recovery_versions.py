"""Quarantine speculative publications after rollback, before reusing version IDs."""
import json
from pathlib import Path


def quarantine(output, committed_version, recovery):
    output, recovery = Path(output), Path(recovery)
    for path in (output / 'task_queue').glob('result-*.json'):
        if json.loads(path.read_text()).get('policy_version', 0) > committed_version:
            destination = recovery / 'task_queue' / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)
    for path in (output / 'task_queue').glob('continuation-*'):
        metadata = path / 'group.json'
        if metadata.exists() and json.loads(metadata.read_text())['policy_version'] > committed_version:
            destination = recovery / 'task_queue' / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)
    for path in (output / 'snapshots').glob('policy-*.safetensors'):
        if int(path.stem.split('-')[1]) > committed_version:
            destination = recovery / 'snapshots' / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.rename(destination)
            packed = path.parent / 'inference' / path.name
            if packed.exists():
                destination = recovery / 'snapshots' / 'inference' / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                packed.rename(destination)
