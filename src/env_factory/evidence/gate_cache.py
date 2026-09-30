"""Reuse outer-workflow gate evidence only for identical code and artifacts."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

GATES = ('business_acceptance', 'sandbox_pytest', 'runtime_genericity', 'outer_conformance',
         'mutation_resistance', 'training_readiness', 'declared_training_policy')
ARTIFACTS = ('acceptance_result.json', 'training_readiness.json', 'agentic_training_value.json',
             'review_report.json', 'mutation_report.json')


def identity(root: Path, project: Path) -> dict:
    from env_factory.sandbox_scoring import evidence_fingerprint
    artifacts = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ARTIFACTS}
    # Shell orchestration is also part of the authority, beyond Python scorers.
    shells = {str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((project / 'scripts').rglob('*.sh'))}
    return {'context': evidence_fingerprint(root, project), 'artifacts': artifacts, 'shells': shells}


def save(root: Path, project: Path, results: dict) -> None:
    if set(results) != set(GATES) or not all(value[0] is True for value in results.values()):
        return
    try:
        key = identity(root, project)
    except OSError:
        return
    path = root / '.node_checkpoints' / 'qualification.json'
    path.parent.mkdir(exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps({'version': 1, 'identity': key, 'results': results}, ensure_ascii=False))
    temporary.replace(path)


def load(root: Path, project: Path) -> dict | None:
    try:
        value = json.loads((root / '.node_checkpoints' / 'qualification.json').read_text())
        results = value['results']
        if (value['version'] != 1 or value['identity'] != identity(root, project)
                or set(results) != set(GATES)
                or any(not isinstance(v, list) or len(v) != 2 or v[0] is not True or not isinstance(v[1], str) for v in results.values())):
            return None
        return results
    except (OSError, ValueError, KeyError, TypeError):
        return None
