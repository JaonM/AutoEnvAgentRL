"""Preserve one fixed business design for a separately validated author retry.

Checkpoints carry source data, never acceptance certificates. Loading one does not
skip compilation, independent review, or a completed author turn.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _publish(target, data):
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.checkpoint-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(root: Path, *, request: dict, blueprint: dict,
                    source: dict | None, evidence: dict, validation_feedback: dict | None = None) -> Path:
    """Write outside the author workspace, publish the manifest last atomically."""
    from .business_blueprint import validate_blueprint
    validate_blueprint(blueprint, request)
    directory = root / 'authoring_checkpoint'
    if directory.is_symlink():
        raise ValueError('AUTHOR_CHECKPOINT: symbolic links are not allowed')
    directory.mkdir(exist_ok=True)
    files = {}
    for label, value in (('request', request), ('blueprint', blueprint), ('source', source),
                         ('validation_feedback', validation_feedback)):
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f'AUTHOR_CHECKPOINT: {label} must be an object')
        data = _encoded(value)
        digest = _digest(data)
        # Content-addressed snapshots avoid mixing old manifests and new files.
        filename = f'{label}-{digest}.json'
        target = directory / filename
        if target.is_symlink():
            raise ValueError('AUTHOR_CHECKPOINT: symbolic links are not allowed')
        if not target.exists():
            _publish(target, data)
        elif target.read_bytes() != data:
            raise ValueError('AUTHOR_CHECKPOINT: existing snapshot has changed')
        files[label] = {'filename': filename, 'sha256': digest}
    manifest = {'version': '1.0', 'scope': 'candidate_only_requires_fresh_validation',
                'files': files, 'previous_status': evidence.get('status'),
                'author_seconds': evidence.get('previous_author_seconds', 0) + evidence.get('author_seconds', 0),
                'repair_budget': evidence.get('repair_budget', {}),
                'source_attempts': evidence.get('previous_source_attempts', 0) + sum(
                    a.get('phase') == 'implementation' and a.get('business_attempt_consumed', True)
                    for a in evidence.get('attempts', [])),
                'previous_issue_ids': sorted(set(evidence.get('previous_issue_ids', [])) |
                    {issue for a in evidence.get('attempts', []) for issue in a.get('issue_ids', [])})}
    _publish(directory / 'manifest.json', _encoded(manifest))
    return directory / 'manifest.json'


def load_checkpoint(path: Path, *, expected_request: dict | None = None) -> dict:
    """Read only declared snapshots; reject drift before starting an author."""
    if path.is_dir():
        path = path / 'manifest.json'
    if path.is_symlink():
        raise ValueError('AUTHOR_CHECKPOINT: symbolic links are not allowed')
    manifest = json.loads(path.read_text())
    if manifest.get('version') != '1.0' or manifest.get('scope') != 'candidate_only_requires_fresh_validation':
        raise ValueError('AUTHOR_CHECKPOINT: unsupported manifest')
    if (type(manifest.get('source_attempts')) is not int or not 0 <= manifest['source_attempts'] <= 3
            or not isinstance(manifest.get('previous_issue_ids'), list)
            or any(not isinstance(issue, str) for issue in manifest['previous_issue_ids'])):
        raise ValueError('AUTHOR_CHECKPOINT: invalid retry history')
    used = manifest.get('repair_budget', {}).get('used', {})
    if (not isinstance(used, dict) or used.keys() - {'structural', 'semantic'}
            or any(type(value) is not int or not 0 <= value <= 1 for value in used.values())):
        raise ValueError('AUTHOR_CHECKPOINT: invalid repair budget')
    files = manifest.get('files', {})
    if not isinstance(files, dict) or not {'request', 'blueprint'} <= files.keys() or files.keys() - {'request', 'blueprint', 'source', 'validation_feedback'}:
        raise ValueError('AUTHOR_CHECKPOINT: invalid snapshot inventory')
    result = {'manifest': manifest}
    for label, entry in files.items():
        digest = entry.get('sha256', '')
        filename = entry.get('filename')
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)
                or filename != f'{label}-{digest}.json'):
            raise ValueError('AUTHOR_CHECKPOINT: invalid snapshot name or digest')
        target = path.parent / filename
        if target.is_symlink():
            raise ValueError('AUTHOR_CHECKPOINT: symbolic links are not allowed')
        data = target.read_bytes()
        if _digest(data) != digest:
            raise ValueError('AUTHOR_CHECKPOINT: snapshot content changed')
        result[label] = json.loads(data)
        if not isinstance(result[label], dict):
            raise ValueError('AUTHOR_CHECKPOINT: snapshot must be an object')
    if expected_request is not None and result['request'] != expected_request:
        raise ValueError('AUTHOR_CHECKPOINT: request differs from fixed design')
    from .business_blueprint import validate_blueprint
    validate_blueprint(result['blueprint'], result['request'])
    result['manifest_sha256'] = _digest(path.read_bytes())
    return result
