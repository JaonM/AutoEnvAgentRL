"""Delivery bundle: the files a training run needs, with content hashes."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from env_factory_v3 import answers
from env_factory_v3.sandbox import load_module
from env_factory_v3.spec import load

PACKAGE_ROOT = Path(__file__).parent
FILES = ('request.json', 'answer_schema.json', 'env.py', 'oracle.py', 'gates.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def platform_identity():
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_ROOT.rglob('*.py')):
        if '__pycache__' in path.parts:
            continue
        digest.update(path.relative_to(PACKAGE_ROOT).as_posix().encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def export(source_folder, request_path, report, destination, *, status):
    destination = Path(destination)
    destination.mkdir(parents=True)
    shutil.copy(request_path, destination / 'request.json')
    for name in ('answer_schema.json', 'env.py', 'oracle.py'):
        shutil.copy(Path(source_folder) / name, destination / name)
    slim = {key: value for key, value in report.items() if key != 'worlds'}
    slim['worlds'] = [{k: w.get(k) for k in ('world', 'status', 'reward')} for w in report.get('worlds', [])]
    (destination / 'gates.json').write_text(json.dumps(slim, ensure_ascii=False, indent=1, default=str))
    manifest = {'version': 'v3-m1', 'platform_identity': platform_identity(),
                'files': {name: sha256(destination / name) for name in FILES}, 'status': status}
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


def load_bundle(folder, *, require_current_platform=True):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha256(folder / name) != digest:
            raise ValueError(f'bundle file {name} was modified after export')
    if require_current_platform and manifest['platform_identity'] != platform_identity():
        raise ValueError('bundle was qualified on a different V3 platform; requalify it with `verify`')
    gates_report = json.loads((folder / 'gates.json').read_text())
    return {'request': load(folder / 'request.json'),
            'schema': answers.load_schema(folder / 'answer_schema.json'),
            'env': load_module(folder / 'env.py', role='env'),
            'oracle': load_module(folder / 'oracle.py', role='oracle'),
            'valid_worlds': gates_report.get('valid_worlds') or ['original'],
            'manifest': manifest, 'gates': gates_report}


def requalify(folder):
    """Re-run every gate on the current platform; on success re-sign the bundle for it.

    Changing any V3 platform file invalidates every bundle (load_bundle refuses a different
    platform identity); this is the only way back, and it never skips a gate.
    """
    from env_factory_v3 import gates
    folder = Path(folder)
    loaded = load_bundle(folder, require_current_platform=False)
    excluded = loaded['gates'].get('excluded_worlds') or []
    report = gates.qualify(loaded['env'], loaded['oracle'], loaded['schema'], loaded['request'], excluded=excluded)
    if report['passed']:
        slim = {key: value for key, value in report.items() if key != 'worlds'}
        slim['worlds'] = [{k: w.get(k) for k in ('world', 'status', 'reward')} for w in report.get('worlds', [])]
        (folder / 'gates.json').write_text(json.dumps(slim, ensure_ascii=False, indent=1, default=str))
        manifest = loaded['manifest']
        manifest.update(platform_identity=platform_identity(),
                        files={name: sha256(folder / name) for name in FILES})
        manifest.setdefault('requalified', []).append(manifest['platform_identity'])
        (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return report
