"""Delivery bundle: the files a training run needs, with content hashes."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from env_factory.core import answers
from env_factory.core.sandbox import load_module
from env_factory.core.spec import load

PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # src/env_factory
FILES = ('request.json', 'answer_schema.json', 'env.py', 'verifier.py', 'gates.json')
RUBRIC_FILES = ('request.json', 'answer_schema.json', 'env.py', 'rubric.json', 'gates.json')


def files_for(request):
    """Business bundles carry the verifier; rubric bundles carry the rubric instead."""
    return RUBRIC_FILES if request.get('reward_type') == 'rubric' else FILES


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# What decides a sandbox's behaviour and reward at training time: loading, the episode service, the
# agent's view, the user simulator, grading, the gates a bundle is certified with and the container
# it runs in. Build-time orchestration (generation, codex sessions, repair audits, the run command,
# the policy gate) is not part of it, so editing those never invalidates certified bundles.
RUNTIME_MODULES = ('__init__.py', 'runtime.py',
                   'core/__init__.py', 'core/answers.py', 'core/disclosure.py', 'core/extract.py', 'core/llm.py', 'core/numerals.py',
                   'core/reward.py', 'core/rubric.py', 'core/sandbox.py', 'core/spec.py', 'core/worlds.py',
                   'serving/__init__.py', 'serving/agent_view.py', 'serving/server.py', 'serving/user_sim.py',
                   'certify/__init__.py', 'certify/bundle.py', 'certify/docker.py', 'certify/gates.py')


def platform_identity():
    digest = hashlib.sha256()
    for name in sorted(RUNTIME_MODULES):
        digest.update(name.encode() + b'\0' + (PACKAGE_ROOT / name).read_bytes())
    return digest.hexdigest()


def export(source_folder, request_path, report, destination, *, status):
    destination = Path(destination)
    destination.mkdir(parents=True)
    shutil.copy(request_path, destination / 'request.json')
    names = files_for(json.loads(Path(request_path).read_text()))
    for name in names:
        if name not in ('request.json', 'gates.json'):
            shutil.copy(Path(source_folder) / name, destination / name)
    slim = {key: value for key, value in report.items() if key != 'worlds'}
    slim['worlds'] = [{k: w.get(k) for k in ('world', 'status', 'reward')} for w in report.get('worlds', [])]
    (destination / 'gates.json').write_text(json.dumps(slim, ensure_ascii=False, indent=1, default=str))
    manifest = {'version': 'v3-m1', 'platform_identity': platform_identity(),
                'files': {name: sha256(destination / name) for name in names}, 'status': status}
    (destination / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return manifest


def load_bundle(folder, *, require_current_platform=True):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    for name, digest in manifest['files'].items():
        if sha256(folder / name) != digest:
            raise ValueError(f'bundle file {name} was modified after export')
    if require_current_platform and manifest['platform_identity'] != platform_identity():
        raise ValueError('bundle was qualified on a different platform; requalify it with `verify`')
    gates_report = json.loads((folder / 'gates.json').read_text())
    rubric = 'rubric.json' in manifest['files']
    # Bundles certified before the rename carry the verifier as oracle.py; requalify moves it.
    verifier_file = 'oracle.py' if 'oracle.py' in manifest['files'] else 'verifier.py'
    return {'request': load(folder / 'request.json'),
            'schema': answers.load_schema(folder / 'answer_schema.json'),
            'env': load_module(folder / 'env.py', role='env'),
            'verifier': None if rubric else load_module(folder / verifier_file, role='verifier'),
            'rubric': json.loads((folder / 'rubric.json').read_text()) if rubric else None,
            'valid_worlds': gates_report.get('valid_worlds') or ['original'],
            'manifest': manifest, 'gates': gates_report}


def default_judge():
    from env_factory.core.llm import ChatClient, load_env_file
    from env_factory.core.rubric import Judge
    load_env_file()
    return Judge(ChatClient('JUDGE'))


def qualify_loaded(loaded, *, judge='auto'):
    """Every gate for a loaded bundle of either reward type. judge=None skips the judge-based
    rubric gates (offline container check)."""
    from env_factory.certify import gates
    if loaded['rubric'] is not None:
        return gates.qualify_rubric(loaded['env'], loaded['rubric'], loaded['schema'], loaded['request'],
                                    judge=default_judge() if judge == 'auto' else judge)
    return gates.qualify(loaded['env'], loaded['verifier'], loaded['schema'], loaded['request'],
                         excluded=loaded['gates'].get('excluded_worlds') or [])


def service(loaded, **options):
    """The episode service for a loaded bundle (rubric bundles grade with the judge)."""
    from env_factory.serving import server
    return server.Service(loaded['env'], loaded['verifier'], loaded['schema'], loaded['request'],
                          valid_worlds=loaded['valid_worlds'], rubric=loaded['rubric'], **options)


def requalify(folder):
    """Re-run every gate on the current platform; on success re-sign the bundle for it.

    Changing any runtime module (RUNTIME_MODULES) invalidates every bundle (load_bundle refuses a
    different platform identity); this is the only way back, and it never skips a gate.
    """
    folder = Path(folder)
    loaded = load_bundle(folder, require_current_platform=False)
    report = qualify_loaded(loaded)
    if report['passed']:
        if (folder / 'oracle.py').exists() and not (folder / 'verifier.py').exists():
            (folder / 'oracle.py').rename(folder / 'verifier.py')  # the file's name before the rename
        slim = {key: value for key, value in report.items() if key != 'worlds'}
        slim['worlds'] = [{k: w.get(k) for k in ('world', 'status', 'reward')} for w in report.get('worlds', [])]
        (folder / 'gates.json').write_text(json.dumps(slim, ensure_ascii=False, indent=1, default=str))
        manifest = loaded['manifest']
        manifest.update(platform_identity=platform_identity(),
                        files={name: sha256(folder / name) for name in files_for(loaded['request'])})
        manifest.setdefault('requalified', []).append(manifest['platform_identity'])
        (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    return report
