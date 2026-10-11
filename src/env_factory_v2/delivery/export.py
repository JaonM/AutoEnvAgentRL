"""Deterministic, dependency-free container delivery of a frozen contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import re
import tempfile

from env_factory_v2.contracts.contract import require
from env_factory_v2.contracts.expressions import canonical
from env_factory_v2.validation.verification import compact_evidence


def export_bundle(compiled, report, output, *, base_image='python:3.14-slim'):
    require(isinstance(base_image, str) and re.fullmatch(r'[A-Za-z0-9._/:@-]+', base_image), 'invalid base image reference')
    output = Path(output)
    require(not output.exists(), 'delivery directory already exists')
    output.mkdir(parents=True)
    package = output / 'env_factory_v2'
    package.mkdir()
    from env_factory_v2.package import package_files
    for name, path in package_files().items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (output / 'contract.json').write_text(canonical(compiled['source']) + '\n')
    if 'demand' in compiled['source']:
        (output / 'task_spec.json').write_text(canonical(compiled['source']['demand']['task_spec']) + '\n')
        (output / 'agent_context.json').write_text(canonical(compiled['source']['demand']['agent_context']) + '\n')
    (output / 'user_profile.json').write_text(canonical({'facts': compiled['source']['user']['facts'],
        'profile': compiled['source']['user'].get('profile', {})}) + '\n')
    (output / 'public.json').write_text(canonical(compiled['public']) + '\n')
    delivery_report = compact_evidence(report)
    delivery_report['full_execution_evidence_identity'] = report['evidence_identity']
    delivery_report['evidence_detail'] = 'verdicts_and_answers'
    delivery_report.pop('evidence_identity', None)
    delivery_report['evidence_identity'] = hashlib.sha256(canonical(delivery_report).encode()).hexdigest()
    (output / 'verification.json').write_text(json.dumps(delivery_report, ensure_ascii=False, indent=2) + '\n')
    (output / 'Dockerfile').write_text(
        f'FROM {base_image}\nWORKDIR /sandbox\n'
        'COPY --chown=65532:65532 env_factory_v2 /sandbox/env_factory_v2\n'
        'COPY --chown=65532:65532 contract.json public.json user_profile.json verification.json /sandbox/\n'
        'USER 65532:65532\nEXPOSE 8080\n'
        'CMD ["python", "-m", "env_factory_v2", "serve", "/sandbox/contract.json", "--host", "0.0.0.0", "--port", "8080"]\n')
    files = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(output.rglob('*')) if p.is_file()}
    manifest = {'contract_identity': compiled['identity'], 'files': files,
                'bundle_identity': hashlib.sha256(canonical(files).encode()).hexdigest(),
                'qualified': False, 'scope': 'exported_before_container_validation'}
    (output / 'manifest.json').write_text(canonical(manifest) + '\n')
    return manifest


def verify_bundle(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    actual = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(output.rglob('*')) if p.is_file()
              and str(p.relative_to(output)) not in {'manifest.json', 'delivery.json', 'training.json'}}
    require(actual == manifest['files'] and hashlib.sha256(canonical(actual).encode()).hexdigest() == manifest['bundle_identity'],
            'delivery content changed after export')
    return manifest


def verify_container(output, *, image, agent=None):
    output = Path(output)
    manifest = verify_bundle(output)
    build = subprocess.run(['docker', 'build', '--tag', image, str(output)], capture_output=True, text=True, timeout=600)
    (output.parent / f'{output.name}-docker-build.log').write_text(build.stdout + build.stderr)
    if build.returncode:
        raise RuntimeError('Docker build failed; see sibling docker build log')
    identity = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
    args = ['docker', 'run', '--rm', '--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges',
            '--memory=256m', '--cpus=1', '--pids-limit=64', '--tmpfs', '/tmp:rw,noexec,nosuid,size=32m',
            '-e', 'PYTHONDONTWRITEBYTECODE=1']
    token_directory = None
    try:
        if agent is not None:
            # Only an ephemeral bridge token is mounted, never host Codex credentials.
            token_directory = tempfile.TemporaryDirectory(prefix='.v2-bridge-', dir=output.parent)
            token_path = Path(token_directory.name) / 'token'
            token_path.write_text(agent.token)
            token_path.chmod(0o644)
            url = agent.base_url.replace('127.0.0.1', 'host.docker.internal').replace('localhost', 'host.docker.internal')
            args += ['--mount', f'type=bind,src={token_path.resolve()},dst=/bridge-token,readonly',
                     '-e', 'V2_BRIDGE_URL=' + url, '-e', 'V2_BRIDGE_TOKEN_FILE=/bridge-token',
                     '-e', 'V2_BRIDGE_TRANSPORT_TIMEOUT=' + str(agent.timeout)]
        else:
            args += ['--network=none']
        run = subprocess.run(args + [identity, 'python', '-m', 'env_factory_v2', 'container-check', '/sandbox/contract.json'],
                             capture_output=True, text=True, timeout=1800)
        (output.parent / f'{output.name}-container-check.log').write_text(run.stdout + run.stderr)
        (output.parent / f'{output.name}-container-status.json').write_text(canonical({
            'exit_code': run.returncode, 'image_id': identity, 'memory_limit': '256m'}) + '\n')
        if run.returncode:
            raise RuntimeError(f'container execution verification failed (exit {run.returncode}); see sibling container log and status')
        evidence = json.loads(run.stdout)
        require(evidence['passed'] is True and evidence['contract_identity'] == manifest['contract_identity'], 'container evidence mismatch')
    finally:
        if token_directory is not None:
            token_directory.cleanup()
    return {'image_id': identity, 'bundle_identity': manifest['bundle_identity'], 'execution': evidence}
