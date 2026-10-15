"""Container delivery: hardened images verified offline inside a container.

build() packs this package and one bundle into an image; build_many() packs several bundles (every
certified task of one domain world, which differ only in their request, answer schema and manifest)
into one image under /tasks/<n>, so a world is built and verified once instead of once per task. Every
gate runs again inside the image with no network, per bundle, and the image is recorded next to each
bundle (docker_image.json, outside the hashed files) with the path it is served from. run() starts a
serving container for training on that path: same hardening, the HTTP port published on 127.0.0.1 only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

from env_factory.certify import bundle

BASE_IMAGE = 'python:3.14-slim'
PORT = 8000
USER = '10001:10001'
HARDENING = ['--read-only', '--cap-drop=ALL', '--security-opt=no-new-privileges', '--user', USER,
             '--memory=512m', '--cpus=1', '--pids-limit=64',
             '--tmpfs', '/tmp:rw,noexec,nosuid,size=32m', '-e', 'PYTHONDONTWRITEBYTECODE=1']
METADATA = 'docker_image.json'
IDLE_EXIT = 900  # seconds without a request before a serving container stops (and --rm removes it)
DOCKERFILE = f"""FROM {{base}}
WORKDIR /app
COPY env_factory /app/env_factory
COPY bundle /bundle
ENV PYTHONPATH=/app PYTHONDONTWRITEBYTECODE=1
USER {USER}
EXPOSE {PORT}
CMD ["python", "-m", "env_factory", "serve", "/bundle", "--host", "0.0.0.0", "--port", "{PORT}"]
"""


def docker(*args, timeout=600, env=None):
    result = subprocess.run(['docker', *args], capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, **env} if env else None)
    if result.returncode:
        raise RuntimeError(f'docker {args[0]} failed: {(result.stderr or result.stdout)[-800:]}')
    return result.stdout.strip()


def identity(folder):
    """A bundle's content identity: its certified files on this platform."""
    manifest = bundle.load_bundle(folder)['manifest']
    return hashlib.sha256(json.dumps([manifest['platform_identity'], manifest['files']],
                                     sort_keys=True).encode()).hexdigest()


def build(folder, *, base_image=BASE_IMAGE):
    """One bundle in its own image, served from /bundle."""
    return build_many([folder], base_image=base_image)[0]


def build_many(folders, *, base_image=BASE_IMAGE, timeout=1800):
    """Several bundles in one image (a world's tasks), each verified offline inside it; returns the metadata
    written next to each bundle. One bundle keeps the single-bundle layout (/bundle)."""
    folders = [Path(f).resolve() for f in folders]
    paths = ['/bundle'] if len(folders) == 1 else [f'/tasks/{i:03d}' for i in range(len(folders))]
    members = [{'path': path, 'identity': identity(folder)} for path, folder in zip(paths, folders)]
    digest = hashlib.sha256(json.dumps(members, sort_keys=True).encode()).hexdigest()
    tag = f'envfactory:{digest[:16]}'
    with tempfile.TemporaryDirectory(prefix='envfactory-docker-') as context:
        context = Path(context)
        package = context / 'env_factory'
        for path in sorted(bundle.PACKAGE_ROOT.rglob('*.py')):
            if '__pycache__' not in path.parts:
                target = package / path.relative_to(bundle.PACKAGE_ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        for path, folder in zip(paths, folders):
            target = context / path.lstrip('/')
            target.mkdir(parents=True)
            for name in (*bundle.load_bundle(folder)['manifest']['files'], 'manifest.json'):
                shutil.copyfile(folder / name, target / name)
        if len(folders) > 1:
            (context / 'bundle').mkdir()  # the Dockerfile copies /bundle; a world image serves /tasks/<n>
        (context / 'Dockerfile').write_text(DOCKERFILE.format(base=base_image).replace(
            'COPY bundle /bundle', 'COPY bundle /bundle' + ('\nCOPY tasks /tasks' if len(folders) > 1 else '')))
        docker('build', '--tag', tag, str(context))
    image = docker('image', 'inspect', '--format', '{{.Id}}', tag)
    # Every gate again, inside the image, with no network at all: one container, every bundle in turn.
    output = docker('run', '--rm', '--network=none', *HARDENING, image,
                    'python', '-m', 'env_factory', 'container-check', *paths, timeout=timeout * len(folders))
    evidence = json.loads(output.splitlines()[-1])
    by_path = evidence['bundles'] if 'bundles' in evidence else {paths[0]: evidence}
    results = []
    for path, folder, member in zip(paths, folders, members):
        manifest = bundle.load_bundle(folder)['manifest']
        found = by_path.get(path) or {}
        if not (found.get('passed') and found.get('platform_identity') == manifest['platform_identity']
                and found.get('files') == manifest['files']):
            raise RuntimeError(f'container verification failed for {folder}: {found}')
        metadata = {'image_id': image, 'tag': tag, 'identity': member['identity'], 'path': path,
                    'image_identity': digest, 'members': len(folders), 'base_image': base_image,
                    'platform_identity': manifest['platform_identity'], 'files': manifest['files'],
                    'container_check': found, 'hardening': HARDENING, 'network_during_check': 'none'}
        (folder / METADATA).write_text(json.dumps(metadata, indent=1))
        results.append(metadata)
    return results


def image_for(folder):
    """(verified image, path of the bundle inside it); refuses stale metadata."""
    folder = Path(folder).resolve()
    path = folder / METADATA
    if not path.exists():
        raise ValueError(f'{folder} has no verified image; run `python -m env_factory docker {folder}`')
    metadata = json.loads(path.read_text())
    if metadata['identity'] != identity(folder):
        raise ValueError(f'{folder}: image was verified for different bundle content or platform')
    return metadata['image_id'], metadata.get('path', '/bundle')


def run(folder, *, user_mode='structured', token=None, timeout=30):
    """Start a serving container; returns {'container', 'url'}. The natural user mode passes the
    simulator's provider settings by name (values never appear in argv)."""
    image, served = image_for(folder)
    extra = []
    rubric = (Path(folder) / 'rubric.json').exists()  # the judge grades rubric episodes inside the container
    if user_mode == 'natural' or rubric:
        for prefix in ('USER_SIM', 'JUDGE', 'LLM'):
            for name in ('API_KEY', 'BASE_URL', 'MODEL'):
                if os.environ.get(f'{prefix}_{name}'):
                    extra += ['--env', f'{prefix}_{name}']
    # --rm with --idle-exit: a container whose trainer died without removing it exits by itself.
    command = ['python', '-m', 'env_factory', 'serve', served, '--host', '0.0.0.0', '--port', str(PORT),
               '--user-mode', user_mode, '--idle-exit', str(IDLE_EXIT)]
    if token:
        extra += ['--env', 'SANDBOX_SERVICE_TOKEN']  # value comes from the docker CLI's environment, not argv
    container = docker('run', '-d', '--rm', '--label', 'envfactory=serve', *HARDENING, *extra,
                       '--publish', f'127.0.0.1::{PORT}', image, *command,
                       env={'SANDBOX_SERVICE_TOKEN': token} if token else None)
    try:
        port = docker('port', container, f'{PORT}/tcp').splitlines()[0].rsplit(':', 1)[1]
        url = f'http://127.0.0.1:{port}'
        opener = build_opener(ProxyHandler({}))
        deadline = time.monotonic() + timeout
        while True:
            try:
                with opener.open(url + '/health', timeout=2) as response:
                    if json.load(response).get('ok'):
                        return {'container': container, 'url': url}
            except OSError:
                pass
            if time.monotonic() > deadline:
                raise RuntimeError(f'container {container[:12]} not healthy within {timeout}s')
            time.sleep(0.2)
    except BaseException:
        remove(container)
        raise


def remove(container):
    subprocess.run(['docker', 'rm', '-f', container], capture_output=True, timeout=60)
