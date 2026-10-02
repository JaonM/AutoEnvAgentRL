"""Prepare/start/stop Docker sandbox services without waiting for Pod readiness."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .checkpoint import atomic_json
from .tasks import load_tasks
from .sandbox_service import artifact_identity
from .remote_environment import container_hashes


def resources(tasks, images, *, namespace, secret, prefix):
    for name in (namespace, secret, prefix):
        if not re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,38}[a-z0-9])?', name):
            raise ValueError('namespace, secret and prefix must be DNS labels of at most 40 characters')
    code = Path(__file__).with_name('sandbox_service.py').read_text()
    items, services = [], {}
    for task in tasks:
        image = images[task.id]
        if not re.fullmatch(r'[^\s]+@sha256:[a-f0-9]{64}', image):
            raise ValueError('images must use immutable registry/name@sha256:... references')
        hashes = container_hashes(task.sandbox)
        identity = artifact_identity(hashes)
        name = prefix + '-' + identity[:12]
        labels = {'envfactory-run': prefix, 'envfactory-sandbox': name}
        metadata = {'name': name, 'namespace': namespace, 'labels': labels}
        config_name = name + '-runtime'
        items.append({'apiVersion': 'v1', 'kind': 'ConfigMap',
                      'metadata': {**metadata, 'name': config_name},
                      'data': {'serve.py': code, 'hashes.json': json.dumps(hashes)}})
        container = {
            'name': 'sandbox', 'image': image, 'imagePullPolicy': 'IfNotPresent',
            'command': ['python3', '/opt/envfactory/serve.py'], 'workingDir': '/app',
            'ports': [{'containerPort': 8000}],
            'envFrom': [{'secretRef': {'name': secret}}],
            'env': [{'name': 'RL_ARTIFACT_HASHES_FILE', 'value': '/opt/envfactory/hashes.json'},
                    {'name': 'PYTHONUNBUFFERED', 'value': '1'}],
            'resources': {'requests': {'cpu': '100m', 'memory': '128Mi'},
                          'limits': {'cpu': '1', 'memory': '1Gi'}},
            'securityContext': {'allowPrivilegeEscalation': False, 'readOnlyRootFilesystem': True,
                                'capabilities': {'drop': ['ALL']}},
            'volumeMounts': [{'name': 'runtime', 'mountPath': '/opt/envfactory', 'readOnly': True},
                             {'name': 'tmp', 'mountPath': '/tmp'},
                             {'name': 'state', 'mountPath': '/app/.runtime'}],
            'readinessProbe': {'httpGet': {'path': '/health', 'port': 8000}, 'periodSeconds': 2},
            'startupProbe': {'httpGet': {'path': '/health', 'port': 8000},
                             'periodSeconds': 2, 'failureThreshold': 300},
        }
        items.append({'apiVersion': 'apps/v1', 'kind': 'Deployment', 'metadata': metadata,
                      'spec': {'replicas': 1, 'strategy': {'type': 'Recreate'},
                               'selector': {'matchLabels': {'envfactory-sandbox': name}},
                               'template': {'metadata': {'labels': labels,
                                            'annotations': {'runtime-sha256': hashlib.sha256(code.encode()).hexdigest()}},
                                            'spec': {'automountServiceAccountToken': False,
                                                     'securityContext': {'runAsNonRoot': True, 'runAsUser': 10001,
                                                                         'runAsGroup': 10001, 'fsGroup': 10001},
                                                     'containers': [container],
                                                     'volumes': [{'name': 'runtime', 'configMap': {'name': config_name}},
                                                                 {'name': 'tmp', 'emptyDir': {'sizeLimit': '1Gi'}},
                                                                 {'name': 'state', 'emptyDir': {'sizeLimit': '1Gi'}}]}}}})
        items.append({'apiVersion': 'v1', 'kind': 'Service', 'metadata': metadata,
                      'spec': {'type': 'NodePort', 'selector': {'envfactory-sandbox': name},
                               'ports': [{'port': 8000, 'targetPort': 8000, 'protocol': 'TCP'}]}})
        services[identity] = {'name': name, 'task_id': task.id, 'image': image}
    return {'apiVersion': 'v1', 'kind': 'List', 'items': items}, services


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('render', 'start', 'stop'))
    parser.add_argument('--sandbox', default='')
    parser.add_argument('--tasks', default='')
    parser.add_argument('--images', type=Path, help='JSON mapping task IDs to immutable Docker image references')
    parser.add_argument('--namespace', default='default')
    parser.add_argument('--secret', default='envfactory-sandbox', help='existing Secret containing trainer and simulator credentials')
    parser.add_argument('--prefix', default='rl-sandbox', help='unique resource prefix per independent deployment')
    parser.add_argument('--host', help='cluster node IP/DNS reachable by external trainer')
    parser.add_argument('--context', default='', help='kubectl context; empty uses current context')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    kubectl = ['kubectl'] + (['--context', args.context] if args.context else [])
    if args.action == 'stop':
        subprocess.run(kubectl + ['delete', '-f', str(args.output / 'resources.json'),
                                  '--ignore-not-found', '--wait=false'], check=True, timeout=120)
        return
    if not args.images or (args.action == 'start' and not args.host):
        parser.error('--images is required; start also requires --host')
    if args.host and not re.fullmatch(r'[A-Za-z0-9_.:\[\]-]+', args.host):
        parser.error('--host must be a node IP or hostname, without scheme/path/port')
    tasks = load_tasks(sandbox=args.sandbox, manifest=args.tasks)
    manifest, services = resources(tasks, json.loads(args.images.read_text()),
                                  namespace=args.namespace, secret=args.secret, prefix=args.prefix)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'resources.json', manifest)
    if args.action == 'render':
        print(f'Rendered {len(tasks)} sandboxes; no cluster changes: {args.output / "resources.json"}')
        return
    # API submission only: image pulls, scheduling and readiness happen asynchronously.
    subprocess.run(kubectl + ['apply', '-f', str(args.output / 'resources.json')], check=True, timeout=120)
    for service in services.values():
        result = subprocess.run(kubectl + ['get', 'service', service['name'], '-n', args.namespace, '-o', 'json'],
                                check=True, capture_output=True, text=True, timeout=30)
        port = json.loads(result.stdout)['spec']['ports'][0]['nodePort']
        host = args.host if ':' not in args.host else '[' + args.host.strip('[]') + ']'
        service['url'] = f'http://{host}:{port}'
    atomic_json(args.output / 'services.json', {'services': services, 'startup_timeout': 600, 'request_timeout': 120})
    print(f'Submitted {len(tasks)} sandboxes without waiting for readiness. Endpoints: {args.output / "services.json"}')


if __name__ == '__main__':
    main()
