"""Command line for EnvFactory V3.

  python -m env_factory_v3 import-v2 --frozen DIR --sample 4 --output request.json
  python -m env_factory_v3 build --request request.json --output DIR [--repairs 2]
  python -m env_factory_v3 batch --frozen DIR --samples 4,5,9 --output DIR [--parallel 1]
  python -m env_factory_v3 verify BUNDLE        # re-run every gate on an exported bundle
  python -m env_factory_v3 serve BUNDLE --port 8090
  python -m env_factory_v3 check {schema|env|oracle|diff}   # same as env_factory_v3.checks
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from env_factory_v3 import bundle, checks, gates, pipeline, server, spec


def _runner(args):
    from env_factory_v3.codex import CodexRunner
    if shutil.which('codex') is None:
        raise SystemExit('codex CLI not found on PATH')
    return CodexRunner(model=args.model)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m env_factory_v3')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('import-v2')
    p.add_argument('--frozen', required=True)
    p.add_argument('--sample', type=int, required=True)
    p.add_argument('--output', required=True)
    for name in ('build', 'batch'):
        p = sub.add_parser(name)
        p.add_argument('--output', required=True)
        p.add_argument('--repairs', type=int, default=2)
        p.add_argument('--perturbed', type=int, default=20)
        p.add_argument('--model', default='gpt-6.1-sol')
        if name == 'build':
            p.add_argument('--request', required=True)
        else:
            p.add_argument('--frozen', required=True)
            p.add_argument('--samples', required=True)
            p.add_argument('--parallel', type=int, default=1)
    p = sub.add_parser('verify')
    p.add_argument('bundle')
    p = sub.add_parser('serve')
    p.add_argument('bundle')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8090)
    p = sub.add_parser('check')
    p.add_argument('kind', choices=sorted(checks.KINDS))
    args = parser.parse_args(argv)

    if args.command == 'import-v2':
        request = spec.from_v2_frozen(args.frozen, args.sample)
        Path(args.output).write_text(json.dumps(request, ensure_ascii=False, indent=1))
        print(args.output)
        return 0
    if args.command == 'build':
        status = pipeline.build(spec.load(args.request), args.output, runner=_runner(args),
                                repairs=args.repairs, perturbed=args.perturbed)
        print(json.dumps({k: status.get(k) for k in ('id', 'training_ready', 'failure_class', 'reason',
                                                       'elapsed_seconds')}, ensure_ascii=False))
        return 0 if status['training_ready'] else 1
    if args.command == 'batch':
        requests = [spec.from_v2_frozen(args.frozen, int(s)) for s in args.samples.split(',')]
        summary = pipeline.batch(requests, args.output, runner=_runner(args), parallel=args.parallel,
                                 repairs=args.repairs, perturbed=args.perturbed)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    if args.command == 'verify':
        loaded = bundle.load_bundle(args.bundle, require_current_platform=False)
        report = gates.qualify(loaded['env'], loaded['oracle'], loaded['schema'], loaded['request'])
        print(json.dumps({g: report[g]['passed'] for g in ('static', 'differential', 'negatives', 'category', 'http')}
                         | {'passed': report['passed']}))
        return 0 if report['passed'] else 1
    if args.command == 'serve':
        loaded = bundle.load_bundle(args.bundle)
        service = server.Service(loaded['env'], loaded['oracle'], loaded['schema'], loaded['request'],
                                 valid_worlds=loaded['valid_worlds'])
        httpd = server.start(service, args.host, args.port)
        print(f'serving {args.bundle} on http://{args.host}:{httpd.server_address[1]}')
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            httpd.shutdown()
        return 0
    if args.command == 'check':
        return checks.main([args.kind])
    return 2


if __name__ == '__main__':
    sys.exit(main())
