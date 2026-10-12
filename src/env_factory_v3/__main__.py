"""Command line for EnvFactory V3.

  python -m env_factory_v3 import-v2 --frozen DIR --sample 4 --output request.json
  python -m env_factory_v3 build --request request.json --output DIR [--repairs 2]
  python -m env_factory_v3 batch --frozen DIR --samples 4,5,9 --output DIR [--parallel 1]
  python -m env_factory_v3 batch --requests DIR --output DIR                 # generated requests
  python -m env_factory_v3 domains --count 12 --split holdout-4 --output domains.json   # graph x business grid
  python -m env_factory_v3 generate --domains domains.json --count 30 --output DIR
  python -m env_factory_v3 serve BUNDLE --port 8090
  python -m env_factory_v3 verify BUNDLE [--resign]   # re-run gates; --resign re-signs for this platform
  python -m env_factory_v3 docker BUNDLE              # hardened image, verified offline in a container
  python -m env_factory_v3 policy BUNDLE [--k 4]      # policy gate with an independent model
  python -m env_factory_v3 findings BUILD             # spec_findings as questions for the business owner
  python -m env_factory_v3 clarify --request R --answer TEXT --output R2   # rebuild R2 with the answers
  python -m env_factory_v3 check {schema|env|oracle|diff}   # same as env_factory_v3.checks
"""
from __future__ import annotations

import argparse
import json
import os
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
            p.add_argument('--requests', help='folder of request .json files (instead of --frozen/--samples)')
            p.add_argument('--frozen')
            p.add_argument('--samples', default='')
            p.add_argument('--parallel', type=int, default=1)
    p = sub.add_parser('verify')
    p.add_argument('bundle')
    p.add_argument('--resign', action='store_true', help='on success, re-sign the bundle for this platform')
    p = sub.add_parser('container-check', help='inside the image: every gate on the bundle, JSON evidence')
    p.add_argument('bundle')
    p = sub.add_parser('policy', help='policy gate: an independent model plays k episodes')
    p.add_argument('bundle')
    p.add_argument('--k', type=int, default=4)
    p.add_argument('--prefix', default='LLM', help='<PREFIX>_API_KEY / _BASE_URL / _MODEL')
    p.add_argument('--user-mode', choices=('structured', 'natural'), default='structured')
    p = sub.add_parser('generate', help='new build requests for new business domains')
    p.add_argument('--domains', required=True, help='JSON list of {"name", "detail"} domains')
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--parallel', type=int, default=2)
    p.add_argument('--rubric-share', type=float, default=0.0,
                   help='fraction of tasks that are subjective (rubric reward); taken from multi-step slots')
    p.add_argument('--model', default='gpt-6.1-sol')
    p = sub.add_parser('domains', help='sample domains from the scene graph x business grid')
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--split', required=True, help='dev, holdout-1 ... holdout-5 (components never shared)')
    p.add_argument('--output', required=True)
    p.add_argument('--hops', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--exclude', nargs='*', default=[], help='domains.json files whose names must not recur')
    p = sub.add_parser('findings', help="open business-rule questions from a build's spec_findings")
    p.add_argument('build')
    p = sub.add_parser('clarify', help="add the business owner's answers to a request")
    p.add_argument('--request', required=True)
    p.add_argument('--answer', action='append', required=True, help='one clarification sentence (repeatable)')
    p.add_argument('--output', required=True)
    p = sub.add_parser('docker', help='build the hardened image and verify it offline in a container')
    p.add_argument('bundle')
    p.add_argument('--base-image', default=None)
    p = sub.add_parser('serve')
    p.add_argument('bundle')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8090)
    p.add_argument('--user-mode', choices=('structured', 'natural'), default='structured',
                   help='natural: free-dialogue user simulated with V3_USER_* (falls back to LLM_*)')
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
        if args.requests:
            requests = [spec.load(path) for path in sorted(Path(args.requests).glob('*.json'))]
        else:
            requests = [spec.from_v2_frozen(args.frozen, int(s)) for s in args.samples.split(',') if s]
        summary = pipeline.batch(requests, args.output, runner=_runner(args), parallel=args.parallel,
                                 repairs=args.repairs, perturbed=args.perturbed)
        print(json.dumps(summary, ensure_ascii=False, indent=1))
        return 0
    if args.command == 'verify':
        if args.resign:
            report = bundle.requalify(args.bundle)
        else:
            report = bundle.qualify_loaded(bundle.load_bundle(args.bundle, require_current_platform=False))
        print(json.dumps({g: report[g]['passed'] for g in gates.gate_names(report)} | {'passed': report['passed']}))
        return 0 if report['passed'] else 1
    if args.command == 'container-check':
        loaded = bundle.load_bundle(args.bundle)
        report = bundle.qualify_loaded(loaded, judge=None)  # offline: no judge model in the container
        print(json.dumps({'passed': report['passed'], 'platform_identity': bundle.platform_identity(),
                          'files': loaded['manifest']['files'],
                          'gates': {g: report[g]['passed'] for g in gates.gate_names(report)}}))
        return 0 if report['passed'] else 1
    if args.command == 'policy':
        from env_factory_v3 import policy
        policy.load_env_file()
        user = policy.ChatClient('V3_USER') if args.user_mode == 'natural' else None
        report = policy.gate(args.bundle, policy.ChatClient(args.prefix), k=args.k, user_mode=args.user_mode,
                             user_client=user)
        print(json.dumps({k: report[k] for k in ('model', 'user_mode', 'k', 'played', 'successes', 'success_rate',
                                                 'mean_reward', 'status')}))
        return 0 if report['status'] == 'passed' else 1
    if args.command == 'generate':
        from env_factory_v3 import generate
        summary = generate.generate(json.loads(Path(args.domains).read_text()), args.count, args.output,
                                    runner=_runner(args), parallel=args.parallel, python=sys.executable,
                                    rubric_share=args.rubric_share)
        print(json.dumps({k: summary[k] for k in ('requested', 'kept', 'domains')}, ensure_ascii=False))
        return 0
    if args.command == 'domains':
        from env_factory_v3 import domains
        from env_factory_v3.llm import ChatClient, load_env_file
        load_env_file()
        graph = domains.SceneGraph.from_neo4j(os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
                                              os.environ.get('NEO4J_USER', 'neo4j'), os.environ['NEO4J_PASSWORD'],
                                              os.environ.get('NEO4J_DATABASE', 'neo4j'))
        exclude = [d['name'] for path in args.exclude for d in json.loads(Path(path).read_text())]
        report = domains.propose(graph, ChatClient('V3_DOMAIN'), args.count, split=args.split, seed=args.seed,
                                 hops=args.hops, exclude=exclude)
        Path(args.output).write_text(json.dumps(report['accepted'], ensure_ascii=False, indent=1))
        Path(args.output).with_suffix('.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(json.dumps({'split': args.split, 'accepted': len(report['accepted']),
                          'rejected': len(report['rejected']), 'nodes_in_split': len(graph.nodes_in(args.split))},
                         ensure_ascii=False))
        return 0
    if args.command == 'findings':
        status = json.loads((Path(args.build) / 'status.json').read_text())
        report = spec.findings_report(status)
        (Path(args.build) / 'findings.md').write_text(report)
        print(report)
        return 0
    if args.command == 'clarify':
        request = spec.clarify(spec.load(args.request), args.answer)
        Path(args.output).write_text(json.dumps(request, ensure_ascii=False, indent=1))
        print(args.output)
        return 0
    if args.command == 'docker':
        from env_factory_v3 import docker
        options = {'base_image': args.base_image} if args.base_image else {}
        print(json.dumps(docker.build(args.bundle, **options), indent=1))
        return 0
    if args.command == 'serve':
        loaded = bundle.load_bundle(args.bundle)
        user = None
        if args.user_mode == 'natural':
            from env_factory_v3.llm import ChatClient, load_env_file
            load_env_file()
            user = ChatClient('V3_USER')
        service = bundle.service(loaded, user_mode=args.user_mode, user_client=user)
        httpd = server.start(service, args.host, args.port, token=os.environ.get('V3_SERVICE_TOKEN') or None)
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
