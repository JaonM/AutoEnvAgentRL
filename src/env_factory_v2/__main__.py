"""python -m env_factory_v2: demo, generate, build, verify, serve."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .code_agent import NativeCodeAgent
from .contract import compile_contract
from .examples import EXAMPLES
from .expressions import canonical
from .pipeline import build
from .server import serve
from .verification import qualify


def environment_agent():
    if not os.environ.get('V2_BRIDGE_URL'):
        return None
    return NativeCodeAgent(base_url=os.environ['V2_BRIDGE_URL'],
        token=Path(os.environ['V2_BRIDGE_TOKEN_FILE']).read_text().strip(),
        evidence_root='/tmp/v2-native-evidence')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    demo = sub.add_parser('demo', help='Offline examples: executable checks only, never native qualification')
    demo.add_argument('--example', choices=EXAMPLES, required=True)
    demo.add_argument('--output', type=Path, required=True)
    for name in ('generate', 'build'):
        command = sub.add_parser(name)
        command.add_argument('--request', type=Path, required=True)
        command.add_argument('--connection', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--repairs', type=int, default=1)
        command.add_argument('--without-docker', action='store_true')
        command.add_argument('--base-image', default='python:3.14-slim')
        if name == 'build':
            command.add_argument('--source', type=Path, required=True)
    for name in ('verify', 'container-check', 'serve'):
        command = sub.add_parser(name)
        command.add_argument('contract', type=Path)
        if name == 'serve':
            command.add_argument('--host', default='127.0.0.1')
            command.add_argument('--port', type=int, default=8080)
        if name == 'verify':
            command.add_argument('--connection', type=Path)
    args = parser.parse_args()
    if args.command == 'demo':
        args.output.mkdir(parents=True, exist_ok=False)
        source = EXAMPLES[args.example]()
        compiled, report = qualify(source)
        (args.output / 'contract.json').write_text(canonical(source) + '\n')
        (args.output / 'public.json').write_text(canonical(compiled['public']) + '\n')
        (args.output / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        executable = report['gates']['joint_execution_and_counterexamples'] and report['gates']['deterministic_reset_and_replay']
        print(canonical({'executable_checks_passed': executable, 'qualified': False,
                         'reason': 'No independent native business challenge or Docker verification.', 'output': str(args.output)}))
        return 0 if executable else 1
    if args.command in {'generate', 'build'}:
        agent = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output.parent / (args.output.name + '-native'))
        request = json.loads(args.request.read_text())
        source = json.loads(args.source.read_text()) if args.command == 'build' else None
        result = build(request, agent=agent, output=args.output, source=source, repairs=args.repairs,
                       docker=not args.without_docker, base_image=args.base_image)
        print(canonical(result))
        return 0 if result.get('qualified') or result.get('gates', {}).get('joint_execution_and_counterexamples') and args.without_docker else 1
    source = json.loads(args.contract.read_text())
    agent = environment_agent()
    if args.command == 'serve':
        server = serve(source, host=args.host, port=args.port, agent=agent)
        try:
            server.serve_forever()
        finally:
            server.server_close()
        return 0
    if args.command == 'verify' and args.connection:
        agent = NativeCodeAgent.from_connection(args.connection, evidence_root=args.contract.parent / 'native-verification')
    review = None
    if args.command == 'container-check':
        review = json.loads((args.contract.parent / 'verification.json').read_text()).get('independent_review')
    compiled, report = qualify(source, judge=agent.judge if agent else None,
                              independent_judge=agent.independent_judge if agent else None, independent_review=review)
    executable = report['gates']['joint_execution_and_counterexamples'] and report['gates']['deterministic_reset_and_replay']
    if args.command == 'container-check':
        from .api_verification import verify_api
        api = verify_api(compiled['source'], report, agent=agent)
        print(canonical({'passed': executable and report['gates']['independent_business_challenge'],
                         'contract_identity': compiled['identity'], 'evidence_identity': report['evidence_identity'],
                         'http_api': api, 'scope': 'container_joint_execution'}))
        return 0 if report['passed'] else 1
    print(canonical(report))
    return 0 if executable else 1


if __name__ == '__main__':
    raise SystemExit(main())
