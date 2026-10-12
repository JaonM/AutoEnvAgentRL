"""Give existing V3 bundles an agent-facing policy (request["agent_policy"]), then re-certify them.

For each bundle: write the policy (policies.write: no internals, coverage checked), store it in the
bundle's request.json, re-run every gate and re-sign (bundle.requalify), rebuild the hardened image,
and run the policy gate in free dialogue, which now reads only the policy. A bundle whose policy,
gates, image or policy gate fails is reported, not trained on. Results are appended per bundle.

    python experiments/v3/add_agent_policies.py --tasks output/v3_sft/tasks.json --output output/v3_policies.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from pathlib import Path


def amend(folder, policy):
    """Store the policy in the bundle's request.json; requalify then re-certifies the changed bundle."""
    from env_factory_v3 import bundle
    path = folder / 'request.json'
    request = json.loads(path.read_text())
    request['agent_policy'] = policy
    path.write_text(json.dumps(request, ensure_ascii=False, indent=1))
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['files']['request.json'] = bundle.sha256(path)
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


def one(task, clients, k):
    from env_factory_v3 import bundle, docker, policies, policy
    folder = Path(task['sandbox'])
    row, started = {'id': task['id'], 'sandbox': str(folder)}, time.monotonic()
    try:
        loaded = bundle.load_bundle(folder, require_current_platform=False)
        request = loaded['request']
        text = request.get('agent_policy')
        if not text:
            text, problems = policies.write(clients['writer'], request, tuple(loaded['env'].make_tools()))
            if text is None:
                return {**row, 'stage': 'policy', 'problems': problems[:5]}
            amend(folder, text)
        row['policy_chars'], row['rules_chars'] = len(text), len(request['business_rules'])
        report = bundle.requalify(folder)
        if not report['passed']:
            return {**row, 'stage': 'gates', 'failed': [g for g in report['gate_names'] if not report[g]['passed']]}
        docker.build(folder)
        gate = policy.gate(folder, clients['agent'], k=k, user_mode='natural', user_client=clients['user'])
        row.update(stage='done' if gate['status'] == 'passed' else 'policy_gate', successes=gate.get('successes'),
                   played=gate.get('played'))
        return row
    except Exception as exc:
        return {**row, 'stage': 'error', 'error': f'{type(exc).__name__}: {exc}'[:300]}
    finally:
        row['seconds'] = round(time.monotonic() - started)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--parallel', type=int, default=6)
    parser.add_argument('--policy-k', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from env_factory_v3.llm import ChatClient, load_env_file
    load_env_file()
    clients = {'writer': ChatClient('V3_OPENING'), 'agent': ChatClient('LLM'), 'user': ChatClient('V3_USER')}
    tasks = json.loads(Path(args.tasks).read_text())['tasks']
    out, lock = Path(args.output), threading.Lock()
    done = {json.loads(line)['id'] for line in out.read_text().splitlines()} if out.exists() else set()

    def run(task):
        row = one(task, clients, args.policy_k)
        with lock:
            with out.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({k: row.get(k) for k in ('id', 'stage', 'successes', 'problems', 'failed', 'error',
                                                  'policy_chars', 'rules_chars', 'seconds')}, ensure_ascii=False),
              flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        rows = list(pool.map(run, [t for t in tasks if t['id'] not in done]))
    print(f"DONE {sum(r['stage'] == 'done' for r in rows)}/{len(rows)} ready", flush=True)


if __name__ == '__main__':
    main()
