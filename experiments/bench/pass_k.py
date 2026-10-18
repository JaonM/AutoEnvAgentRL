"""Report pass@k (any success) and pass^k (all succeed) separately, rejecting incomplete runs."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path

try:
    from .behavior_funnel import is_write
except ImportError:
    from behavior_funnel import is_write

ROOT = Path('.bench/tau2-repo/data/simulations/v3bench')


def estimates(n, c, k):
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError('require 0 <= successes <= trials and 1 <= k <= trials')
    denominator = math.comb(n, k)
    return {'pass_at_k': 1 - math.comb(n - c, k) / denominator,
            'pass_hat_k': math.comb(c, k) / denominator}


def summarize(rows, k=8):
    if not rows:
        return {'tasks': 0, 'pass_at_1': None, f'pass_at_{k}': None, f'pass_hat_{k}': None,
                'mixed_group_share': None, 'success_count_histogram': {}}
    values = [estimates(r['trials'], r['successes'], k) for r in rows]
    return {'tasks': len(rows), 'pass_at_1': sum(r['successes'] / r['trials'] for r in rows) / len(rows),
            f'pass_at_{k}': sum(v['pass_at_k'] for v in values) / len(values),
            f'pass_hat_{k}': sum(v['pass_hat_k'] for v in values) / len(values),
            'mixed_group_share': sum(0 < r['successes'] < r['trials'] for r in rows) / len(rows),
            'success_count_histogram': dict(sorted(collections.Counter(r['successes'] for r in rows).items()))}


def task_rows(data, domain, expected_ids, trials):
    tasks = {t['id']: t for t in data['tasks']}
    grouped = collections.defaultdict(list)
    for sim in data['simulations']:
        grouped[sim['task_id']].append(sim)
    if set(grouped) != set(expected_ids):
        raise ValueError(f'{domain}: missing or unexpected task ids')
    out = []
    for task_id in sorted(expected_ids):
        sims = grouped[task_id]
        if len(sims) != trials or {s['trial'] for s in sims} != set(range(trials)):
            raise ValueError(f'{domain}/{task_id}: incomplete or duplicate trials')
        if any(s.get('termination_reason') == 'infrastructure_error' or s.get('reward_info') is None
               or s['reward_info'].get('reward') is None for s in sims):
            raise ValueError(f'{domain}/{task_id}: infrastructure failure or missing reward')
        writes = any(is_write(a['name']) for a in (tasks[task_id].get('evaluation_criteria') or {}).get('actions') or [])
        out.append({'domain': domain, 'task_id': task_id, 'write': writes, 'trials': trials,
                    'successes': sum(s['reward_info']['reward'] >= 1 for s in sims),
                    'seeds': [s['seed'] for s in sorted(sims, key=lambda s: s['trial'])]})
    return out


def report(arm, trials, selected=None):
    rows, protocol, hashes = [], {}, {}
    for domain in ('airline', 'retail'):
        path = ROOT / f'{arm}-{domain}/results.json'
        data = json.loads(path.read_text())
        expected = selected[domain] if selected else [t['id'] for t in data['tasks']]
        rows.extend(task_rows(data, domain, expected, trials))
        info = data['info']
        protocol[domain] = {'num_trials': info['num_trials'], 'seed': info['seed'], 'max_steps': info['max_steps'],
                            'agent_model': info['agent_info']['llm'],
                            'agent_temperature': info['agent_info']['llm_args'].get('temperature'),
                            'user_model': info['user_info']['llm'],
                            'user_temperature': info['user_info']['llm_args'].get('temperature')}
        if info['num_trials'] != trials:
            raise ValueError('trial metadata mismatch')
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'arm': arm, 'protocol': protocol, 'overall': summarize(rows),
            'write': summarize([r for r in rows if r['write']]),
            'no_write': summarize([r for r in rows if not r['write']]),
            'domains': {d: summarize([r for r in rows if r['domain'] == d]) for d in protocol},
            'tasks': rows, 'source_sha256': hashes,
            'notes': ['pass_at_8: at least one success in eight attempts; pass_hat_8: all eight succeed.',
                      'A fixed stratified subset is a development probe, not the full benchmark score.',
                      'Repeated API user simulation and floating-point execution are not guaranteed bitwise deterministic.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm')
    parser.add_argument('--trials', type=int, default=8)
    parser.add_argument('--task-ids')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = report(args.arm, args.trials, json.loads(Path(args.task_ids).read_text()) if args.task_ids else None)
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('arm', 'overall', 'write', 'no_write')}, indent=2))


if __name__ == '__main__':
    main()
