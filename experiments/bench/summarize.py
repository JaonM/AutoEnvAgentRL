"""Summarize τ²-bench arms and compare them with paired tests (plan §6).

pass^1 per domain and overall (airline + retail), split by whether the task's gold actions contain a
write (a breakdown added after the baseline showed the no-write tasks pass by doing nothing); paired
per-task comparison of two arms: Wilcoxon signed-rank on per-task success rates and a 10,000-sample
paired bootstrap 95% CI of the difference.

    .bench/tau2/bin/python experiments/bench/summarize.py A C
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / '.bench/tau2-repo/data/simulations/v3bench'  # quick evals: arm qNAME
DOMAINS = ('airline', 'retail')
WRITE = ('book', 'update', 'cancel', 'send_certificate', 'transfer', 'modify', 'exchange', 'return')


def needs_write(task):
    actions = (task.get('evaluation_criteria') or {}).get('actions') or []
    return any(any(word in action['name'] for word in WRITE) for action in actions)


def outcome(sim):
    """Why one simulation passed or failed: pass, infrastructure, max_steps, too_many_errors, db (wrong final
    database), communicate / nl (database right, required information or assertion missed)."""
    info = sim.get('reward_info') or {}
    if (info.get('reward') or 0.0) >= 1 - 1e-9:
        return 'pass'
    reason = sim.get('termination_reason')
    if reason in ('infrastructure_error', 'max_steps', 'too_many_errors'):
        return reason
    breakdown = info.get('reward_breakdown') or {}
    if breakdown.get('DB', 1.0) < 1:
        return 'db'
    return 'communicate' if breakdown.get('COMMUNICATE', 1.0) < 1 else 'nl'


def per_task(arm):
    """{(domain, task id): {'rate': success rate over trials, 'write': bool}}."""
    out = {}
    for domain in DOMAINS:
        path = ROOT / f'{arm}-{domain}' / 'results.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        tasks = {t['id']: t for t in data['tasks']}
        runs = {}
        kinds = {}
        for sim in data['simulations']:
            reward = (sim.get('reward_info') or {}).get('reward') or 0.0
            runs.setdefault(sim['task_id'], []).append(1.0 if reward >= 1 - 1e-9 else 0.0)
            kinds.setdefault(sim['task_id'], []).append(outcome(sim))
        for task_id, rewards in runs.items():
            out[(domain, task_id)] = {'rate': sum(rewards) / len(rewards), 'write': needs_write(tasks[task_id]),
                                      'trials': len(rewards), 'outcomes': kinds[task_id]}
    return out


def table(rows):
    def mean(keys):
        return round(sum(rows[k]['rate'] for k in keys) / len(keys), 3) if keys else None
    groups = {'overall': list(rows)}
    for domain in DOMAINS:
        groups[domain] = [k for k in rows if k[0] == domain]
    groups['needs write'] = [k for k in rows if rows[k]['write']]
    groups['no write'] = [k for k in rows if not rows[k]['write']]
    from collections import Counter
    report = {name: {'tasks': len(keys), 'pass^1': mean(keys)} for name, keys in groups.items()}
    report['outcomes'] = dict(Counter(o for k in rows for o in rows[k]['outcomes']).most_common())
    return report


def wilcoxon(differences):
    """Two-sided Wilcoxon signed-rank p-value (normal approximation, zeros dropped, tie-corrected ranks)."""
    import math
    d = [x for x in differences if abs(x) > 1e-12]
    n = len(d)
    if n == 0:
        return 1.0
    ordered = sorted(range(n), key=lambda i: abs(d[i]))
    ranks, i = [0.0] * n, 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[ordered[j + 1]]) == abs(d[ordered[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[ordered[k]] = (i + j) / 2 + 1
        i = j + 1
    w = sum(r for r, x in zip(ranks, d) if x > 0)
    mu, sigma = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sigma if sigma else 0.0
    return math.erfc(abs(z) / math.sqrt(2))


def compare(a, b, keys):
    diffs = [b[k]['rate'] - a[k]['rate'] for k in keys]
    if not diffs:
        return {'tasks': 0}
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(10000))
    return {'tasks': len(keys), 'difference': round(sum(diffs) / len(diffs), 3),
            'ci95': [round(boots[249], 3), round(boots[9749], 3)], 'wilcoxon_p': round(wilcoxon(diffs), 5)}


def main():
    arms = sys.argv[1:] or ['A']
    rows = {arm: per_task(arm) for arm in arms}
    report = {'arms': {arm: table(r) for arm, r in rows.items() if r}}
    for other in arms[1:]:
        shared = sorted(set(rows[arms[0]]) & set(rows[other]))
        report[f'{other} vs {arms[0]}'] = {
            'overall': compare(rows[arms[0]], rows[other], shared),
            'needs write': compare(rows[arms[0]], rows[other], [k for k in shared if rows[arms[0]][k]['write']]),
            'no write': compare(rows[arms[0]], rows[other], [k for k in shared if not rows[arms[0]][k]['write']])}
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
