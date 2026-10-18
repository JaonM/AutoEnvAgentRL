"""Fixed 32-task, eight-rollout probe: base model and both repaired SFT seeds."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

from behavior_funnel import is_write
from pass_k import report

ROOT = Path(__file__).resolve().parents[2]


def select_tasks(size=32, seed=20261017):
    # Use task definitions only, never rewards or model trajectories to select tasks.
    strata = collections.defaultdict(list)
    task_hashes = {}
    for domain in ('airline', 'retail'):
        data = json.loads(Path(f'.bench/tau2-repo/data/simulations/v3bench/qA3m-{domain}/results.json').read_text())
        task_hashes[domain] = hashlib.sha256(json.dumps(data['tasks'], sort_keys=True).encode()).hexdigest()
        for task in data['tasks']:
            write = any(is_write(a['name']) for a in (task.get('evaluation_criteria') or {}).get('actions') or [])
            strata[(domain, write)].append(task['id'])
    total = sum(map(len, strata.values()))
    quotas = {s: len(ids) * size // total for s, ids in strata.items()}
    order = sorted(strata, key=lambda s: (-(len(strata[s]) * size % total), s))
    for s in order[:size-sum(quotas.values())]:
        quotas[s] += 1
    selected = {'airline': [], 'retail': []}
    for stratum, ids in sorted(strata.items()):
        ids = sorted(ids)
        random.Random(f'{seed}:{stratum}').shuffle(ids)
        selected[stratum[0]].extend(ids[:quotas[stratum]])
    selected = {d: sorted(ids) for d, ids in selected.items()}
    assert sum(map(len, selected.values())) == size
    return selected, {'selection_seed': seed, 'population_tasks': total,
                      'strata': {f'{d}:{"write" if w else "no_write"}': quotas[(d, w)] for d, w in strata},
                      'task_definitions_sha256': task_hashes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='output/I2nfix/pass8')
    parser.add_argument('--after', default='output/I2nfix/all.done')
    args = parser.parse_args()
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    selected, manifest = select_tasks()
    ids_path = out / 'task_ids.json'
    ids_path.write_text(json.dumps(selected, indent=2))
    arms = [('I2nfix-p8-base', 'models/Qwen3-0.6B'),
            ('I2nfix-p8-s0', 'output/I2nfix/seed0/sft/model'),
            ('I2nfix-p8-s1', 'output/I2nfix/seed1/sft/model')]
    manifest.update(trials=8, temperature=1.0, max_steps=100, benchmark_seed=300, tasks=selected,
                    arms=dict(arms), scope='fixed stratified development subset, not full-benchmark pass@8',
                    code_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [
                        'experiments/bench/run_pass8_control.py', 'experiments/bench/pass_k.py',
                        'experiments/bench/serve_openai.py', 'experiments/bench/quick_eval.sh',
                        'experiments/bench/run_tau2.sh']})
    (out / 'experiment.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    print(f'waiting for {args.after}', flush=True)
    while not Path(args.after).exists():
        if 'Traceback' in Path('/tmp/i2nfix-control.log').read_text():
            raise RuntimeError('preceding control failed; pass@8 not started')
        time.sleep(30)
    jobs = []
    for i, (arm, model) in enumerate(arms):
        assert not Path(f'output/tau2/q{arm}-airline.log').exists(), 'arm already exists'
        log = (out / f'{arm}.log').open('w')
        p = subprocess.Popen(['bash', 'experiments/bench/quick_eval.sh', arm, model, '8'],
                             env={**os.environ, 'PYTHONPATH': 'src', 'TASK_IDS_FILE': str(ids_path.resolve()),
                                  'AGENT_TEMPERATURE': '1.0', 'BENCH_PORT': str(9110+i*34),
                                  'BATCH': '0', 'THINK': '0', 'SERVERS': '3', 'BENCH_MAX_EVALS': '2'},
                             stdout=log, stderr=subprocess.STDOUT)
        jobs.append((arm, p, log))
    results = {}
    failures = []
    for arm, p, log in jobs:
        code = p.wait(); log.close()
        if code:
            failures.append((arm, code)); continue
        r = report('q'+arm, 8, selected)
        assert all(x['agent_temperature'] == 1.0 and x['max_steps'] == 100 for x in r['protocol'].values())
        results[arm] = r
    if failures:
        raise RuntimeError(f'evaluation failed: {failures}')
    (out / 'report.json').write_text(json.dumps(results, indent=2))
    lines = ['32-task fixed subset, 8 trials, agent temperature 1.0',
             'arm | mean pass@1 | pass@8 | write pass@8 | pass^8 | mixed groups']
    for arm, r in results.items():
        a, w = r['overall'], r['write']
        lines.append(f"{arm} | {a['pass_at_1']:.1%} | {a['pass_at_8']:.1%} | {w['pass_at_8']:.1%} | "
                     f"{a['pass_hat_8']:.1%} | {a['mixed_group_share']:.1%}")
    (out / 'report.txt').write_text('\n'.join(lines)+'\n')
    (out / 'all.done').write_text('done\n')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
