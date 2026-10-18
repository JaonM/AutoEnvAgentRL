"""Run the repaired I2n two-seed control with a frozen task split and augmentations.

The two arms differ only in training RNG/order. Comparison with historical I2n
also changes the split, so it is not a causal ablation of the masking fix alone.
Runs in a fresh output directory; never resumes from a partial checkpoint/log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='output/I2nfix')
    parser.add_argument('--port', type=int, default=9010)
    args = parser.parse_args()
    os.chdir(ROOT)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    sources = ['output/I2n/selected.jsonl', 'output/v3_I2/probe/replay.jsonl']
    episodes = [json.loads(line) for path in sources for line in Path(path).read_text().splitlines()]
    tasks = sorted({e['id'] for e in episodes})
    random.Random(20261017).shuffle(tasks)
    held = sorted(tasks[:round(len(tasks) * .1)])
    (out / 'valid_ids.json').write_text(json.dumps(held, indent=2))
    train_tasks = {e['id'] for e in episodes if e['id'] not in held}
    valid_tasks = {e['id'] for e in episodes if e['id'] in held}
    assert train_tasks.isdisjoint(valid_tasks)
    manifest = {'split_seed': 0, 'augmentation_seed': 0, 'task_selection_seed': 20261017,
                'train_tasks': len(train_tasks), 'valid_tasks': len(valid_tasks),
                'train_episodes': sum(e['id'] in train_tasks for e in episodes),
                'valid_episodes': sum(e['id'] in valid_tasks for e in episodes),
                'sources': {p: digest(Path(p)) for p in sources},
                'code': {p: digest(Path(p)) for p in ['experiments/sft_train.py', 'experiments/tool_rename.py',
                                                     'experiments/bench/i2n_control.py']}}
    (out / 'experiment.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    env = {**os.environ, 'PYTHONPATH': 'src'}
    data = []
    for seed in (0, 1):
        run = out / f'seed{seed}'
        run.mkdir()
        cmd = [sys.executable, '-u', 'experiments/sft_train.py', '--teacher', sources[0], '--replay', sources[1],
               '--model', 'models/Qwen3-0.6B', '--output', str(run / 'sft'), '--tuning', 'full',
               '--learning-rate', '2e-5', '--grad-accumulation', '16', '--epochs', '1', '--evals-per-epoch', '8',
               '--seed', str(seed), '--split-seed', '0', '--augmentation-seed', '0',
               '--valid-ids', str(out / 'valid_ids.json'), '--rename-tools', '.7', '--vary-results', '.6',
               '--greet', '.5', '--relayout', '.5', '--no-error-turns', '--max-tokens', '32768',
               '--render', 'episode', '--example-values', '.5', '--inject-errors', '.5',
               '--context-only-announcements']
        print(f'training seed {seed}', flush=True)
        with (run / 'sft.log').open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        data.append(json.loads((run / 'sft/data_manifest.json').read_text()))
        assert data[-1] == data[0], 'rendered training/validation data differ across seeds'
    jobs = []
    for seed in (0, 1):
        arm = f'{out.name}-s{seed}-full'
        for domain in ('airline', 'retail'):
            assert not Path(f'output/tau2/q{arm}-{domain}.log').exists(), 'evaluation arm already exists'
        log = (out / f'eval_seed{seed}.log').open('w')
        process = subprocess.Popen(['bash', 'experiments/bench/quick_eval.sh', arm,
                                    str(out / f'seed{seed}/sft/model'), '1'],
                                   env={**env, 'BENCH_PORT': str(args.port + seed * 34)},
                                   stdout=log, stderr=subprocess.STDOUT)
        jobs.append((process, log, arm))
    failures = []
    for process, log, arm in jobs:
        code = process.wait()
        log.close()
        if code:
            failures.append((arm, code))
        for domain, count in [('airline', 50), ('retail', 114)]:
            path = Path(f'.bench/tau2-repo/data/simulations/v3bench/q{arm}-{domain}/results.json')
            result = json.loads(path.read_text())
            simulations = result['simulations']
            assert len(simulations) == count and len({s['task_id'] for s in simulations}) == count
            assert all(s.get('reward_info') is not None and s.get('termination_reason') != 'infrastructure_error'
                       for s in simulations), f'incomplete/failed evaluation: {arm} {domain}'
    assert not failures, failures
    (out / 'all.done').write_text('done\n')
    print('training and evaluation complete', flush=True)


if __name__ == '__main__':
    main()
