"""Merge trainable tasks of one or more pools and hold out whole domains (plan §4.4).

Writes, under --output:
  tasks_all.json   every trainable task, split "train" (teacher demonstrations are collected on all)
  tasks_rl.json    the same tasks; held-out domains marked split "eval" (RL checkpoint selection only)
  valid_ids.json   ids of the held-out tasks (SFT validation set)
  split.json       which domains were held out

    python experiments/v3/bench/make_split.py --pools output/v3_bench_pool output/v3_bench_pool2 \
        --output output/v3_bench --heldout-share 0.15 [--exclude flagged.json]
--exclude takes a disagreement report (env_factory_v3.production.disagreement) or a JSON list of
task ids; those tasks are left out of every split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--pools', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--heldout-share', type=float, default=0.15)
    parser.add_argument('--exclude', help='disagreement report or JSON list of request ids to leave out')
    args = parser.parse_args()
    excluded = set()
    if args.exclude:
        raw = json.loads(Path(args.exclude).read_text())
        excluded = set(raw['summary']['flagged_ids'] if isinstance(raw, dict) else raw)
    tasks, domain_of = [], {}
    for number, pool in enumerate(args.pools, 1):
        manifest = json.loads((Path(pool) / 'tasks.json').read_text())
        for task in manifest['tasks']:
            detail = manifest['details'][task['id']]
            if task['id'] in excluded or detail.get('id') in excluded:
                continue
            tid = f"p{number}-{task['id']}"
            tasks.append({'id': tid, 'sandbox': task['sandbox'], 'split': 'train', 'weight': 1.0})
            domain_of[tid] = detail.get('domain') or detail.get('family') or tid
    domains = sorted(set(domain_of.values()), key=lambda d: hashlib.sha256(d.encode()).hexdigest())
    wanted = args.heldout_share * len(tasks)
    held, count = [], 0
    for domain in domains:  # deterministic order; whole domains only
        if count >= wanted:
            break
        held.append(domain)
        count += sum(1 for d in domain_of.values() if d == domain)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    valid = [t['id'] for t in tasks if domain_of[t['id']] in held]
    (out / 'tasks_all.json').write_text(json.dumps({'tasks': tasks}, ensure_ascii=False, indent=1))
    rl = [{**t, 'split': 'eval' if t['id'] in set(valid) else 'train'} for t in tasks]
    (out / 'tasks_rl.json').write_text(json.dumps({'tasks': rl}, ensure_ascii=False, indent=1))
    (out / 'valid_ids.json').write_text(json.dumps(valid))
    (out / 'split.json').write_text(json.dumps({'heldout_domains': held, 'tasks': len(tasks),
                                                'heldout_tasks': len(valid), 'domain_of': domain_of},
                                               ensure_ascii=False, indent=1))
    print(f'{len(tasks)} tasks, {len(valid)} held out in {len(held)} domains'
          + (f', {len(excluded)} excluded ids' if excluded else ''))


if __name__ == '__main__':
    main()
