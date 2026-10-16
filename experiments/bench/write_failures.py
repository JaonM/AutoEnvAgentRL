"""Why the writes of τ²-bench write tasks fail: for every simulation of a task whose gold actions include a write,
each gold write is matched against the agent's calls of the same tool —
  matched        a call with exactly the gold arguments
  wrong_args     the tool was called, never with the gold arguments (the differing argument names are counted)
  missing        the tool was never called
plus the agent's writes that no gold write asked for (extra), and whether the gold reads before the write matched.

    .bench/tau2/bin/python experiments/bench/write_failures.py qA3 qI2s-s0-full [--examples 4]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize  # noqa: E402


def is_write(name):
    return any(word in name for word in summarize.WRITE) and not name.startswith('transfer_to_human')


def diff_keys(gold, made):
    keys = set(gold) | set(made)
    return sorted(k for k in keys if json.dumps(gold.get(k), sort_keys=True) != json.dumps(made.get(k), sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('arms', nargs='+')
    parser.add_argument('--examples', type=int, default=3)
    args = parser.parse_args()
    for arm in args.arms:
        kinds, arg_keys, tools, sims, solved, extra, reads = (collections.Counter(), collections.Counter(),
                                                              collections.Counter(), 0, 0, 0, [0, 0])
        examples = collections.defaultdict(list)
        for domain in summarize.DOMAINS:
            data = json.loads((summarize.ROOT / f'{arm}-{domain}' / 'results.json').read_text())
            tasks = {t['id']: t for t in data['tasks']}
            for sim in data['simulations']:
                gold = (tasks[sim['task_id']].get('evaluation_criteria') or {}).get('actions') or []
                gold_writes = [a for a in gold if is_write(a['name'])]
                if not gold_writes:
                    continue
                sims += 1
                solved += ((sim.get('reward_info') or {}).get('reward') or 0) >= 1
                made = [(c['name'], c.get('arguments') or {}) for m in sim['messages'] if m.get('role') == 'assistant'
                        for c in (m.get('tool_calls') or [])]
                checks = (sim.get('reward_info') or {}).get('action_checks') or []
                for check in checks:
                    if not is_write(check['action']['name']):
                        reads[0] += bool(check.get('action_match'))
                        reads[1] += 1
                used = []
                for action in gold_writes:
                    same = [(i, a) for i, (n, a) in enumerate(made) if n == action['name']]
                    hit = next((i for i, a in same if not diff_keys(action['arguments'], a)), None)
                    if hit is not None:
                        kinds['matched'] += 1
                        used.append(hit)
                    elif same:
                        kinds['wrong_args'] += 1
                        closest = min(same, key=lambda s: len(diff_keys(action['arguments'], s[1])))
                        used.append(closest[0])
                        for key in diff_keys(action['arguments'], closest[1]):
                            arg_keys[key] += 1
                        if len(examples[('wrong_args', domain)]) < args.examples:
                            d = {k: (action['arguments'].get(k), closest[1].get(k)) for k in diff_keys(action['arguments'], closest[1])}
                            examples[('wrong_args', domain)].append(f"{domain} task {sim['task_id']} {action['name']}: gold vs made {d}")
                    else:
                        kinds['missing'] += 1
                        tools[action['name']] += 1
                extra += sum(1 for i, (n, _) in enumerate(made) if is_write(n) and i not in used)
        total = max(1, sum(kinds.values()))
        print(f"\n{arm}: {sims} write-task simulations, {solved} solved; gold writes {sum(kinds.values())}: "
              + ', '.join(f'{k} {v} ({v / total:.0%})' for k, v in kinds.most_common())
              + f"; extra writes {extra}; gold reads matched {reads[0]}/{reads[1]}")
        print('  arguments that differ (wrong_args):', ', '.join(f'{k} {v}' for k, v in arg_keys.most_common(12)))
        print('  tools never called (missing):', ', '.join(f'{k} {v}' for k, v in tools.most_common(8)))
        for lines in examples.values():
            for line in lines:
                print('   ', line[:300])


if __name__ == '__main__':
    main()
