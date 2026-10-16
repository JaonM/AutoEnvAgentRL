"""Where τ²-bench write tasks fail, step by step, and how each behaviour fares (step 1 of proving a bottleneck).

Each write task's gold actions define the steps it needs; each simulation is checked against them in order:
  identify       the customer was found (retail: a find_user_id_* call that returned an id; airline: get_user_details
                 of the gold user succeeded)
  look_up        every gold order / reservation was read (get_order_details / get_reservation_details on its id)
  inspect        every gold product / flight search the task needs was made (get_product_details, search_*)
  write_tool     every gold write tool was called at least once
  write_args     some call of each gold write tool had the gold arguments, judged per argument kind (record id, item
                 id list, object list, payment method, other) — the first wrong kind is recorded
  pass           the task's reward was 1
A simulation's "stuck at" is its first failing step. Behaviours are reported apart: identification when the user does
not know the primary identifier (the scenario says so), payment choice, id lists, object lists, variant choice.

    .bench/tau2/bin/python experiments/bench/behavior_funnel.py qA3 qI2s-s0-full qI2f-s0-full
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize  # noqa: E402

STEPS = ('identify', 'look_up', 'inspect', 'write_tool', 'write_args', 'pass')
UNKNOWN_ID = re.compile(r"(do not|don't|can ?not|cannot) (remember|recall|know|find)|forgot|not remember", re.I)


def is_write(name):
    return any(w in name for w in summarize.WRITE) and not name.startswith('transfer_to_human')


def arg_kind(key, value):
    if 'payment' in key:
        return 'payment'
    if isinstance(value, list):
        return 'object list' if value and isinstance(value[0], dict) else 'id list'
    if key in ('order_id', 'reservation_id', 'user_id'):
        return 'record id'
    return 'other'


def same(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def compared(action):
    """The gold arguments τ² compares (compare_args when the task names them, else all)."""
    keys = action.get('compare_args')
    return {k: v for k, v in action['arguments'].items() if keys is None or k in keys}


def analyse(sim, task):
    gold = (task.get('evaluation_criteria') or {}).get('actions') or []
    writes = [a for a in gold if is_write(a['name'])]
    if not writes:
        return None
    msgs = sim['messages']
    results = {m.get('id'): (m.get('content') or '') for m in msgs if m.get('role') == 'tool'}
    calls = [(c['name'], c.get('arguments') or {}, results.get(c.get('id'), ''))
             for m in msgs if m.get('role') == 'assistant' for c in (m.get('tool_calls') or [])]
    ok = lambda r: bool(r) and not r.startswith('Error')
    steps = {}
    gold_user = next((a['arguments'].get('user_id') for a in gold if 'user_id' in a['arguments']), None)
    if any(a['name'].startswith('find_user_id') for a in gold):
        steps['identify'] = any(n.startswith('find_user_id') and ok(r) for n, _, r in calls)
    elif gold_user:
        steps['identify'] = any(n == 'get_user_details' and a.get('user_id') == gold_user and ok(r) for n, a, r in calls)
    else:
        steps['identify'] = True
    lookups = [a for a in gold if a['name'] in ('get_order_details', 'get_reservation_details')]
    steps['look_up'] = all(any(n == g['name'] and same(a, g['arguments']) and ok(r) for n, a, r in calls) for g in lookups)
    inspects = [a for a in gold if a['name'] == 'get_product_details' or a['name'].startswith('search_')]
    steps['inspect'] = all(any(n == g['name'] and same(a, g['arguments']) for n, a, _ in calls) for g in inspects)
    steps['write_tool'] = all(any(n == g['name'] for n, _, _ in calls) for g in writes)
    wrong_kinds = []
    for g in writes:
        made = [a for n, a, _ in calls if n == g['name']]
        if not made:
            continue
        if any(all(same(a.get(k), v) for k, v in compared(g).items()) for a in made):
            continue
        best = max(made, key=lambda a: sum(same(a.get(k), v) for k, v in compared(g).items()))
        wrong_kinds += [arg_kind(k, v) for k, v in compared(g).items() if not same(best.get(k), v)]
    steps['write_args'] = steps['write_tool'] and not wrong_kinds
    steps['pass'] = ((sim.get('reward_info') or {}).get('reward') or 0) >= 1
    stuck = next((s for s in STEPS if not steps.get(s, True)), None)
    scenario = json.dumps(task.get('user_scenario') or {}, ensure_ascii=False)
    behaviours = {
        'identify (user lacks the primary id)': steps['identify'] if UNKNOWN_ID.search(scenario) and
        any(a['name'] == 'find_user_id_by_name_zip' for a in gold) else None,
        'payment choice': None, 'id list': None, 'object list': None,
        'variant choice': None,
    }
    if any('new_item_ids' in g['arguments'] for g in writes):
        made = [a for n, a, _ in calls if any(n == g['name'] and 'new_item_ids' in g['arguments'] for g in writes)]
        target = next(g['arguments']['new_item_ids'] for g in writes if 'new_item_ids' in g['arguments'])
        behaviours['variant choice'] = any(same(sorted(a.get('new_item_ids') or []), sorted(target)) for a in made)
    kinds = {arg_kind(k, v) for g in writes for k, v in compared(g).items()}
    for kind, label in (('payment', 'payment choice'), ('id list', 'id list'), ('object list', 'object list')):
        if kind in kinds:
            behaviours[label] = steps['write_tool'] and kind not in wrong_kinds
    return {'steps': steps, 'stuck': stuck, 'wrong_kinds': wrong_kinds, 'behaviours': behaviours}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('arms', nargs='+')
    args = parser.parse_args()
    for arm in args.arms:
        rows = []
        for domain in summarize.DOMAINS:
            data = json.loads((summarize.ROOT / f'{arm}-{domain}' / 'results.json').read_text())
            tasks = {t['id']: t for t in data['tasks']}
            for sim in data['simulations']:
                row = analyse(sim, tasks[sim['task_id']])
                if row:
                    rows.append(row)
        n = len(rows)
        reached = {s: sum(all(r['steps'].get(t, True) for t in STEPS[:i + 1]) for r in rows) for i, s in enumerate(STEPS)}
        stuck = collections.Counter(r['stuck'] for r in rows if r['stuck'])
        kinds = collections.Counter(k for r in rows for k in set(r['wrong_kinds']))
        print(f'\n{arm}: {n} write-task simulations')
        print('  funnel (passed every step so far): ' + ' -> '.join(f'{s} {reached[s]}' for s in STEPS))
        print('  first failing step: ' + ', '.join(f'{s} {stuck[s]} ({stuck[s] / n:.0%})' for s in STEPS if stuck[s]))
        print('  wrong argument kinds (sims): ' + ', '.join(f'{k} {v}' for k, v in kinds.most_common()))
        for label in ('identify (user lacks the primary id)', 'payment choice', 'id list', 'object list', 'variant choice'):
            vals = [r['behaviours'][label] for r in rows if r['behaviours'][label] is not None]
            if vals:
                print(f'  {label:38s} needed in {len(vals):3d}, done right {sum(vals):3d} ({sum(vals) / len(vals):.0%})')


if __name__ == '__main__':
    main()
