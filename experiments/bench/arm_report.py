"""Every metric of τ²-bench quick-eval arms in one table, compared with a reference arm (the first).

Overall pass^1 alone misled us (2026-10-16): every SFT arm scored 4/134 on the tasks that need a write, and all of
their overall gains came from the 30 no-write tasks, which pass by doing little — the "best" arms barely called
tools. So each arm is reported on
  score        overall, needs-write (134 tasks), no-write (30), airline, retail pass^1
  tool health  calls per simulation, simulations with no tool call, assistant turns announcing a lookup
               ("let me check") without making it, simulations that attempted a write when the task needs one
  habits       call_habits.py: too many errors, wide turns, repeated failed calls, ungrounded / example ids
and paired against the reference on overall, needs-write and no-write tasks (bootstrap CI, Wilcoxon).

    .bench/tau2/bin/python experiments/bench/arm_report.py qA3 qI2s-s0-full qI2s-s1-full
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import call_habits  # noqa: E402
import summarize  # noqa: E402

ANNOUNCE = re.compile(r"\b(let me (check|look|get|find|verify|pull|see)|i'll (check|look|get|find|pull)|"
                      r"i will (check|look|get|find))\b", re.I)


def tool_health(arm):
    sims = calls = silent = announced = write_tasks = attempted = 0
    for domain in summarize.DOMAINS:
        path = summarize.ROOT / f'{arm}-{domain}' / 'results.json'
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        tasks = {t['id']: t for t in data['tasks']}
        for sim in data['simulations']:
            assistant = [m for m in sim['messages'] if m.get('role') == 'assistant']
            made = [c['name'] for m in assistant for c in (m.get('tool_calls') or [])]
            sims += 1
            calls += len(made)
            silent += not made
            announced += sum(1 for m in assistant if not m.get('tool_calls') and ANNOUNCE.search(m.get('content') or ''))
            if summarize.needs_write(tasks[sim['task_id']]):
                write_tasks += 1
                attempted += any(any(w in name for w in summarize.WRITE) for name in made)
    n = max(1, sims)
    return {'calls_per_sim': round(calls / n, 1), 'no_call_sims': round(silent / n, 3),
            'announce_without_call': round(announced / n, 2), 'write_attempted': round(attempted / max(1, write_tasks), 3)}


def main():
    arms = sys.argv[1:]
    if not arms:
        sys.exit(__doc__)
    rows = {arm: summarize.per_task(arm) for arm in arms}
    reference = arms[0]
    print(f"{'arm':22s} {'overall':>8s} {'write':>9s} {'no-write':>9s} {'airline':>8s} {'retail':>7s} | "
          f"{'calls':>5s} {'no-call':>7s} {'announce':>8s} {'wrote':>6s} | {'tme':>3s} {'repeat':>6s} {'ungr.':>6s} {'examp.':>6s}")
    for arm in arms:
        if not rows[arm]:
            print(f'{arm:22s} (no results)')
            continue
        table = summarize.table(rows[arm])
        write, nowrite = table['needs write'], table['no write']
        health, habits = tool_health(arm), call_habits.summary(arm)
        print(f"{arm:22s} {table['overall']['pass^1']:8.3f} {write['pass^1']:5.3f}({round(write['pass^1'] * write['tasks']):3d}) "
              f"{nowrite['pass^1']:5.3f}({round(nowrite['pass^1'] * nowrite['tasks']):2d}) "
              f"{table['airline']['pass^1']:8.3f} {table['retail']['pass^1']:7.3f} | "
              f"{health['calls_per_sim']:5.1f} {health['no_call_sims']:7.0%} {health['announce_without_call']:8.2f} "
              f"{health['write_attempted']:6.0%} | {habits['too_many_errors']:3d} {habits['repeat_failures']:6.0%} "
              f"{habits['ungrounded_ids']:6.0%} {habits['example_copies']:6.0%}")
    for arm in arms[1:]:
        shared = sorted(set(rows[reference]) & set(rows[arm]))
        parts = {'overall': shared, 'write': [k for k in shared if rows[reference][k]['write']],
                 'no-write': [k for k in shared if not rows[reference][k]['write']]}
        cells = []
        for name, keys in parts.items():
            c = summarize.compare(rows[reference], rows[arm], keys)
            if c.get('tasks'):
                cells.append(f"{name} {c['difference']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}] p={c['wilcoxon_p']:.2f}")
        print(f'  {arm} vs {reference}: ' + '; '.join(cells))


if __name__ == '__main__':
    main()
