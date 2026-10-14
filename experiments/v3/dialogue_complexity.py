"""Dialogue complexity of a batch, measured on its policy-gate episodes (a strong model in free dialogue):
a rich user script does not make a long conversation unless the simulated user plays it and the agent has
to elicit what it needs. Reports, with acceptance lines:
  user_replies        median user messages after the opening (>= 4)
  agent_turns         median assistant turns (>= 8)
  asks_first          share of episodes whose agent asks the user something before its first tool call,
                      among tasks with an identity, ambiguous, clarify or forgotten element (>= 0.7)
  triggered           per script element, share of episodes where it visibly happened (>= 0.8 overall):
                      wrong_first: the wrong value was said; inquisitive: the user asked a question after the
                      opening; multi_goal: at least as many user replies as goals; pushback: at least rounds+1
                      user replies; ambiguous/clarify: the agent asked before its first write.

    python experiments/v3/dialogue_complexity.py output/v3_world_pilot/*/tasks/*/bundle
    python experiments/v3/dialogue_complexity.py --tasks output/run/tasks.json
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

LINES = {'user_replies': 4, 'agent_turns': 8, 'asks_first': 0.7, 'triggered': 0.8}
ASKING = ('identity', 'ambiguous', 'clarify', 'forgotten')


def asks(message):
    text = message.get('content') or ''
    return message['role'] == 'assistant' and not message.get('tool_calls') and ('?' in text or '？' in text)


def episode_stats(request, episode):
    messages = episode['messages']
    users = [m for m in messages if m['role'] == 'user'][1:]
    first_call = next((i for i, m in enumerate(messages) if m['role'] == 'assistant' and m.get('tool_calls')),
                      len(messages))
    script = request.get('user_script') or {}
    elements = [e for e in (request.get('source') or {}).get('script') or []]
    asked_first = any(asks(m) for m in messages[:first_call])
    writes = [i for i, m in enumerate(messages) if m['role'] == 'assistant'
              and any('approv' in (c['function'].get('arguments') or '') or 'consent' in (c['function'].get('arguments') or '')
                      for c in m.get('tool_calls') or [])]
    first_write = writes[0] if writes else len(messages)
    said = ' '.join(m.get('content') or '' for m in users)
    triggered = {}
    for element in elements:
        if element == 'wrong_first' and isinstance(script.get('wrong_first'), dict):
            triggered[element] = any(str(v) in said for v in script['wrong_first'].values())
        elif element == 'inquisitive':
            triggered[element] = '?' in said or '？' in said
        elif element == 'multi_goal':
            triggered[element] = len(users) >= len(script.get('goals') or [1, 1])
        elif element == 'pushback' and isinstance(script.get('pushback'), dict):
            triggered[element] = len(users) >= int(script['pushback'].get('rounds') or 1) + 1
        elif element in ('ambiguous', 'clarify'):
            triggered[element] = any(asks(m) for m in messages[:first_write])
    return {'user_replies': len(users), 'agent_turns': sum(m['role'] == 'assistant' for m in messages),
            'asks_first': asked_first if set(elements) & set(ASKING) else None, 'triggered': triggered}


def report(bundles):
    rows = []
    for folder in bundles:
        folder = Path(folder)
        gate = folder / 'policy-natural.json'
        if not gate.exists():
            continue
        request = json.loads((folder / 'request.json').read_text())
        rows += [episode_stats(request, e) for e in json.loads(gate.read_text()).get('episodes') or []
                 if e.get('messages')]
    if not rows:
        return {'episodes': 0}
    asked = [r['asks_first'] for r in rows if r['asks_first'] is not None]
    per = collections.defaultdict(list)
    for r in rows:
        for element, hit in r['triggered'].items():
            per[element].append(hit)
    hits = [h for values in per.values() for h in values]
    out = {'episodes': len(rows),
           'user_replies': statistics.median(r['user_replies'] for r in rows),
           'agent_turns': statistics.median(r['agent_turns'] for r in rows),
           'asks_first': round(sum(asked) / len(asked), 3) if asked else None,
           'triggered': round(sum(hits) / len(hits), 3) if hits else None,
           'triggered_by_element': {k: f'{sum(v)}/{len(v)}' for k, v in sorted(per.items())}}
    out['passed'] = all(out[k] is None or out[k] >= line for k, line in LINES.items())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('bundles', nargs='*')
    parser.add_argument('--tasks', help='a tasks.json whose sandboxes are measured')
    args = parser.parse_args()
    bundles = list(args.bundles)
    if args.tasks:
        root = Path(args.tasks).parent
        bundles += [root / t['sandbox'] for t in json.loads(Path(args.tasks).read_text())['tasks']]
    print(json.dumps(report(bundles), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
