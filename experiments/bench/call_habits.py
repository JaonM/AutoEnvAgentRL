"""Tool-call habits of quick-eval arms on τ²-bench, with dev_check's metrics: widest turn, invented and
example-copied identifiers ("such as '#W0000000'" in τ²'s tool docstrings), repeating a call right after it
failed, and episodes ended by too many errors.

    python experiments/bench/call_habits.py qA3 qI2p-full qI2p-w0.5
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments'))
from dev_check import call_habits  # noqa: E402

SIMULATIONS = ROOT / '.bench/tau2-repo/data/simulations/v3bench'
TOOLS = str(ROOT / '.bench/tau2-repo/src/tau2/domains/{}/tools.py')


def rows(arm, domain):
    examples = re.findall(r"such as '([^']+)'", Path(TOOLS.format(domain)).read_text())
    tools = [{'function': {'description': ' '.join(f"such as '{e}'" for e in examples)}}]
    data = json.loads((SIMULATIONS / f'{arm}-{domain}' / 'results.json').read_text())
    for simulation in data['simulations']:
        messages = []
        for m in simulation['messages']:
            calls = [{'id': c.get('id'), 'function': {'name': c['name'], 'arguments': json.dumps(c.get('arguments') or {})}}
                     for c in m.get('tool_calls') or []]
            messages.append({'role': m['role'], 'content': m.get('content'), 'tool_calls': calls,
                             'tool_call_id': m.get('id'), 'error': m.get('error')})
        yield {'tools': tools, 'messages': messages,
               'too_many_errors': 'error' in str(simulation.get('termination_reason', '')).lower()}


def summary(arm):
    played = [r for domain in ('airline', 'retail') for r in rows(arm, domain)]
    habits = [call_habits(r) for r in played]
    ids = max(1, sum(h[1] for h in habits))
    repeats = 0
    for row in played:
        failed = {m['tool_call_id'] for m in row['messages'] if m['role'] == 'tool' and m.get('error')}
        previous = None
        for m in row['messages']:
            for call in m['tool_calls'] if m['role'] == 'assistant' else []:
                key = (call['function']['name'], call['function']['arguments'])
                if previous and previous[0] == key and previous[1]:
                    repeats += 1
                    break
                previous = (key, call['id'] in failed)
            else:
                continue
            break
    n = max(1, len(played))
    return {'episodes': len(played), 'too_many_errors': sum(r['too_many_errors'] for r in played),
            'wide_turns': round(sum(h[0] > 3 for h in habits) / n, 3),
            'repeat_failures': round(repeats / n, 3),
            'ungrounded_ids': round(sum(h[2] for h in habits) / ids, 3),
            'example_copies': round(sum(h[3] for h in habits) / ids, 3)}


if __name__ == '__main__':
    for arm in sys.argv[1:]:
        print(arm, json.dumps(summary(arm)))
