"""Behaviour check of a model on held-out tasks before any τ²-bench evaluation.

B1 failed τ²-bench by habits that show on our own tasks too: inventing an identifier and retrying the
same failing call, calling tools that do not exist, talking until the step limit. This plays the
model (served by serve_openai.py) on held-out tasks through the training interface and reports:
  success rate and mean reward,
  repeat_failures   share of episodes that repeat an identical call right after it failed,
  unknown_tools     share of episodes calling a tool that is not in their tool list,
  step_limit        share of episodes the step limit ended (the user never left; rows record terminated),
  asks_before_lookup share of episodes whose first account lookup comes after asking the user.
A model passes when none of these is worse than the base model's by more than the margins.

    python experiments/dev_check.py --tasks DEV.json --port 8790 --model-name m --output OUT.jsonl
    python experiments/dev_check.py --report OUT.jsonl [--against BASE.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def metrics(rows):
    played = [r for r in rows if r.get('messages')]
    n = max(1, len(played))
    repeat = unknown = limit = asks = lookups = 0
    for row in played:
        names = {t['function']['name'] for t in row.get('tools') or []}
        results = {m.get('tool_call_id'): m.get('content') or '' for m in row['messages'] if m['role'] == 'tool'}
        previous, repeated, unknown_here, asked, looked = None, False, False, False, False
        for message in row['messages']:
            if message['role'] != 'assistant':
                continue
            calls = message.get('tool_calls') or []
            if not calls and (message.get('content') or '').strip():
                if not looked and ('?' in message['content'] or '？' in message['content']):
                    asked = True
            for call in calls:
                key = (call['function']['name'], call['function'].get('arguments'))
                failed = '"ok": false' in results.get(call.get('id'), '') or \
                    results.get(call.get('id'), '').startswith('Error')
                if previous and previous[0] == key and previous[1]:
                    repeated = True
                previous = (key, failed)
                if call['function']['name'] not in names:
                    unknown_here = True
                looked = True
        repeat += repeated
        unknown += unknown_here
        limit += row.get('terminated') is False  # the user never left: the step limit ended it
        if looked:
            lookups += 1
            asks += asked
    return {'episodes': len(played), 'success': round(sum(bool(r.get('success')) for r in played) / n, 3),
            'mean_reward': round(sum(r.get('reward') or 0 for r in played) / n, 3),
            'repeat_failures': round(repeat / n, 3), 'unknown_tools': round(unknown / n, 3),
            'step_limit': round(limit / n, 3), 'asks_before_lookup': round(asks / max(1, lookups), 3)}


MARGINS = {'success': -0.05, 'repeat_failures': 0.05, 'unknown_tools': 0.03, 'step_limit': 0.05}


def verdict(model, base):
    problems = []
    if model['success'] - base['success'] < MARGINS['success']:
        problems.append(f"success {model['success']} < base {base['success']}")
    for key in ('repeat_failures', 'unknown_tools', 'step_limit'):
        if model[key] - base[key] > MARGINS[key]:
            problems.append(f'{key} {model[key]} > base {base[key]}')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks')
    parser.add_argument('--port', type=int, default=8790)
    parser.add_argument('--model-name', default='candidate')
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--output')
    parser.add_argument('--report', help='only report metrics of an existing output')
    parser.add_argument('--against', help="the base model's output, for a pass/fail verdict")
    args = parser.parse_args()
    if not args.report:
        env = {**os.environ, 'DEV_BASE_URL': f'http://127.0.0.1:{args.port}/v1', 'DEV_MODEL': args.model_name,
               'DEV_API_KEY': 'local', 'PYTHONPATH': 'src', 'ENVFACTORY_CONSENT': 'plain',
               'ENVFACTORY_RESULT_FORMAT': 'random'}
        # collect_teacher plays only train-split tasks: hand it the held-out tasks marked as such
        tasks = json.loads(Path(args.tasks).read_text())
        playable = Path(args.output).with_suffix('.tasks.json')
        playable.write_text(json.dumps({'tasks': [{**t, 'split': 'train'} for t in tasks['tasks']]}, ensure_ascii=False))
        subprocess.run([sys.executable, '-u', 'experiments/collect_teacher.py', '--tasks', str(playable),
                        '--episodes', str(args.episodes), '--parallel', '8', '--prefix', 'DEV', '--no-style',
                        '--output', args.output], env=env, check=True)
    path = args.report or args.output
    report = metrics([json.loads(line) for line in Path(path).read_text().splitlines()])
    if args.against:
        base = metrics([json.loads(line) for line in Path(args.against).read_text().splitlines()])
        report = {'model': report, 'base': base, 'problems': verdict(report, base)}
        report['passed'] = not report['problems']
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
