"""Behaviour check of a model on held-out V3 tasks before any τ²-bench evaluation.

B1 failed τ²-bench by habits that show on our own tasks too: inventing an identifier and retrying the
same failing call, calling tools that do not exist, talking until the step limit. This plays the
model (served by serve_openai.py) on held-out tasks through the training interface and reports:
  success rate and mean reward,
  repeat_failures   share of episodes that repeat an identical call right after it failed,
  unknown_tools     share of episodes calling a tool that is not in their tool list,
  step_limit        share of episodes the step limit ended (the user never left; rows record terminated),
  asks_before_lookup share of episodes whose first account lookup comes after asking the user,
  wide_turns        share of episodes with a turn making more than 3 tool calls,
  ungrounded_ids    share of identifier arguments never seen before in the conversation (invented),
  example_copies    share of identifier arguments equal to an example value of a tool description.
A model passes when none of these is worse than the base model's by more than the margins.

Sampling matches τ²-bench's quick eval: the candidate decodes greedily (temperature 0) and the simulated user runs at
temperature 0, so two models meet the same user; with the earlier 0.7 / 0.3 the same model scored 62% then 3.5% step
limits. Eight episodes per task by default, and every metric carries a 95% interval from a bootstrap over tasks (the
episodes of one task are not independent); with --against, the per-task paired difference gets one too.

    python experiments/v3/dev_check.py --tasks DEV.json --port 8790 --model-name m --output OUT.jsonl
    python experiments/v3/dev_check.py --report OUT.jsonl [--against BASE.jsonl]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ID_KEY = re.compile(r'(^|_)(id|ids|ref|reference|number|no|code)$', re.IGNORECASE)
EXAMPLE = re.compile(r"(?:such as|e\.g\.,?|例如)\s*'([^']+)'", re.IGNORECASE)


def call_habits(row):
    """(widest turn, identifier arguments, invented ones, example copies) of one episode."""
    examples = set(EXAMPLE.findall(json.dumps(row.get('tools') or [], ensure_ascii=False)))
    widest = ids = invented = copies = 0
    seen = ''
    for message in row['messages']:
        calls = message.get('tool_calls') or [] if message['role'] == 'assistant' else []
        widest = max(widest, len(calls))
        for call in calls:
            arguments = call['function'].get('arguments') or '{}'
            try:
                values = json.loads(arguments) if isinstance(arguments, str) else arguments
            except ValueError:
                values = {}
            for key, value in (values if isinstance(values, dict) else {}).items():
                if ID_KEY.search(key) and isinstance(value, str) and value:
                    ids += 1
                    invented += value not in seen
                    copies += value in examples
        seen += json.dumps({k: message.get(k) for k in ('content', 'tool_calls')}, ensure_ascii=False) \
            if message['role'] != 'assistant' else ''
    return widest, ids, invented, copies


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
    habits = [call_habits(r) for r in played]
    id_total = max(1, sum(h[1] for h in habits))
    return {'episodes': len(played), 'success': round(sum(bool(r.get('success')) for r in played) / n, 3),
            'mean_reward': round(sum(r.get('reward') or 0 for r in played) / n, 3),
            'repeat_failures': round(repeat / n, 3), 'unknown_tools': round(unknown / n, 3),
            'step_limit': round(limit / n, 3), 'asks_before_lookup': round(asks / max(1, lookups), 3),
            'wide_turns': round(sum(h[0] > 3 for h in habits) / n, 3),
            'ungrounded_ids': round(sum(h[2] for h in habits) / id_total, 3),
            'example_copies': round(sum(h[3] for h in habits) / id_total, 3)}


CI_KEYS = ('success', 'mean_reward', 'repeat_failures', 'step_limit', 'wide_turns')


def _by_task(rows):
    """{task: [per-episode metric values]}: each played episode's metrics computed once."""
    import collections
    groups = collections.defaultdict(list)
    for row in rows:
        if row.get('messages'):
            groups[row['id']].append({k: float(v) for k, v in metrics([row]).items() if k in CI_KEYS})
    return groups


def _mean(episodes, key):
    return sum(e[key] for e in episodes) / max(1, len(episodes))


def intervals(rows, draws=2000, seed=0):
    """{metric: [low, high]}: a 95% bootstrap interval, resampling tasks (not episodes)."""
    import random
    groups = _by_task(rows)
    tasks = sorted(groups)
    if len(tasks) < 2:
        return {}
    rng, samples = random.Random(seed), {k: [] for k in CI_KEYS}
    for _ in range(draws):
        drawn = [e for task in (rng.choice(tasks) for _ in tasks) for e in groups[task]]
        for key in CI_KEYS:
            samples[key].append(_mean(drawn, key))
    return {k: [round(sorted(v)[int(0.025 * draws)], 3), round(sorted(v)[int(0.975 * draws) - 1], 3)]
            for k, v in samples.items()}


def paired(rows, base_rows, draws=2000, seed=0):
    """{metric: [low, high]} of (model - base), resampling the tasks both played."""
    import random
    model, base = _by_task(rows), _by_task(base_rows)
    tasks = sorted(set(model) & set(base))
    if len(tasks) < 2:
        return {}
    rng, samples = random.Random(seed), {k: [] for k in CI_KEYS}
    for _ in range(draws):
        drawn = [rng.choice(tasks) for _ in tasks]
        a = [e for t in drawn for e in model[t]]
        b = [e for t in drawn for e in base[t]]
        for key in CI_KEYS:
            samples[key].append(_mean(a, key) - _mean(b, key))
    return {k: [round(sorted(v)[int(0.025 * draws)], 3), round(sorted(v)[int(0.975 * draws) - 1], 3)]
            for k, v in samples.items()}


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
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--temperature', type=float, default=0.0, help="the candidate's sampling temperature")
    parser.add_argument('--output')
    parser.add_argument('--report', help='only report metrics of an existing output')
    parser.add_argument('--against', help="the base model's output, for a pass/fail verdict")
    args = parser.parse_args()
    if not args.report:
        env = {**os.environ, 'DEV_BASE_URL': f'http://127.0.0.1:{args.port}/v1', 'DEV_MODEL': args.model_name,
               'DEV_API_KEY': 'local', 'PYTHONPATH': 'src', 'ENVFACTORY_V3_CONSENT': 'plain',
               'ENVFACTORY_V3_RESULT_FORMAT': 'random', 'ENVFACTORY_V3_USER_TEMPERATURE': '0'}
        # collect_teacher plays only train-split tasks: hand it the held-out tasks marked as such
        tasks = json.loads(Path(args.tasks).read_text())
        playable = Path(args.output).with_suffix('.tasks.json')
        playable.write_text(json.dumps({'tasks': [{**t, 'split': 'train'} for t in tasks['tasks']]}, ensure_ascii=False))
        subprocess.run([sys.executable, '-u', 'experiments/v3/collect_teacher.py', '--tasks', str(playable),
                        '--episodes', str(args.episodes), '--parallel', '8', '--prefix', 'DEV', '--no-style',
                        '--temperature', str(args.temperature),
                        '--output', args.output], env=env, check=True)
    path = args.report or args.output
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    report = {**metrics(rows), 'ci95': intervals(rows)}
    if args.against:
        base_rows = [json.loads(line) for line in Path(args.against).read_text().splitlines()]
        base = {**metrics(base_rows), 'ci95': intervals(base_rows)}
        report = {'model': report, 'base': base, 'problems': verdict(report, base),
                  'paired_difference_ci95': paired(rows, base_rows)}
        report['passed'] = not report['problems']
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
