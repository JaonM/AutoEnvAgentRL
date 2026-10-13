"""Select the SFT cold-start set from teacher episodes (plan E13): balance, not just success.

The first collection showed why plain "keep every success" fails a small model: easy tasks succeed
12/12 and flood the set (77% of it answered without a single tool call) while hard tasks contribute
nothing. This keeps successful episodes only, then
- caps each task at --per-task episodes (different worlds and tool namings first),
- caps answer-only (direct_response) tasks at --direct-share of the selection,
- prefers English episodes up to --english-share (τ²-bench is English),
- reports how many selected episodes demonstrate each τ²-relevant behaviour (asking consent, handling a
  refusal, writing, verifying identity, several goals, out-of-scope requests, pushback), so gaps can be
  filled by targeted collection,
- drops episodes whose speech leaks internal field names or cents, and episodes with more than
  --max-failed failed tool results (turns that failed are kept as context but not trained, see
  sft_train.py --no-error-turns).

    python experiments/v3/sft_select.py --teacher output/v3_step1/teacher.jsonl --tasks output/v3_step1/tasks_all.json \
        --output output/v3_step1/teacher_selected.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def failed_results(messages):
    count = 0
    for message in messages:
        if message.get('role') == 'tool':
            try:
                count += json.loads(message.get('content') or '{}').get('ok') is False
            except ValueError:
                pass
    return count


IDENTITY = ('verify', 'identity', 'authenticat', 'lookup_customer', 'find_customer', 'find_user', 'get_user',
            'lookup_user', 'get_customer')


def skills(episode, request):
    """The τ²-relevant behaviours one episode demonstrates (counted for coverage, not filtered on)."""
    consent = (episode.get('naming') or {}).get('consent', 'consent')
    results = {m.get('tool_call_id'): m.get('content') or '' for m in episode['messages'] if m['role'] == 'tool'}
    calls = [c for m in episode['messages'] if m['role'] == 'assistant' for c in m.get('tool_calls') or []]
    found = set()
    for call in calls:
        name, arguments = call['function']['name'], call['function'].get('arguments') or ''
        result = results.get(call.get('id'), '')
        if '"proposal"' in arguments:
            found.add('asks_consent')
            if '"consent": null' in result or '"consent": None' in result:
                found.add('handles_refusal')
        if f'"{consent}"' in arguments and '"ok": true' in result:
            found.add('writes')
        if any(word in name.lower() for word in IDENTITY):
            found.add('verifies_identity')
    script = request.get('user_script') or {}
    goals = (request.get('user_facts') or {}).get('goals') or script.get('goals') or []
    if len(goals) >= 2:
        found.add('multi_goal')
    if script.get('out_of_scope'):
        found.add('out_of_scope')
    if script.get('pushback'):
        found.add('pushback')
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--tasks', required=True, help='task list with sandbox paths (to read category and language)')
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-task', type=int, default=4)
    parser.add_argument('--direct-share', type=float, default=0.15)
    parser.add_argument('--english-share', type=float, default=0.7)
    parser.add_argument('--max-failed', type=int, default=3)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    sandboxes = {t['id']: t['sandbox'] for t in json.loads(Path(args.tasks).read_text())['tasks']}
    meta, requests = {}, {}
    for task_id, sandbox in sandboxes.items():
        request = json.loads((Path(sandbox) / 'request.json').read_text())
        meta[task_id], requests[task_id] = (request['category'], request.get('language', 'zh')), request
    rng = random.Random(args.seed)
    episodes = [r for r in map(json.loads, Path(args.teacher).read_text().splitlines())
                if r.get('success') and r.get('messages') and r['id'] in meta
                and not r.get('speech_leaks') and failed_results(r['messages']) <= args.max_failed]
    by_task = collections.defaultdict(list)
    for episode in episodes:
        by_task[episode['id']].append(episode)
    chosen = []
    for task_id, group in by_task.items():
        rng.shuffle(group)
        seen, picked = set(), []
        for episode in group:  # distinct worlds first, then the rest
            if episode.get('world') not in seen:
                seen.add(episode.get('world'))
                picked.append(episode)
        picked += [e for e in group if e not in picked]
        chosen += picked[:args.per_task]
    tools = [e for e in chosen if meta[e['id']][0] != 'direct_response']
    direct = [e for e in chosen if meta[e['id']][0] == 'direct_response']
    rng.shuffle(direct)
    allowed = int(args.direct_share / max(1e-9, 1 - args.direct_share) * len(tools))
    selected = tools + direct[:allowed]
    english = [e for e in selected if meta[e['id']][1] == 'en']
    other = [e for e in selected if meta[e['id']][1] != 'en']
    if english and len(english) / len(selected) < args.english_share:
        keep = int(len(english) * (1 - args.english_share) / args.english_share)
        rng.shuffle(other)
        selected = english + other[:max(keep, 0)]
    rng.shuffle(selected)
    Path(args.output).write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in selected))
    count = lambda key: dict(collections.Counter(meta[e['id']][key] for e in selected))
    print(json.dumps({'successful': len(episodes), 'selected': len(selected), 'tasks': len({e['id'] for e in selected}),
                      'category': count(0), 'language': count(1),
                      'skills': dict(collections.Counter(k for e in selected for k in skills(e, requests[e['id']])))},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
