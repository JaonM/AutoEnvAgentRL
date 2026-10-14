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
- caps each domain at --domain-share of the selection (a small model memorizes one domain's tools),
- drops near-duplicate episodes of a task (same tool-call sequence and same closing words),
- prints a diversity report against the targets (≥2000 episodes, ≥500 tasks, no domain above the cap,
  every skill at --skill-min episodes) and exits with status 3 when a target is missed, so an
  undersized set is never trained by accident (--allow-short overrides),
- drops episodes whose speech leaks internal field names or cents, and episodes with more than
  --max-failed failed tool results (turns that failed are kept as context but not trained, see
  sft_train.py --no-error-turns).

    python experiments/v3/sft_select.py --teacher A.jsonl B.jsonl --tasks tasks_a.json tasks_b.json \
        --output selected.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path


def is_failure(content):
    """A failed tool result in any of the formats episodes carry: our envelope, "Error: ..." text."""
    content = content or ''
    if content.startswith('Error'):
        return True
    try:
        value = json.loads(content)
    except ValueError:
        return False
    return isinstance(value, dict) and value.get('ok') is False


def failed_results(messages):
    return sum(is_failure(m.get('content')) for m in messages if m.get('role') == 'tool')


IDENTITY = ('verify', 'identity', 'authenticat', 'lookup_customer', 'find_customer', 'find_user', 'get_user',
            'lookup_user', 'get_customer')


def skills(episode, request, write_tools=()):
    """The τ²-relevant behaviours one episode demonstrates (counted for coverage, not filtered on)."""
    consent = (episode.get('naming') or {}).get('consent', 'consent')
    results = {m.get('tool_call_id'): m.get('content') or '' for m in episode['messages'] if m['role'] == 'tool'}
    calls = [c for m in episode['messages'] if m['role'] == 'assistant' for c in m.get('tool_calls') or []]
    found = set()
    # plain-dialogue consent: the agent's last words before a write asked the user something
    last_words = ''
    for message in episode['messages']:
        if message['role'] == 'assistant':
            for call in message.get('tool_calls') or []:
                if call['function']['name'] in write_tools and any(q in last_words for q in ('?', '？')):
                    found.add('asks_consent')
            if not message.get('tool_calls') and (message.get('content') or '').strip():
                last_words = message['content']
    for call in calls:
        name, arguments = call['function']['name'], call['function'].get('arguments') or ''
        result = results.get(call.get('id'), '')
        if '"proposal"' in arguments:
            found.add('asks_consent')
            if '"consent": null' in result or '"consent": None' in result:
                found.add('handles_refusal')
        if (name in write_tools or f'"{consent}"' in arguments) and not is_failure(result):
            found.add('writes')
        if any(word in name.lower() for word in IDENTITY):
            found.add('verifies_identity')
    failed_then = False
    asks = lambda m: m['role'] == 'assistant' and not m.get('tool_calls') and any(q in (m.get('content') or '')
                                                                                 for q in ('?', '？'))
    for index, message in enumerate(episode['messages']):
        content = message.get('content') or ''
        if message['role'] == 'tool' and is_failure(content):
            later = episode['messages'][index + 1:]
            # recovered: a later call succeeded, or the agent's next words asked the user
            if any(m['role'] == 'tool' and not is_failure(m.get('content')) for m in later) or \
                    any(asks(m) for m in later[:2]):
                failed_then = True
    if failed_then:
        found.add('recovers_from_error')
    if str(episode.get('world') or '').startswith('flip'):
        found.add('refusal_world')
    script = request.get('user_script') or {}
    goals = (request.get('user_facts') or {}).get('goals') or script.get('goals') or []
    if len(goals) >= 2:
        found.add('multi_goal')
    if script.get('out_of_scope'):
        found.add('out_of_scope')
    if script.get('pushback'):
        found.add('pushback')
    return found


def signature(episode):
    """Near-duplicate key: the tool-call sequence and the last thing the agent said."""
    calls = tuple(c['function']['name'] for m in episode['messages'] if m['role'] == 'assistant'
                  for c in m.get('tool_calls') or [])
    said = [m.get('content') or '' for m in episode['messages'] if m['role'] == 'assistant' and m.get('content')]
    return calls, ' '.join((said[-1] if said else '').lower().split())[:160]


SKILLS = ('asks_consent', 'handles_refusal', 'writes', 'verifies_identity', 'multi_goal', 'out_of_scope', 'pushback',
          'recovers_from_error', 'refusal_world')


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--teacher', nargs='+', required=True)
    parser.add_argument('--tasks', nargs='+', required=True, help='task lists with sandbox paths')
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-task', type=int, default=4)
    parser.add_argument('--direct-share', type=float, default=0.15)
    parser.add_argument('--english-share', type=float, default=0.7)
    parser.add_argument('--domain-share', type=float, default=0.05)
    parser.add_argument('--max-failed', type=int, default=3)
    parser.add_argument('--min-episodes', type=int, default=2000)
    parser.add_argument('--min-tasks', type=int, default=500)
    parser.add_argument('--skill-min', type=int, default=100)
    parser.add_argument('--check-skills', nargs='*', default=None,
                        help='skills the minimum applies to (default: all of SKILLS)')
    parser.add_argument('--allow-short', action='store_true', help='write the selection even if targets are missed')
    parser.add_argument('--near-miss-share', type=float, default=0.0,
                        help='up to this share of near misses (safe, final state right, reward >= 0.85, at most one '
                             'answer field wrong), at most one per task')
    parser.add_argument('--dev-domains', type=int, default=4,
                        help='domains held out of the selection for the behaviour check (dev_check.py); '
                             'airline and retail-like domains always stay in training')
    parser.add_argument('--dev-output', help='where to write the held-out task list')
    parser.add_argument('--dev-from', help='a held-out task list whose domains stay held out (keeps the dev set fixed)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    sandboxes = {t['id']: t['sandbox'] for path in args.tasks for t in json.loads(Path(path).read_text())['tasks']}
    meta, requests, write_tools = {}, {}, {}
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
    from env_factory_v3.core.sandbox import load_module
    from env_factory_v3.runtime import is_write
    for task_id, sandbox in sandboxes.items():
        request = json.loads((Path(sandbox) / 'request.json').read_text())
        try:
            tools = load_module(Path(sandbox) / 'env.py', role='env').make_tools()
            write_tools[task_id] = {name for name, fn in tools.items() if is_write(fn)}
        except Exception:  # noqa: BLE001 - a bundle that will not load contributes no write names
            write_tools[task_id] = set()
        domain = (request.get('source') or {}).get('generated') or request.get('family') or task_id
        meta[task_id], requests[task_id] = (request['category'], request.get('language', 'zh'), domain), request
    rng = random.Random(args.seed)
    import hashlib
    all_domains = sorted({m[2] for m in meta.values()}, key=lambda d: hashlib.sha256(str(d).encode()).hexdigest())
    core = ('airline', 'flight', 'retail', 'e-commerce', 'ecommerce', 'online shop', 'order')
    held = set([d for d in all_domains if not any(word in str(d).lower() for word in core)][:args.dev_domains])
    if args.dev_from:
        held = set(json.loads(Path(args.dev_from).read_text())['domains'])
    if args.dev_output:
        dev = [{'id': t, 'sandbox': sandboxes[t], 'split': 'eval', 'weight': 1.0} for t in meta if meta[t][2] in held]
        Path(args.dev_output).write_text(json.dumps({'tasks': dev, 'domains': sorted(held)}, ensure_ascii=False, indent=1))
    meta = {t: m for t, m in meta.items() if m[2] not in held}
    episodes = [r for path in args.teacher for r in map(json.loads, Path(path).read_text().splitlines())
                if r.get('success') and r.get('messages') and r['id'] in meta
                and not r.get('speech_leaks') and failed_results(r['messages']) <= args.max_failed]
    near = []
    if args.near_miss_share:
        for path in args.teacher:
            for r in map(json.loads, Path(path).read_text().splitlines()):
                scores = r.get('field_scores') or {}
                wrong = sum(1 for v in scores.values() if v < 1)
                if (not r.get('success') and r.get('messages') and r['id'] in meta and r.get('state_ok')
                        and r.get('safety') is not False and (r.get('reward') or 0) >= 0.85 and wrong <= 1
                        and not r.get('speech_leaks')):
                    near.append({**r, 'near_miss': True})
    by_task = collections.defaultdict(list)
    for episode in episodes:
        by_task[episode['id']].append(episode)
    chosen, duplicates = [], 0
    for task_id, group in by_task.items():
        rng.shuffle(group)
        unique, keys = [], set()
        for episode in group:
            key = signature(episode)
            if key in keys:
                duplicates += 1
                continue
            keys.add(key)
            unique.append(episode)
        # plain-dialogue consent first (τ²-style), refusal (flip) worlds next, then the rest
        unique.sort(key=lambda e: (e.get('consent') != 'plain', not str(e.get('world') or '').startswith('flip')))
        seen, picked = set(), []
        for episode in unique:  # distinct worlds first, then the rest
            if episode.get('world') not in seen:
                seen.add(episode.get('world'))
                picked.append(episode)
        picked += [e for e in unique if e not in picked]
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
    if near:
        rng.shuffle(near)
        room, used = int(args.near_miss_share / max(1e-9, 1 - args.near_miss_share) * len(selected)), set()
        for episode in near:
            if room <= 0:
                break
            if episode['id'] not in used:
                used.add(episode['id'])
                selected.append(episode)
                room -= 1
    rng.shuffle(selected)
    cap = max(1, int(args.domain_share * len(selected)))
    per_domain, capped = collections.Counter(), []
    for episode in selected:
        domain = meta[episode['id']][2]
        if per_domain[domain] < cap:
            per_domain[domain] += 1
            capped.append(episode)
    selected = capped
    count = lambda index: dict(collections.Counter(meta[e['id']][index] for e in selected).most_common())
    skill_counts = collections.Counter(k for e in selected for k in skills(e, requests[e['id']], write_tools[e['id']]))
    sequences = collections.Counter(signature(e)[0] for e in selected)
    domains = collections.Counter(meta[e['id']][2] for e in selected)
    report = {'successful': len(episodes), 'near_duplicates_dropped': duplicates, 'selected': len(selected),
              'tasks': len({e['id'] for e in selected}), 'domains': len(domains),
              'largest_domain_share': round(max(domains.values()) / len(selected), 3) if selected else None,
              'category': count(0), 'language': count(1),
              'distinct_tool_sequences': round(len(sequences) / max(1, len(selected)), 3),
              'consent_protocol': dict(collections.Counter(e.get('consent') or 'token' for e in selected)),
              'near_misses': sum(bool(e.get('near_miss')) for e in selected),
              'turns': dict(collections.Counter(
                  '1-4' if t <= 4 else '5-9' if t <= 9 else '10+'
                  for t in (sum(m['role'] == 'assistant' for m in e['messages']) for e in selected))),
              'skills': {k: skill_counts.get(k, 0) for k in SKILLS}}
    missed = []
    if len(selected) < args.min_episodes:
        missed.append(f'episodes {len(selected)} < {args.min_episodes}')
    if report['tasks'] < args.min_tasks:
        missed.append(f"tasks {report['tasks']} < {args.min_tasks}")
    checked = args.check_skills if args.check_skills is not None else SKILLS
    missed += [f'skill {k} {v} < {args.skill_min}' for k, v in report['skills'].items()
               if k in checked and v < args.skill_min]
    report['targets_missed'] = missed
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if missed and not args.allow_short:
        raise SystemExit(3)
    Path(args.output).write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in selected))


if __name__ == '__main__':
    main()
