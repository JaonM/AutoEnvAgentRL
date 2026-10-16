"""Repair old teacher episodes to the current corpus requirements instead of collecting them again.

The pool1/pool3 corpus is 80-87% outcome reports (teacher_review: report), 7-8% lookups announced without being
made, and was graded on contracts of up to 35 fields. relabel_contracts.py trimmed each task's contract to its
outcome and deciding factors; this script brings the episodes of those tasks in line:
  1. exact repeats of an earlier read call (same tool, same arguments) are removed with their results — a read changes
     nothing, and a repeat answers nothing new; other reads stay (an unused-looking lookup may still have decided
     something, and dropping it would teach conclusions without lookups);
  2. text-only turns announcing a lookup without making it stay as context only (not trained);
  3. every report-shaped reply (teacher_review.REPORT, or longer than REPORT_CHARS) is rewritten by a model into at
     most three plain sentences that keep any question or confirmation request and, when it delivers the result, the
     kept fields; the model also says whether the user's next message still follows — when it does not, the episode
     is dropped;
  4. the repaired episode is replayed and graded on the relabelled bundle (as regrade_teacher.py does) and reviewed
     again; it is kept only when it succeeds with no style flag.

    python experiments/repair_episodes.py --teacher A.jsonl ... --relabel output/relabel --output out.jsonl [--limit 100]
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from teacher_review import ANNOUNCE, REPORT, REPORT_CHARS, STYLE, flags  # noqa: E402

REWRITE = """You shorten one reply a customer-service agent wrote, so it reads like a good agent talking: at most three
plain sentences, no headings, bullet lists or "here is the outcome" summaries, in {language}.

Keep, in substance:
- any question the reply asks the user, and any request to confirm a change together with the exact change proposed
  (what, which item, the amount) — the user's next message answers it;
- when the reply delivers the result, these facts exactly as the reply states them (same amounts, dates, ids, outcome):
{kept}
Drop everything else: restated rules, intermediate numbers, lookups recounted, things that did not change.

The user's previous message:
{before}

The reply to shorten:
{reply}

The user's next message (it must still make sense after your rewrite; empty if the conversation ends):
{after}

Answer with one JSON object: {{"reply": "<the shortened reply>"}}"""

FOLLOWS = """An agent said this to a customer:
{reply}

The customer answered:
{after}

Does the customer's answer refer to anything the agent's message does not contain (a number, an item, an option, a
question it did not ask)? Answer with one JSON object: {{"refers_to_missing": true|false}}"""

TOKEN = re.compile(r'[A-Za-z]*\d[\w.-]*|\$\d[\d,.]*')

CJK = re.compile(r'[一-鿿]')


def language_of(row):
    if row.get('language') in ('en', 'zh'):
        return row['language']
    text = ' '.join(m.get('content') or '' for m in row['messages'] if m.get('role') == 'user')
    return 'zh' if len(CJK.findall(text)) > 0.1 * max(1, len(text)) else 'en'


def drop_repeated_reads(messages, writes):
    """Remove read calls identical to an earlier one, with their tool results."""
    seen, dropped, out = set(), set(), []
    for message in messages:
        if message.get('role') == 'assistant' and message.get('tool_calls'):
            kept = []
            for call in message['tool_calls']:
                key = (call['function']['name'], call['function'].get('arguments'))
                if call['function']['name'] not in writes and key in seen:
                    dropped.add(call.get('id'))
                    continue
                seen.add(key)
                kept.append(call)
            if not kept and not (message.get('content') or '').strip():
                continue
            message = {**message, 'tool_calls': kept}
        if message.get('role') == 'tool' and message.get('tool_call_id') in dropped:
            continue
        out.append(message)
    return out, len(dropped)


def repair(row, bundle_dir, kept, client, service_for):
    from env_factory.serving import agent_view
    svc = service_for(bundle_dir)
    writes = {t['name'] for t in svc.instructions['tools'] if t.get('kind') == 'write'}
    messages, repeats = drop_repeated_reads(copy.deepcopy(row['messages']), writes)
    language = 'Chinese' if language_of(row) == 'zh' else 'English'
    fields = '\n'.join(f'  - {name}: {spec.get("description", "")[:160]}' for name, spec in kept.items()) or '  (none)'
    announced = rewritten = 0
    for index, message in enumerate(messages):
        if message.get('role') != 'assistant' or message.get('tool_calls') or not (message.get('content') or '').strip():
            continue
        text = message['content']
        if ANNOUNCE.search(text) and len(text) < 200:
            message['context_only'] = True
            announced += 1
            continue
        if not (len(text) > REPORT_CHARS or REPORT.search(text)):
            continue
        before = next((m.get('content') or '' for m in reversed(messages[:index]) if m.get('role') == 'user'), '')
        after = next((m.get('content') or '' for m in messages[index + 1:] if m.get('role') == 'user'), '')
        prompt = REWRITE.format(language=language, kept=fields, before=before[:1500], reply=text[:4000], after=after[:1500])
        answer = client.chat([{'role': 'user', 'content': prompt}], temperature=0).get('content') or ''
        try:
            result = json.loads(answer[answer.index('{'):answer.rindex('}') + 1])
        except ValueError:
            return None, 'rewrite unparseable'
        shorter = (result.get('reply') or '').strip()
        if not shorter:
            return None, 'rewrite empty'
        if after:
            # a number or id the customer repeats must not have been cut from the reply they answered
            lost = [t for t in set(TOKEN.findall(after)) if t in text and t not in shorter]
            if lost:
                return None, 'next user message refers to cut details'
            check = client.chat([{'role': 'user', 'content': FOLLOWS.format(reply=shorter, after=after[:1500])}],
                                temperature=0).get('content') or ''
            try:
                if json.loads(check[check.index('{'):check.rindex('}') + 1]).get('refers_to_missing'):
                    return None, 'next user message no longer follows'
            except ValueError:
                return None, 'continuity check unparseable'
        message['content'] = shorter
        rewritten += 1
    repaired = {**row, 'messages': messages, 'repaired': {'repeated_reads_dropped': repeats, 'announcements_masked':
                                                          announced, 'replies_rewritten': rewritten, 'kept': list(kept)}}
    # replay and grade on the relabelled bundle (regrade_teacher.py)
    names = {t['name'] for t in svc.instructions['tools']}
    episode = svc.reset(world=row['world'], consent=row.get('consent'))['episode']
    for message in messages:
        if message['role'] != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            try:
                action = agent_view.action(call['function']['name'], json.loads(call['function'].get('arguments') or '{}'),
                                           names, row.get('naming'))
            except (KeyError, ValueError):
                continue
            if action is not None:
                svc.step(episode, action)
        if not message.get('tool_calls') and (message.get('content') or '').strip():
            svc.step(episode, {'type': 'chat', 'message': message['content']})
    grade = svc.reward(episode)
    svc.release(episode)
    repaired.update(reward=grade.get('reward'), success=grade.get('success') is True, state_ok=grade.get('state_ok'),
                    safety=grade.get('safety'), field_scores=grade.get('field_scores'),
                    extracted_answer=grade.get('answer'), previous={'reward': row.get('reward'), 'success': row.get('success')})
    review = [f for f in flags({**repaired, 'messages': [m for m in messages if not m.get('context_only')]})]
    repaired['review'] = review
    if not repaired['success']:
        return repaired, 'fails on the relabelled contract'
    if set(review) & set(STYLE):
        return repaired, f'still flagged {sorted(set(review) & set(STYLE))}'
    return repaired, 'repaired'


class Stub:
    model = 'stub'

    def chat(self, messages, tools=None, **parameters):
        return {'content': 'ok'}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--teacher', nargs='+', required=True)
    parser.add_argument('--relabel', required=True, help='relabel_contracts.py output folder')
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    import random
    from env_factory.certify import bundle
    from env_factory.core.extract import Extractor
    from env_factory.core.llm import ChatClient, load_env_file
    load_env_file(ROOT / '.env')
    client = ChatClient('LLM', timeout=180)
    relabelled = {}
    for line in (Path(args.relabel) / 'report.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r.get('status') == 'passed':
            relabelled[r['id']] = r
    rows = [json.loads(line) for path in args.teacher for line in Path(path).read_text().splitlines()]
    rows = [r for r in rows if r.get('success') and r.get('messages') and r['id'] in relabelled]
    random.Random(args.seed).shuffle(rows)
    rows = rows[:args.limit] if args.limit else rows
    print(f'{len(rows)} successful episodes on relabelled tasks to repair', flush=True)
    services, lock = {}, threading.Lock()

    def service_for(bundle_dir):
        with lock:
            if bundle_dir not in services:
                loaded = bundle.load_bundle(bundle_dir)
                services[bundle_dir] = (bundle.service(loaded, user_mode='natural', user_client=Stub(), audit=False,
                                                       extractor=Extractor(ChatClient('JUDGE', timeout=60))), loaded)
            return services[bundle_dir][0]

    out, counts = Path(args.output), {}

    def one(row):
        info = relabelled[row['id']]
        try:
            service_for(info['bundle'])
            schema = services[info['bundle']][1]['schema']
            kept = {k: v for k, v in schema['fields'].items() if v.get('graded', True)}
            repaired, status = repair(row, info['bundle'], kept, client, service_for)
        except Exception as exc:  # noqa: BLE001 - one episode never stops the rest
            repaired, status = None, f'error: {type(exc).__name__}: {exc}'[:200]
        with lock:
            counts[status.split(':')[0] if status.startswith('error') else status] = counts.get(
                status.split(':')[0] if status.startswith('error') else status, 0) + 1
            if repaired is not None:
                with out.open('a') as handle:
                    handle.write(json.dumps({**repaired, 'repair_status': status}, ensure_ascii=False) + '\n')
            if sum(counts.values()) % 20 == 0:
                print(json.dumps(counts), flush=True)

    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(one, rows))
    print('final', json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
