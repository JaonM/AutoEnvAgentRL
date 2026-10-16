"""Review teacher episodes as they are collected, so a bad corpus is caught after a few dozen episodes instead of
after thousands (2026-10-16: the pool1/pool3 corpus turned out to be 64% outcome reports, which the student copied
instead of acting, and was found only after training and a τ²-bench evaluation).

Every episode gets rule-based flags (cheap, no model call):
  error             the episode raised (provider or sandbox)
  report            a text-only reply longer than REPORT_CHARS or shaped as an outcome report
  announce          a text-only reply announcing a lookup ("let me check") without making it
  repeat            the same reply (over 40 characters) twice in a row
  leak              internal field names or cents in what the agent said
  language          an English task answered in Chinese, or the other way round
  invented_id       a call with an identifier that never appeared earlier in the conversation
  repeated_failure  the same call made again right after it failed
  unsafe            a write without consent (safety violation)
  no_call           no tool call at all although the task has business tools
  max_turns         the episode never terminated (turn limit)
STYLE flags are the teacher's habits a student copies; their rate over the last WINDOW episodes trips the breaker
(collection stops and an alert is written) when it passes BREAKER, errors at ERROR_BREAKER.
Per task, the pilot seeds decide whether the rest are worth playing (task_verdict): two pilot failures on the same
answer field point at the task or its verifier, not the teacher.

    python experiments/teacher_review.py output/I2_pool4/teacher_pool4.jsonl     # summary of a corpus so far
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sft_select import call_profile, is_failure  # noqa: E402

REPORT_CHARS = 600
REPORT = re.compile(r"(here'?s (the|a|your) (outcome|summary|picture|result|breakdown|overview)|处理结果如下|结果如下|"
                    r"^\s*#{1,4} |\n\s*\*\*[^*\n]{2,40}\*\*\s*\n\s*[-•])", re.I | re.M)
ANNOUNCE = re.compile(r"\b(let me (check|look|get|find|verify|pull|see)|i'll (check|look|get|find|pull)|"
                      r"i will (check|look|get|find))\b|我(来|先)?(查|看|核实)一下|稍等", re.I)
INTERNAL = re.compile(r'[A-Za-z]+_[A-Za-z_]+|\d+\s*分(?!钟)|\b\d+\s*cents?\b')
CJK = re.compile(r'[一-鿿]')
STYLE = ('report', 'announce', 'repeat', 'leak', 'language')
WINDOW, BREAKER, ERROR_BREAKER = 40, 0.25, 0.2


def replies(row):
    return [m.get('content') or '' for m in row.get('messages') or []
            if m.get('role') == 'assistant' and not m.get('tool_calls') and (m.get('content') or '').strip()]


def flags(row):
    """The rule-based flags of one collected episode (see the module docstring)."""
    if row.get('error') or not row.get('messages'):
        return ['error']
    out = []
    texts = replies(row)
    if any(len(t) > REPORT_CHARS or REPORT.search(t) for t in texts):
        out.append('report')
    if any(ANNOUNCE.search(t) for t in texts):
        out.append('announce')
    if any(texts[i] == texts[i - 1] and len(texts[i]) > 40 for i in range(1, len(texts))):
        out.append('repeat')
    if any(INTERNAL.search(t) for t in texts):
        out.append('leak')
    spoken = ' '.join(texts)
    if spoken and row.get('language') in ('en', 'zh'):
        share = len(CJK.findall(spoken)) / len(spoken)
        if (row['language'] == 'en' and share > 0.1) or (row['language'] == 'zh' and share < 0.05):
            out.append('language')
    messages = row['messages']
    if call_profile(messages, (), row.get('tools') or ())[2]:
        out.append('invented_id')
    results = {m.get('tool_call_id'): m.get('content') or '' for m in messages if m.get('role') == 'tool'}
    previous = None
    for m in messages:
        for call in m.get('tool_calls') or [] if m.get('role') == 'assistant' else []:
            key = (call['function']['name'], call['function'].get('arguments'))
            if previous and previous[0] == key and previous[1]:
                out.append('repeated_failure')
                break
            previous = (key, is_failure(results.get(call.get('id'), '')))
        if out and out[-1] == 'repeated_failure':
            break
    if row.get('safety') is False:
        out.append('unsafe')
    if row.get('tools') and not any(m.get('tool_calls') for m in messages if m.get('role') == 'assistant'):
        out.append('no_call')
    if row.get('terminated') is False:
        out.append('max_turns')
    return out


def task_verdict(rows):
    """(play the remaining seeds?, reason) from a task's pilot episodes."""
    if rows and all(r.get('error') for r in rows):
        return False, 'every pilot episode raised'
    failed = [r for r in rows if not r.get('success') and not r.get('error')]
    if len(rows) >= 2 and len(failed) == len(rows):
        wrong = [frozenset(f for f, v in (r.get('field_scores') or {}).items() if v < 1) for r in failed]
        shared = frozenset.intersection(*wrong) if wrong else frozenset()
        if shared and all(r.get('state_ok') for r in failed):
            return False, f'every pilot failed on the same fields with the right final state: {sorted(shared)[:4]}'
    return True, None


class Breaker:
    """Rolling style and error rates over the last WINDOW reviewed episodes."""

    def __init__(self):
        self.recent = collections.deque(maxlen=WINDOW)

    def add(self, row_flags):
        self.recent.append(set(row_flags))
        if len(self.recent) < WINDOW:
            return None
        rates = {f: sum(f in r for r in self.recent) / len(self.recent) for f in (*STYLE, 'error')}
        tripped = {f: round(v, 2) for f, v in rates.items()
                   if v > (ERROR_BREAKER if f == 'error' else BREAKER)}
        return tripped or None


def summary(rows):
    counted = collections.Counter(f for r in rows for f in (r.get('review') or flags(r)))
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r['id']].append(r)
    stopped = {t: r.get('task_stopped') for t, group in by_task.items() for r in group if r.get('task_stopped')}
    n = max(1, len(rows))
    return {'episodes': len(rows), 'successes': sum(bool(r.get('success')) for r in rows),
            'tasks': len(by_task), 'tasks_with_success': sum(any(r.get('success') for r in g) for g in by_task.values()),
            'flag_rates': {f: round(c / n, 3) for f, c in counted.most_common()}, 'tasks_stopped': stopped}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('corpus')
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.corpus).read_text().splitlines()]
    print(json.dumps(summary(rows), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
