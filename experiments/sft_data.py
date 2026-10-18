"""SFT examples from successful teacher episodes (collect_teacher.py), for mlx_lm.lora --mask-prompt.

Each assistant turn of a successful episode becomes one chat example: the conversation up to and
including that turn, with the episode's tools. --mask-prompt trains only on the last message, so
every tool call, every message to the user and the final submit is learned once, in exactly the
context the policy sees during RL. Dedup runs first; train/valid is split by task, never by turn.

    python experiments/sft_data.py --teacher output/v3_sft/teacher.jsonl --output output/v3_sft/data
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from trajectory_dedup import add_arguments, deduplicate, task_key, validation_keys

KEEP = ('role', 'content', 'tool_calls', 'tool_call_id', 'name')


def examples(episode):
    messages = [{k: v for k, v in m.items() if k in KEEP} for m in episode['messages']]
    for index, message in enumerate(messages):
        if message['role'] == 'assistant' and index > 0:
            yield {'messages': messages[:index + 1], 'tools': episode['tools']}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--teacher', required=True)
    add_arguments(parser)
    parser.add_argument('--output', required=True)
    parser.add_argument('--valid-share', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    episodes = [r for r in map(json.loads, Path(args.teacher).read_text().splitlines())
                if r.get('success') and r.get('messages')
                and (r.get('trajectory_policy') or {}).get('passed') is not False]
    episodes, dedup_report = deduplicate(episodes, args.dedup_threshold, enabled=not args.no_dedup)
    random.Random(args.seed).shuffle(episodes)
    held = validation_keys(episodes, args.valid_share)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, part in (('valid', [e for e in episodes if task_key(e) in held]),
                       ('train', [e for e in episodes if task_key(e) not in held])):
        rows = [example for episode in part for example in examples(episode)]
        random.Random(args.seed).shuffle(rows)
        (out / f'{name}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        counts[name] = {'episodes': len(part), 'examples': len(rows)}
    by_task = {}
    for episode in episodes:
        by_task[episode['id']] = by_task.get(episode['id'], 0) + 1
    summary = {**counts, 'tasks': len(by_task), 'episodes_per_task': dict(sorted(by_task.items()))}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (out / 'dedup_report.json').write_text(json.dumps(dedup_report, ensure_ascii=False, indent=2))
    print(json.dumps(counts))


if __name__ == '__main__':
    main()
