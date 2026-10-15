"""Start library for the reverse curriculum (rl/rollout/curriculum.py): exact snapshots after each turn of
successful teacher episodes.

Each episode is replayed in the training runtime (rl.environments.v3.V3Episode) with its own seed and world; the
simulated user is replaced by its recorded replies, and after every assistant turn the replayed conversation must
equal the recording (platform notices aside, which the runtime adds itself) before the snapshot is kept. A replay that
drifts is dropped, never kept half-right.

    python experiments/v3/start_states.py --teacher T.jsonl --tasks TASKS.json --task-ids HARD.json \
        --per-task 4 --output output/start_library.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

NOTICE = re.compile(r'\s*(\(System notice:[^)]*\)|（系统提示：[^）]*）)')


class Recorded:
    """Stands in for the user model: the recorded replies, in order."""

    model = 'recorded'

    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, messages, tools=None, **parameters):
        return {'content': self.replies.pop(0) if self.replies else ''}


def comparable(messages):
    keep = []
    for message in messages:
        if message.get('role') in ('user', 'assistant', 'tool'):
            keep.append((message['role'], NOTICE.sub('', str(message.get('content') or '')).strip()))
    return keep


def settings(row):
    """The runtime settings the episode was played under: older collections predate the random result format
    (always the envelope) and plain-dialogue consent (always tokens)."""
    first = next((str(m.get('content') or '') for m in row['messages'] if m.get('role') == 'tool'), '')
    envelope = first.startswith('{"ok"')
    return {'ENVFACTORY_V3_RESULT_FORMAT': 'envelope' if envelope else 'random',
            'ENVFACTORY_V3_CONSENT': row.get('consent') or 'token'}


def replay(row, sandbox):
    """[snapshot after turn 1, ..., after turn n-1] of one recorded episode, or None when the replay drifts."""
    import os
    from rl.environments.v3 import V3Episode
    messages = row['messages']
    previous = {k: os.environ.get(k) for k in settings(row)}
    os.environ.update(settings(row))
    assistant = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
    # replies the user gave after each chat turn (the message following an assistant turn without tool calls)
    replies = [NOTICE.sub('', str(messages[i + 1].get('content') or '')).strip()
               for i in assistant if not messages[i].get('tool_calls') and i + 1 < len(messages)
               and messages[i + 1]['role'] == 'user']
    try:
        env = V3Episode(sandbox)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    try:
        env.service.user_client = Recorded(replies)
        env.reset(row['seed'], world=row.get('world'))
        if comparable(env.messages)[:1] != comparable(messages)[:1]:
            return None  # another opening: the episode was played under other settings
        snapshots = []
        for turn, index in enumerate(assistant[:-1]):
            message = {k: v for k, v in messages[index].items() if k in ('role', 'content', 'tool_calls')}
            env.step(message)
            following = assistant[turn + 1]
            if comparable(env.messages) != comparable(messages[:following]):
                return None
            snapshots.append(env.snapshot())
        return snapshots
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--teacher', nargs='+', required=True)
    parser.add_argument('--tasks', required=True, help='task list with sandbox paths')
    parser.add_argument('--task-ids', help='JSON list of the tasks to build starts for (default: all)')
    parser.add_argument('--per-task', type=int, default=4)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    sandboxes = {t['id']: t['sandbox'] for t in json.loads(Path(args.tasks).read_text())['tasks']}
    wanted = set(json.loads(Path(args.task_ids).read_text())) if args.task_ids else set(sandboxes)
    rows = [r for path in args.teacher for r in map(json.loads, Path(path).read_text().splitlines())
            if r.get('success') and r.get('messages') and r['id'] in wanted and r['id'] in sandboxes]
    library, dropped = {}, 0
    for row in rows:
        if len(library.get(row['id'], [])) >= args.per_task:
            continue
        snapshots = replay(row, sandboxes[row['id']])
        if not snapshots:
            dropped += 1
            continue
        library.setdefault(row['id'], []).append({'seed': row['seed'], 'world': row.get('world'),
                                                 'snapshots': snapshots})
    Path(args.output).write_text(json.dumps(library, ensure_ascii=False))
    print(json.dumps({'tasks': len(library), 'recordings': sum(len(v) for v in library.values()),
                      'dropped_replays': dropped}))


if __name__ == '__main__':
    main()
