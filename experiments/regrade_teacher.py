"""Re-grade saved teacher episodes under the current grading, without new teacher calls.

Each episode is replayed through a fresh service (a stub user: the final state depends only on the
agent's calls and the platform's consent decisions, which are deterministic) and graded again; the
answer is re-extracted from the agent's speech. Rows are rewritten with the new reward, success and
field scores, keeping the old ones under "previous". Rows are appended as they finish (a rerun skips
rows already in the output) and progress is printed every 25 episodes.

    python experiments/regrade_teacher.py --tasks TASKS.json --teacher IN.jsonl [IN2.jsonl ...] --output OUT.jsonl
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
from pathlib import Path


class Stub:
    model = 'stub'

    def chat(self, messages, tools=None, **parameters):
        return {'content': 'ok'}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--tasks', required=True)
    parser.add_argument('--teacher', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parallel', type=int, default=12)
    args = parser.parse_args()
    from env_factory.certify import bundle
    from env_factory.core.llm import load_env_file
    from env_factory.serving import agent_view
    load_env_file()
    sandboxes = {t['id']: t['sandbox'] for t in json.loads(Path(args.tasks).read_text())['tasks']}
    rows = [{**json.loads(line), 'source_file': Path(path).name}
            for path in args.teacher for line in Path(path).read_text().splitlines()]
    rows = [r for r in rows if r.get('messages') and r['id'] in sandboxes]
    key = lambda r: (r['id'], r['seed'], r.get('teacher_model') or r.get('source_file'))
    out_path = Path(args.output)
    done = {key(r) for r in map(json.loads, out_path.read_text().splitlines())} if out_path.exists() else set()
    rows = [r for r in rows if key(r) not in done]
    print(f'{len(rows)} episodes to regrade ({len(done)} already done)', flush=True)
    services, lock, written = {}, threading.Lock(), [0]

    def service(task_id):
        with lock:
            if task_id not in services:
                loaded = bundle.load_bundle(sandboxes[task_id])
                from env_factory.core.extract import Extractor
                from env_factory.core.llm import ChatClient
                services[task_id] = bundle.service(loaded, user_mode='natural', user_client=Stub(), audit=False,
                                                   extractor=Extractor(ChatClient('JUDGE', timeout=60)))
            return services[task_id]

    def one(row):
        from env_factory.production import offpeak
        offpeak.wait(lambda text: print(text, flush=True))  # re-extraction calls the judge model
        svc = service(row['id'])
        names = {t['name'] for t in svc.instructions['tools']}
        # The episode's own consent protocol: a plain-dialogue episode replayed under the token protocol turns every
        # agreed write into a safety violation (2026-10-15: 261 successes regraded as failures that way).
        episode = svc.reset(world=row['world'], consent=row.get('consent'))['episode']
        for message in row['messages']:
            if message['role'] != 'assistant':
                continue
            for call in message.get('tool_calls') or []:
                try:
                    action = agent_view.action(call['function']['name'],
                                               json.loads(call['function'].get('arguments') or '{}'), names,
                                               row.get('naming'))
                except (KeyError, ValueError):
                    continue
                if action is not None:
                    svc.step(episode, action)
            if not message.get('tool_calls') and (message.get('content') or '').strip():
                svc.step(episode, {'type': 'chat', 'message': message['content']})
        grade = svc.reward(episode)
        svc.release(episode)
        previous = {k: row.get(k) for k in ('reward', 'success')}
        result = {**row, 'previous': previous, 'reward': grade.get('reward'),
                  'success': grade.get('success') is True, 'state_ok': grade.get('state_ok'),
                  'safety': grade.get('safety'), 'field_scores': grade.get('field_scores')}
        with lock:
            with open(out_path, 'a') as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            written[0] += 1
            if written[0] % 25 == 0:
                print(f'{written[0]}/{len(rows)} regraded', flush=True)
        return result

    def safe(row):
        try:
            return one(row)
        except Exception as error:  # noqa: BLE001 - one bad replay never stops the rest
            print(f"skip {row['id']} seed {row['seed']}: {type(error).__name__}: {error}"[:200], flush=True)
            return None

    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(safe, rows))
    out = [json.loads(line) for line in out_path.read_text().splitlines()]
    before, after = sum(r['previous']['success'] for r in out), sum(r['success'] for r in out)
    print(json.dumps({'episodes': len(out), 'success_before': before, 'success_after': after,
                      'tasks_with_success_before': len({r['id'] for r in out if r['previous']['success']}),
                      'tasks_with_success_after': len({r['id'] for r in out if r['success']})}))


if __name__ == '__main__':
    main()
