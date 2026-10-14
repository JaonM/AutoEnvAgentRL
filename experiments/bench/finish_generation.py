"""Finish a run's generation stage in parallel after its request-author sessions are done.

generate.generate writes openings and agent policies one request at a time; with long policies that
is several minutes per task. This finishes the same stage for an interrupted run: requests already
written are kept, the rest get an opening and a policy in parallel (same functions, same plan),
then near-duplicates are dropped in index order, repetitive openings are respread, and
generation.json is written so `run` resumes at the build stage. Every finished request is written
at once (rejections go to generate/finish_rows.jsonl), so a rerun skips them; a request that takes
longer than --task-timeout (a stalled provider connection) is recorded as rejected, and the process
exits without waiting for such threads.

    python experiments/bench/finish_generation.py --run output/v3_bench_pool --parallel 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import threading
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--task-timeout', type=int, default=1200, help='seconds per request before giving up')
    parser.add_argument('--mix', default=None, help='the run\'s --mix (run_options.json does not record it)')
    args = parser.parse_args()
    from env_factory.core.llm import ChatClient, load_env_file
    from env_factory.production import agent_policy, generate, openings
    load_env_file()
    client = ChatClient('OPENING')
    run = Path(args.run)
    options = json.loads((run / 'run_options.json').read_text())
    domains = json.loads((run / 'domains.json').read_text())
    count = options['count']
    mix = generate.parse_mix(args.mix) if args.mix else None
    plan = generate.plan(domains, count, options.get('rubric_share', 0.0), **({'mix': mix} if mix else {}))
    langs = generate.languages(count, options.get('english_share', 0.0))
    output = run / 'generate'
    (output / 'requests').mkdir(exist_ok=True)

    rows_path = output / 'finish_rows.jsonl'
    rejected = {}
    if rows_path.exists():
        rejected = {r['index']: r for r in map(json.loads, rows_path.read_text().splitlines())}
    lock = threading.Lock()

    def base_row(index):
        domain, category, shape = plan[index]
        return {'index': index, 'domain': domain['name'], 'category': category, 'shape': shape,
                'language': langs[index]}

    def reject(row):
        with lock:
            with rows_path.open('a') as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')

    def one(index):
        domain, category, shape = plan[index]
        row = base_row(index)
        folder = output / 'sessions' / f'{index:03d}'
        report = json.loads((folder / 'check-request.json').read_text()) if (folder / 'check-request.json').exists() else None
        if not report or not report['passed']:
            return reject(row | {'passed': False, 'problems': (report or {}).get('problems', ['no request'])[:5]})
        request = json.loads((folder / 'request.json').read_text())
        request.setdefault('time', {'start': 10, 'tick': 0})
        request['language'] = langs[index]
        opening, persona, problems = generate.write_opening(client, request, f'gen-{index:03d}')
        if opening is None:
            return reject(row | {'passed': False, 'persona': persona, 'problems': problems})
        request['user_request'] = opening
        policy, problems = agent_policy.write(client, request)
        if policy is None:
            return reject(row | {'passed': False, 'persona': persona,
                                 'problems': [f'agent policy: {p}' for p in problems[:4]]})
        request['agent_policy'] = policy
        request['id'] = f'gen-{index:03d}'
        request['source'] = {'generated': domain['name'], 'family': domain.get('family'), 'shape': shape,
                             'persona': persona, 'language': langs[index], 'opening_writer': client.model}
        (output / 'requests' / f"{request['id']}.json").write_text(json.dumps(request, ensure_ascii=False, indent=1))

    todo = [i for i in range(count) if not (output / 'requests' / f'gen-{i:03d}.json').exists() and i not in rejected]
    print(f'{len(todo)} requests to finish', flush=True)
    pool = concurrent.futures.ThreadPoolExecutor(args.parallel)
    started = {}

    def timed(index):
        started[index] = time.monotonic()
        return one(index)

    futures = {pool.submit(timed, i): i for i in todo}
    pending = set(futures)
    while pending:  # per-request watchdog: the client timeout does not always fire through the proxy
        done, pending = concurrent.futures.wait(pending, timeout=30)
        for future in done:
            if future.exception():
                reject(base_row(futures[future]) | {'passed': False, 'problems': [repr(future.exception())[:300]]})
        now = time.monotonic()
        for future in [f for f in pending if now - started.get(futures[f], now) > args.task_timeout]:
            reject(base_row(futures[future]) | {'passed': False, 'problems': ['timed out (stalled provider connection)']})
            pending.discard(future)
        print(f'{len(todo) - len(pending)}/{len(todo)} finished', flush=True)
    pool.shutdown(wait=False, cancel_futures=True)
    rejected = {r['index']: r for r in map(json.loads, rows_path.read_text().splitlines())} if rows_path.exists() else {}
    kept, rows = [], []
    for index in range(count):  # index order, as generate.generate does
        path = output / 'requests' / f'gen-{index:03d}.json'
        if not path.exists():
            rows.append(rejected.get(index, base_row(index) | {'passed': False, 'problems': ['not finished']}))
            continue
        request = json.loads(path.read_text())
        twin = next((k for k in kept if generate.similarity(generate.signature(k), generate.signature(request))
                     >= generate.SIMILAR), None)
        if twin is not None:
            rows.append(base_row(index) | {'passed': False, 'problems': [f'near-duplicate of {twin["id"]}']})
            path.unlink()
            continue
        kept.append(request)
        rows.append(base_row(index) | {'passed': True, 'problems': []})
    rewritten = []
    worker = threading.Thread(target=lambda: rewritten.extend(generate.respread(kept, client, output)), daemon=True)
    worker.start()
    worker.join(timeout=args.task_timeout * 2)  # repetition rewrites are optional; never wait on a stall
    summary = {'requested': count, 'kept': len(kept), 'domains': [d['name'] for d in domains],
               'openings_rewritten_for_repetition': rewritten,
               'opening_diversity': openings.diversity([k['user_request'] for k in kept]), 'rows': rows,
               'finished_by': 'experiments/bench/finish_generation.py'}
    (output / 'generation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"{len(kept)}/{count} requests kept", flush=True)
    os._exit(0)  # do not wait for threads stuck on a stalled connection


if __name__ == '__main__':
    main()
