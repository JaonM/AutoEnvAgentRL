"""Finish a run's generation stage in parallel after its request-author sessions are done.

generate.generate writes openings and agent policies one request at a time; with long policies that
is several minutes per task. This finishes the same stage for an interrupted run: requests already
written are kept, the rest get an opening and a policy in parallel (same functions, same plan),
then near-duplicates are dropped in index order, repetitive openings are respread, and
generation.json is written so `run` resumes at the build stage.

    python experiments/v3/bench/finish_generation.py --run output/v3_bench_pool --parallel 8
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--run', required=True)
    parser.add_argument('--parallel', type=int, default=8)
    parser.add_argument('--mix', default=None, help='the run\'s --mix (run_options.json does not record it)')
    args = parser.parse_args()
    from env_factory_v3.core.llm import ChatClient, load_env_file
    from env_factory_v3.production import agent_policy, generate, openings
    load_env_file()
    client = ChatClient('V3_OPENING')
    run = Path(args.run)
    options = json.loads((run / 'run_options.json').read_text())
    domains = json.loads((run / 'domains.json').read_text())
    count = options['count']
    mix = generate.parse_mix(args.mix) if args.mix else None
    plan = generate.plan(domains, count, options.get('rubric_share', 0.0), **({'mix': mix} if mix else {}))
    langs = generate.languages(count, options.get('english_share', 0.0))
    output = run / 'generate'
    (output / 'requests').mkdir(exist_ok=True)

    def one(index):
        domain, category, shape = plan[index]
        row = {'index': index, 'domain': domain['name'], 'category': category, 'shape': shape, 'language': langs[index]}
        done = output / 'requests' / f'gen-{index:03d}.json'
        if done.exists():
            return row | {'passed': True, 'problems': []}, json.loads(done.read_text())
        folder = output / 'sessions' / f'{index:03d}'
        report = json.loads((folder / 'check-request.json').read_text()) if (folder / 'check-request.json').exists() else None
        if not report or not report['passed']:
            return row | {'passed': False, 'problems': (report or {}).get('problems', ['no request'])[:5]}, None
        request = json.loads((folder / 'request.json').read_text())
        request.setdefault('time', {'start': 10, 'tick': 0})
        request['language'] = langs[index]
        opening, persona, problems = generate.write_opening(client, request, f'gen-{index:03d}')
        row['persona'] = persona
        if opening is None:
            return row | {'passed': False, 'problems': problems}, None
        request['user_request'] = opening
        policy, problems = agent_policy.write(client, request)
        if policy is None:
            return row | {'passed': False, 'problems': [f'agent policy: {p}' for p in problems[:4]]}, None
        request['agent_policy'] = policy
        request['source'] = {'generated': domain['name'], 'family': domain.get('family'), 'shape': shape,
                             'persona': persona, 'language': langs[index], 'opening_writer': client.model}
        return row | {'passed': True, 'problems': []}, request

    with concurrent.futures.ThreadPoolExecutor(args.parallel) as pool:
        results = list(pool.map(one, range(count)))
    kept, rows = [], []
    for row, request in results:  # index order, as generate.generate does
        if request is not None:
            twin = next((k for k in kept if generate.similarity(generate.signature(k), generate.signature(request))
                         >= generate.SIMILAR), None)
            if twin is not None:
                row.update(passed=False, problems=[f'near-duplicate of {twin["id"]}'])
                (output / 'requests' / f"gen-{row['index']:03d}.json").unlink(missing_ok=True)
            else:
                request['id'] = f"gen-{row['index']:03d}"
                kept.append(request)
                (output / 'requests' / f"{request['id']}.json").write_text(json.dumps(request, ensure_ascii=False,
                                                                                      indent=1))
        rows.append(row)
    rewritten = generate.respread(kept, client, output)
    summary = {'requested': count, 'kept': len(kept), 'domains': [d['name'] for d in domains],
               'openings_rewritten_for_repetition': rewritten,
               'opening_diversity': openings.diversity([k['user_request'] for k in kept]), 'rows': rows,
               'finished_by': 'experiments/v3/bench/finish_generation.py'}
    (output / 'generation.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"{len(kept)}/{count} requests kept", flush=True)


if __name__ == '__main__':
    main()
