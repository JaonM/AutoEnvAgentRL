"""Audit immutable natural replay transcripts before selecting SFT demonstrations."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from env_factory.certify import bundle
from env_factory.core.llm import ChatClient, load_env_file
from env_factory.production import dialogue_audit
from env_factory.runtime import is_write, tool_specs
from recheck_world_tasks import save
from trajectory_dedup import deduplicate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--parallel', type=int, default=2)
    parser.add_argument('--wait-seconds', type=int, default=0)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('use a separate output directory')
    load_env_file()
    plan = json.loads((args.source / 'plan.json').read_text())
    judge = ChatClient('JUDGE')
    identity = {'audit_code': bundle.sha256(dialogue_audit.__file__), 'runner': bundle.sha256(__file__),
                'source_identity': json.loads((args.source / 'identity.json').read_text()),
                'model': judge.model, 'endpoint': judge.base_url}
    frozen = args.output / 'identity.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('audit configuration changed; use a new output')
    save(frozen, identity)
    jobs = [(domain, assignment) for domain in plan['domains'] for assignment in domain['assignments']]

    def one(job):
        domain, assignment = job
        folder = args.source / domain['name'] / 'tasks' / f't{assignment["index"]:03d}'
        deadline = time.monotonic() + args.wait_seconds
        while True:
            paths = [p for p in (folder / 'natural.json', folder / 'diagnostic_natural.json') if p.exists()]
            if paths:
                source = paths[0]
                # Source writers are not atomic; wait for a complete JSON document.
                try:
                    record = json.loads(source.read_text())
                    break
                except json.JSONDecodeError:
                    pass
            if time.monotonic() >= deadline:
                return {'domain': domain['name'], 'index': assignment['index'], 'pending': True}
            time.sleep(2)
        loaded = bundle.load_bundle(folder / 'bundle')
        request = loaded['request']
        functions = loaded['env'].make_tools()
        write_names = {name for name, fn in functions.items() if is_write(fn)}
        sources = [(f'raw:{i}', episode) for i, episode in enumerate(record['raw_episodes'])]
        if record.get('bounded_episode'):
            sources.append(('bounded', record['bounded_episode']))
        rows, candidates = [], []
        for name, episode in sources:
            target = args.output / domain['name'] / f't{assignment["index"]:03d}' / f'{name.replace(":", "-")}.json'
            episode_hash = hashlib.sha256(json.dumps(episode, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if target.exists():
                row = json.loads(target.read_text())
                if row['episode_sha256'] != episode_hash or row['bundle_files'] != loaded['manifest']['files']:
                    raise ValueError('source transcript/bundle changed')
            else:
                try:
                    row = dialogue_audit.audit(ChatClient('JUDGE'), request, episode,
                                                tool_specs(functions, consent=False), write_names)
                except Exception as exc:
                    row = {'passed': False, 'infrastructure_error': f'{type(exc).__name__}: {exc}'[:500]}
                row.update(source=str(source), source_episode=name, episode_sha256=episode_hash,
                           bundle_files=loaded['manifest']['files'])
                save(target, row)
            accepted = bool(row['passed'] and episode.get('pilot_accepted') and record.get('task_certified')
                            and not record.get('diagnostic_only'))
            rows.append({'source_episode': name, 'audit_passed': row['passed'], 'candidate': accepted,
                         'business_success': bool(episode.get('success')),
                         'errors': row.get('errors', []), 'infrastructure_error': row.get('infrastructure_error')})
            if accepted:
                candidates.append({**episode, 'id': request['id'], 'source_episode': name,
                                   **{k: record.get(k) for k in ('domain', 'world', 'consent', 'platform_identity',
                                                                'evaluation_identity')}})
        selected, report = deduplicate(candidates, .85)
        result = {'domain': domain['name'], 'index': assignment['index'], 'episodes': rows,
                  'selected_sources': [e['source_episode'] for e in selected], 'dedup': report,
                  'training_ready': False}
        save(args.output / domain['name'] / f't{assignment["index"]:03d}' / 'selection.json', result)
        return result

    rows = []
    with ThreadPoolExecutor(args.parallel) as pool:
        for future in as_completed([pool.submit(one, job) for job in jobs]):
            rows.append(future.result())
            report = {'completed': sum(not r.get('pending') for r in rows), 'total': len(jobs),
                      'tasks_with_candidate': sum(bool(r.get('selected_sources')) for r in rows),
                      'selected_episodes': sum(len(r.get('selected_sources', [])) for r in rows),
                      'infrastructure_errors': sum(bool(e.get('infrastructure_error')) for r in rows for e in r.get('episodes', [])),
                      'results': rows, 'training_ready': False}
            save(args.output / 'summary.json', report)
            print(json.dumps({k: v for k, v in report.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
