"""Replay a fixed task pilot against the staged sufficiency checker; preserve original evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from env_factory.certify import bundle, sufficiency
from env_factory.core.llm import ChatClient, load_env_file


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--parallel', type=int, default=4)
    parser.add_argument('--k', type=int, default=2)
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    if source.resolve() == output.resolve():
        parser.error('use a new output directory')
    load_env_file()
    plan = json.loads((source / 'plan.json').read_text())
    checker = hashlib.sha256(Path(sufficiency.__file__).read_bytes()).hexdigest()
    jobs = [(d, a) for d in plan['domains'] for a in d['assignments']]

    def check(job):
        domain, assignment = job
        task = source / domain['name'] / 'tasks' / f't{assignment["index"]:03d}'
        original = json.loads((task / 'pilot_result.json').read_text())
        target = Path(original['bundle'])
        loaded = bundle.load_bundle(target)
        identity = {'checker_sha256': checker, 'files': loaded['manifest']['files'], 'k': args.k}
        path = output / domain['name'] / f't{assignment["index"]:03d}.json'
        if path.exists():
            cached = json.loads(path.read_text())
            if cached.get('identity') != identity:
                raise ValueError('replay configuration changed; use a new output directory')
            if not cached.get('infrastructure_error'):
                return cached
        row = {'domain': domain['name'], 'index': assignment['index'], 'source_bundle': str(target),
               'original_passed': original['passed'], 'identity': identity, 'training_ready': False}
        try:
            row['sufficiency'] = sufficiency.check(loaded, ChatClient('LLM'), k=args.k)
            row['passed'] = row['sufficiency']['passed']
        except Exception as exc:
            row.update(passed=False, infrastructure_error=f'{type(exc).__name__}: {exc}'[:500])
        save(path, row)
        return row

    rows = []
    with ThreadPoolExecutor(args.parallel) as pool:
        for future in as_completed([pool.submit(check, job) for job in jobs]):
            row = future.result()
            rows.append(row)
            summary = {'completed': len(rows), 'total': len(jobs), 'passed': sum(r['passed'] for r in rows),
                       'infrastructure_errors': sum(bool(r.get('infrastructure_error')) for r in rows),
                       'checker_sha256': checker, 'training_ready': False,
                       'scope': 'Policy sufficiency replay only; natural dialogue and other gates remain separate.',
                       'tasks': [{k: v for k, v in r.items() if k not in ('sufficiency', 'identity')} for r in rows]}
            save(output / 'summary.json', summary)
            print(json.dumps({k: summary[k] for k in ('completed', 'total', 'passed', 'infrastructure_errors')}), flush=True)


if __name__ == '__main__':
    main()
