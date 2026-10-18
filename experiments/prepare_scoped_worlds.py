"""Materialize caller-owned train/dev customer partitions and independently certify each world view."""
import argparse
import copy
import json
from pathlib import Path
import shutil

from env_factory.certify import bundle
from env_factory.production.data_scope import check_references, partition
from env_factory.runtime import canonical, run_reference
from recheck_world_tasks import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plans = json.loads(args.plan.read_text())
    identity = {'plan': plans, 'platform': bundle.platform_identity(), 'runner': bundle.sha256(__file__),
                'scope_code': bundle.sha256(Path(__file__).resolve().parents[1] / 'src/env_factory/production/data_scope.py')}
    frozen = args.output / 'identity.json'
    if frozen.exists() and json.loads(frozen.read_text()) != identity:
        raise ValueError('scope inputs changed; use a new output')
    save(frozen, identity)
    results = []
    for domain, plan in plans.items():
        for split in ('train', 'dev'):
            target = args.output / domain / split
            receipt = target / 'result.json'
            if receipt.exists():
                result = json.loads(receipt.read_text())
                bundle.load_bundle(result['bundle'])
                results.append(result)
                continue
            failures = []
            candidates = sorted((args.pilot / domain / 'tasks').glob('t*/bundle'))
            for source in candidates:
                # The new scope is fully qualified below; immutable source files may
                # come from a prior platform version, as with bundle.requalify.
                loaded = bundle.load_bundle(source, require_current_platform=False)
                request = copy.deepcopy(loaded['request'])
                scoped = partition(request['tables'], plan['ownership'], seed=plan['seed'],
                                   heldout_count=plan['heldout_count'])['partitions'][split]
                if scoped['owners'] != plan['owner_groups'][split]:
                    raise ValueError('owner partition drift')
                problems = check_references(scoped['tables'], plan['references'])
                if problems:
                    failures.append({'source': str(source), 'problems': problems})
                    continue
                original = run_reference(loaded['env'], request['tables'], request['user_facts'],
                                         start=request['time']['start'])
                request['tables'] = scoped['tables']
                current = run_reference(loaded['env'], request['tables'], request['user_facts'],
                                        start=request['time']['start'])
                if canonical(original['answer']) != canonical(current['answer']):
                    failures.append({'source': str(source), 'problems': ['sample customer not solvable in partition']})
                    continue
                request['id'] += f'-{split}'
                request['source'] = {**(request.get('source') or {}), 'data_split': split,
                                     'owner_partition_seed': plan['seed'], 'scoped_from': str(source.resolve())}
                report = bundle.qualify_loaded({**loaded, 'request': request})
                if not report['passed']:
                    failures.append({'source': str(source), 'report': report})
                    continue
                stage = target / 'source'
                stage.mkdir(parents=True, exist_ok=True)
                for name in ('env.py', 'verifier.py', 'answer_schema.json'):
                    shutil.copyfile(source / name, stage / name)
                save(stage / 'request.json', request)
                bundle.export(stage, stage / 'request.json', report, target / 'bundle', status='qualified')
                result = {'domain': domain, 'split': split, 'bundle': str((target / 'bundle').resolve()),
                          'owners': scoped['owners'], 'private_row_ids': scoped['private_row_ids'],
                          'table_counts': {k: len(v) for k, v in scoped['tables'].items()},
                          'source_sample_bundle': str(source.resolve()), 'passed': True, 'training_ready': False}
                save(target / 'qualification.json', report)
                save(target / 'sample_selection.json', failures)
                save(receipt, result)
                results.append(result)
                print(json.dumps({k: v for k, v in result.items() if k in ('domain', 'split', 'passed', 'table_counts')}), flush=True)
                break
            else:
                save(target / 'sample_selection.json', failures)
                raise ValueError(f'no valid sample for {domain}/{split}')
    for domain in plans:
        train, dev = [next(r for r in results if r['domain'] == domain and r['split'] == s) for s in ('train', 'dev')]
        assert set(train['owners']).isdisjoint(dev['owners'])
        for table in train['private_row_ids']:
            assert set(train['private_row_ids'][table]).isdisjoint(dev['private_row_ids'][table])
    save(args.output / 'summary.json', {'results': results, 'private_data_disjoint': True,
                                       'training_ready': False,
                                       'limits': 'Customer/record isolation only; scenario-family split and task/trajectory admission remain.'})


if __name__ == '__main__':
    main()
