"""Expand certified synthetic worlds without changing policy, APIs or verifiers."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from env_factory.core import spec as request_spec
from env_factory.certify import bundle
from env_factory.core.reward import grade
from env_factory.production.data_scope import partition
from env_factory.production.world import task_schema
from env_factory.production.world_data import author_context, integrity, profile
from env_factory.runtime import canonical, run_reference
from world_extensions import scale_recipes as recipes


def save(path, value):
    path = Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def certify(source, request, output):
    loaded = bundle.load_bundle(source, require_current_platform=False)
    schema = task_schema(request['goal_schema'], request['user_facts']['goals']) or loaded['schema']
    loaded = {**loaded, 'request': request, 'schema': schema}
    report = bundle.qualify_loaded(loaded)
    save(output / 'qualification.json', report)
    if not report['passed']:
        raise ValueError(f'expanded world failed qualification: {output}')
    stage = output / 'source';stage.mkdir(parents=True, exist_ok=True)
    for name in ('env.py', 'verifier.py'):
        shutil.copyfile(source / name, stage / name)
    request_spec.write(request, stage / 'request.json');save(stage / 'answer_schema.json', schema)
    bundle.export(stage, stage / 'request.json', report, output / 'bundle', status='qualified')
    return loaded


def probes(loaded, domain):
    request = loaded['request'];tables = request['tables'];schema = request['goal_schema']
    users = {r['user_id']: r for r in tables['users']};cases = []
    if domain == 'retail':
        for order in [r for r in tables['orders'] if r['id'].startswith('#SX')][::17]:
            params = {'user_id': order['user_id'], 'order_id': order['order_id']}
            process = 'cancel';params['reason'] = 'no longer needed'
            cases.append((process, params))
    else:
        for r in [r for r in tables['reservations'] if r['id'].startswith('SR')][::37]:
            cases.append(('eligible_cancellation', {'user_id': r['user_id'], 'reservation_id': r['reservation_id'],
                                                     'cancellation_reason': 'health'}))
    results = []
    for index, (process, params) in enumerate(cases):
        user = users[params['user_id']]
        params = {k: v for k, v in params.items() if k in schema[process]['params']}
        facts = {'goals': [{'process': process, 'params': params}]}
        if domain == 'retail':
            facts['email'] = user['email']
        approval = schema[process].get('approval')
        if approval:
            for key, value in params.items():
                approval = approval.replace('<' + key + '>', str(value))
            facts[approval] = index % 4 != 0
        ref = run_reference(loaded['env'], tables, facts, start=request['time']['start'])
        expected = loaded['verifier'].expected(copy.deepcopy(tables), copy.deepcopy(facts), request['time']['start'])
        score = grade(ref, expected, task_schema(schema, facts['goals']), initial=tables, reference_calls=ref['tool_calls'])
        okay = (canonical(ref['answer']) == canonical(expected['answer']) and
                canonical(ref['final_state']) == canonical(expected['final_state']) and score['success'])
        results.append({'index': index, 'process': process, 'params': params, 'passed': bool(okay),
                        'answer': ref['answer'], 'tool_calls': ref['tool_calls']})
    if not results or not all(r['passed'] for r in results):
        raise ValueError('generated-record reference/verifier probes failed: ' + str([r for r in results if not r['passed']][:3]))
    outcomes = {a.get('status') for row in results for a in row['answer'].values()}
    if not {'completed', 'refused'} <= outcomes:
        raise ValueError('probes must exercise both successful writes and genuine refusals: ' + str(outcomes))
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('output/world_coverage_upgrade_v8'))
    p.add_argument('--pilot', type=Path, default=Path('output/world_tau2_task_dialogue_v4'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=20261020)
    args = p.parse_args();args.output.mkdir(parents=True, exist_ok=False)
    targets = {'retail': {'users': 500, 'orders': 1000, 'products': 50, 'variants': 600},
               'airline': {'users': 500, 'reservations': 2000, 'flights': 300, 'dates': 30, 'routes': 108}}
    summary = {}
    for domain, target in targets.items():
        source = args.source / domain / 'bundle';loaded = bundle.load_bundle(source, require_current_platform=False)
        request = copy.deepcopy(loaded['request']);before = request['tables']
        tables = getattr(recipes, domain)(before, target, args.seed)
        contract = recipes.contract(domain, target)
        errors = integrity(tables, contract) + recipes.semantics(domain, tables)
        save(args.output / domain / 'integrity.json', {'passed': not errors, 'errors': errors, 'contract': contract})
        if errors:
            raise ValueError(errors[:10])
        request.update(id=f'{domain}-scaled-v1', request_kind='world', tables=tables)
        request.pop('world_id', None)
        child = 'orders' if domain == 'retail' else 'reservations'
        ownership = {'owner_table': 'users', 'owner_key': 'user_id', 'tables': {'users': 'user_id', child: 'user_id'}}
        request['source'] = {**request.get('source', {}), 'data_expansion_seed': args.seed,
            'source_bundle': str(source.resolve()), 'source_files': loaded['manifest']['files'],
            'authoring_scope': {'ownership': ownership, 'references': contract['references'], 'seed': args.seed,
                                'owners_per_task': 4, 'catalog_rows': 12, 'max_bytes': 600000}}
        grown = certify(source, request, args.output / domain / 'world')
        record_probes = probes(grown, domain)
        save(args.output / domain / 'record_probes.json', record_probes)
        split = partition(tables, ownership, seed=args.seed, heldout_count=100)
        groups = []
        samples = [loaded['request']] + [request_spec.read(p) for p in sorted((args.pilot / domain / 'tasks').glob('t*/bundle/request.json'))]
        for name, scope in split['partitions'].items():
            scoped = copy.deepcopy(request);scoped['tables'] = scope['tables'];scoped['id'] += '-' + name
            selected = None
            for sample in samples:
                facts = sample['user_facts']
                # Scope must contain every private owner/record used by the sample.
                owners_in_facts = {v for g in facts.get('goals', []) for k, v in g.get('params', {}).items() if k == 'user_id'}
                if not owners_in_facts or not owners_in_facts <= set(scope['owners']):
                    continue
                for field in ('user_facts', 'user_scene', 'user_request', 'user_script', 'fact_descriptions'):
                    if field in sample:
                        scoped[field] = copy.deepcopy(sample[field])
                    else:
                        scoped.pop(field, None)
                full_ref = run_reference(grown['env'], tables, facts, start=request['time']['start'])
                view_ref = run_reference(grown['env'], scope['tables'], facts, start=request['time']['start'])
                if canonical(full_ref['answer']) != canonical(view_ref['answer']):
                    continue
                selected = sample['id'];break
            if selected is None:
                raise ValueError('no valid sample for expanded scope ' + domain + '/' + name)
            target_dir = args.output / domain / name
            certify(source, scoped, target_dir)
            views = [author_context(scoped, i) for i in range(30)]
            for view in views:
                errs = integrity(view['tables'], {'references': contract['references'], 'inverse': contract['inverse']})
                if errs:
                    raise ValueError('author view lost graph closure: ' + str(errs[:3]))
            row = {'name': f'{domain}_{name}', 'business_domain': domain, 'split': name,
                   'bundle': str((target_dir / 'bundle').resolve()), 'sample': selected,
                   'data_scope': {'ownership': ownership, 'owners': scope['owners'], 'private_row_ids': scope['private_row_ids'],
                       'references': [{'table': child, 'field': 'user_id', 'target_table': 'users', 'target_field': 'user_id'},
                                      {'table': 'users', 'field': child, 'target_table': child,
                                       'target_field': 'order_id' if domain == 'retail' else 'reservation_id'}]},
                   'table_counts': {k: len(v) for k, v in scope['tables'].items()},
                   'author_view_max_bytes': max(len(json.dumps(v).encode()) for v in views), 'passed': True}
            save(target_dir / 'result.json', row);groups.append(row)
        stats = profile(tables, recipes.metrics(domain))
        if domain == 'airline':
            stats['airports'] = len({f[k] for f in tables['flights'] for k in ('origin', 'destination')})
            stats['routes'] = len({(f['origin'], f['destination']) for f in tables['flights']})
            stats['dated_flights'] = sum(len(f['dates']) for f in tables['flights'])
        else:
            stats['variants'] = sum(len(p['variants']) for p in tables['products'])
        save(args.output / domain / 'profile.json', stats)
        summary[domain] = {'targets': target, 'profile': stats, 'groups': groups,
                           'record_probes': len(record_probes), 'passed': True}
        save(args.output / 'progress.json', summary)
        print(json.dumps({'domain': domain, 'tables': stats['tables'], 'probes': len(record_probes), 'passed': True}), flush=True)
    save(args.output / 'summary.json', {'passed': True, 'domains': summary, 'seed': args.seed,
         'source': str(args.source), 'benchmark_records_used': False, 'training_ready': False,
         'limits': ['Aggregate size is not evidence of equal task difficulty.',
                    'Only domain data recipes encode business relationships; generic platform helpers are domain independent.']})


if __name__ == '__main__':
    main()
