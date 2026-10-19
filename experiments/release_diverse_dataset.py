"""Revalidate diversity candidates and publish a complete, immutable SFT release.

No model calls and no transcript rewriting. Saved model judgments are checked
against their actual quotes, reference execution and the certified bundle.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from env_factory.certify import bundle, trajectory
from env_factory.coverage import matches
from env_factory.diversity import AdmissionLedger, Normalizer, digest, features, repetition
from env_factory.diversity_context import signature
from env_factory.production import data_scope, dialogue_audit
from env_factory.runtime import canonical, is_write, run_reference


TARGETS = {'retail_train': 500, 'retail_dev': 100, 'airline_train': 500, 'airline_dev': 100}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def selected_records(producer, *, complete):
    progress = read(producer / 'progress.json')
    identity = read(producer / 'identity.json')
    for name, expected in identity['code'].items():
        require(sha(name) == expected, 'generation input changed: ' + name)
    plan = read(identity['plan'])
    require(plan['fill_targets'] == TARGETS, 'unexpected target quotas')
    if complete:
        require(progress['phase'] == 'completed', 'producer has not completed')
        manifest = read(producer / 'candidate_manifest.json')
        require(manifest['data_target_met'] and manifest['counts'] == TARGETS, 'candidate quotas incomplete')
        for name, expected in manifest['source_hashes'].items():
            require(sha(producer / name) == expected, 'candidate export checksum mismatch')
        return manifest['records'], plan, identity
    baseline = read(Path(plan['source_audit']) / 'records.json')
    records = [{'episode': r['episode'], 'split': r['split'], 'group': r['group'],
                'key': r['contextual_task_key']} for r in baseline]
    records += [{'episode': r['collection']['selected_episode'], 'split': r['domain'].rsplit('_', 1)[1],
                 'group': r['domain'], 'key': r['task_key']}
                for r in progress['results'] if r.get('passed') and r['stage'] == 'admitted']
    return records, plan, identity


def validate_population(episodes, records, *, complete):
    ids = [e['id'] for e in episodes]
    require(len(set(ids)) == len(ids), 'duplicate episode IDs')
    counts = Counter(f"{e['domain']}_{e['split']}" for e in episodes)
    require(set(counts) <= set(TARGETS), 'unexpected dataset group')
    require(all(n <= TARGETS[g] for g, n in counts.items()), 'dataset quota overflow')
    if complete:
        require(counts == TARGETS, 'dataset requires 1000 train and 200 dev tasks')
    require(len(episodes) == len(records), 'record count mismatch')
    for e, r in zip(episodes, records):
        require(r['split'] == e['split'] and r['group'] == f"{e['domain']}_{e['split']}",
                'candidate split metadata mismatch')
    families = {s: {e['scenario_family'] for e in episodes if e['split'] == s} for s in ('train', 'dev')}
    require(not families['train'] & families['dev'], 'scenario families overlap')
    AdmissionLedger([{'split': r['split'], 'task_key': r['key']} for r in records], {'train': 3, 'dev': 1})
    return dict(counts)


def validate_evidence(episode, candidate, pilot, loaded):
    require(candidate.get('accepted') is True and candidate.get('required_actions_passed') is True,
            'source demonstration not admitted')
    require(pilot.get('passed') is True and loaded['gates'].get('passed') is True,
            'task or bundle gates failed')
    for key in ('success', 'policy_aware_success', 'quality_audit_passed'):
        require(episode.get(key) is True, 'unqualified selected episode: ' + key)
    require(bool(episode.get('messages')), 'missing transcript')
    require(all(episode.get(k) == v for k, v in candidate['episode'].items()),
            'selected episode differs from source demonstration')
    migration=episode.get('migration') or {}
    if migration.get('kind')=='deterministic_replay':
        original=Path(migration['original_episode'])
        require(sha(original)==migration['original_sha256'],'replay original evidence changed')
        old=read(original)
        demonstration=Path(migration['original_demonstration'])
        require(sha(demonstration)==migration['original_demonstration_sha256'],'original demonstration changed')
        require(old['messages']==episode['messages'],'replay transcript was rewritten')
        require(old['answer']==episode['answer'],'replay answer was rewritten')
        require(migration.get('new_teacher_collection') is False,'replay mislabeled as collection')
    request = loaded['request']
    require(episode['source_task_id'] == request['id'], 'task identity mismatch')
    rules = trajectory.constraints(request)
    require(trajectory.qualify(episode, rules)['policy_aware_success'], 'dialogue policy failed')
    functions = loaded['env'].make_tools()
    writes = dialogue_audit.writes_of(episode['messages'], {n for n, fn in functions.items() if is_write(fn)})
    elements = [k for k in dialogue_audit.SCRIPT_KEYS if (request.get('user_script') or {}).get(k)]
    audit = candidate.get('audit', {})
    require(audit.get('passed') is True, 'source audit failed')
    checked = dialogue_audit.validate(audit.get('raw'), episode['messages'], writes, elements)
    require(checked['passed'], 'audit evidence invalid: ' + str(checked['errors']))


def coverage_report(rows, audit_root):
    result = {}
    for domain in ('retail', 'airline'):
        subset = [r for r in rows if r['domain'] == domain]
        branches, matrix = [], []
        prior = read(audit_root / domain / 'matrix.json')
        for cell in read(audit_root / domain / 'branches.json'):
            hits = [r for r in subset if any(all(matches(g, p) for p in cell['predicates']) for g in r['goals'])]
            branches.append({'id': cell['id'], 'natural_target': cell['natural_target'], 'count': len(hits)})
            # Retain the previously declared matrix denominator, rather than inventing new cells.
            obstacles = [m['obstacle'] for m in prior if m['branch'] == cell['id']]
            for obstacle in obstacles:
                matrix.append({'branch': cell['id'], 'obstacle': obstacle,
                               'count': sum(obstacle in r['witnesses'] for r in hits)})
        natural = [b for b in branches if b['natural_target']]
        result[domain] = {'tasks': len(subset), 'natural_branches_covered': sum(b['count'] > 0 for b in natural),
                          'natural_branches_total': len(natural), 'branches': branches,
                          'matrix': matrix, 'matrix_covered': sum(m['count'] > 0 for m in matrix),
                          'matrix_total': len(matrix), 'repetition': {}}
        for name, field in [('contextual_structure', 'key'), ('actual_tool_path', 'tool_path'),
                            ('normalized_actual_calls', 'normalized_calls')]:
            stats = repetition(subset, lambda r: r[field])
            result[domain]['repetition'][name] = {k: v for k, v in stats.items() if k != 'clusters'}
    return result


def plan_groups(plan):
    # Fully populated baseline groups may have no remaining generation jobs.
    return {d['name']:d for d in (plan.get('domains') or [j['domain'] for j in plan['jobs']])}


def audit_and_release(producer, output, audit_root, *, complete=True):
    output.mkdir(parents=True, exist_ok=False)
    records, plan, identity = selected_records(producer, complete=complete)
    rules = read(plan['context_rules'])
    groups = plan_groups(plan)
    episodes = [read(r['episode']) for r in records]
    counts = validate_population(episodes, records, complete=complete)
    receipts, rows, pinned = [], [], {}
    for index, (record, episode) in enumerate(zip(records, episodes)):
        source = Path(episode['source_episode'])
        pilot_path = source.parent / 'pilot_result.json'
        candidate, pilot = read(source), read(pilot_path)
        folder = Path(episode['sandbox'])
        require(folder.resolve() == Path(pilot['bundle']).resolve(), 'pilot bundle mismatch')
        loaded = bundle.load_bundle(folder)
        require(episode['platform_identity'] == loaded['manifest']['platform_identity'] == identity['platform'],
                'episode platform identity mismatch')
        validate_evidence(episode, candidate, pilot, loaded)
        request = loaded['request']
        limits = groups[record['group']]['data_scope']
        require(not data_scope.check_references(request['tables'], limits['references']), 'broken table references')
        for table, field in limits['ownership']['tables'].items():
            table_rows = request['tables'][table]
            require(all(r[field] in limits['owners'] for r in table_rows) and
                    {r['id'] for r in table_rows} == set(limits['private_row_ids'][table]), 'private data split changed')
        reference = run_reference(loaded['env'], request['tables'], request['user_facts'], start=request['time']['start'])
        require(canonical(reference['answer']) == canonical(pilot['reference_answer']), 'reference answer changed')
        f = features(request, pilot, episode)
        key, _ = signature(request, pilot, f, Normalizer(request), rules[episode['domain']], digest)
        require(key == record['key'], 'normalized structure changed')
        witnesses = dict(f['observed_obstacles'])
        for event in candidate['audit']['raw']['script_events']:
            witnesses[event['element']] = event['evidence']
        goals = [dict(process=g['process'], params=g['params'], observed=reference['answer'][f'goal_{i}'])
                 for i, g in enumerate(request['user_facts']['goals'], 1)]
        rows.append({'id': episode['id'], 'domain': episode['domain'], 'split': episode['split'],
                     'key': key, 'goals': goals, 'witnesses': witnesses, 'tool_path': f['tool_path'],
                     'normalized_calls': f['normalized_calls'], 'task_type': pilot['assignment']['type']})
        paths = [Path(record['episode']), source, pilot_path, folder / 'manifest.json',
                 *[folder / name for name in loaded['manifest']['files']]]
        hashes = {str(p.resolve()): sha(p) for p in paths}
        pinned.update(hashes)
        receipts.append({'id': episode['id'], 'structure': key, 'source_hashes': hashes})
        if index % 50 == 0:
            save(output / 'progress.json', {'checked': index + 1, 'total': len(records), 'training_ready': False})
    # Original baseline is retained: verify coverage of each previously selected branch survives.
    baseline_ids = {read(r['episode'])['id'] for r in read(Path(plan['source_audit']) / 'records.json')}
    require(baseline_ids <= {e['id'] for e in episodes}, 'selected baseline was lost')
    coverage = coverage_report(rows, audit_root)
    save(output / 'coverage.json', coverage)
    save(output / 'records.json', rows)
    save(output / 'evidence_receipts.json', receipts)
    for name, expected in {**identity['code'], **pinned}.items():
        require(sha(name) == expected, 'input changed during review: ' + name)
    report = {'passed': True, 'complete_dataset': complete, 'counts': counts, 'checked': len(records),
              'cross_split_structure_overlap': 0, 'structure_caps': {'train': 3, 'dev': 1},
              'baseline_retained': len(baseline_ids), 'limits': [
                  'Saved semantic model judgments are evidence-checked, not rerun or treated as formal proofs.',
                  'Coverage gaps are reported; this is not an assertion of exhaustive benchmark coverage.',
                  'Matching tool paths alone do not reject a task.']}
    save(output / 'review.json', report)
    if complete:
        # Match the producer's sealed export exactly; release.json is written last.
        exported = [json.loads(line) for split in ('train', 'dev')
                    for line in (producer / (split + '.jsonl')).read_text().splitlines()]
        require({e['id']: e for e in exported} == {e['id']: e for e in episodes} and len(exported) == 1200,
                'exported episodes differ from reviewed source')
        (output / 'teacher.jsonl').write_text(''.join(json.dumps(e, ensure_ascii=False) + '\n' for e in episodes))
        save(output / 'valid_ids.json', sorted(e['id'] for e in episodes if e['split'] == 'dev'))
        save(output / 'tasks.json', {'tasks': [{'id': e['id'], 'sandbox': e['sandbox']} for e in episodes]})
        names = ['teacher.jsonl', 'valid_ids.json', 'tasks.json', 'review.json', 'coverage.json',
                 'records.json', 'evidence_receipts.json']
        save(output / 'release.json', {'training_ready': True, 'train_tasks': 1000, 'dev_tasks': 200,
             'one_episode_per_task': True, 'producer': str(producer), 'platform': identity['platform'],
             'source_hashes': {name: sha(output / name) for name in names}, 'limits': report['limits']})
    save(output / 'progress.json', {'checked': len(records), 'total': len(records), 'training_ready': complete})
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--producer', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--audit-root', type=Path, required=True)
    p.add_argument('--inspect-current', action='store_true')
    args = p.parse_args()
    print(json.dumps(audit_and_release(args.producer, args.output, args.audit_root,
                                      complete=not args.inspect_current), ensure_ascii=False))


if __name__ == '__main__':
    main()
