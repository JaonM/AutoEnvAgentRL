"""Summarize the fixed world-task pilot without reading benchmark task data."""
from __future__ import annotations
import argparse
import collections
import json
import statistics
from pathlib import Path


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def histogram(values):
    return dict(sorted(collections.Counter(values).items()))


def distribution(values):
    return {'n': len(values), 'min': min(values), 'median': statistics.median(values), 'max': max(values)} if values else {'n': 0}


def summarize(root):
    plan = read(root / 'plan.json')
    domains, cases = {}, []
    for domain in plan['domains']:
        rows, episodes, natural_reports, diagnostic_reports = [], [], [], []
        process = {p: {'assigned': 0, 'generated': 0, 'certified': 0, 'raw_solved': 0, 'bounded_solved': 0,
                       'raw_business_solved': 0, 'diagnostic_business_solved': 0}
                   for a in domain['assignments'] for p in a['processes']}
        for assignment in domain['assignments']:
            folder = root / domain['name'] / 'tasks' / f't{assignment["index"]:03d}'
            original = read(folder / 'pilot_result.json')
            row = read(folder / 'requalification.json') or original
            natural = read(folder / 'natural.json')
            diagnostic = read(folder / 'diagnostic_natural.json')
            case = {'domain': domain['name'], 'index': assignment['index'], 'type': assignment['type'],
                    'processes': assignment['processes'], 'script': assignment.get('script', []),
                    'requested_shape': assignment.get('shape'), 'generated': original is not None,
                    'original_passed': bool(original and original['passed']),
                    'requalified': (folder / 'requalification.json').exists(),
                    'certified': bool(row and row['passed']), 'stage': row.get('stage') if row else 'pending',
                    'folder': str(folder), 'requirements': assignment['requirements']}
            if row:
                rows.append(row)
                case.update(bundle=row.get('bundle'), actual_shape=row.get('actual_shape'),
                            reference_actual=(row.get('task_quality') or {}).get('actual'),
                            reference_answer=row.get('reference_answer'),
                            repairs_used=row.get('task_repairs_used'), problems=[] if row['passed'] else row.get('problems', []))
                if row.get('bundle'):
                    request = read(Path(row['bundle']) / 'request.json')
                    case['opening'] = request['user_request']
                    case['user_script'] = request.get('user_script')
                suff = read(folder / 'sufficiency_recheck.json')
                if suff and not suff['passed']:
                    case['sufficiency_failure'] = suff
            if natural:
                natural_reports.append(natural)
                episodes.extend(natural['raw_episodes'])
                case.update(raw_business_successes=natural['raw_successes'],
                            raw_strict_successes=natural['raw_policy_successes'],
                            bounded_success=natural['bounded_success'],
                            raw_valid_business_successes=sum(bool(e.get('success')) and e.get('valid', True)
                                is not False and e.get('required_actions_passed', True) for e in natural['raw_episodes']))
            if diagnostic:
                diagnostic_reports.append(diagnostic)
                case['diagnostic_only'] = {'raw_business_successes': diagnostic['raw_successes'],
                                          'raw_episodes': len(diagnostic['raw_episodes']),
                                          'legacy_strict_successes': diagnostic['raw_policy_successes']}
            for p in assignment['processes']:
                process[p]['assigned'] += 1
                process[p]['generated'] += bool(original)
                process[p]['certified'] += case['certified']
                process[p]['raw_solved'] += bool(natural and natural['raw_policy_successes'])
                process[p]['bounded_solved'] += bool(natural and natural['bounded_success'])
                process[p]['raw_business_solved'] += bool(natural and any(e.get('success') and
                    e.get('valid', True) is not False and e.get('required_actions_passed', True)
                    for e in natural['raw_episodes']))
                process[p]['diagnostic_business_solved'] += bool(diagnostic and any(e.get('success') and
                    e.get('valid', True) is not False and e.get('required_actions_passed', True)
                    for e in diagnostic['raw_episodes']))
            cases.append(case)
        measured = [(r.get('task_quality') or {}).get('actual') for r in rows if r['passed']]
        measured = [m for m in measured if m is not None]
        failures = collections.Counter()
        for e in episodes:
            if e.get('pilot_accepted'):
                continue
            if e.get('error'): failures['execution_error'] += 1
            if e.get('valid') is False: failures['invalid_simulator'] += 1
            if e.get('state_ok') is False: failures['wrong_state'] += 1
            if e.get('fields') and not all(e['fields'].values()): failures['answer_fields'] += 1
            if e.get('safety') is False: failures['safety'] += 1
            if e.get('trajectory_policy', {}).get('passed') is False: failures['dialogue_policy'] += 1
            if e.get('required_actions_passed') is False: failures['required_action_missing'] += 1
        domains[domain['name']] = {
            'planned': len(domain['assignments']), 'generated': len(rows),
            'original_certified': sum(bool(r.get('original_passed', r['passed'])) for r in rows),
            'certified': sum(r['passed'] for r in rows),
            'rejection_stages': histogram(r['stage'] for r in rows if not r['passed']),
            'process_coverage': process,
            'certified_shapes': histogram(r.get('actual_shape', 'unknown') for r in rows if r['passed']),
            'reference_complexity': {key: distribution([m[key] for m in measured]) for key in ('tool_calls', 'writes', 'goals')},
            'natural_tasks': len(natural_reports), 'raw_episodes': len(episodes),
            'diagnostic_tasks': len(diagnostic_reports),
            'diagnostic_episodes': sum(len(r['raw_episodes']) for r in diagnostic_reports),
            'diagnostic_business_successes': sum(r['raw_successes'] for r in diagnostic_reports),
            'raw_business_successes': sum(bool(e.get('success')) for e in episodes),
            'raw_valid_business_successes': sum(bool(e.get('success')) and e.get('valid', True) is not False
                and e.get('required_actions_passed', True) for e in episodes),
            'raw_strict_successes': sum(bool(e.get('pilot_accepted')) for e in episodes),
            'raw_solved_tasks': sum(bool(r['raw_policy_successes']) for r in natural_reports),
            'raw_business_solved_tasks': sum(any(e.get('success') and e.get('valid', True) is not False
                and e.get('required_actions_passed', True) for e in r['raw_episodes']) for r in natural_reports),
            'bounded_attempts': sum(r.get('bounded_episode') is not None for r in natural_reports),
            'bounded_solved_tasks': sum(r['bounded_success'] for r in natural_reports),
            'raw_failures_overlapping': dict(failures),
            'raw_valid_episodes': sum(not e.get('error') and e.get('valid', True) is not False for e in episodes),
            'natural_turns': distribution([e['turns'] for e in episodes if 'turns' in e]),
            'natural_tool_calls': distribution([e['tool_calls'] for e in episodes if 'tool_calls' in e]),
        }
    runs = [read(path) for path in sorted(root.glob('*/tasks/*/*natural.json'))]
    all_episodes = [e for run in runs for e in run['raw_episodes']]
    overall = {'tasks': len(runs), 'episodes': len(all_episodes),
               'business_successes': sum(bool(e.get('success')) for e in all_episodes),
               'state_matches': sum(e.get('state_ok') is True for e in all_episodes),
               'invalid_simulator_flags': sum(e.get('valid') is False for e in all_episodes),
               'safety_failure_flags': sum(e.get('safety') is False for e in all_episodes),
               'legacy_strict_successes': sum(bool(e.get('pilot_accepted')) for e in all_episodes),
               'tasks_with_business_success': sum(any(e.get('success') for e in run['raw_episodes']) for run in runs),
               'includes_noncertified_diagnostics': True}
    return {'complete': all(c['requalified'] for c in cases) and all(
                d['generated'] == d['planned'] and d['natural_tasks'] == d['certified'] for d in domains.values()),
            'all_raw_including_diagnostics': overall, 'domains': domains, 'cases': cases,
            'scoring_interpretation': {
                'primary': 'Business success with valid simulator and required actions; format diagnostics are separate.',
                'legacy_strict': 'raw_strict_successes applies the frozen public-contract single-call/no-mixed-turn constraints. It is not tau2 reward or a universal demonstration-quality rule.',
                'parallel_calls': 'Multiple tool calls do not alone prove an error; dependency, argument provenance and write conflicts require review.',
                'mixed_turns': 'tau2 default parsing accepts mixed messages and graders can inspect saved text, but the half-duplex user simulator does not receive tool-attached text in that turn.',
            },
            'limits': ['Stratified pilot; not a random sample of benchmark difficulty or its task distribution.',
                       'Two original-world episodes per certified task; not benchmark pass@8.',
                       'Bounded teacher retries are reported separately from raw model outcomes.',
                       'Rejected candidates remain in the denominator. No task is marked ready for SFT.',
                       'Structural/label coverage does not prove every intended dialogue event was enacted.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    report = summarize(args.root)
    (args.root / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': report['complete'], 'domains': report['domains']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
