"""Independent validation branches with one complete, actionable defect report."""
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

from .execution_progress import ExecutionProgress
from .semantic_reward import SemanticCalibrationUnavailable


def verify_independent_routes(artifacts, root, verify, prepare, *, workers=2, **options):
    if type(workers) is not int or not 1 <= workers <= 2:
        raise ValueError('validation workers must be 1 or 2')
    jobs = [('references', artifacts, root, {**options, 'external_routes': True})]
    for probe in artifacts.get('execution_route_probes', []):
        candidate, probe_root = prepare(artifacts, root, probe)
        jobs.append((probe['route_id'], candidate, probe_root,
            {'initial_changes': probe['initial_changes'], 'verify_route_probes': False,
             'judgment_cache': options.get('judgment_cache')}))
    reports, failures, exceptions, unavailable = {}, [], [], []
    progress = ExecutionProgress(root)
    # Every job constructs its own SQLite episode/data and writes its own
    # directory. Two workers bound per-task pressure; the bridge also enforces
    # a process-wide model concurrency limit across simultaneous pipelines.
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        pending = {executor.submit(verify, candidate, directory, **kwargs): identity
                   for identity, candidate, directory, kwargs in jobs}
        for future in as_completed(pending):
            if future.cancelled():
                continue
            identity = pending[future]
            try:
                reports[identity] = future.result()
            except (ValueError, RuntimeError) as exc:
                exceptions.append(exc)
                if isinstance(exc, SemanticCalibrationUnavailable):
                    unavailable.append(exc)
                    for other in pending:
                        other.cancel()
                failures.append({'route_id': identity, 'error_type': type(exc).__name__, 'error': str(exc)})
                progress.feedback('validation_branch', route_id=identity, error_type=type(exc).__name__)
    failures.sort(key=lambda item: item['route_id'])
    (root / 'validation_findings.json').write_text(json.dumps({
        'completed_branches': sorted(reports), 'findings': failures}, ensure_ascii=False, indent=2) + '\n')
    if unavailable:
        raise unavailable[0]
    if failures:
        if len(failures) == 1:
            raise exceptions[0]
        raise ValueError('VALIDATION_BATCH_FAILED: ' + json.dumps(failures, ensure_ascii=False))
    report = reports['references']
    conflicts = {key: value for observed in reports.values()
                 for key, value in observed.get('semantic_judgment_reuse', {}).get('conflicting_judgments', {}).items()}
    shared_cache = options.get('judgment_cache')
    if shared_cache is not None:
        dependencies = {key for observed in reports.values()
                        for key in observed.get('semantic_judgment_reuse', {}).get('dependency_keys', [])}
        conflicts.update({key: value for key, value in shared_cache.snapshot()[1].items() if key in dependencies})
    if conflicts:
        raise SemanticCalibrationUnavailable('SEMANTIC_JUDGMENT_CONFLICT: ' + json.dumps(conflicts, ensure_ascii=False))
    coverage = report['execution_route_coverage']
    for probe in artifacts.get('execution_route_probes', []):
        observed = reports[probe['route_id']]
        directory = root / 'route_probes' / probe['route_id']
        (directory / 'verification.json').write_text(json.dumps(observed, ensure_ascii=False, indent=2) + '\n')
        report['realism_observations'].extend(observed.get('realism_observations', []))
        coverage['probes'].append({'route_id': probe['route_id'], 'initial_changes': probe['initial_changes'],
            'user_variables': probe.get('user_variables', {}), 'passed': observed['passed'], 'checks': observed['checks']})
    routes = artifacts['task_spec'].get('execution_routes', [])
    if routes and set(coverage['references']) | {p['route_id'] for p in coverage['probes']} != {r['id'] for r in routes}:
        raise ValueError('EXECUTION_ROUTE_UNCOVERED: every declared route needs a passing reference/probe')
    from .realism_requirements import require_realism_coverage
    require_realism_coverage(artifacts, report['realism_observations'])
    complexity = artifacts['task_spec'].get('interaction_complexity', {})
    if (complexity.get('require_behavioral_diversity') and complexity.get('variants', 0) > 1
            and report['distinct_executed_interaction_behaviors'] < 2):
        raise ValueError('INTERACTION_BEHAVIOR_DIVERSITY: successful references execute only one user behavior')
    return report
