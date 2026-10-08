"""Model-free execution diagnostics. A clean report never certifies delivery."""
import copy
import json

from env_factory.sandbox_runtime import SandboxError
from .report_values import report_value


def check_all_routes(artifacts, root, verify, prepare):
    """Collect independent deterministic defects before any semantic call."""
    reports = [{'route_id': 'references', **verify(artifacts, root,
                verify_route_probes=False, deterministic_only=True)}]
    for probe in artifacts.get('execution_route_probes', []):
        candidate, probe_root = prepare(artifacts, root, probe)
        reports.append({'route_id': probe['route_id'], **verify(candidate, probe_root,
            initial_changes=probe['initial_changes'], verify_route_probes=False, deterministic_only=True)})
    summary = {'scope': 'all_routes_deterministic_diagnostics_only', 'training_ready': False,
               'model_calls': 0, 'routes': reports}
    from .dynamic_choice_coverage import coverage, REPAIR_HINT
    choices = coverage(artifacts, [item for report in reports
                       for item in report.get('dynamic_choice_observations', [])])
    # A reward-dependent trajectory may not have been executable without a
    # judge. Missing evidence there is pending, not a proven coverage defect.
    choices['deferred'] = bool(choices['missing'] and any(r.get('deferred') for r in reports))
    summary['dynamic_choice_coverage'] = choices
    (root / 'deterministic_preflight_all_routes.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    findings = [{'route_id': r['route_id'], **f} for r in reports for f in r['findings']]
    if not choices['passed'] and not choices['deferred']:
        findings.append({'code': 'DYNAMIC_CHOICE_UNCOVERED', 'missing': choices['missing'],
                         'message': REPAIR_HINT})
        from .execution_progress import ExecutionProgress
        ExecutionProgress(root).feedback('dynamic_choice_coverage', code='DYNAMIC_CHOICE_UNCOVERED')
    if findings:
        raise ValueError('DETERMINISTIC_PREFLIGHT_FAILED: ' + json.dumps(findings, ensure_ascii=False))
    return summary


def run_scenarios(scenarios, metrics, runner, call, proof, diagnostic, progress, root, deferred, *, observe=None):
    findings, completed, pending = [], [], []
    choice_observations = []
    deterministic = {m['id'] for m in metrics if m.get('type') == 'rule-based'}
    all_known = deterministic == {m['id'] for m in metrics}
    for scenario in scenarios:
        identity = scenario.get('scenario_id')
        trial = copy.deepcopy(scenario)
        rewards = {f"step:{s['step_id']}" for s in trial['steps']
                   if s.get('operation') == 'reward' and s.get('step_id')}
        last_reward = trial['steps'][-1].get('operation') == 'reward'
        if not all_known:
            # Reward captures can feed tools or dialogue. Never substitute an
            # optimistic bound into downstream business behavior.
            if any(s.get('capture') for s in trial['steps'] if s.get('operation') == 'reward'):
                pending.append({'scenario_id': identity, 'reason': 'reward-dependent trajectory'})
                continue
            trial['assertions'] = [a for a in trial.get('assertions', [])
                if a.get('source') not in rewards and not (last_reward
                    and a.get('source') != 'variables' and not str(a.get('source')).startswith('step:'))]
        call('POST', '/v1/reset', {}, {})
        try:
            with progress.measure('deterministic_scenario', scenario_id=identity):
                runner.run(trial)
                bound = call('GET', '/v1/reward', {}, {})[1]['reward']
                if scenario['kind'] == 'goal_success' and bound < 1 - 1e-9:
                    raise SandboxError('DETERMINISTIC_REWARD_UNREACHABLE',
                        'Reference cannot reach full reward even with full semantic credit', 500,
                        copy.deepcopy(proof))
            completed.append(identity)
            if observe is not None and scenario['kind'] == 'goal_success':
                choice_observations.extend({**item, 'scenario_id': identity} for item in observe())
        except deferred as exc:
            pending.append({'scenario_id': identity, 'reason': str(exc)})
            if observe is not None and scenario['kind'] == 'goal_success' and last_reward:
                choice_observations.extend({**item, 'scenario_id': identity} for item in observe())
        except SandboxError as exc:
            finding = {'scenario_id': identity, 'code': exc.code, 'message': str(exc),
                       'details': exc.details}
            try:
                finding['diagnostic'] = diagnostic()
            except (SandboxError, ValueError, TypeError, KeyError) as error:
                finding['diagnostic_error'] = str(error)
            findings.append(report_value(finding))
            progress.feedback('deterministic_scenario', scenario_id=identity, code=exc.code)
    report = {'scope': 'deterministic_diagnostics_only', 'training_ready': False,
              'model_calls': 0, 'findings': findings, 'completed': completed, 'deferred': pending,
              'dynamic_choice_observations': choice_observations,
              'pending': ['full_execution', 'semantic_calibration', 'delivery', 'independent_review']}
    (root / 'deterministic_preflight.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if findings:
        (root / 'preflight_failure.json').write_text(json.dumps(findings[0], ensure_ascii=False, indent=2) + '\n')
    return report
