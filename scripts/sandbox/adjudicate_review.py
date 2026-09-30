#!/usr/bin/env python3
"""Replay declarative review counterexamples before allowing automatic repair.

No reviewer-supplied code or shell command is executed. Unsupported semantic
claims stay unresolved and never become automatic code-repair instructions.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

from env_factory.sandbox_scoring import review_source_hashes

REQUIRED_MODULES = {'business_tools', 'reward', 'user_simulator', 'runtime_contract'}


def adjudicate(root: Path, report: dict) -> dict:
    findings = report.get('findings', [])
    verdicts = []
    for index, finding in enumerate(findings):
        if finding.get('severity') not in {'high', 'critical'}:
            continue
        verdict = {'finding_index': index, 'status': 'unresolved', 'evidence': 'No supported executable reproduction.'}
        probe = finding.get('reproduction')
        if isinstance(probe, dict) and probe.get('kind') in {'invalid_goal_reward', 'no_tool_reward'}:
            try:
                verdict.update(replay_probe(root, probe))
            except Exception as exc:
                verdict['evidence'] = f'Reproduction unavailable: {type(exc).__name__}: {exc}'
        verdicts.append(verdict)
    checked = set(report.get('checked_modules', []))
    unresolved = any(v['status'] == 'unresolved' for v in verdicts)
    confirmed = any(v['status'] == 'confirmed' for v in verdicts)
    coverage = REQUIRED_MODULES <= checked
    # A bare fail or low score without a reproducible finding is not a repair.
    unsupported_failure = (report.get('status') == 'fail' and not verdicts) or not isinstance(report.get('score'), (int, float))
    status = 'confirmed_defect' if confirmed else ('review_unresolved' if unresolved or not coverage or unsupported_failure else 'resolved')
    return {'version': '1.0', 'status': status, 'source_hashes': review_source_hashes(root),
            'review_run_id': report.get('review_run_id'), 'coverage_complete': coverage, 'findings': verdicts}


def replay_probe(root: Path, probe: dict) -> dict:
    from env_factory.sandbox_runtime import AcceptanceScenarioRunner, BusinessGoalEvaluator
    steps = probe.get('steps')
    allowed = {'reset', 'tool_call', 'agent_response', 'reward', 'business_snapshot', 'mutate_business_state', 'replay', 'observation'}
    if (not isinstance(steps, list) or not 1 <= len(steps) <= 30
            or any(not isinstance(s, dict) or s.get('operation') not in allowed for s in steps)):
        raise ValueError('Only 1..30 declarative scenario steps are supported')
    task = json.loads((root / 'task.json').read_text())
    if any(m.get('type') != 'rule-based' for m in task.get('metrics', [])):
        raise ValueError('Model-judged claims require independent live evidence')
    os.environ['SANDBOX_EVALUATOR_MOCK'] = '1'
    os.environ['SANDBOX_MUTATION_MODE'] = 'disabled'
    os.environ['SANDBOX_TRAINER_API_KEY'] = 'review-reproduction-local'
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location('review_probe_app', root / 'app.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="envfactory-review-probe-") as temporary:
        app = module.create_app(db_path=Path(temporary) / "episode.sqlite3")
        headers = {'Authorization': 'Bearer review-reproduction-local'}
        runner = AcceptanceScenarioRunner(app.handle, trainer_headers=headers,
                                         business_snapshot=app.business_snapshot,
                                         mutate_business_state=app.mutate_business_state)
        app.handle('POST', '/v1/reset', {'episode_id': 'review-probe', 'seed': 1701}, headers)
        baseline = copy.deepcopy(app.business_snapshot())
        if any(s['operation'] == 'reset' for s in steps):
            raise ValueError('Reset is owned by the adjudicator')
        run = runner.run({'steps': steps, 'assertions': []})
        state = app.business_snapshot()
        goal = task.get('task_spec', {}).get('goal_contract', {})
        if probe['kind'] == 'invalid_goal_reward':
            if not goal.get('row_predicates'):
                raise ValueError('No executable state goal for this claim')
            invalid = not BusinessGoalEvaluator.evaluate(goal['row_predicates'], state, baseline) or not BusinessGoalEvaluator.preserves_unrelated(goal, baseline, state)
        else:
            if task.get('training_contract', {}).get('category', task.get('training_category')) == 'direct_response':
                raise ValueError('No-tool execution is valid for direct_response')
            invalid = not any(s['operation'] == 'tool_call' for s in steps)
        if not invalid:
            raise ValueError('Counterexample does not violate the named contract obligation')
        status, reward, _ = app.handle('GET', '/v1/reward', headers=headers)
        if status != 200 or not isinstance(reward.get('reward'), (int, float)):
            raise ValueError('Reward execution unavailable')
        value = reward['reward']
        return {'status': 'confirmed' if value > 0.2 else 'refuted',
                'evidence': {'invariant_violated': True, 'reward': value, 'maximum': 0.2, 'steps_executed': len(run['history'])}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    path = root / 'review_report.json'
    report = json.loads(path.read_text())
    result = adjudicate(root, report)
    report['adjudication'] = result
    if result['status'] == 'resolved':
        for verdict in result['findings']:
            if verdict['status'] == 'refuted':
                report['findings'][verdict['finding_index']]['severity'] = 'low'
                report['findings'][verdict['finding_index']]['adjudication_status'] = 'refuted'
        if report.get('status') == 'fail':
            # Refuting a counterexample does not establish the whole sandbox's
            # semantic quality. Require a new review, never invent a pass score.
            result['status'] = 'review_unresolved'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'findings': result['findings']}, ensure_ascii=False))
    return 76 if result['status'] == 'review_unresolved' else 0


if __name__ == '__main__':
    raise SystemExit(main())
