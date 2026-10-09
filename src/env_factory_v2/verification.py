"""Joint user/environment execution and evidence-based executable qualification."""
from __future__ import annotations

import copy
import hashlib

from .contract import compile_contract, require
from .expressions import ContractError, canonical, evaluate
from .reward import grade
from .runtime import Sandbox


def replay_identity(run):
    value = copy.deepcopy(run)
    for turn in value['trace']:
        if turn.get('status') == 'ok' and 'next_cursor' in turn['result']:
            if turn['result']['next_cursor'] is not None:
                turn['result']['next_cursor'] = '<opaque cursor>'
    for evidence in value['outcome'].get('semantic_evidence', {}).values():
        evidence.pop('provenance', None)
    return canonical(value)


def run_scenario(compiled, scenario, *, judge=None, recipes_only=False):
    sandbox = Sandbox(compiled)
    sandbox.reset(scenario.get('mutations', []))
    error = None
    try:
        for step in scenario['steps']:
            kind = step['kind']
            if kind in {'call', 'traverse'}:
                function = sandbox.call if kind == 'call' else sandbox.traverse
                result = function(step['tool'], step['arguments'], call_id=step['id'],
                                  **({'cursor': sandbox.resolve(step['cursor'])} if 'cursor' in step else {}))
            elif kind == 'user':
                result = sandbox.interact(step['stage'], question=step['question'],
                    proposal=step.get('proposal'), source_id=step.get('source_id'))
            elif kind == 'wait':
                result = sandbox.wait(step['seconds'])
            elif kind == 'revoke':
                result = sandbox.revoke(step['stage'])
            else:
                raise ContractError('unsupported scenario action')
        # Scenario answer recipes are test fixtures, never runtime reward targets.
        answer = {key: evaluate(expression, sandbox.context(private=True))
                  for key, expression in scenario['answer'].items()}
        outcome = {'pending_grade': True} if recipes_only else grade(sandbox, answer, judge=judge)
    except ContractError as exc:
        error = str(exc)
        answer = None
        outcome = {'reward': 0.0, 'success': False}
    return {'scenario': scenario['id'], 'answer': answer, 'outcome': outcome,
            'error': error, 'trace': sandbox.trace, 'final_state': sandbox.data,
            'time': sandbox.time()}


def preflight_recipes(compiled):
    """Cheap real trajectories and answer rendering; never fake semantic grades."""
    checks = []
    for scenario in compiled['source']['scenarios']:
        run = run_scenario(compiled, scenario, recipes_only=True)
        expected_error = scenario.get('expect_error')
        passed = (run['error'] is not None and expected_error in run['error']) if expected_error else run['error'] is None
        checks.append({'id': scenario['id'], 'passed': passed, 'error': run['error']})
    return {'scope': 'trajectory_and_recipe_validation_no_semantic_judgment',
            'passed': all(check['passed'] for check in checks), 'checks': checks}


def expected_result(scenario, run):
    if scenario['expect_success']:
        return run['error'] is None and run['outcome']['success'] is True and abs(run['outcome']['reward'] - 1) < 1e-9
    if scenario.get('expect_error'):
        return run['error'] is not None and scenario['expect_error'] in run['error']
    # An invalid negative fixture is not evidence of reward discrimination.
    return run['error'] is None and run['outcome']['success'] is False and run['outcome']['reward'] < 1


def qualify(source, *, judge=None, independent_judge=None, independent_review=None):
    compiled = compile_contract(source)
    records = []
    for scenario in compiled['source']['scenarios']:
        first = run_scenario(compiled, scenario, judge=judge)
        repeat = run_scenario(compiled, scenario, judge=independent_judge or judge)
        records.append({'id': scenario['id'], 'kind': 'scenario',
                        'passed': expected_result(scenario, first) and expected_result(scenario, repeat)
                            and replay_identity(first) == replay_identity(repeat),
                        'first': first, 'repeat': repeat})
    scenarios = {s['id']: s for s in compiled['source']['scenarios']}
    for probe in compiled['source']['probes']:
        scenario = copy.deepcopy(scenarios[probe['scenario']])
        scenario.update(id=probe['id'], mutations=scenario.get('mutations', []) + probe['mutations'],
                        expect_success=probe['expect_success'])
        if 'answer' in probe:
            scenario['answer'] = probe['answer']
        if 'expect_error' in probe:
            scenario['expect_error'] = probe['expect_error']
        first = run_scenario(compiled, scenario, judge=judge)
        repeat = run_scenario(compiled, scenario, judge=independent_judge or judge)
        records.append({'id': probe['id'], 'kind': 'probe', 'rule_ids': probe['rule_ids'],
                        'passed': expected_result(scenario, first) and expected_result(scenario, repeat)
                            and replay_identity(first) == replay_identity(repeat), 'first': first, 'repeat': repeat})
    execution_passed = all(record['passed'] for record in records)
    review_passed = (independent_review is not None and independent_review.get('passed') is True
                    and independent_review.get('contract_identity') == compiled['identity']
                    and independent_review.get('scope') == 'native_independent_contract_challenge')
    gates = {'structural_contract': True, 'joint_execution_and_counterexamples': execution_passed,
             'independent_business_challenge': review_passed,
             'deterministic_reset_and_replay': all(replay_identity(r['first']) == replay_identity(r['repeat']) for r in records)}
    report = {'version': '2.0', 'basis': 'contract_execution_v2', 'contract_identity': compiled['identity'],
              'passed': all(gates.values()), 'gates': gates, 'records': records,
              'independent_review': independent_review,
              'scope': 'executable_contract_not_learning_effectiveness', 'container_verified': False}
    report['evidence_identity'] = hashlib.sha256(canonical(report).encode()).hexdigest()
    return compiled, report
