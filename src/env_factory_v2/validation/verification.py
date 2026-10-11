"""Joint user/environment execution and evidence-based executable qualification."""
from __future__ import annotations

import copy
import hashlib

from env_factory_v2.contracts.contract import compile_contract, require, validate_user_overrides
from env_factory_v2.contracts.expressions import ContractError, canonical, evaluate
from env_factory_v2.execution.reward import assess_process, assess_answer_fields, grade
from env_factory_v2.execution.runtime import Sandbox


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
    overrides = scenario.get('user_overrides', {})
    validate_user_overrides(sandbox.source, overrides)
    sandbox.user.facts.update(copy.deepcopy(overrides))
    error = None
    failure_integrity = None
    try:
        for step in scenario['steps']:
            kind = step['kind']
            if kind in {'call', 'traverse'}:
                function = sandbox.call if kind == 'call' else sandbox.traverse
                expected_failure_state = None
                if kind == 'call' and sandbox.tools[step['tool']]['kind'] == 'transaction':
                    # Time-triggered events legitimately precede an attempted
                    # transaction. Compare against that state, not episode reset
                    # or the state before earlier successful transactions.
                    expected_failure_state = copy.deepcopy(sandbox)
                    try:
                        expected_failure_state._advance()
                    except ContractError:
                        pass
                try:
                    result = function(step['tool'], step['arguments'], call_id=step['id'],
                                      **({'cursor': sandbox.resolve(step['cursor'])} if 'cursor' in step else {}))
                except ContractError:
                    if expected_failure_state is not None:
                        failure_integrity = {
                            'business_state_preserved': canonical(sandbox.data) == canonical(expected_failure_state.data),
                            'authorization_state_preserved': canonical(vars(sandbox.user)) == canonical(vars(expected_failure_state.user)),
                            'state_version_preserved': sandbox.version == expected_failure_state.version,
                        }
                        failure_integrity['passed'] = all(failure_integrity.values())
                    raise
            elif kind == 'user':
                result = sandbox.interact(step['stage'], question=step['question'],
                    proposal=step.get('proposal'), source_id=step.get('source_id'))
            elif kind == 'wait':
                result = sandbox.wait(step['seconds'])
            elif kind == 'time':
                result = sandbox.observe_time()
            elif kind == 'revoke':
                result = sandbox.revoke(step['stage'])
            else:
                raise ContractError('unsupported scenario action')
        # Scenario answer recipes are test fixtures, never runtime reward targets.
        answer = {key: evaluate(expression, sandbox.context(private=True))
                  for key, expression in scenario['answer'].items()}
        if recipes_only:
            outcome = {'pending_grade': True}
            if not sandbox.source.get('explanations'):
                outcome['deterministic_grade'] = grade(sandbox, answer)
            elif scenario['expect_success']:
                outcome['workflow'] = assess_process(sandbox)
                outcome['structured_components'] = assess_answer_fields(sandbox, answer)[0]
        else:
            outcome = grade(sandbox, answer, judge=judge)
    except ContractError as exc:
        error = str(exc)
        answer = None
        outcome = {'reward': 0.0, 'success': False}
    return {'scenario': scenario['id'], 'answer': answer, 'outcome': outcome,
            'error': error, 'trace': sandbox.trace, 'final_state': sandbox.data,
            'time': sandbox.time(),
            **({'failure_integrity': failure_integrity} if failure_integrity is not None else {})}


def authored_cases(compiled):
    """Materialize the same authored cases for preflight and qualification."""
    scenarios = {s['id']: s for s in compiled['source']['scenarios']}
    for scenario in scenarios.values():
        yield 'scenario', scenario, None
    for probe in compiled['source']['probes']:
        scenario = copy.deepcopy(scenarios[probe['scenario']])
        scenario.update(id=probe['id'], mutations=scenario.get('mutations', []) + probe['mutations'],
                        expect_success=probe['expect_success'])
        scenario['user_overrides'] = {**scenario.get('user_overrides', {}), **probe.get('user_overrides', {})}
        for key in ('answer', 'expect_error'):
            if key in probe:
                scenario[key] = probe[key]
        yield 'probe', scenario, probe['rule_ids']


def preflight_recipes(compiled):
    """Cheap real trajectories and answer rendering; never fake semantic grades."""
    checks = []
    for kind, scenario, _ in authored_cases(compiled):
        run = run_scenario(compiled, scenario, recipes_only=True)
        expected_error = scenario.get('expect_error')
        passed = (run['error'] is not None and expected_error in run['error']) if expected_error else run['error'] is None
        passed = passed and run.get('failure_integrity', {}).get('passed', True)
        workflow = run['outcome'].get('workflow')
        deterministic = run['outcome'].get('deterministic_grade')
        if deterministic is not None:
            passed = expected_result(scenario, {**run, 'outcome': deterministic})
        elif scenario['expect_success'] and run['error'] is None:
            passed = passed and workflow['passed'] and all(
                item['passed'] for item in run['outcome']['structured_components'].values())
        checks.append({'id': scenario['id'], 'kind': kind, 'passed': passed, 'error': run['error'],
                       **({'failure_integrity': run['failure_integrity']} if 'failure_integrity' in run else {}),
                       **({'failure_evidence': {
                           'expected_success': scenario['expect_success'],
                           'expected_error': expected_error,
                           'user_overrides': copy.deepcopy(scenario.get('user_overrides', {})),
                           'executed_tools': [{key: turn.get(key) for key in
                               ('tool', 'arguments', 'status', 'error')} for turn in run['trace']
                               if turn.get('kind') == 'tool'],
                           'answer': run['answer']}} if not passed else {}),
                       **({'workflow': workflow} if workflow is not None else {}),
                       **({'structured_components': run['outcome']['structured_components']}
                          if 'structured_components' in run['outcome'] else {}),
                       **({'deterministic_grade': deterministic} if deterministic is not None else {})})
    return {'scope': 'trajectory_and_recipe_validation_no_semantic_judgment',
            'passed': all(check['passed'] for check in checks), 'checks': checks}


def expected_result(scenario, run):
    if run.get('failure_integrity', {}).get('passed') is False:
        return False
    if scenario['expect_success']:
        return run['error'] is None and run['outcome']['success'] is True
    if scenario.get('expect_error'):
        return run['error'] is not None and scenario['expect_error'] in run['error']
    # An invalid negative fixture is not evidence of reward discrimination.
    return run['error'] is None and run['outcome']['success'] is False and run['outcome']['reward'] < 1


def compact_evidence(value):
    """Retain verdicts and reference answers without duplicating worlds/traces."""
    if isinstance(value, list):
        return [compact_evidence(item) for item in value]
    if isinstance(value, dict):
        is_run = {'outcome', 'trace', 'final_state', 'time'} <= value.keys()
        return {key: compact_evidence(item) for key, item in value.items()
                if not (is_run and key in {'trace', 'final_state'})}
    return value


def qualify(source, *, judge=None, independent_judge=None, independent_review=None, compact=False):
    compiled = compile_contract(source)
    independent_evidence = {}
    def review_once(payload):
        # This cache lives only inside one qualification with a fixed compiled
        # contract and judge. Every scenario still executes and replays; only
        # identical semantic evidence reuses its independent review.
        identity = canonical(payload)
        if identity not in independent_evidence:
            independent_evidence[identity] = copy.deepcopy(independent_judge(payload))
            return copy.deepcopy(independent_evidence[identity])
        result = copy.deepcopy(independent_evidence[identity])
        result['provenance'].update(reused=True, reuse_scope='qualification_independent_evidence')
        return result
    records = []
    for kind, scenario, rule_ids in authored_cases(compiled):
        first = run_scenario(compiled, scenario, judge=judge)
        repeat = run_scenario(compiled, scenario, judge=review_once if independent_judge else judge)
        replay_passed = replay_identity(first) == replay_identity(repeat)
        expectations_passed = expected_result(scenario, first) and expected_result(scenario, repeat)
        records.append({'id': scenario['id'], 'kind': kind, 'replay_passed': replay_passed,
                        **({'rule_ids': rule_ids} if rule_ids is not None else {}),
                        'expectations_passed': expectations_passed,
                        'passed': expectations_passed and replay_passed,
                        'first': compact_evidence(first) if compact else first,
                        'repeat': compact_evidence(repeat) if compact else repeat})
    execution_passed = all(record['expectations_passed'] for record in records)
    review_passed = (independent_review is not None and independent_review.get('passed') is True
                    and independent_review.get('contract_identity') == compiled['identity']
                    and independent_review.get('scope') == 'native_independent_contract_challenge')
    gates = {'joint_execution_and_counterexamples': execution_passed,
             'independent_business_challenge': review_passed,
             'deterministic_reset_and_replay': all(r['replay_passed'] for r in records)}
    report = {'version': '2.0', 'basis': 'contract_execution_v2', 'contract_identity': compiled['identity'],
              'passed': all(gates.values()), 'gates': gates, 'records': records,
              'independent_review': compact_evidence(independent_review) if compact else independent_review,
              'scope': 'executable_contract_not_learning_effectiveness', 'container_verified': False}
    report['evidence_detail'] = 'verdicts_and_answers' if compact else 'full_execution'
    report['semantic_replay'] = ('explicit_independent_judge' if independent_judge
                                 else 'same_judge_exact_evidence_reuse_when_supported')
    report['evidence_identity'] = hashlib.sha256(canonical(report).encode()).hexdigest()
    return compiled, report


def attach_independent_review(compiled, execution, review):
    """Bind a later review to unchanged execution evidence without rerunning it."""
    from env_factory_v2.contracts.contract import require
    report = copy.deepcopy(execution)
    identity = report.pop('evidence_identity', None)
    require(identity == hashlib.sha256(canonical(report).encode()).hexdigest(),
            'execution evidence changed before independent review')
    require(report['contract_identity'] == compiled['identity'], 'execution contract identity mismatch')
    require(review.get('contract_identity') == compiled['identity']
            and review.get('scope') == 'native_independent_contract_challenge',
            'independent review does not match executed contract')
    report['independent_review'] = (compact_evidence(review)
        if report['evidence_detail'] == 'verdicts_and_answers' else copy.deepcopy(review))
    report['gates']['independent_business_challenge'] = review.get('passed') is True
    report['passed'] = all(report['gates'].values())
    report['evidence_identity'] = hashlib.sha256(canonical(report).encode()).hexdigest()
    return report
