"""Independent business interpretation, grounded in executable counterexamples."""
from __future__ import annotations

import copy

from .code_agent import object_schema
from .contract import require
from .verification import expected_result, replay_identity, run_scenario


CASE_SCHEMA = object_schema({
    'id': {'type': 'string'}, 'rule_ids': {'type': 'array', 'items': {'type': 'string'}},
    'scenario_id': {'type': 'string'}, 'reason': {'type': 'string'},
    'mutations_json': {'type': 'string'}, 'answer_json': {'type': 'string'},
    'expect_success': {'type': 'boolean'},
})
CHALLENGE_SCHEMA = object_schema({'business_consistent': {'type': 'boolean'},
    'reason': {'type': 'string'}, 'cases': {'type': 'array', 'items': CASE_SCHEMA}})
AUDIT_SCHEMA = object_schema({'business_consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
    'cases': {'type': 'array', 'items': object_schema({'id': {'type': 'string'},
              'expect_success': {'type': 'boolean'}, 'reason': {'type': 'string'}})}})


def challenge(compiled, original_request, agent):
    from .contract import ContractError
    try:
        report = _challenge(compiled, original_request, agent)
        all_passed = all(record['passed'] for record in report['records'])
        if report['business_consistent'] != all_passed:
            report = _audit(compiled, original_request, agent, report)
        if not report['business_consistent'] and all(record['passed'] for record in report['records']):
            raise RuntimeError('independent review is inconclusive: no executed counterexample supports its rejection')
        return report
    except (ContractError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError('independent challenge invalid; do not repair source from an ungrounded claim: ' + str(exc)) from exc


def _challenge(compiled, original_request, agent):
    import json
    from .fixtures import review_contract
    review_source, fixture_evidence = review_contract(compiled['source'])
    value, provenance = agent.ask(
        'Independently compare the ORIGINAL request with this proposed business contract. compiled_public '
        'is the actual agent-visible interface; authored tool descriptions are not independently published. Do not trust its '
        'reference answers, tool predicates or reward targets as business truth. Propose concrete executable '
        'positive and negative counterexamples. Cover every public rule, extra identity/display-name filters, '
        'mutable selected-record facts, ordering, absence, actual menu choices, authorization and transactions '
        'including private-information leakage and unnecessary user questions '
        'where applicable. Stay within declared data schemas, foreign keys and actual user stages; no invented '
        'user values. Each case reuses an actual scenario, changes initial records by primary key, and supplies '
        'an independently correct answer recipe (the expression grammar is provided). answer_json and '
        'mutations_json are JSON strings; use {} to retain the original answer recipe. The verifier owns '
        'intentional-invalid-action expectations inherited from each referenced scenario; never invent runtime '
        'error strings. Include at least one positive and one negative. '
        'For stateless direct tasks, empty data mutations are appropriate. Validate boundary predicates '
        'by invoking the actual named public rule with boundary arguments in a conditional answer recipe: '
        'return the independently correct original-task answer only when the boundary predicate has its '
        'business-correct value, otherwise a wrong schema-valid answer. This is evaluator validation; '
        'do not invent an extra agent explanation obligation. Assess evaluator coverage across the authored '
        'probes AND your executable cases; a boundary covered by your own case does not require an extra '
        'agent answer field or a duplicate authored probe. '
        'For generated tables the exact generator plan is authoritative; previews are concrete examples, not '
        'the whole table. Derive unshown values from the documented plan, never invent hidden rows. '
        'Unexecuted model opinions are not proof. Do not execute tools.',
        {'original_request': original_request, 'contract': review_source,
         'compiled_public': compiled['public'], 'fixture_evidence': fixture_evidence,
         'grammar': (__import__('pathlib').Path(__file__).parent / 'CONTRACT.md').read_text()}, CHALLENGE_SCHEMA)
    require(2 <= len(value['cases']) <= 32, 'independent challenge requires 2..32 concrete cases')
    require(any(c['expect_success'] for c in value['cases']) and any(not c['expect_success'] for c in value['cases']),
            'challenge needs positive and negative behavior')
    rules = {r['id'] for r in compiled['source']['rules']}
    covered = {rule for case in value['cases'] for rule in case['rule_ids']}
    require(covered == rules, 'independent challenges must cover actual public rules')
    scenarios = {s['id']: s for s in compiled['source']['scenarios']}
    records = []
    identities = set()
    for case in value['cases']:
        require(case['id'] and case['id'] not in identities and case['scenario_id'] in scenarios, 'invalid challenge identity')
        identities.add(case['id'])
        scenario = copy.deepcopy(scenarios[case['scenario_id']])
        mutations, answer = json.loads(case['mutations_json']), json.loads(case['answer_json'])
        require(isinstance(mutations, list) and isinstance(answer, dict), 'invalid challenge recipes')
        scenario.update(id=case['id'], mutations=scenario.get('mutations', []) + mutations,
                        expect_success=case['expect_success'])
        # A reviewer cannot turn a correctly rejected action into an author defect
        # by guessing a different runtime error substring.
        if answer:
            scenario['answer'] = answer
        first = run_scenario(compiled, scenario, judge=agent.judge)
        repeat = run_scenario(compiled, scenario, judge=agent.independent_judge)
        passed = expected_result(scenario, first) and expected_result(scenario, repeat)
        records.append({'case': case, 'passed': passed and replay_identity(first) == replay_identity(repeat),
                        'first': first, 'repeat': repeat})
    return {'scope': 'native_independent_contract_challenge', 'contract_identity': compiled['identity'],
            'passed': value['business_consistent'] and all(r['passed'] for r in records),
            'business_consistent': value['business_consistent'], 'reason': value['reason'],
            'provenance': provenance, 'records': records}


def _audit(compiled, original_request, agent, report):
    """Resolve contradictory review labels without editing source or discarding cases."""
    from .fixtures import review_contract
    from .expressions import canonical
    source, fixtures = review_contract(compiled['source'])
    observations = []
    for record in report['records']:
        run = record['first']
        answer = run['answer']
        observations.append({'case': record['case'], 'matched_expectation': record['passed'],
                             'actual_error': run['error'], 'actual_outcome': run['outcome'],
                             'actual_answer': answer if len(canonical(answer)) <= 8000 else
                             {'omitted': 'large answer; derive exact values from generator and case recipe'}})
    value, provenance = agent.ask(
        'Independently audit a contradictory business review against the ORIGINAL request. '
        'The prior review conclusion and executed cases disagree. For EVERY unchanged case determine '
        'whether its actual proposed behavior/answer SHOULD succeed according to the original business. '
        'A deliberately wrong answer must have expect_success=false even if a prior label says true. '
        'Do not change expectations merely to match the observed grader; the grader may itself be wrong. '
        'Distinguish evaluator validation obligations from agent answer obligations; require additional '
        'answer prose only where the original business asks for it. Keep every case ID, mutation, recipe '
        'and trajectory unchanged. Explain each expectation from original business facts. No dropped cases, '
        'invented data, weaker rules or tools. business_consistent describes the contract, not the reviewer. '
        'If a business omission remains, keep business_consistent=false even if cases cannot expose it. '
        'Do not execute tools.',
        {'original_request': original_request, 'contract': source, 'compiled_public': compiled['public'],
         'fixture_evidence': fixtures, 'prior_review': {'business_consistent': report['business_consistent'],
         'reason': report['reason']}, 'executed_cases': observations}, AUDIT_SCHEMA)
    labels = {case['id']: case for case in value['cases']}
    require(len(labels) == len(value['cases']) and set(labels) == {r['case']['id'] for r in report['records']},
            'audit must preserve every independent case exactly once')
    require(value['reason'].strip() and all(case['reason'].strip() for case in value['cases']), 'audit reasons required')
    require(any(case['expect_success'] for case in value['cases'])
            and any(not case['expect_success'] for case in value['cases']), 'audit must preserve positive and negative coverage')
    scenarios = {s['id']: s for s in compiled['source']['scenarios']}
    records = []
    for old in report['records']:
        case = copy.deepcopy(old['case'])
        label = labels[case['id']]
        case['expect_success'], case['reason'] = label['expect_success'], label['reason']
        # Same executed evidence, independently corrected business labels. No
        # source, mutations, answers or semantic judgments are regenerated here.
        scenario = {**scenarios[case['scenario_id']], 'expect_success': case['expect_success']}
        records.append({**old, 'case': case, 'passed': expected_result(scenario, old['first'])
                        and expected_result(scenario, old['repeat'])
                        and replay_identity(old['first']) == replay_identity(old['repeat'])})
    return {**report, 'passed': value['business_consistent'] and all(r['passed'] for r in records),
            'business_consistent': value['business_consistent'], 'reason': value['reason'], 'records': records,
            'review_audit': {'provenance': provenance, 'result': value, 'original_review': report}}
