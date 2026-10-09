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


def challenge(compiled, original_request, agent):
    from .contract import ContractError
    try:
        return _challenge(compiled, original_request, agent)
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
    if not value['business_consistent'] and all(record['passed'] for record in records):
        raise RuntimeError('independent review is inconclusive: no executed counterexample supports its rejection')
    return {'scope': 'native_independent_contract_challenge', 'contract_identity': compiled['identity'],
            'passed': value['business_consistent'] and all(r['passed'] for r in records),
            'business_consistent': value['business_consistent'], 'reason': value['reason'],
            'provenance': provenance, 'records': records}
