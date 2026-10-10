"""Independent business interpretation, grounded in executable counterexamples."""
from __future__ import annotations

import copy
import json

from .code_agent import object_schema
from .contract import require
from .json_syntax import content_tokens as _json_content_tokens
from .verification import expected_result, replay_identity, run_scenario


CASE_SCHEMA = object_schema({
    'id': {'type': 'string'}, 'rule_ids': {'type': 'array', 'items': {'type': 'string'}},
    'scenario_id': {'type': 'string'}, 'reason': {'type': 'string'},
    'mutations_json': {'type': 'string'}, 'answer_json': {'type': 'string'},
    'user_overrides_json': {'type': 'string'},
    'expect_success': {'type': 'boolean'},
})
CHALLENGE_SCHEMA = object_schema({'business_consistent': {'type': 'boolean'},
    'reason': {'type': 'string'}, 'cases': {'type': 'array', 'items': CASE_SCHEMA}})
AUDIT_SCHEMA = object_schema({'business_consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
    'cases': {'type': 'array', 'items': object_schema({'id': {'type': 'string'},
              'expect_success': {'type': 'boolean'}, 'reason': {'type': 'string'}})}})


def _repair_recipe_json(value, agent):
    broken = []
    for index, case in enumerate(value['cases']):
        for field in ('mutations_json', 'answer_json', 'user_overrides_json'):
            try:
                json.loads(case.get(field, '{}'))
            except json.JSONDecodeError as exc:
                _json_content_tokens(case[field])
                broken.append({'index': index, 'field': field, 'json': case[field], 'error': str(exc)})
    if not broken:
        return value, None
    schema = object_schema({'repairs': {'type': 'array', 'items': object_schema({
        'index': {'type': 'integer'}, 'field': {'type': 'string'}, 'json': {'type': 'string'}})}})
    repaired, provenance = agent.ask(
        'Repair JSON delimiters ONLY in these malformed independent-review recipes. Preserve every key, '
        'string, number, boolean and null token exactly and in the same order. Preserve business intent '
        'and expression nesting. Return each requested index/field exactly once. Do not change labels, '
        'cases, source, values or rules. Do not execute tools.', {'broken_recipes': broken}, schema)
    expected = {(item['index'], item['field']): item for item in broken}
    result = copy.deepcopy(value)
    seen = set()
    for item in repaired['repairs']:
        key = (item['index'], item['field'])
        require(key in expected and key not in seen, 'JSON repair must preserve requested fields exactly once')
        seen.add(key)
        require(_json_content_tokens(item['json']) == _json_content_tokens(expected[key]['json']),
                'JSON repair changed business content')
        json.loads(item['json'])
        result['cases'][item['index']][item['field']] = item['json']
    require(seen == set(expected), 'JSON repair omitted recipe')
    return result, {'provenance': provenance, 'original_review': value, 'repairs': repaired}


def challenge(compiled, original_request, agent):
    from .contract import ContractError
    try:
        report = _challenge(compiled, original_request, agent)
        all_passed = all(record['passed'] for record in report['records'])
        if report['business_consistent'] != all_passed:
            report = _audit(compiled, original_request, agent, report)
        if not report['business_consistent'] and all(record['passed'] for record in report['records']):
            # A structurally unsolvable answer or absent user branch need not be
            # expressible by initial-record mutations. Preserve uncertainty and
            # the failed gate; a candidate revision still needs a fresh review.
            return {**report, 'passed': False, 'status': 'inconclusive',
                    'rejection_proven': False,
                    'instruction': 'Investigate these unresolved concerns against the original request. '
                    'They are not proven counterexamples. Preserve original obligations and valid tests; '
                    'any revised candidate must pass a fresh independent review and all execution gates.'}
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
        'user keys or incompatible fact types. user_overrides_json is a JSON object of existing private fact '
        'overrides for this qualification case only; use {} to retain inherited facts. You may vary approval '
        'booleans to exercise real refusal. Never change the base fixture or public request. Each case reuses an actual scenario, changes initial records by primary key, and supplies '
        'an independently correct answer recipe (the expression grammar is provided). answer_json and '
        'mutations_json and user_overrides_json are JSON strings; use {} to retain the original answer recipe. The verifier owns '
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
    value, completion = _complete_review(compiled, original_request, agent, value)
    require(2 <= len(value['cases']) <= 32, 'independent challenge requires 2..32 concrete cases')
    value, format_repair = _repair_recipe_json(value, agent)
    require(any(c['expect_success'] for c in value['cases']) and any(not c['expect_success'] for c in value['cases']),
            'challenge needs positive and negative behavior')
    rules = {r['id'] for r in compiled['source']['rules']}
    covered = {rule for case in value['cases'] for rule in case['rule_ids']}
    require(covered == rules, 'independent challenges must cover actual public rules')
    scenarios = {s['id']: s for s in compiled['source']['scenarios']}
    records = []
    identities = set()
    mutation_repair_budget = [1]
    for case in value['cases']:
        require(case['id'] and case['id'] not in identities and case['scenario_id'] in scenarios, 'invalid challenge identity')
        identities.add(case['id'])
        records.append(_execute_case(compiled, case, scenarios, agent, mutation_repair_budget))
    return {'scope': 'native_independent_contract_challenge', 'contract_identity': compiled['identity'],
            'passed': value['business_consistent'] and all(r['passed'] for r in records),
            'business_consistent': value['business_consistent'], 'reason': value['reason'],
            'provenance': provenance, 'records': records,
            **({'completeness_recovery': completion} if completion else {}),
            **({'format_repair': format_repair} if format_repair else {})}


def _complete_review(compiled, original_request, agent, value):
    """One append-only recovery for incomplete review, never a source repair."""
    rules = {r['id'] for r in compiled['source']['rules']}
    cases = value['cases']
    covered = {r for c in cases for r in c['rule_ids']}
    require(len(cases) <= 32 and covered <= rules, 'invalid challenge size or unknown public rules')
    complete = (len(cases) >= 2 and covered == rules
                and any(c['expect_success'] for c in cases)
                and any(not c['expect_success'] for c in cases))
    if complete:
        return value, None
    from .fixtures import review_contract
    source, fixtures = review_contract(compiled['source'])
    supplement, provenance = agent.ask(
        'Complete an incomplete independent business review by APPENDING concrete executable cases. '
        'Preserve every existing case, business_consistent verdict and reason; never edit the contract. '
        'The combined review must contain 2..32 cases, both positive and negative, and cover every '
        'actual public rule. Use unique new IDs and actual scenario IDs. Recipes follow the supplied '
        'grammar, mutations must reference actual records. Return only the additional cases. '
        'All appended cases will execute independently; do not invent runtime error expectations. '
        'Do not execute tools.',
        {'original_request': original_request, 'existing_review': value, 'contract': source,
         'fixture_evidence': fixtures, 'compiled_public': compiled['public'],
         'grammar': (__import__('pathlib').Path(__file__).parent / 'CONTRACT.md').read_text()},
        object_schema({'cases': {'type': 'array', 'items': CASE_SCHEMA}}))
    result = {**value, 'cases': copy.deepcopy(cases) + supplement['cases']}
    return result, {'provenance': provenance, 'original_review': value,
                    'appended_cases': supplement['cases'], 'budget': 1}


def _execute_case(compiled, case, scenarios, agent, mutation_repair_budget=None):
    from .expressions import ContractError
    scenario = copy.deepcopy(scenarios[case['scenario_id']])
    mutations, answer = json.loads(case['mutations_json']), json.loads(case['answer_json'])
    require(isinstance(mutations, list) and isinstance(answer, dict), 'invalid challenge recipes')
    scenario.update(id=case['id'], mutations=scenario.get('mutations', []) + mutations,
                    expect_success=case['expect_success'])
    overrides = json.loads(case.get('user_overrides_json', '{}'))
    from .contract import validate_user_overrides
    validate_user_overrides(compiled['source'], overrides)
    scenario['user_overrides'] = {**scenario.get('user_overrides', {}), **overrides}
    if answer:
        scenario['answer'] = answer
    mutation_repair = None
    try:
        first = run_scenario(compiled, scenario, judge=agent.judge)
    except ContractError as exc:
        if ('write/mutation target must identify exactly one row' not in str(exc)
                or mutation_repair_budget is None or mutation_repair_budget[0] == 0):
            raise
        mutation_repair_budget[0] -= 1
        from .fixtures import review_contract
        source, fixtures = review_contract(compiled['source'])
        value, provenance = agent.ask(
            'Repair ONLY the order of this independent case mutation list. Mutations execute sequentially; '
            'renaming a primary key changes how later mutations identify that row. Preserve every mutation '
            'exactly; return a permutation of 0-based indices. No additions, removals, edits, labels, answers '
            'or source changes. Preserve the intended business changes; order independent non-key changes '
            'before renaming their target. If reordering cannot resolve it, return the original order. '
            'Do not execute tools.',
            {'case': case, 'mutations': mutations, 'unchanged_scenario_mutations': scenario.get('mutations', [])[:-len(mutations)] if mutations else scenario.get('mutations', []),
             'error': str(exc), 'contract': source, 'fixture_evidence': fixtures},
            object_schema({'order': {'type': 'array', 'items': {'type': 'integer'}}}))
        order = value['order']
        require(len(order) == len(mutations) and all(type(index) is int for index in order)
                and set(order) == set(range(len(mutations))), 'mutation repair must preserve every mutation exactly once')
        reordered = [mutations[index] for index in order]
        original_case = copy.deepcopy(case)
        case = {**case, 'mutations_json': json.dumps(reordered, ensure_ascii=False)}
        scenario['mutations'] = copy.deepcopy(scenarios[case['scenario_id']].get('mutations', [])) + reordered
        first = run_scenario(compiled, scenario, judge=agent.judge)
        mutation_repair = {'provenance': provenance, 'order': order, 'original_case': original_case}
    repeat = run_scenario(compiled, scenario, judge=agent.independent_judge)
    passed = expected_result(scenario, first) and expected_result(scenario, repeat)
    return {'case': case, 'passed': passed and replay_identity(first) == replay_identity(repeat),
            'first': first, 'repeat': repeat,
            **({'mutation_order_repair': mutation_repair} if mutation_repair else {})}


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
    supplement = None
    missing = {True, False} - {record['case']['expect_success'] for record in records}
    if missing:
        extra, extra_provenance = agent.ask(
            'Append grounded independent counterexamples to restore missing positive/negative coverage after '
            'correcting contradictory labels. Never relabel or discard any existing case. Each new case must '
            'reuse an actual scenario and provide schema-valid JSON mutations and answer recipes. Derive '
            'expected success from the ORIGINAL business, never from the observed grader. A negative case '
            'must propose genuinely wrong behavior/answer. No invented data, runtime error strings, weaker '
            'rules or tools. Supply 1..4 new cases with unique IDs and the requested missing labels. Do not execute tools.',
            {'original_request': original_request, 'contract': source, 'compiled_public': compiled['public'],
             'fixture_evidence': fixtures, 'existing_cases': [record['case'] for record in records],
             'missing_expect_success': sorted(missing),
             'grammar': (__import__('pathlib').Path(__file__).parent / 'CONTRACT.md').read_text()},
            object_schema({'cases': {'type': 'array', 'items': CASE_SCHEMA}}))
        require(1 <= len(extra['cases']) <= 4 and len(records) + len(extra['cases']) <= 32,
                'audit supplemental case limit exceeded')
        extra, format_repair = _repair_recipe_json(extra, agent)
        identities = {record['case']['id'] for record in records}
        rules = {rule['id'] for rule in compiled['source']['rules']}
        mutation_repair_budget = [1]
        for case in extra['cases']:
            require(case['id'] and case['id'] not in identities and case['scenario_id'] in scenarios,
                    'invalid supplemental challenge identity')
            require(case['reason'].strip() and set(case['rule_ids']) <= rules,
                    'supplemental challenge must use actual rules and reasons')
            require(case['expect_success'] in missing, 'supplemental challenge must restore missing label')
            identities.add(case['id'])
            records.append(_execute_case(compiled, case, scenarios, agent, mutation_repair_budget))
        supplement = {'provenance': extra_provenance, 'result': extra, 'format_repair': format_repair}
    require({record['case']['expect_success'] for record in records} == {True, False},
            'audit must preserve positive and negative coverage')
    return {**report, 'passed': value['business_consistent'] and all(r['passed'] for r in records),
            'business_consistent': value['business_consistent'], 'reason': value['reason'], 'records': records,
            'review_audit': {'provenance': provenance, 'result': value, 'original_review': report,
                             'supplement': supplement}}
