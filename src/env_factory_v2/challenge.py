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
    'steps_json': {'type': 'string'},
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
        for field in ('mutations_json', 'answer_json', 'user_overrides_json', 'steps_json'):
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
        from .communication_coverage import communication_coverage
        report['communication_coverage'] = communication_coverage(compiled, report['records'])
        if report.get('requirement_review', {}).get('passed') is False:
            return {**report, 'passed': False, 'status': 'inconclusive',
                    'instruction': 'Resolve the independently identified public-clause discrepancies. '
                    'Case relabeling cannot override a failed requirement interpretation review.'}
        all_passed = all(record['passed'] for record in report['records'])
        if report['business_consistent'] != all_passed:
            report = _audit(compiled, original_request, agent, report)
            report['communication_coverage'] = communication_coverage(compiled, report['records'])
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
    from .requirement_review import obligations, review_schema, validate_review, implementation_paths, mandatory_obligations
    requirements = obligations(compiled, original_request)
    value, provenance = agent.ask(
        'Independently compare the ORIGINAL request with this proposed business contract. compiled_public '
        'is the actual agent-visible interface; authored tool descriptions are not independently published. Do not trust its '
        'reference answers, tool predicates or reward targets as business truth. Propose concrete executable '
        'positive and negative counterexamples. Cover every public rule, extra identity/display-name filters, '
        'mutable selected-record facts, ordering, absence, actual menu choices, authorization and transactions '
        'including private-information leakage and unnecessary user questions '
        'where applicable. Stay within declared data schemas, foreign keys and actual user stages; '
        'do not invent user keys or incompatible fact types. '
        'Apply empty-result requirements consistently: if the ORIGINAL public rules do not guarantee a '
        'matching identity or available candidate, assess a truthful successful no-write completion, '
        'conditional downstream tools, and absence of nonexistent result dependencies. A missing-row '
        'exception is not a successful no-result episode. State the concrete public obligation and '
        'executable evidence for any rejection; an unexecutable boundary remains explicitly inconclusive. '
        'Do not require preliminary authorization for clarification or changing a preference unless '
        'the original business rules explicitly require it. '
        'user_overrides_json is a JSON object of existing private fact '
        'overrides for this qualification case only; use {} to retain inherited facts. You may vary approval '
        'booleans to exercise real refusal. Never change the base fixture or public request. Each case names an actual base scenario, changes initial records by primary key, and supplies '
        'an independently correct answer recipe (the expression grammar is provided). answer_json and '
        'mutations_json and user_overrides_json are JSON strings; use {} to retain the original answer recipe. The verifier owns '
        'intentional-invalid-action expectations inherited from each referenced scenario; never invent runtime '
        'error strings. When an inherited scenario expects an intentional runtime error, expect_success '
        'must be false; a safely rejected illegal action is a negative execution fixture, not a successful '
        'completed episode. Successful no-write paths must finish without an exception. '
        'steps_json is a JSON string: null inherits the base trajectory; an array supplies your own '
        'complete trajectory using the existing public tools and user stages. Use independent steps '
        'concisely: an array item {"$step":N} copies the zero-based Nth step of the named base '
        'scenario exactly. Mix references with new full step objects to omit, reorder or add actions. '
        'A reference cannot change any arguments; write a full replacement step for changes. '
        'All references are expanded before the same grammar, dependency and runtime checks. '
        'to exercise missing branches instead of accepting the author scenario catalog as complete. '
        'An empty array explicitly tests no actions. Independent steps do not inherit expect_error: '
        'they must complete normally, so a runtime error cannot count as a successful negative case. '
        'Each wait step accepts 0..86400 seconds; longer elapsed durations require multiple valid '
        'wait steps within the action budget, never an out-of-range wait argument. '
        'For conditional communication criteria inspect when against the ORIGINAL obligation and '
        'exercise applicable and inapplicable branches when reachable within that obligation. '
        'Do not demand an unreachable branch or invent an obligation to recite unrelated business rules. '
        'Include at least one positive and one negative. '
        'For stateless direct tasks, empty data mutations are appropriate. Validate boundary predicates '
        'by invoking the actual named public rule with boundary arguments in a conditional answer recipe: '
        'return the independently correct original-task answer only when the boundary predicate has its '
        'business-correct value, otherwise a wrong schema-valid answer. This is evaluator validation; '
        'do not invent an extra agent explanation obligation. Assess evaluator coverage across the authored '
        'probes AND your executable cases; a boundary covered by your own case does not require an extra '
        'agent answer field or a duplicate authored probe. '
        'For generated tables the exact generator plan is authoritative; previews are concrete examples, not '
        'the whole table. Derive unshown values from the documented plan, never invent hidden rows. '
        'When public_requirements is nonempty, review EVERY listed clause exactly once in requirement_reviews. '
        'Also review EVERY mandatory_obligation_catalog entry exactly once in obligation_reviews. '
        'Use paths arrays to group entries ONLY when the same original requirements, applicability, '
        'verdict, rationale and executable cases justify EACH member independently. Otherwise use '
        'separate groups, including every unsupported field. Grouping saves repeated prose, never '
        'exempts a member from semantic review; the host expands groups and checks exact coverage. '
        'For each mandatory output field, nested property, communication or success constraint, identify '
        'the original requirement_ids that justify requiring it from this user, and actual case_ids that '
        'exercise it. Existence in a business table or tool response does not justify mandatory output '
        'or reward weight. Distinguish necessary evidence to decide the requested result from unrelated '
        'record columns; returning a full record must not make every column an extra task obligation. '
        'Derived outputs and necessary decision evidence may be justified implicitly by the original '
        'request: exact keyword matches are not required. If an obligation is unsupported, mark '
        'consistent=false, explain the added burden, and allow requirement_ids=[]; do not invent a '
        'citation or remove genuine safety and transaction requirements. A citation is not proof: '
        'assess the actual business meaning in both directions. '
        'Declare applicability as applicable, conditional, or outside_request. Determine scope from the '
        'original natural request and the clause conditions, never merely from missing generated tools '
        'or rules. A business dossier can describe other operations that this user did not request. '
        'Do not turn those operations into extra task obligations or require the agent to recite them. '
        'For outside_request explain the absent triggering operation and cite actual executable cases '
        'that verify the requested behavior and implementation paths that establish its actual scope. '
        'Never exclude a user_request clause. For conditional clauses test applicable branches when '
        'schema-valid inputs can activate them within this task. Every applicability status still '
        'requires genuine implementation references and nonempty actual case_ids; do not invent cases '
        'for unrelated operations or claim missing required functionality is outside_request. '
        'For each clause explain whether executable behavior faithfully implements it, cite specific JSON '
        'Pointer implementation_paths selected EXACTLY from implementation_citation_catalog, relative to '
        'the contract itself (no /contract or /compiled_public prefix), '
        'and actual independent case_ids which exercise that obligation. Private facts and the demand text '
        'are not implementation evidence. Do not invent an implementation or omit an unmet obligation: '
        'mark consistent false and explain the discrepancy. Trace communication and authorization obligations '
        'to actual stages and trajectories too. A citation is not proof by itself: inspect semantics. '
        'Unexecuted model opinions are not proof. Do not execute tools.',
        {'original_request': original_request, 'contract': review_source,
         'compiled_public': compiled['public'], 'fixture_evidence': fixture_evidence,
         'public_requirements': requirements,
         'mandatory_obligation_catalog': mandatory_obligations(compiled['source']) if requirements else [],
         'implementation_citation_catalog': implementation_paths(compiled['source']) if requirements else [],
         'grammar': (__import__('pathlib').Path(__file__).parent / 'CONTRACT.md').read_text()},
        review_schema(CHALLENGE_SCHEMA, requirements, compiled['source']))
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
    requirement_review = validate_review(compiled, requirements, value)
    return {'scope': 'native_independent_contract_challenge', 'contract_identity': compiled['identity'],
            'passed': value['business_consistent'] and all(r['passed'] for r in records)
                      and (requirement_review is None or requirement_review['passed']),
            'business_consistent': value['business_consistent'], 'reason': value['reason'],
            'provenance': provenance, 'records': records,
            **({'requirement_review': requirement_review} if requirement_review else {}),
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
        'actual public rule. Use unique new IDs and actual base scenario IDs. steps_json may be null '
        'to inherit or an array of independent steps using existing tools/stages. Recipes follow the supplied '
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


def _case_scenario(compiled, case, scenarios):
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
    steps = json.loads(case.get('steps_json', 'null'))
    require(steps is None or isinstance(steps, list), 'independent steps must be null or an array')
    if steps is not None:
        inherited_steps = scenario['steps']
        expanded = []
        for step in steps:
            if isinstance(step, dict) and '$step' in step:
                require(set(step) == {'$step'} and type(step['$step']) is int
                        and 0 <= step['$step'] < len(inherited_steps), 'invalid independent step reference')
                expanded.append(copy.deepcopy(inherited_steps[step['$step']]))
            else:
                expanded.append(copy.deepcopy(step))
        steps = expanded
        tool_ids = {tool['id'] for tool in compiled['source']['tools']}
        for step in steps:
            require(isinstance(step, dict), 'independent step must be an object')
            if step.get('kind') in {'call', 'traverse'}:
                require(step.get('tool') in tool_ids, 'unknown independent tool')
            if step.get('kind') == 'wait':
                seconds = step.get('seconds')
                require(type(seconds) in {int, float} and 0 <= seconds <= 86400,
                        'invalid independent wait; each wait must be 0..86400 seconds')
        scenario['steps'] = steps
        scenario.pop('expect_error', None)
        # Validate the new trajectory with the production grammar, without
        # changing the contract executed or its scoring identity.
        from .contract import compile_contract
        validation_source = copy.deepcopy(compiled['source'])
        validation_case = copy.deepcopy(scenario)
        validation_case['id'] = '__independent_case'
        while any(s['id'] == validation_case['id'] for s in validation_source['scenarios']):
            validation_case['id'] += '_'
        validation_source['scenarios'].append(validation_case)
        compile_contract(validation_source)
    return scenario


def _execute_case(compiled, case, scenarios, agent, mutation_repair_budget=None):
    from .expressions import ContractError
    scenario = _case_scenario(compiled, case, scenarios)
    mutations = json.loads(case['mutations_json'])
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
         'reason': report['reason']}, 'executed_cases': observations,
         'grammar': (__import__('pathlib').Path(__file__).parent / 'CONTRACT.md').read_text()}, AUDIT_SCHEMA)
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
        scenario = _case_scenario(compiled, case, scenarios)
        records.append({**old, 'case': case, 'passed': expected_result(scenario, old['first'])
                        and expected_result(scenario, old['repeat'])
                        and replay_identity(old['first']) == replay_identity(old['repeat'])})
    supplement = None
    missing = {True, False} - {record['case']['expect_success'] for record in records}
    if missing:
        extra, extra_provenance = agent.ask(
            'Append grounded independent counterexamples to restore missing positive/negative coverage after '
            'correcting contradictory labels. Never relabel or discard any existing case. Each new case must '
            'name an actual base scenario and provide schema-valid JSON mutations and answer recipes. '
            'steps_json is null to inherit or an array to construct an independent trajectory using '
            'existing tools and stages; new trajectories cannot inherit runtime-error expectations. Derive '
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
