"""Predeclared representation-neutral business acceptance for new free rounds."""
from env_factory_v2.contract import require
from env_factory_v2.fixtures import review_contract
from env_factory_v2.reward import grade
from env_factory_v2.expressions import canonical
from v2_natural_semantics import schema, fill_projections, wrong_projection, rename_observations


def business_truth(spec, business, user, *, user_request, **kwargs):
    """Use predeclared public answer obligations without relaxing state checks."""
    from audit_v2_natural import canonical_truth
    truth = canonical_truth(spec, business, user, **kwargs)
    if 'answer_obligations' not in spec:
        return truth
    obligations = spec['answer_obligations']
    require(isinstance(obligations, list), 'answer obligations must be predeclared')
    roles = []
    for obligation in obligations:
        require(isinstance(obligation, dict) and set(obligation) == {'role', 'source', 'text'},
                'answer obligation provenance required')
        role, source, text = (obligation[k] for k in ('role', 'source', 'text'))
        require(role in truth['answer'] and role not in roles, 'unknown or duplicate answer obligation')
        public = {'user_request': user_request, 'business_policy': business['policy']}
        require(source in public and isinstance(text, str) and text.strip() and text in public[source],
                'answer obligation must cite public demand or policy')
        roles.append(role)
    truth['state_result_values'] = truth['answer']
    truth['answer'] = {role: truth['answer'][role] for role in roles}
    return truth


def state_counterexamples(compiled, scenario, truth, judge):
    """Undo each required changed cell while retaining the successful trace."""
    import copy
    from measure_v2_natural import execute_reference
    from audit_v2_natural import reference_answer
    checks = []
    for table, rows in truth['final_state'].items():
        initial = {row['id']: row for row in truth['initial'][table]}
        for row in rows:
            before = initial[row['id']]
            for column, value in row.items():
                if canonical(value) == canonical(before[column]):
                    continue
                world = execute_reference(compiled, scenario)
                answer = reference_answer(world, scenario)
                actual = next(item for item in world.data[table] if item['id'] == row['id'])
                actual[column] = copy.deepcopy(before[column])
                score = grade(world, answer, judge=judge)
                checks.append({'id': f"wrong_state_{table}_{row['id']}_{column}",
                               'passed': not score['success'], 'outcome': score})
    return checks


def accept(compiled, row, business, user, spec, agent):
    # Calibration helpers are shared with the previous evidence-preserving audit;
    # no old audit scores, model mappings or traces are reused in a new round.
    from audit_v2_natural import canonical_truth, value_checks, audited_observations, reference_answer
    from measure_v2_natural import execute_reference, sorted_data
    truth = business_truth(spec, business, user, user_request=row['raw_request'])
    aliases = {r['id']: r['name'] for r in spec['params'].get('candidates', [])}
    source, previews = review_contract(compiled['source'])
    refusal_evidence = []
    if any(tool['kind'] == 'transaction' for tool in source['tools']):
        for scenario in source['scenarios']:
            if not scenario['expect_success'] or scenario.get('mutations'):
                continue
            try:
                refused_world = execute_reference(compiled, scenario, refuse=True)
                refused_truth = business_truth(spec, business, user,
                    user_request=row['raw_request'], refuse=True)
                refusal_evidence.append({'scenario': scenario['id'], 'trace': refused_world.trace,
                    'user_state': refused_world.user.snapshot(),
                    'state_unchanged': canonical(sorted_data(refused_world.data))
                    == canonical(sorted_data(refused_truth['initial']))})
            except (ValueError, KeyError, TypeError) as error:
                refusal_evidence.append({'scenario': scenario['id'], 'error': str(error)})
    review, provenance = agent.ask('Independently audit business obligations against the user request and dossier. '
        'Map every semantic role exactly once using closed paths into declared answer fields. Allow nested '
        'Each selector field selects the top-level answer field; path is relative to its value: '
        'use an empty string for the whole value, payable for a nested property, or items.*.id for array IDs. '
        'Do not repeat the selected field name in path. single packing requires exactly one selector; '
        'never select multiple conditional alternatives as one scalar. Allow nested '
        'objects, singleton scalar packing, minutes/seconds conversion for hours, and public lecturer ID aliases. '
        'Do not compute or invent values. Never require a particular answer layout. Check missing obligations, '
        'hidden requirements, privacy, unsupported restrictions and actual refusal handling. '
        'Direct refusal is consent.declined; withdrawal after approval is consent.revoked. '
        'A revoked-only success branch does not cover direct refusal. Verify an actual declined '
        'no-write path earns success with truthful submission fields. The supplied executed refusal '
        'evidence also exercises this path even if no separate named fixture declares it; do not '
        'require duplicate fixtures when actual execution evidence exists. No tools.',
        {'user_request': row['raw_request'], 'business': business, 'private_user_fixture': user,
         'contract': source, 'fixture_previews': previews, 'public': compiled['public'],
         'executed_direct_refusal': refusal_evidence,
         'required_roles': [{'id': k, 'canonical_type': type(v).__name__} for k,v in truth['answer'].items()],
         'public_name_aliases': aliases}, schema(truth['answer'], [f['id'] for f in source['answers']]))
    entries = review['mapping']
    require(len(entries) == len(truth['answer']) and {e['role'] for e in entries} == set(truth['answer']), 'incomplete semantic mapping')
    selected, checks, reference_attempts = None, [], []
    for scenario in compiled['source']['scenarios']:
        if not scenario['expect_success'] or scenario.get('mutations') or scenario.get('user_overrides'):
            continue
        try:
            world = execute_reference(compiled, scenario)
            answer = reference_answer(world, scenario)
            current = value_checks(answer, truth, entries, aliases) + audited_observations(world, truth, spec['kind'], business)
            current.append({'id': 'full_business_state', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(truth['final_state']))})
            score = grade(world, answer, judge=agent.judge)
            current.append({'id': 'positive_reward', 'passed': score['success'], 'outcome': score})
            checks = current
            reference_attempts.append({'scenario': scenario['id'], 'checks': current})
            if all(c['passed'] for c in current):
                selected = scenario
                break
        except (ValueError, KeyError, TypeError) as error:
            checks = [{'id': 'positive_execution', 'passed': False, 'error': str(error)}]
            reference_attempts.append({'scenario': scenario['id'], 'checks': checks})
    if selected is None:
        return {'passed': False, 'mapping': entries, 'review': review, 'provenance': provenance,
                'checks': checks, 'reference_attempts': reference_attempts,
                'error': 'no independently correct positive reference',
                'scope': 'predeclared_independent_business_checks_not_posthoc'}
    checks.extend(state_counterexamples(compiled, selected, truth, agent.judge))
    for entry in entries:
        world = execute_reference(compiled, selected)
        score = grade(world, wrong_projection(reference_answer(world, selected), entry), judge=agent.judge)
        checks.append({'id': 'wrong_' + entry['role'], 'passed': not score['success'], 'outcome': score})
    world = execute_reference(compiled, rename_observations(selected))
    score = grade(world, reference_answer(world, rename_observations(selected)), judge=agent.judge)
    checks.append({'id': 'caller_id_invariance', 'passed': score['success']})
    if spec['mutation']:
        baseline_answer = reference_answer(execute_reference(compiled, selected), selected)
        changed = business_truth(spec, business, user, user_request=row['raw_request'], mutation=spec['mutation'])
        answer_changed = canonical(changed['answer']) != canonical(truth['answer'])
        for stale in ([False, True] if answer_changed else [False]):
            world = execute_reference(compiled, selected, mutation=spec['mutation'])
            answer = baseline_answer if stale else fill_projections(reference_answer(world, selected), entries, changed['answer'], name_aliases=aliases)
            score = grade(world, answer, judge=agent.judge)
            checks.append({'id': 'stale_answer' if stale else 'changed_answer',
                'passed': not score['success'] if stale else score['success']
                and all(c['passed'] for c in value_checks(answer, changed, entries, aliases))
                and canonical(sorted_data(world.data)) == canonical(sorted_data(changed['final_state'])), 'outcome': score})
    if any(t['kind'] == 'transaction' for t in compiled['source']['tools']):
        world = execute_reference(compiled, selected, refuse=True)
        refused = business_truth(spec, business, user, user_request=row['raw_request'], refuse=True)
        answer = fill_projections(reference_answer(world, selected), entries, refused['answer'], name_aliases=aliases)
        for field in compiled['source']['answers']:
            text = field['meaning'].lower()
            if field['schema']['type'] == 'boolean' and ('本次' in text or 'this execution' in text) and any(w in text for w in ['提交', 'commit', '登记']):
                answer[field['id']] = False
        score = grade(world, answer, judge=agent.judge)
        checks.append({'id': 'actual_decline', 'passed': score['success'] and canonical(sorted_data(world.data)) == canonical(sorted_data(refused['initial']))
            and all(c['passed'] for c in value_checks(answer, refused, entries, aliases)), 'outcome': score})
    return {'passed': review['business_consistent'] and all(c['passed'] for c in checks),
        'mapping': entries, 'review': review, 'provenance': provenance, 'checks': checks,
        'scope': 'predeclared_independent_business_checks_not_posthoc'}
