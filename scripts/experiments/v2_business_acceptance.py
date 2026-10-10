"""Predeclared representation-neutral business acceptance for new free rounds."""
from env_factory_v2.contract import require
from env_factory_v2.fixtures import review_contract
from env_factory_v2.reward import grade
from env_factory_v2.expressions import canonical
from v2_natural_semantics import schema, fill_projections, wrong_projection, rename_observations


def accept(compiled, row, business, user, spec, agent):
    # Calibration helpers are shared with the previous evidence-preserving audit;
    # no old audit scores, model mappings or traces are reused in a new round.
    from audit_v2_natural import canonical_truth, value_checks, audited_observations, reference_answer
    from measure_v2_natural import execute_reference, sorted_data
    truth = canonical_truth(spec, business, user)
    aliases = {r['id']: r['name'] for r in spec['params'].get('candidates', [])}
    source, previews = review_contract(compiled['source'])
    review, provenance = agent.ask('Independently audit business obligations against the user request and dossier. '
        'Map every semantic role exactly once using closed paths into declared answer fields. Allow nested '
        'objects, singleton scalar packing, minutes/seconds conversion for hours, and public lecturer ID aliases. '
        'Do not compute or invent values. Never require a particular answer layout. Check missing obligations, '
        'hidden requirements, privacy, unsupported restrictions and actual refusal handling. No tools.',
        {'user_request': row['raw_request'], 'business': business, 'private_user_fixture': user,
         'contract': source, 'fixture_previews': previews, 'public': compiled['public'],
         'required_roles': [{'id': k, 'canonical_type': type(v).__name__} for k,v in truth['answer'].items()],
         'public_name_aliases': aliases}, schema(truth['answer'], [f['id'] for f in source['answers']]))
    entries = review['mapping']
    require(len(entries) == len(truth['answer']) and {e['role'] for e in entries} == set(truth['answer']), 'incomplete semantic mapping')
    selected, checks = None, []
    for scenario in compiled['source']['scenarios']:
        if not scenario['expect_success'] or scenario.get('mutations'):
            continue
        try:
            world = execute_reference(compiled, scenario)
            answer = reference_answer(world, scenario)
            current = value_checks(answer, truth, entries, aliases) + audited_observations(world, truth, spec['kind'], business)
            current.append({'id': 'full_business_state', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(truth['final_state']))})
            score = grade(world, answer, judge=agent.judge)
            current.append({'id': 'positive_reward', 'passed': score['success'], 'outcome': score})
            checks = current
            if all(c['passed'] for c in current):
                selected = scenario
                break
        except (ValueError, KeyError, TypeError) as error:
            checks = [{'id': 'positive_execution', 'passed': False, 'error': str(error)}]
    require(selected is not None, 'no independently correct positive reference')
    for entry in entries:
        world = execute_reference(compiled, selected)
        score = grade(world, wrong_projection(reference_answer(world, selected), entry), judge=agent.judge)
        checks.append({'id': 'wrong_' + entry['role'], 'passed': not score['success'], 'outcome': score})
    world = execute_reference(compiled, rename_observations(selected))
    score = grade(world, reference_answer(world, rename_observations(selected)), judge=agent.judge)
    checks.append({'id': 'caller_id_invariance', 'passed': score['success']})
    if spec['mutation']:
        baseline_answer = reference_answer(execute_reference(compiled, selected), selected)
        changed = canonical_truth(spec, business, user, mutation=spec['mutation'])
        for stale in [False, True]:
            world = execute_reference(compiled, selected, mutation=spec['mutation'])
            answer = baseline_answer if stale else fill_projections(reference_answer(world, selected), entries, changed['answer'], name_aliases=aliases)
            score = grade(world, answer, judge=agent.judge)
            checks.append({'id': 'stale_answer' if stale else 'changed_answer',
                'passed': not score['success'] if stale else score['success'] and all(c['passed'] for c in value_checks(answer, changed, entries, aliases)), 'outcome': score})
    if any(t['kind'] == 'transaction' for t in compiled['source']['tools']):
        world = execute_reference(compiled, selected, refuse=True)
        refused = canonical_truth(spec, business, user, refuse=True)
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
