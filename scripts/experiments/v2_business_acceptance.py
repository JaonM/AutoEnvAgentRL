"""Predeclared representation-neutral business acceptance for new free rounds."""
from env_factory_v2.contract import require
from env_factory_v2.fixtures import review_contract
from env_factory_v2.reward import grade
from env_factory_v2.expressions import canonical
from v2_natural_semantics import schema, fill_projections, wrong_projection, rename_observations


def runtime_business_truth(spec, business, user, *, user_request, world,
                           time_call_id, authorization_index=None, refuse=False):
    """Explicit runtime-to-oracle adapter; missing evidence never gets defaults."""
    import copy
    from v2_natural_oracle import expand
    from v2_runtime_evidence import (observed_time, authorized_call_context, target_attempts,
                                    target_authorizations, event_time, state_before_event)
    initial = expand(business)
    require(all(table in world.initial and canonical(sorted(world.initial[table], key=canonical))
                == canonical(sorted(rows, key=canonical))
                for table, rows in initial.items()), 'runtime initial business rows differ from independent fixture')
    options = {'now': observed_time(world, time_call_id)}
    if refuse:
        options['refuse'] = True
    private = copy.deepcopy(user)
    evidence = {'time_call_id': time_call_id, 'virtual_seconds': options['now']}
    if spec['kind'] in ('procurement_partial', 'access_temporary'):
        table = 'orders' if spec['kind'] == 'procurement_partial' else 'grants'
        identities = private['facts']['order_ids' if table == 'orders' else 'grant_ids']
        timings, sources = {}, {}
        for identity in identities:
            attempts = target_attempts(world, table, identity)
            require(len(attempts) <= 1, 'multiple transaction attempts require explicit retry semantics')
            decisions = target_authorizations(world, table, identity)
            if attempts:
                timings[identity] = attempts[0]['time']
                sources[identity] = {'kind': 'actual_attempt', 'call_id': attempts[0]['call_id']}
            elif decisions:
                timings[identity] = decisions[-1]['time']
                sources[identity] = {'kind': 'authorization_without_write', 'trace_index': decisions[-1]['trace_index']}
            else:
                # Eligibility may correctly prevent both consent and submission.
                # Use an actual trace event; expected state still requires any
                # eligible authorized write, so omission cannot fabricate success.
                timings[identity] = event_time(world.trace[-1])
                sources[identity] = {'kind': 'no_transaction_or_consent', 'trace_index': len(world.trace) - 1}
        if table == 'orders':
            options['now'] = [timings[identity] for identity in identities]
        else:
            options['submission_times'] = timings
        evidence['individual_times'] = timings
        evidence['individual_time_sources'] = sources
    if spec['kind'] == 'procurement_large':
        identity = private['facts']['order_id']
        attempts = target_attempts(world, 'orders', identity)
        decisions = target_authorizations(world, 'orders', identity)
        require(len(attempts) <= 1 and decisions, 'large purchase requires its actual proposal and unambiguous attempt evidence')
        if authorization_index is None:
            authorization_index = decisions[-1]['trace_index']
        decision = next((d for d in decisions if d['trace_index'] == authorization_index), None)
        require(decision is not None, 'large purchase authorization does not identify its order')
        if attempts:
            context = authorized_call_context(world, attempts[0]['call_id'], authorization_index)
            snapshot = context['authorized_state']
            options['now'] = context['call_time']
            proposal_time = context['authorization_time']
        else:
            snapshot = state_before_event(world, authorization_index)
            proposal_time = decision['time']
            options['now'] = event_time(world.trace[-1])
        require('quotes' in snapshot and len(snapshot.get('budgets', [])) == 1,
                'authorized procurement snapshot lacks its quotes or budget')
        options['authorized_snapshot'] = {'quotes': snapshot['quotes'],
            'budget_cents': snapshot['budgets'][0]['remaining_cents']}
        private['facts']['proposal_time'] = proposal_time
        evidence['authorization_index'] = authorization_index
        evidence['submission_attempts'] = attempts
    truth = business_truth(spec, business, private, user_request=user_request, **options)
    # Executable distractors may have their own tables. Preserve them in complete
    # state comparison, rather than discarding unauthorized side effects there.
    for table, rows in world.initial.items():
        if table not in initial:
            truth['initial'][table] = copy.deepcopy(rows)
            truth['final_state'][table] = copy.deepcopy(rows)
    truth['runtime_evidence'] = evidence
    return truth


TIMED_READ_TABLES = {'procurement_quote': {'quotes'}, 'procurement_combination': {'quotes'},
                    'access_query': {'nodes', 'grants', 'memberships'},
                    'access_compare': {'nodes', 'grants', 'memberships'}}
TIMED_TRANSACTION_TABLES = {'procurement_partial': {'quotes', 'budgets', 'orders'},
                            'procurement_large': {'quotes', 'budgets', 'orders'},
                            'access_temporary': {'nodes', 'grants', 'memberships'},
                            'access_revoke': {'nodes', 'grants', 'memberships'}}
TIMED_TABLES = {**TIMED_READ_TABLES, **TIMED_TRANSACTION_TABLES}


def business_truth(spec, business, user, *, user_request, world=None, **kwargs):
    """Use predeclared public answer obligations without relaxing state checks."""
    if world is not None and spec['kind'] in TIMED_TABLES:
        calls = [event for event in world.trace if event.get('kind') == 'tool'
                 and event.get('status') == 'ok' and world.tools[event['tool']]['kind'] == 'query'
                 and world.tools[event['tool']]['table'] in TIMED_TABLES[spec['kind']]]
        require(calls, 'timed business truth requires an actual relevant query')
        require(set(kwargs) <= {'refuse'}, 'timed variants require their own executed runtime evidence')
        return runtime_business_truth(spec, business, user, user_request=user_request,
                                      world=world, time_call_id=calls[-1]['id'], **kwargs)
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
            if row['id'] not in initial:
                world = execute_reference(compiled, scenario)
                answer = reference_answer(world, scenario)
                world.data[table] = [item for item in world.data[table] if item['id'] != row['id']]
                score = grade(world, answer, judge=judge)
                checks.append({'id': f"missing_created_record_{table}_{row['id']}",
                               'passed': not score['success'], 'outcome': score})
                continue
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
    initial_world = None
    if spec['kind'] in TIMED_TABLES:
        first = next((scenario for scenario in compiled['source']['scenarios']
            if scenario['expect_success'] and not scenario.get('mutations') and not scenario.get('user_overrides')), None)
        require(first is not None, 'timed acceptance requires an executed positive scenario')
        initial_world = execute_reference(compiled, first)
    truth = business_truth(spec, business, user, user_request=row['raw_request'], world=initial_world)
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
                    user_request=row['raw_request'], refuse=True, world=refused_world)
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
        'Use representation=set_membership only for the independently offered boolean roles when a complete identifier set answers that question; the checker validates the entire set before membership. Use representation=total_quantity only for an independently offered quantity role when the selected value is the total supplied quantity; the host subtracts the declared demand. For offered record roles use representation=records and record_layout: identity_path selects the object ID, details_path selects the detail object or array, scope_field selects the permission in each detail (empty for flat records), columns maps each canonical field to a detail-relative path. The host selects the independently declared scope and enum aliases. For other representations provide empty record_layout paths and columns. Otherwise use native. Do not compute or invent values. Never require a particular answer layout. Check missing obligations, '
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
         'record_roles': truth.get('record_roles', {}), 'quantity_offsets': truth.get('quantity_offsets', {}), 'public_name_aliases': aliases, 'set_membership_roles': list(truth.get('set_memberships', {}))}, schema(truth['answer'], [f['id'] for f in source['answers']], set_memberships=truth.get('set_memberships'), quantity_offsets=truth.get('quantity_offsets'), record_roles=truth.get('record_roles')))
    entries = review['mapping']
    require(len(entries) == len(truth['answer']) and {e['role'] for e in entries} == set(truth['answer']), 'incomplete semantic mapping')
    selected, checks, reference_attempts = None, [], []
    for scenario in compiled['source']['scenarios']:
        if not scenario['expect_success'] or scenario.get('mutations') or scenario.get('user_overrides'):
            continue
        try:
            world = execute_reference(compiled, scenario)
            if spec['kind'] in TIMED_TABLES:
                truth = business_truth(spec, business, user, user_request=row['raw_request'], world=world)
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
        score = grade(world, wrong_projection(reference_answer(world, selected), entry, record_roles=truth.get('record_roles')), judge=agent.judge)
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
            answer = baseline_answer if stale else fill_projections(reference_answer(world, selected), entries, changed['answer'], name_aliases=aliases, set_memberships=changed.get('set_memberships'), quantity_offsets=changed.get('quantity_offsets'), record_roles=changed.get('record_roles'))
            score = grade(world, answer, judge=agent.judge)
            checks.append({'id': 'stale_answer' if stale else 'changed_answer',
                'passed': not score['success'] if stale else score['success']
                and all(c['passed'] for c in value_checks(answer, changed, entries, aliases))
                and canonical(sorted_data(world.data)) == canonical(sorted_data(changed['final_state'])), 'outcome': score})
    if any(t['kind'] == 'transaction' for t in compiled['source']['tools']):
        world = execute_reference(compiled, selected, refuse=True)
        refused = business_truth(spec, business, user, user_request=row['raw_request'], refuse=True, world=world)
        answer = fill_projections(reference_answer(world, selected), entries, refused['answer'], name_aliases=aliases, set_memberships=refused.get('set_memberships'), quantity_offsets=refused.get('quantity_offsets'), record_roles=refused.get('record_roles'))
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
