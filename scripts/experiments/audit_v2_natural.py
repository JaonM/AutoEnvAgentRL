"""Post-seal exploratory correction of representation-only measurement errors.

Never edit a sealed pilot, repair its sources, or label this as a fresh blind
round. Extra audit calls and relaxed layout assumptions are explicitly recorded.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from decimal import Decimal
from pathlib import Path

from measure_v2_natural import (PROJECT, read, save, digest, verify_assets, fixture_alignment,
    execute_reference, observation_checks, NativeCodeAgent, require, compile_contract,
    certify, V2Episode, equivalent, canonical, sorted_data)
from env_factory_v2.contracts.expressions import evaluate
from env_factory_v2.execution.reward import grade
from v2_natural_oracle import expected
from v2_natural_semantics import schema, normalize, rename_observations, wrong_projection, fill_projections


def canonical_truth(spec, business, user, **kwargs):
    truth = expected(spec, business, user, **kwargs)
    p = spec['params']
    if spec['kind'] == 'energy':
        energy = sum(Decimal(str(w)) * Decimal(str(h)) / Decimal(1000) for w, h in zip(p['watts'], p['hours']))
        truth['answer'] = {'energy_kwh': float(energy), 'cost': float(energy * Decimal(str(p['price'])))}
    elif spec['kind'] == 'discount':
        total = sum(Decimal(str(v)) for v in p['prices'])
        payable = max(Decimal(0), total * (Decimal(str(p['rate'])) if total >= p['threshold'] else 1) - Decimal(str(p['coupon'])))
        truth['answer'] = {'payable': float(payable), 'savings': float(total - payable)}
    # Commit is checked in state and trace. These natural requests did not require
    # an extra boolean answer field about this execution's submission.
    truth['answer'].pop('submitted', None)
    return truth


def value_checks(answer, truth, entries, aliases):
    actual = normalize(answer, entries, name_aliases=aliases, role_types={k: type(v) for k, v in truth['answer'].items()}, set_memberships=truth.get('set_memberships'), quantity_offsets=truth.get('quantity_offsets'), record_roles=truth.get('record_roles'))
    return [{'id': 'answer_' + k, 'passed': k in actual and equivalent(actual[k], v, unordered=k == 'ids')}
            for k, v in truth['answer'].items()]


def reference_answer(world, scenario):
    return {k: evaluate(e, world.context(private=True)) for k, e in scenario['answer'].items()}


def scalar_literals(answer, truth, entries):
    answer = copy.deepcopy(answer)
    for e in entries:
        if e['packing'] == 'single' and len(e['selectors']) == 1 and not e['selectors'][0]['path'] and e['names'] == 'native':
            value = truth['answer'][e['role']]
            if e['time_unit'] != 'native':
                value *= 60 if e['time_unit'] == 'minutes' else 3600
            if type(answer[e['selectors'][0]['field']]) is int and type(value) is float and value.is_integer():
                value = int(value)
            answer[e['selectors'][0]['field']] = copy.deepcopy(value)
    return answer


def audited_observations(world, truth, kind, business, agent=None):
    from v2_communication_evidence import communication_checks
    checks = observation_checks(world, truth, kind) + communication_checks(world, truth, agent)
    if 'transaction_groups' in truth:
        from v2_fresh_business_oracle import transaction_group_checks
        checks += transaction_group_checks(world, truth['transaction_groups'])
    # An atomic receipt is authoritative evidence of a committed update unless
    # the dossier explicitly requires a post-commit query. No invented read gate.
    if '提交后查询' in business['policy'] or '提交后核验' in business['policy']:
        return checks
    for check in checks:
        if not check['id'].startswith('post_commit_') or check['passed']:
            continue
        for table, rows in truth['final_state'].items():
            for row in rows:
                if check['id'] != 'post_commit_' + table + '_' + row['id']:
                    continue
                before = next(r for r in truth['initial'][table] if r['id'] == row['id'])
                changes = {k: v for k, v in row.items() if v != before[k]}
                receipt = any(t.get('kind') == 'tool' and t.get('status') == 'ok' and
                    world.tools[t['tool']]['kind'] == 'transaction' and any(c['table'] == table and c['key'] == row['id']
                    and all(k in c['set'] and equivalent(c['set'][k], v) for k, v in changes.items())
                    for c in t['result']['changes']) for t in world.trace)
                if receipt:
                    check.update(passed=True, evidence='actual_atomic_commit_receipt_no_postquery_obligation')
    return checks


def rollout(bundle, row, truth, entries, aliases, agent, root, business):
    from measure_v2_natural import ACTION_SCHEMA
    env = V2Episode(bundle)
    start = time.monotonic()
    try:
        env.reset(row['id'])
        for turn in range(40):
            response, _ = agent.ask(
                'Fulfill the natural user request using only public tools and actual observations. '
                'Ask for missing user facts and exact consent. Use actual observation IDs in bindings. '
                'Return 1..8 sequential tool intents per turn, stop to inspect unknown results and cursors. '
                'No hidden evidence, evaluator targets, reference trajectories or Codex tools. '
                'Finish with v2_submit and the declared answer fields.',
                {'messages': env.messages, 'tools': env.tools, 'remaining_turns': 40 - turn}, ACTION_SCHEMA)
            require(1 <= len(response['calls']) <= 8, 'audit policy batch size invalid')
            calls = [{'id': c['id'], 'type': 'function', 'function': {'name': c['name'],
                'arguments': c['arguments_json']}} for c in response['calls']]
            env.step({'role': 'assistant', 'content': '', 'tool_calls': calls})
            save(root / 'rollout_progress.json', {'turns': turn + 1, 'terminated': env.terminated})
            if env.terminated:
                break
        world = env.server.episodes[env.episode]
        outcome = env.finish()
        checks = value_checks(env.answer or {}, truth, entries, aliases) + audited_observations(world, truth, row['kind'], business, agent)
        checks.append({'id': 'full_business_state', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(truth['final_state']))})
        return {'passed': env.terminated and outcome['score']['success'] and all(c['passed'] for c in checks),
            'answer': env.answer, 'outcome': outcome, 'checks': checks, 'messages': env.messages,
            'seconds': time.monotonic() - start, 'scope': 'fresh_native_policy_on_unchanged_original_bundle_posthoc_audit'}
    finally:
        env.close()


def audit_sample(row, assets, original, root, connection, reuse=None):
    business, user, spec = (read(assets / row[k]) for k in ('business', 'user', 'acceptance'))
    progress = read(original / 'result.json')
    result = {'id': row['id'], 'stratum': row['stratum'], 'title': row['title'], 'qualified': False,
              'source_repaired': False, 'original_pipeline_qualified': progress['sandbox_qualified']}
    if not progress['sandbox_qualified']:
        result['status'] = 'original_pipeline_failed'
        return result
    root.mkdir()
    bundle = original / 'build/sandbox'
    compiled = fixture_alignment(read(bundle / 'contract.json'), row, business, user)
    truth = canonical_truth(spec, business, user)
    aliases = {r['id']: r['name'] for r in spec['params'].get('candidates', [])}
    from env_factory_v2.contracts.fixtures import review_contract
    source, previews = review_contract(compiled['source'])
    agent = NativeCodeAgent.from_connection(connection, evidence_root=root / 'native')
    mapping_path = reuse / f'sample-{row["id"]:02d}/mapping.json' if reuse else None
    if mapping_path is not None and mapping_path.exists():
        old = read(mapping_path)
        review, provenance = old['response'], {**old['provenance'], 'reused_mapping_sha256': digest(mapping_path)}
    else:
        review, provenance = agent.ask(
        'Audit business meaning against the natural request and external dossier. Map every required '
        'semantic role exactly once using declared answer fields and closed dot paths (numeric array '
        'indices or * for array projection). Multiple roles may use different paths of the SAME object '
        'field. Ordered results may be split across fields; packing=list concatenates in declared order. '
        'Duration may be minutes/seconds and converts to canonical hours. Public lecturer IDs may map '
        'to names using only the dossier aliases. Do not compute values, reorder results, invent missing '
        'facts or edit the sandbox. Use representation=total_quantity only for offered quantity roles; the host subtracts the independently declared demand. For offered record roles representation=records uses identity_path, details_path, scope_field (empty for flat records), and columns of canonical name and detail-relative path; the host owns scope and enum aliases. Other representations use empty record_layout paths and columns. Task did not require an extra committed/submitted boolean; actual '
        'commit correctness is independently checked against full state and trace. Do not require flat '
        'fields or one output layout. Identify genuine missing business obligations. No tools.',
        {'raw_request': row['raw_request'], 'business': business, 'private_user_fixture': user,
         'contract': source, 'fixture_previews': previews, 'public': compiled['public'],
         'required_roles': [{'id': k, 'canonical_type': type(v).__name__} for k, v in truth['answer'].items()], 'public_name_aliases': aliases, 'quantity_offsets': truth.get('quantity_offsets', {}), 'record_roles': truth.get('record_roles', {})},
        schema(truth['answer'], [f['id'] for f in compiled['source']['answers']], quantity_offsets=truth.get('quantity_offsets'), record_roles=truth.get('record_roles')))
    save(root / 'mapping.json', {'response': review, 'provenance': provenance})
    entries = review['mapping']
    require(len(entries) == len(truth['answer']) and {e['role'] for e in entries} == set(truth['answer']), 'posthoc mapping incomplete')
    checks, selected = [], None
    for scenario in compiled['source']['scenarios']:
        if not scenario['expect_success'] or scenario.get('mutations'):
            continue
        world = execute_reference(compiled, scenario)
        answer = reference_answer(world, scenario)
        current = value_checks(answer, truth, entries, aliases) + audited_observations(world, truth, spec['kind'], business, agent)
        current.append({'id': 'full_business_state', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(truth['final_state']))})
        if all(c['passed'] for c in current):
            selected, checks = scenario, current
            break
    require(selected is not None, 'no independently correct baseline reference execution')
    from env_factory_v2.contracts.contract import ContractError
    for entry in entries:
        world = execute_reference(compiled, selected)
        answer = wrong_projection(reference_answer(world, selected), entry, record_roles=truth.get('record_roles'))
        outcome = grade(world, answer, judge=agent.judge)
        checks.append({'id': 'independent_wrong_' + entry['role'],
                       'passed': outcome['reward'] < 1 and not outcome['success'], 'outcome': outcome})
    if spec['mutation']:
        for stale in [False, True]:
            world = execute_reference(compiled, selected, mutation=spec['mutation'])
            changed_truth = canonical_truth(spec, business, user, mutation=spec['mutation'])
            answer = (reference_answer(execute_reference(compiled, selected), selected) if stale else
                fill_projections(reference_answer(world, selected), entries, changed_truth['answer'], name_aliases=aliases, quantity_offsets=changed_truth.get('quantity_offsets'), record_roles=changed_truth.get('record_roles')))
            outcome = grade(world, answer, judge=agent.judge)
            passed = outcome['reward'] < 1 and not outcome['success'] if stale else (
                outcome['success'] is True
                and all(c['passed'] for c in value_checks(answer, changed_truth, entries, aliases)))
            checks.append({'id': 'independent_stale_answer' if stale else 'independent_changed_fact', 'passed': passed, 'outcome': outcome})
    for label, scenario, literal in [('caller_id_invariance', rename_observations(selected), False),
                                      ('correct_literal_reward', selected, True)]:
        world = execute_reference(compiled, scenario)
        answer = reference_answer(world, scenario)
        if literal:
            answer = scalar_literals(answer, truth, entries)
        outcome = grade(world, answer, judge=agent.judge)
        check = {'id': label, 'passed': outcome['success'] is True, 'answer': answer, 'outcome': outcome}
        checks.append(check)
    # Use independently computed canonical answers for scalar fields during refusal;
    # retain independently inspected authored structures for nested fields.
    if any(t['kind'] == 'transaction' for t in compiled['source']['tools']):
        world = execute_reference(compiled, selected, refuse=True)
        refused_truth = canonical_truth(spec, business, user, refuse=True)
        answer = fill_projections(reference_answer(world, selected), entries, refused_truth['answer'], name_aliases=aliases, quantity_offsets=refused_truth.get('quantity_offsets'), record_roles=refused_truth.get('record_roles'))
        for field in compiled['source']['answers']:
            meaning = field['meaning'].lower()
            if (field['schema']['type'] == 'boolean' and ('本次' in meaning or 'this execution' in meaning)
                    and ('提交' in meaning or 'commit' in meaning or '登记' in meaning)):
                answer[field['id']] = False
        outcome = grade(world, answer, judge=agent.judge)
        checks.append({'id': 'actual_user_decline', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(refused_truth['initial']))
            and all(c['passed'] for c in value_checks(answer, refused_truth, entries, aliases))
            and outcome['success'] is True, 'answer': answer, 'outcome': outcome})
        # The adapter publishes a policy-callable revoke action. Without an
        # actual user withdrawal it can manufacture a no-write completion.
        for case in compiled['source']['scenarios']:
            if case['expect_success'] and not case.get('mutations') and any(s['kind'] == 'revoke' for s in case['steps']):
                adverse = execute_reference(compiled, case)
                adverse_answer = reference_answer(adverse, case)
                adverse_score = grade(adverse, adverse_answer, judge=agent.judge)
                wrong_state = canonical(sorted_data(adverse.data)) != canonical(sorted_data(truth['final_state']))
                manufactured = all(v is True for k, v in user['facts'].items() if k.startswith('approve_'))
                checks.append({'id': 'policy_cannot_manufacture_user_revocation',
                    'passed': not (manufactured and wrong_state and adverse_score['success'] is True),
                    'outcome': adverse_score, 'state_differs_from_approved_request': wrong_state,
                    'scope': 'adversarial_public_policy_action_not_an_actual_user_withdrawal'})
    result['business_checks'] = checks
    result['semantic_review'] = review
    result['external_qualified'] = review['business_consistent'] and all(c['passed'] for c in checks)
    save(root / 'acceptance.json', result)
    if not result['external_qualified']:
        result['status'] = 'external_business_or_reward_failed'
        return result
    # Training certification writes only a new COPY, never the sealed bundle.
    copy_bundle = root / 'sandbox'
    shutil.copytree(bundle, copy_bundle)
    os.environ['ENVFACTORY_V2_CONNECTION'] = str(connection.resolve())
    certificate = certify(copy_bundle)
    result['training_ready'] = certificate['training_ready'] is True
    old_rollout = original / 'rollout.json'
    if not old_rollout.exists() and reuse:
        alternative = reuse / f'sample-{row["id"]:02d}/rollout.json'
        if alternative.exists():
            old_rollout = alternative
    if old_rollout.exists():
        old = read(old_rollout)
        result['policy_evidence'] = {'source': str(old_rollout), 'sha256': digest(old_rollout), 'reused': True}
        policy_passed = old['outcome']['score']['success']
        policy_passed = policy_passed and all(c['passed'] for c in value_checks(old['answer'] or {}, truth, entries, aliases))
        policy_passed = policy_passed and all(c['passed'] for c in old.get('independent_checks', old.get('checks', [])) if not c['id'].startswith('answer_'))
        result['policy_success'] = bool(policy_passed)
    else:
        policy = rollout(copy_bundle, {**row, 'kind': spec['kind']}, truth, entries, aliases, agent, root, business)
        save(root / 'rollout.json', policy)
        result['policy_evidence'] = {'source': str(root / 'rollout.json'), 'reused': False}
        result['policy_success'] = policy['passed']
    result['qualified'] = result['policy_success']
    result['status'] = 'passed' if result['qualified'] else 'policy_validation_failed_or_inconclusive'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--round', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--reuse-mappings', type=Path, help='Reuse sealed semantic mapping and actual policy evidence, no repeat native review')
    args = parser.parse_args()
    seal = read(args.round / 'sealed_manifest.json')
    require(all(digest(args.round / p) == h for p, h in seal['files'].items()), 'original seal changed')
    manifest = verify_assets(args.assets)
    if args.reuse_mappings:
        reuse_seal = read(args.reuse_mappings / 'sealed_manifest.json')
        require(all(digest(args.reuse_mappings / p) == h for p, h in reuse_seal['files'].items()), 'previous audit seal changed')
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {'scope': 'posthoc_exploratory_measurement_correction_not_fresh_blind_yield',
        'original_seal_sha256': digest(args.round / 'sealed_manifest.json'), 'planned': 20,
        'source_repairs': 0, 'changes': ['closed nested/split answer projections', 'duration unit normalization',
            'public lecturer ID aliases', 'no unrequested submission boolean', 'decimal canonical scalar positives',
            'caller-owned observation-ID invariance', 'actual user-decline verification'],
        'representation_revision': 3, 'reuse_mappings': str(args.reuse_mappings) if args.reuse_mappings else None,
        'additional_calibration': ['typed singleton projection', 'preserve integer answer schema during unit inversion',
            'independent changed-data and decline answers replace fixed reference answers',
            'commit receipts accepted when dossier has no post-query obligation', 'adversarial policy-controlled revocation guard'],
        'limitation': 'Extra independent audit and some additional policy calls occur after viewing results; exploratory evidence only.'}
    save(args.output / 'protocol.json', protocol)
    results, started = [], time.monotonic()
    for row in manifest['samples']:
        folder = args.output / f'sample-{row["id"]:02d}'
        try:
            result = audit_sample(row, args.assets, args.round / f'sample-{row["id"]:02d}', folder, args.connection, args.reuse_mappings)
        except Exception as exc:
            folder.mkdir(exist_ok=True)
            result = {'id': row['id'], 'stratum': row['stratum'], 'title': row['title'], 'qualified': False,
                'status': 'audit_inconclusive', 'error': f'{type(exc).__name__}: {exc}'}
        if folder.exists():
            save(folder / 'result.json', result)
        results.append(result)
        save(args.output / 'results.json', {'scope': protocol['scope'], 'planned': 20, 'terminal': len(results),
            'qualified': sum(r['qualified'] for r in results), 'exploratory_verified_yield': sum(r['qualified'] for r in results) / 20 if len(results) == 20 else None,
            'seconds': time.monotonic() - started, 'samples': results})
        print(json.dumps({k: result.get(k) for k in ('id', 'qualified', 'status', 'error')}, ensure_ascii=False), flush=True)
    require(all(digest(args.round / p) == h for p, h in seal['files'].items()), 'posthoc audit altered original seal')
    hashes = {str(p.relative_to(args.output)): digest(p) for p in args.output.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    save(args.output / 'sealed_manifest.json', {'files': hashes, 'scope': protocol['scope']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
