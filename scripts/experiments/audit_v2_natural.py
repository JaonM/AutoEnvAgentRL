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
    evaluate, grade, certify, V2Episode, expected, equivalent, canonical, sorted_data)
from v2_natural_semantics import schema, normalize, rename_observations, wrong_projection


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
    actual = normalize(answer, entries, name_aliases=aliases)
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
            answer[e['selectors'][0]['field']] = copy.deepcopy(value)
    return answer


def rollout(bundle, row, truth, entries, aliases, agent, root):
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
        checks = value_checks(env.answer or {}, truth, entries, aliases) + observation_checks(world, truth, row['kind'])
        checks.append({'id': 'full_business_state', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(truth['final_state']))})
        return {'passed': env.terminated and outcome['final_reward'] == 1 and outcome['score']['success'] and all(c['passed'] for c in checks),
            'answer': env.answer, 'outcome': outcome, 'checks': checks, 'messages': env.messages,
            'seconds': time.monotonic() - start, 'scope': 'fresh_native_policy_on_unchanged_original_bundle_posthoc_audit'}
    finally:
        env.close()


def audit_sample(row, assets, original, root, connection):
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
    from env_factory_v2.fixtures import review_contract
    source, previews = review_contract(compiled['source'])
    agent = NativeCodeAgent.from_connection(connection, evidence_root=root / 'native')
    review, provenance = agent.ask(
        'Audit business meaning against the natural request and external dossier. Map every required '
        'semantic role exactly once using declared answer fields and closed dot paths (numeric array '
        'indices or * for array projection). Multiple roles may use different paths of the SAME object '
        'field. Ordered results may be split across fields; packing=list concatenates in declared order. '
        'Duration may be minutes/seconds and converts to canonical hours. Public lecturer IDs may map '
        'to names using only the dossier aliases. Do not compute values, reorder results, invent missing '
        'facts or edit the sandbox. Task did not require an extra committed/submitted boolean; actual '
        'commit correctness is independently checked against full state and trace. Do not require flat '
        'fields or one output layout. Identify genuine missing business obligations. No tools.',
        {'raw_request': row['raw_request'], 'business': business, 'private_user_fixture': user,
         'contract': source, 'fixture_previews': previews, 'public': compiled['public'],
         'required_roles': list(truth['answer']), 'public_name_aliases': aliases},
        schema(truth['answer'], [f['id'] for f in compiled['source']['answers']]))
    save(root / 'mapping.json', {'response': review, 'provenance': provenance})
    entries = review['mapping']
    require(len(entries) == len(truth['answer']) and {e['role'] for e in entries} == set(truth['answer']), 'posthoc mapping incomplete')
    checks, selected = [], None
    for scenario in compiled['source']['scenarios']:
        if not scenario['expect_success'] or scenario.get('mutations'):
            continue
        world = execute_reference(compiled, scenario)
        answer = reference_answer(world, scenario)
        current = value_checks(answer, truth, entries, aliases) + observation_checks(world, truth, spec['kind'])
        current.append({'id': 'full_business_state', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(truth['final_state']))})
        if all(c['passed'] for c in current):
            selected, checks = scenario, current
            break
    require(selected is not None, 'no independently correct baseline reference execution')
    from env_factory_v2.contract import ContractError
    for entry in entries:
        world = execute_reference(compiled, selected)
        answer = wrong_projection(reference_answer(world, selected), entry)
        outcome = grade(world, answer, judge=agent.judge)
        checks.append({'id': 'independent_wrong_' + entry['role'],
                       'passed': outcome['reward'] < 1 and not outcome['success'], 'outcome': outcome})
    if spec['mutation']:
        for stale in [False, True]:
            world = execute_reference(compiled, selected, mutation=spec['mutation'])
            changed_truth = canonical_truth(spec, business, user, mutation=spec['mutation'])
            answer = (reference_answer(execute_reference(compiled, selected), selected) if stale else reference_answer(world, selected))
            outcome = grade(world, answer, judge=agent.judge)
            passed = outcome['reward'] < 1 and not outcome['success'] if stale else (
                outcome['reward'] == 1 and outcome['success']
                and all(c['passed'] for c in value_checks(answer, changed_truth, entries, aliases)))
            checks.append({'id': 'independent_stale_answer' if stale else 'independent_changed_fact', 'passed': passed, 'outcome': outcome})
    for label, scenario, literal in [('caller_id_invariance', rename_observations(selected), False),
                                      ('correct_literal_reward', selected, True)]:
        world = execute_reference(compiled, scenario)
        answer = reference_answer(world, scenario)
        if literal:
            answer = scalar_literals(answer, truth, entries)
        outcome = grade(world, answer, judge=agent.judge)
        check = {'id': label, 'passed': outcome['reward'] == 1 and outcome['success'], 'answer': answer, 'outcome': outcome}
        checks.append(check)
    # Use independently computed canonical answers for scalar fields during refusal;
    # retain independently inspected authored structures for nested fields.
    if any(t['kind'] == 'transaction' for t in compiled['source']['tools']):
        world = execute_reference(compiled, selected, refuse=True)
        refused_truth = canonical_truth(spec, business, user, refuse=True)
        answer = reference_answer(world, selected)
        outcome = grade(world, answer, judge=agent.judge)
        checks.append({'id': 'actual_user_decline', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(refused_truth['initial']))
            and all(c['passed'] for c in value_checks(answer, refused_truth, entries, aliases))
            and outcome['reward'] == 1 and outcome['success'], 'answer': answer, 'outcome': outcome})
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
    if old_rollout.exists():
        old = read(old_rollout)
        result['policy_evidence'] = {'source': str(old_rollout), 'sha256': digest(old_rollout), 'reused': True}
        policy_passed = old['outcome']['score']['success'] and old['outcome']['final_reward'] == 1
        policy_passed = policy_passed and all(c['passed'] for c in value_checks(old['answer'] or {}, truth, entries, aliases))
        policy_passed = policy_passed and all(c['passed'] for c in old['independent_checks'] if not c['id'].startswith('answer_'))
        result['policy_success'] = bool(policy_passed)
    else:
        policy = rollout(copy_bundle, {**row, 'kind': spec['kind']}, truth, entries, aliases, agent, root)
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
    args = parser.parse_args()
    seal = read(args.round / 'sealed_manifest.json')
    require(all(digest(args.round / p) == h for p, h in seal['files'].items()), 'original seal changed')
    manifest = verify_assets(args.assets)
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {'scope': 'posthoc_exploratory_measurement_correction_not_fresh_blind_yield',
        'original_seal_sha256': digest(args.round / 'sealed_manifest.json'), 'planned': 20,
        'source_repairs': 0, 'changes': ['closed nested/split answer projections', 'duration unit normalization',
            'public lecturer ID aliases', 'no unrequested submission boolean', 'decimal canonical scalar positives',
            'caller-owned observation-ID invariance', 'actual user-decline verification'],
        'limitation': 'Extra independent audit and some additional policy calls occur after viewing results; exploratory evidence only.'}
    save(args.output / 'protocol.json', protocol)
    results, started = [], time.monotonic()
    for row in manifest['samples']:
        folder = args.output / f'sample-{row["id"]:02d}'
        try:
            result = audit_sample(row, args.assets, args.round / f'sample-{row["id"]:02d}', folder, args.connection)
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
