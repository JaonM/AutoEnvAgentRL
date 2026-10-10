"""Frozen natural-demand pilot with external truth and actual native policy rollout.

Synthetic, operator-designed sampling. No production population claim. No
acceptance fixture or reference trajectory is visible to author/policy roles.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'src'))
from env_factory_v2.code_agent import NativeCodeAgent, object_schema
from env_factory_v2.contract import compile_contract, require, validate_value
from env_factory_v2.expressions import ContractError, canonical, evaluate
from env_factory_v2.pipeline import build, implementation_identity
from env_factory_v2.runtime import Sandbox
from env_factory_v2.reward import grade
from env_factory_v2.training import certify
from rl.environments.v2 import V2Episode
from v2_natural_oracle import expected, expand, equivalent, wrong_value, UNORDERED


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runner_identity():
    return {p.name: digest(p) for p in [Path(__file__), Path(__file__).with_name('v2_natural_oracle.py'),
        PROJECT / 'src/rl/environments/v2.py', PROJECT / 'src/rl/data/contracts.py',
        PROJECT / 'src/rl/data/trajectory.py']}


def verify_assets(root):
    frozen = read(root / 'frozen.json')['files']
    require(all(digest(root / path) == checksum for path, checksum in frozen.items()), 'natural pilot assets changed')
    manifest = read(root / 'manifest.json')
    require(len(manifest['samples']) == 20 and Counter(r['stratum'] for r in manifest['samples']) == manifest['mix']
            == {'direct': 4, 'simple': 4, 'ordinary': 4, 'complex': 8}, 'natural pilot mix invalid')
    return manifest


def sorted_data(data):
    return {table: sorted(rows, key=lambda r: canonical(r['id'])) for table, rows in data.items()}


def fixture_alignment(source, row, business, user):
    compiled = compile_contract(source)
    require(source['task'] == row['raw_request'], 'public task must preserve original natural request verbatim')
    require(source['user']['facts'] == user['facts'], 'generated user facts differ from frozen simulator facts')
    actual = {t['id']: t['rows'] for t in compiled['source']['tables']}
    require(canonical(sorted_data(actual)) == canonical(sorted_data(expand(business))),
            'generated business records differ from frozen external business data')
    if row['stratum'] == 'complex':
        for table in business['tables']:
            if 'generator' in table:
                tools = [t for t in compiled['source']['tools'] if t.get('table') == table['id']]
                cap = 64 if table['id'] == 'expenses' else 128
                require(tools and all(t['page_size'] <= cap for t in tools), 'business pagination limit changed')
    public_text = canonical(compiled['public'])
    allowed_text = row['raw_request'] + business['policy']
    for value in user['facts'].values():
        if isinstance(value, str) and len(value) >= 2 and value not in allowed_text:
            require(value not in public_text, 'private user value leaked into initial public interface')
    return compiled


def execute_reference(compiled, scenario, *, mutation=None, refuse=False):
    source = copy.deepcopy(compiled['source'])
    if refuse:
        for name in source['user']['facts']:
            if name.startswith('approve_'):
                source['user']['facts'][name] = False
    world = Sandbox(source)
    world.reset([mutation] if mutation else [])
    for step in scenario['steps']:
        kind = step['kind']
        if kind in {'call', 'traverse'}:
            tool = world.tools[step['tool']]
            if refuse and tool['kind'] == 'transaction':
                continue
            (world.call if kind == 'call' else world.traverse)(step['tool'], step['arguments'], call_id=step['id'],
                **({'cursor': world.resolve(step['cursor'])} if 'cursor' in step else {}))
        elif kind == 'user':
            world.interact(step['stage'], question=step['question'], proposal=step.get('proposal'), source_id=step.get('source_id'))
        elif kind == 'wait':
            world.wait(step['seconds'])
        elif kind == 'revoke':
            world.revoke(step['stage'])
    return world


def observation_checks(world, truth, kind):
    checks = []
    changed = {table: {r['id']: {k for k, v in r.items() if v != next(x for x in truth['initial'][table] if x['id'] == r['id'])[k]}
                       for r in rows if r != next(x for x in truth['initial'][table] if x['id'] == r['id'])}
               for table, rows in truth['final_state'].items()}
    for table, identifiers in truth['observations'].items():
        observed = set()
        for turn in world.trace:
            if turn.get('kind') == 'tool' and turn.get('status') == 'ok' and world.tools[turn['tool']].get('table') == table:
                observed.update(r.get('id') for r in turn['result']['items'])
        checks.append({'id': 'observed_' + table, 'passed': set(identifiers) <= observed})
    for table, records in changed.items():
        for identifier, columns in records.items():
            fresh = any(turn.get('kind') == 'tool' and turn.get('status') == 'ok'
                and world.tools[turn['tool']].get('table') == table
                and turn['result']['state_version'] == world.version
                and any(r.get('id') == identifier and columns <= set(r) for r in turn['result']['items'])
                for turn in world.trace)
            checks.append({'id': 'post_commit_' + table + '_' + identifier, 'passed': fresh})
    if kind == 'sequential':
        writes = [i for i, t in enumerate(world.trace) if t.get('kind') == 'tool' and t.get('status') == 'ok'
                  and world.tools[t['tool']]['kind'] == 'transaction']
        refresh = len(writes) == 2 and all(any(t.get('status') == 'ok' and t.get('kind') == 'tool'
            and world.tools[t['tool']].get('table') == table and any(column in r for r in t['result']['items'])
            for t in world.trace[writes[0] + 1:writes[1]]) for table, column in [('accounts', 'quota'), ('features', 'active')])
        checks.append({'id': 'refresh_between_transactions', 'passed': refresh})
    if kind == 'correction':
        corrections = [t for t in world.trace if t.get('kind') == 'user' and world.user.stages[t['stage']]['kind'] == 'correction']
        checks.append({'id': 'actual_user_correction', 'passed': bool(corrections)})
    if kind == 'enroll':
        writes = [t for t in world.trace if t.get('kind') == 'tool' and t.get('status') == 'ok'
                  and world.tools[t['tool']]['kind'] == 'transaction']
        checks.append({'id': 'wait_until_open', 'passed': bool(writes) and all(t['time']['elapsed_seconds'] >= 15 for t in writes)})
    return checks


def answer_checks(answer, truth, mapping):
    return [{'id': 'answer_' + role, 'passed': field in answer and equivalent(answer[field], value, unordered=role in UNORDERED)}
            for role, value in truth['answer'].items() for field in [mapping[role]]]


def external_acceptance(compiled, row, business, user, spec, agent):
    truth = expected(spec, business, user)
    roles = list(truth['answer'])
    schema = object_schema({'business_consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
        'mapping': {'type': 'array', 'items': object_schema({'role': {'type': 'string', 'enum': roles},
            'field': {'type': 'string', 'enum': [f['id'] for f in compiled['source']['answers']]}})}})
    from env_factory_v2.fixtures import review_contract
    source, previews = review_contract(compiled['source'])
    value, provenance = agent.ask(
        'Independently audit this sandbox against the original NATURAL user request and frozen business '
        'dossier. Do not treat generated rules, expected expressions or reference answers as business truth. '
        'Check meaning, privacy, missing obligations and extra restrictions. Map each evaluator semantic '
        'role exactly once to a distinct declared answer field. Map meanings only; do not invent answers '
        'or change business evidence. If a required role is absent or only a vague explanation can express '
        'it, business_consistent=false. No tools. The evaluator will independently calculate all values '
        'and execute positive and negative tests; your opinion alone never proves qualification.',
        {'raw_request': row['raw_request'], 'business': business, 'private_user_fixture': user,
         'contract': source, 'fixture_previews': previews, 'compiled_public': compiled['public'],
         'semantic_roles': roles}, schema)
    mapping = {entry['role']: entry['field'] for entry in value['mapping']}
    require(len(value['mapping']) == len(roles) and set(mapping) == set(roles)
            and len(set(mapping.values())) == len(roles), 'independent answer mapping incomplete or ambiguous')
    scenarios = [s for s in compiled['source']['scenarios'] if s['expect_success'] and not s.get('mutations')]
    checks, selected = [], None
    for scenario in scenarios:
        try:
            world = execute_reference(compiled, scenario)
            authored_answer = {k: evaluate(e, world.context(private=True)) for k, e in scenario['answer'].items()}
            current = answer_checks(authored_answer, truth, mapping) + observation_checks(world, truth, spec['kind'])
            current.append({'id': 'full_business_state', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(truth['final_state']))})
            current.append({'id': 'positive_reward', 'passed': grade(world, authored_answer, judge=agent.judge)['reward'] == 1})
            if all(c['passed'] for c in current):
                selected, checks = scenario, current
                break
            checks = current
        except (ContractError, KeyError, TypeError, ValueError) as exc:
            checks = [{'id': 'external_positive_execution', 'passed': False, 'error': str(exc)}]
    if selected is not None:
        # Independently wrong, schema-valid answers must not receive full reward.
        for role, field in mapping.items():
            world = execute_reference(compiled, selected)
            answer = {k: evaluate(e, world.context(private=True)) for k, e in selected['answer'].items()}
            answer[field] = wrong_value(truth['answer'][role])
            try:
                validate_value(answer[field], next(f for f in compiled['source']['answers'] if f['id'] == field)['schema'])
                outcome = grade(world, answer, judge=agent.judge)
                checks.append({'id': 'wrong_' + role, 'passed': outcome['reward'] < 1 and outcome['success'] is False})
            except ContractError as exc:
                # A schema rejection is still a real guard, but label it separately.
                checks.append({'id': 'wrong_' + role, 'passed': True, 'protection': 'schema', 'error': str(exc)})
        if spec['mutation']:
            for stale in [False, True]:
                world = execute_reference(compiled, selected, mutation=spec['mutation'])
                changed = expected(spec, business, user, mutation=spec['mutation'])
                answer = {k: evaluate(e, world.context(private=True)) for k, e in selected['answer'].items()}
                for role, field in mapping.items():
                    answer[field] = (truth if stale else changed)['answer'][role]
                score = grade(world, answer, judge=agent.judge)
                checks.append({'id': 'stale_changed_fact' if stale else 'correct_changed_fact',
                    'passed': (score['reward'] < 1 and not score['success']) if stale else score['reward'] == 1 and score['success']})
        if any(t['kind'] == 'transaction' for t in compiled['source']['tools']):
            try:
                world = execute_reference(compiled, selected, refuse=True)
                refused_truth = expected(spec, business, user, refuse=True)
                answer = {k: evaluate(e, world.context(private=True)) for k, e in selected['answer'].items()}
                for role, field in mapping.items():
                    answer[field] = refused_truth['answer'][role]
                outcome = grade(world, answer, judge=agent.judge)
                checks.append({'id': 'actual_refusal_no_write', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(refused_truth['initial']))
                    and outcome['success'] is True and outcome['reward'] == 1})
            except (ContractError, KeyError, TypeError, ValueError) as exc:
                checks.append({'id': 'actual_refusal_no_write', 'passed': False, 'error': str(exc)})
            # Correct claimed final values without actual writes must fail.
            world = Sandbox(compiled)
            answer = {mapping[k]: v for k, v in truth['answer'].items()}
            for field in compiled['source'].get('explanations', []):
                answer[field['id']] = '尚未执行任何查询或提交。'
            outcome = grade(world, answer, judge=agent.judge)
            checks.append({'id': 'claimed_write_without_execution', 'passed': outcome['reward'] < 1 and not outcome['success']})
    return {'passed': value['business_consistent'] and selected is not None and all(c['passed'] for c in checks),
            'semantic_review': value, 'provenance': provenance, 'mapping': mapping, 'checks': checks,
            'reference': selected['id'] if selected else None,
            'scope': 'external_business_truth_not_generated_reward_targets'}


ACTION_SCHEMA = object_schema({'calls': {'type': 'array', 'items': object_schema({
    'id': {'type': 'string'}, 'name': {'type': 'string'}, 'arguments_json': {'type': 'string'}})}})


def policy_rollout(bundle, row, business, user, spec, mapping, agent, root):
    env = V2Episode(bundle)
    started = time.monotonic()
    try:
        env.reset(row['id'])
        for turn in range(40):
            value, provenance = agent.ask(
                'Act as the agent fulfilling the natural user request using ONLY supplied public tools '
                'and actual observations. Hidden business records, user preferences, reference trajectories '
                'and evaluator targets are unavailable. Ask declared user stages when information or '
                'authorization is missing. Use $result and $user bindings with actual observation/stage '
                'IDs. You may return 1..8 calls whose dependencies refer to earlier actual observations '
                'or earlier calls in this same batch, never invented results. Pagination must use actual '
                'returned cursors; stop a batch when you need to inspect a new value before deciding. '
                'Return tool intent only, no Codex tools. The host executes intents through the real '
                'training HTTP adapter and returns observations. Finish with v2_submit using declared '
                'fields. Do not consult acceptance data or run tools outside this interface.',
                {'messages': env.messages, 'tools': env.tools, 'remaining_turns': 40 - turn}, ACTION_SCHEMA)
            require(1 <= len(value['calls']) <= 8, 'policy call batch must contain 1..8 calls')
            calls = []
            for call in value['calls']:
                params = json.loads(call['arguments_json'])
                require(isinstance(params, dict), 'policy arguments must be objects')
                calls.append({'id': call['id'], 'type': 'function',
                    'function': {'name': call['name'], 'arguments': json.dumps(params, ensure_ascii=False)}})
            env.step({'role': 'assistant', 'content': '', 'tool_calls': calls})
            save(root / 'rollout_progress.json', {'turns': turn + 1, 'terminated': env.terminated, 'seconds': time.monotonic() - started})
            if env.terminated:
                break
        outcome = env.finish()
        world = env.server.episodes[env.episode]
        truth = expected(spec, business, user)
        checks = answer_checks(env.answer or {}, truth, mapping) + observation_checks(world, truth, spec['kind'])
        checks.append({'id': 'full_business_state', 'passed': canonical(sorted_data(world.data)) == canonical(sorted_data(truth['final_state']))})
        success = env.terminated and outcome['score']['success'] and outcome['final_reward'] == 1 and all(c['passed'] for c in checks)
        result = {'passed': bool(success), 'outcome': outcome, 'independent_checks': checks,
                  'messages': env.messages, 'tools': env.tools, 'answer': env.answer,
                  'seconds': time.monotonic() - started, 'policy_input_boundary': 'public_interface_and_actual_observations_only',
                  'scope': 'actual_native_policy_http_rollout_not_reference_replay'}
        save(root / 'rollout.json', result)
        return result
    finally:
        env.close()


def worker(args):
    assets, root = args.assets.resolve(), args.output.resolve()
    manifest = verify_assets(assets)
    row = next(r for r in manifest['samples'] if r['id'] == args.worker)
    business, user, spec = (read(assets / row[k]) for k in ('business', 'user', 'acceptance'))
    root.mkdir(parents=True, exist_ok=False)
    result = {'id': row['id'], 'qualified': False, 'sandbox_qualified': False, 'external_business_qualified': False,
              'training_ready': False, 'policy_success': None, 'status': 'running', 'phases': {}}
    start = time.monotonic()
    phase_name = 'business_retrieval'
    def phase(name):
        nonlocal phase_name
        phase_name = name
        result['active_phase'] = name
        save(root / 'result.json', result)
        return time.monotonic()
    try:
        base = NativeCodeAgent.from_connection(args.connection, evidence_root=root / 'native')
        tick = phase('business_retrieval')
        catalog = [{'document_id': candidate['business'], 'title': candidate['title'],
                    'policy_summary': read(assets / candidate['business'])['policy']}
                   for candidate in manifest['samples']]
        selected, retrieval_provenance = base.ask(
            'Select the business dossier needed to ground this natural request from the supplied catalog. '
            'Use request meaning, not ordering. Do not invent a document or execute tools.',
            {'raw_request': row['raw_request'], 'catalog': catalog},
            object_schema({'document_id': {'type': 'string', 'enum': [c['document_id'] for c in catalog]},
                           'reason': {'type': 'string'}}))
        save(root / 'retrieval.json', {'selected': selected, 'provenance': retrieval_provenance,
            'sha256': digest(assets / selected['document_id']), 'method': 'native_selection_from_frozen_business_catalog',
            'knowledge_graph_used': False})
        require(selected['document_id'] == row['business'], 'business retrieval selected an unrelated dossier')
        result['phases'][phase_name] = time.monotonic() - tick
        tick = phase('demand_interpretation')
        interpretation, provenance = base.ask(
            'Interpret this natural user request against the retrieved business dossier. Return a concise '
            'public business intent and missing user information names. Do not invent facts, tools, '
            'rules, answers or implementation requirements. Resolve business rules only from the dossier. '
            'A request requiring runtime clarification is supported, not invalid. No tools.',
            {'raw_request': row['raw_request'], 'business_policy': business['policy']},
            object_schema({'supported': {'type': 'boolean'}, 'intent': {'type': 'string'},
                'missing_user_information': {'type': 'array', 'items': {'type': 'string'}}, 'reason': {'type': 'string'}}))
        save(root / 'interpretation.json', {'response': interpretation, 'provenance': provenance})
        require(interpretation['supported'], 'demand interpreter could not ground the natural request')
        result['phases'][phase_name] = time.monotonic() - tick
        request = {'category': row['category'], 'goal': row['raw_request'], 'title': row['title'],
            'business_evidence': business, 'private_user_fixture': user,
            'demand_interpretation': interpretation,
            'requirements': '实现原始自然需求和给定业务规则。task必须逐字保留goal。业务表id、字段、记录和生成计划来自固定外部资料，不得增删或改值；这些记录只能通过工具观察，不能发布到初始提示。user.facts必须恰好等于private_user_fixture.facts，私有事实仅通过模拟用户披露，approve_STAGE对应授权stage的id。不能自造授权或偏好。其他用户阶段、工具、答案字段、评分与场景由生成器设计。公开规则须说明实际业务政策；时间基准2026-10-10T00:00:00+00:00，每动作0秒，等待显式推进。不得使用固定答案替代真实观察。'}
        save(root / 'author_request.json', request)
        class GroundedAuthor(NativeCodeAgent):
            def generate(self, request, *, feedback=None):
                source, provenance = super().generate(request, feedback=feedback)
                fixture_alignment(source, row, business, user)
                return source, provenance
        author = GroundedAuthor.from_connection(args.connection, evidence_root=root / 'author-native')
        tick = phase('sandbox_generation_and_delivery')
        delivery = build(request, agent=author, output=root / 'build', repairs=1, docker=True, base_image=args.base_image)
        result['phases'][phase_name] = time.monotonic() - tick
        result['sandbox_qualified'] = delivery.get('qualified') is True
        require(result['sandbox_qualified'], 'sandbox pipeline did not qualify')
        bundle = root / 'build/sandbox'
        compiled = fixture_alignment(read(bundle / 'contract.json'), row, business, user)
        tick = phase('external_business_acceptance')
        acceptance = external_acceptance(compiled, row, business, user, spec, base)
        save(root / 'external_acceptance.json', acceptance)
        result['phases'][phase_name] = time.monotonic() - tick
        result['external_business_qualified'] = acceptance['passed']
        require(acceptance['passed'], 'independent external business acceptance failed')
        tick = phase('training_readiness')
        os.environ['ENVFACTORY_V2_CONNECTION'] = str(args.connection.resolve())
        certificate = certify(bundle)
        result['training_ready'] = certificate['training_ready'] is True
        result['phases'][phase_name] = time.monotonic() - tick
        tick = phase('native_policy_rollout')
        rollout = policy_rollout(bundle, row, business, user, spec, acceptance['mapping'], base, root)
        result['phases'][phase_name] = time.monotonic() - tick
        result['policy_success'] = rollout['passed']
        # Conservative verified yield. Policy failure is inconclusive about the
        # sandbox, and must be reported separately rather than called its defect.
        result['qualified'] = rollout['passed']
        result['status'] = 'passed' if result['qualified'] else 'policy_validation_inconclusive'
        if not result['qualified']:
            result['failed_phase'] = phase_name
    except Exception as exc:
        result.update(status='failed', failed_phase=phase_name, error=f'{type(exc).__name__}: {exc}')
    result.pop('active_phase', None)
    result['seconds'] = time.monotonic() - start
    save(root / 'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('id', 'qualified', 'status', 'failed_phase', 'error', 'seconds')}, ensure_ascii=False), flush=True)
    return 0 if result['qualified'] else 1


def summarize(protocol, samples, wall):
    terminal = sum(s['status'] not in {'pending', 'running'} for s in samples)
    names = ['sandbox_qualified', 'external_business_qualified', 'training_ready', 'qualified']
    counts = {name: sum(s.get(name) is True for s in samples) for name in names}
    return {'scope': protocol['scope'], 'planned': 20, 'terminal': terminal, **counts,
        'verified_end_to_end_yield': counts['qualified'] / 20 if terminal == 20 else None,
        'sandbox_pipeline_yield': counts['sandbox_qualified'] / 20 if terminal == 20 else None,
        'external_business_yield': counts['external_business_qualified'] / 20 if terminal == 20 else None,
        'policy_attempted': sum(s.get('policy_success') is not None for s in samples),
        'policy_successes': sum(s.get('policy_success') is True for s in samples),
        'groups': {g: {'planned': n, 'qualified': sum(s.get('qualified') is True for s in samples if s['stratum'] == g)}
                   for g, n in protocol['mix'].items()}, 'wall_seconds': wall, 'samples': samples,
        'limitation': 'Synthetic operator-designed natural demands, fixed 4:4:4:8 mix; not production population or independent double-blind sampling. Policy failures are inconclusive sandbox validation, separately classified.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--base-image', default='docker.m.daocloud.io/library/python:3.14-slim')
    parser.add_argument('--sample-timeout', type=int, default=1800)
    parser.add_argument('--worker', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        return worker(args)
    manifest = verify_assets(args.assets.resolve())
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    protocol = {'version': 1, 'scope': manifest['scope'], 'planned': 20, 'mix': manifest['mix'],
        'sampling': manifest['sampling'], 'model': 'gpt-6.1-sol', 'concurrency': 1, 'candidate_repairs': 1,
        'platform_identity': implementation_identity(), 'runner_identity': runner_identity(),
        'assets_identity': digest(args.assets / 'frozen.json'), 'base_image': args.base_image,
        'bridge_timeout_seconds': read(args.connection)['timeout_seconds'], 'sample_timeout_seconds': args.sample_timeout,
        'policy_turn_budget': 40, 'policy_calls_per_turn': 8,
        'policy': 'All 20 natural requests stay in denominator. No within-round fixes, terminal retries, hidden-oracle feedback to author or policy, or reference-script policy replay.',
        'qualification': 'Five delivery gates + external frozen business acceptance + training certificate + actual native policy HTTP rollout consistent with independent truth.',
        'acceptance_budget': 'One external semantic mapping review, all deterministic tests; no acceptance-driven author repair.'}
    save(root / 'protocol.json', protocol)
    samples = [{k: row[k] for k in ('id', 'title', 'stratum', 'raw_request')} | {'status': 'pending', 'qualified': False}
               for row in manifest['samples']]
    start = time.monotonic()
    def publish():
        save(root / 'results.json', summarize(protocol, samples, time.monotonic() - start))
    publish()
    for sample in samples:
        verify_assets(args.assets.resolve())
        require(implementation_identity() == protocol['platform_identity'] and runner_identity() == protocol['runner_identity']
                and digest(args.assets / 'frozen.json') == protocol['assets_identity'], 'experiment implementation or evidence changed')
        sample['status'] = 'running'
        publish()
        folder = root / f'sample-{sample["id"]:02d}'
        command = [sys.executable, str(Path(__file__).resolve()), '--assets', str(args.assets.resolve()),
            '--output', str(folder), '--connection', str(args.connection.resolve()), '--base-image', args.base_image,
            '--worker', str(sample['id'])]
        tick = time.monotonic()
        with (root / f'sample-{sample["id"]:02d}.log').open('w') as stream:
            try:
                completed = subprocess.run(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, timeout=args.sample_timeout)
                exit_code = completed.returncode
            except subprocess.TimeoutExpired:
                exit_code = None
        if (folder / 'result.json').exists():
            value = read(folder / 'result.json')
            if exit_code is None or value.get('status') == 'running':
                value.update(status='timeout', qualified=False, failed_phase=value.get('active_phase'))
            sample.update(value)
        else:
            sample.update(status='process_failed', error='Worker did not publish results', qualified=False)
        sample.update(exit_code=exit_code, seconds=time.monotonic() - tick)
        publish()
        print(json.dumps({k: sample.get(k) for k in ('id', 'title', 'status', 'qualified', 'failed_phase', 'seconds')}, ensure_ascii=False), flush=True)
    require(implementation_identity() == protocol['platform_identity'] and runner_identity() == protocol['runner_identity'], 'implementation changed before seal')
    hashes = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*'))
              if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    save(root / 'sealed_manifest.json', {'scope': 'completed_natural_demand_pilot_evidence', 'files': hashes})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
