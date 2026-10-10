"""Explicit V2 readiness certificate, separate from executable qualification."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .contract import compile_contract, require
from .export import verify_bundle
from .expressions import canonical, evaluate
from .pipeline import implementation_identity


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity():
    from rl.environments import v2
    from rl.data import contracts, trajectory
    return {'v2_platform': implementation_identity(),
            'adapter': digest(v2.__file__), 'segments': digest(contracts.__file__),
            'trajectory': digest(trajectory.__file__), 'protocol': 'v2-local-http-1'}


def verify_delivery(root):
    root = Path(root)
    manifest = verify_bundle(root)
    delivery = json.loads((root / 'delivery.json').read_text())
    source = compile_contract(json.loads((root / 'contract.json').read_text()))
    gates = delivery.get('gates', {})
    require(delivery.get('qualified') is True and set(gates) == {
        'structural_contract', 'joint_execution_and_counterexamples', 'independent_business_challenge',
        'deterministic_reset_and_replay', 'container_execution'} and all(v is True for v in gates.values()), 'V2 delivery not qualified')
    require(delivery.get('platform_identity') == implementation_identity(), 'V2 platform changed; requalification required')
    require(delivery.get('bundle_identity') == manifest['bundle_identity']
            and delivery.get('contract_identity') == source['identity'] == manifest['contract_identity'], 'V2 delivery identity mismatch')
    container = delivery.get('container') or {}
    require(container.get('execution', {}).get('passed') is True
            and container.get('execution', {}).get('http_api', {}).get('passed') is True
            and container.get('bundle_identity') == manifest['bundle_identity'], 'V2 container evidence missing')
    return {'bundle': manifest['bundle_identity'], 'contract': source['identity'],
            'delivery_sha256': digest(root / 'delivery.json'), 'manifest_sha256': digest(root / 'manifest.json')}


def verify_training_bundle(root):
    root = Path(root)
    delivery = verify_delivery(root)
    certificate = json.loads((root / 'training.json').read_text())
    require(certificate.get('version') == '1.0' and certificate.get('training_ready') is True,
            'V2 training readiness not certified')
    require(certificate.get('delivery') == delivery and certificate.get('runtime') == runtime_identity(),
            'V2 training certificate dependencies changed')
    require(certificate.get('conformance', {}).get('passed') is True
            and certificate.get('privacy', {}).get('passed') is True, 'V2 readiness evidence missing')
    unsigned = {k: v for k, v in certificate.items() if k != 'identity'}
    require(certificate.get('identity') == hashlib.sha256(canonical(unsigned).encode()).hexdigest(), 'V2 certificate identity mismatch')
    return certificate


def certify(root):
    from rl.environments.v2 import V2Episode
    from env_factory.evidence.material_privacy import _forbidden_key_paths
    root = Path(root)
    delivery = verify_delivery(root)
    source = compile_contract(json.loads((root / 'contract.json').read_text()))['source']
    scenario = next(s for s in source['scenarios'] if s['expect_success'] and not s.get('mutations') and not s.get('user_overrides'))
    environment = V2Episode(root, _qualification_only=True)
    steps = []
    def call(name, params, identifier):
        if name in environment.names:
            params = {**params['arguments'], **({'_v2_cursor': params['cursor']} if 'cursor' in params else {})}
        action = {'role': 'assistant', 'content': '', 'tool_calls': [{'id': identifier, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(params, ensure_ascii=False)}}]}
        environment.step(action)
        result = json.loads(environment.messages[-1]['content'])
        require(result['status'] == 200, 'V2 adapter conformance request failed')
        steps.append({'tool': name, 'id': identifier})
        return result['tool_result']
    try:
        environment.reset(0)
        initial_messages = copy.deepcopy(environment.messages)
        for index, step in enumerate(scenario['steps']):
            kind = step['kind']
            if kind in {'call', 'traverse'}:
                if kind == 'call':
                    call(step['tool'], {'arguments': step['arguments'],
                         **({'cursor': step['cursor']} if 'cursor' in step else {})}, step['id'])
                else:
                    pages, cursor = [], None
                    while True:
                        identifier = step['id'] + '_page' + str(len(pages))
                        value = call(step['tool'], {'arguments': step['arguments'], 'cursor': cursor}, identifier)
                        pages.append(identifier)
                        cursor = value['next_cursor']
                        if cursor is None:
                            break
                    call('v2_collect', {'pages': pages}, step['id'])
            elif kind == 'user':
                call('v2_user', {k: v for k, v in step.items() if k in {'stage', 'question', 'proposal', 'source_id'}}, f'user_{index}')
            elif kind == 'wait':
                call('v2_wait', {'seconds': step['seconds']}, f'wait_{index}')
            elif kind == 'revoke':
                call('v2_revoke', {'stage': step['stage']}, f'revoke_{index}')
        world = environment.server.episodes[environment.episode]
        answer = {key: evaluate(expr, world.context(private=True)) for key, expr in scenario['answer'].items()}
        environment.step(json.dumps(answer, ensure_ascii=False))
        outcome = environment.finish()
        require(outcome['score']['success'] is True, 'V2 adapter reference conformance not full success')
        privacy_errors = list(_forbidden_key_paths({'messages': environment.messages, 'tools': environment.tools}))
        require(not privacy_errors, 'V2 policy payload contains trainer-only fields')
        # The initial policy view is exactly compiler-owned public context plus
        # platform instructions. No initial fixture/private/reference rows.
        require(json.loads(initial_messages[1]['content']) == environment.public, 'V2 initial policy boundary changed')
        certificate = {'version': '1.0', 'training_ready': True, 'delivery': delivery,
            'runtime': runtime_identity(), 'supported_backend': 'local_http',
            'conformance': {'passed': True, 'scenario': scenario['id'], 'steps': steps,
                            'reward': outcome['final_reward'], 'scope': 'trusted_reference_adapter_check_not_policy_rollout'},
            'privacy': {'passed': True, 'scope': 'public_prompt_tools_and_actual_reference_messages'},
            'seed_semantics': 'contract-fixed initial world; seed identifies policy rollout group',
            'learning_verified': False,
            'user_simulation': {'runtime_default': 'free_dialogue',
                'reference_certification': 'structured_only', 'free_dialogue_rollout_verified': False}}
        certificate['identity'] = hashlib.sha256(canonical(certificate).encode()).hexdigest()
        (root / 'training.json').write_text(json.dumps(certificate, ensure_ascii=False, indent=2) + '\n')
        return verify_training_bundle(root)
    finally:
        environment.close()
