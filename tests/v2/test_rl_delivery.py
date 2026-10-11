"""Adapter/certificate unit checks; mocked qualification is never yield evidence."""
import copy
import json

import pytest

from env_factory_v2.contract import ContractError
from env_factory_v2.examples import direct, simple, transaction
from env_factory_v2.export import export_bundle, verify_bundle
from env_factory_v2.verification import qualify
from env_factory_v2 import training
from rl.environments.v2 import V2Episode
from rl.data.tasks import load_tasks
from rl.data.contracts import attach_contract, validate_contract


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    def make(source):
        compiled, report = qualify(source)
        root = tmp_path / source['id']
        export_bundle(compiled, report, root)
        (root / 'delivery.json').write_text('{}')
        return root
    def unit_delivery(root):
        manifest = verify_bundle(root)
        return {'bundle': manifest['bundle_identity'], 'scope': 'unit_only_mocked_qualification'}
    monkeypatch.setattr(training, 'verify_delivery', unit_delivery)
    return make


def action(name, args, identifier):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': identifier, 'type': 'function',
        'function': {'name': name, 'arguments': json.dumps(args)}}]}


def test_training_certificate_executes_explicit_platform_time_step(bundle):
    source = direct()
    source['success'] = [{'id': 'clock', 'public': 'Read current virtual time.',
        'when': {'literal': True}, 'required_tools': [], 'required_stages': [],
        'required_platform_observations': [{'capability': 'time', 'freshness': 'current'}]}]
    for case in source['scenarios']:
        case['steps'] = [{'kind': 'time'}]
    root = bundle(source)
    assert training.certify(root)['training_ready']


def test_policy_cannot_manufacture_user_withdrawal_even_with_a_known_tool_name(bundle):
    environment = V2Episode(bundle(transaction()), _qualification_only=True)
    # Mocked qualification is unit-only. Exercise the ordinary policy dispatcher.
    environment._qualification_only = False
    try:
        environment.reset(0)
        assert 'v2_revoke' not in {t['function']['name'] for t in environment.tools}
        environment.step(action('accounts', {}, 'read'))
        args = {'from_id': {'$result': 'read', 'path': 'items.0.id'},
                'to_id': {'$result': 'read', 'path': 'items.1.id'}, 'amount': 20}
        environment.step(action('v2_user', {'stage': 'approve', 'question': '批准？',
            'proposal': {'tool': 'transfer', 'arguments': args}}, 'consent'))
        environment.step(action('v2_revoke', {'stage': 'approve'}, 'fabricated'))
        assert json.loads(environment.messages[-1]['content'])['status'] == 400
        world = environment.server.episodes[environment.episode]
        assert world.user.states['approve'] == 'consent.authorized'
        environment.step(json.dumps({'accounts': world.data['accounts']}))
        assert environment.finish()['final_reward'] == 0
    finally:
        environment.close()


def test_trusted_user_withdrawal_remains_available_to_reference_qualification(bundle):
    environment = V2Episode(bundle(transaction()), _qualification_only=True)
    try:
        environment.reset(0)
        environment.step(action('accounts', {}, 'read'))
        args = {'from_id': {'$result': 'read', 'path': 'items.0.id'},
                'to_id': {'$result': 'read', 'path': 'items.1.id'}, 'amount': 20}
        environment.step(action('v2_user', {'stage': 'approve', 'question': '批准？',
            'proposal': {'tool': 'transfer', 'arguments': args}}, 'consent'))
        environment.step(action('v2_revoke', {'stage': 'approve'}, 'actual_user_event'))
        assert json.loads(environment.messages[-1]['content'])['status'] == 200
        assert environment.server.episodes[environment.episode].user.states['approve'] == 'consent.revoked'
    finally:
        environment.close()


def test_direct_adapter_scores_actual_answer_and_finish_is_idempotent(bundle):
    source = direct()
    root = bundle(source)
    environment = V2Episode(root, _qualification_only=True)
    try:
        environment.reset(2)
        answer = {f['id']: f['expected']['literal'] for f in source['answers']}
        _, done = environment.step(json.dumps(answer))
        assert done and environment.finish()['final_reward'] == 1
        assert environment.finish() == environment.finish()
        with pytest.raises(ValueError, match='terminated'):
            environment.step('{}')
    finally:
        environment.close()


def test_policy_view_does_not_include_private_recipes_and_restore_is_exact(bundle):
    root = bundle(simple())
    environment = V2Episode(root, _qualification_only=True)
    restored = V2Episode(root, _qualification_only=True)
    try:
        environment.reset(9)
        assert set(json.loads(environment.messages[1]['content'])) == set(environment.public)
        assert 'scenarios' not in environment.public and 'facts' not in environment.public
        stage = environment.public['user_stages'][0]['id']
        environment.step(action('v2_user', {'stage': stage, 'question': '尺码？'}, 'ask'))
        snapshot = json.loads(json.dumps(environment.snapshot()))
        restored.restore(snapshot)
        assert restored.messages == environment.messages
        assert restored.server.episodes[restored.episode].user.completed == environment.server.episodes[environment.episode].user.completed
        snapshot['bundle_identity'] = 'wrong'
        with pytest.raises(ValueError, match='mismatch'):
            restored.restore(snapshot)
    finally:
        environment.close()
        restored.close()


def test_native_tool_calls_without_transport_ids_execute_actual_user_stage(bundle):
    root = bundle(simple())
    environment = V2Episode(root, _qualification_only=True)
    try:
        environment.reset(0)
        stage = environment.public['user_stages'][0]['id']
        native = action('v2_user', {'stage': stage, 'question': '尺码？'}, 'unused')
        del native['tool_calls'][0]['id']
        environment.step(native)
        reply = json.loads(environment.messages[-1]['content'])
        assert reply['status'] == 200
        assert stage in environment.server.episodes[environment.episode].user.completed
        assert environment.messages[-1]['tool_call_id'] == reply['observation_id']
    finally:
        environment.close()


def test_native_business_parameters_match_public_contract_and_finish_full_success(bundle):
    root = bundle(simple())
    environment = V2Episode(root, _qualification_only=True)
    try:
        environment.reset(0)
        ask = action('v2_user', {'stage': 'size', 'question': '尺码？'}, 'unused')
        del ask['tool_calls'][0]['id']
        environment.step(ask)
        disclosed = json.loads(environment.messages[-1]['content'])['tool_result']['value']
        native = action('find_items', {'size': disclosed}, 'unused')
        del native['tool_calls'][0]['id']
        environment.step(native)
        reply = json.loads(environment.messages[-1]['content'])
        assert reply['status'] == 200
        environment.step(json.dumps({'ids': [row['id'] for row in reply['tool_result']['items']]}))
        assert environment.finish()['final_reward'] == 1
    finally:
        environment.close()


def test_invalid_answer_and_truncation_are_policy_failure_not_infrastructure(bundle):
    root = bundle(direct())
    environment = V2Episode(root, _qualification_only=True)
    try:
        environment.reset(0)
        environment.step('not a JSON answer')
        assert environment.finish()['final_reward'] == 0 and environment.terminated
        environment.reset(0)
        assert environment.finish()['final_reward'] == 0 and not environment.terminated
    finally:
        environment.close()


def test_certificate_task_loader_and_tamper_rejection(bundle):
    root = bundle(direct())
    certificate = training.certify(root)
    assert certificate['training_ready'] and not certificate['learning_verified']
    assert load_tasks(sandbox=str(root))[0].identity == certificate['identity']
    changed = copy.deepcopy(certificate)
    changed['training_ready'] = False
    (root / 'training.json').write_text(json.dumps(changed))
    with pytest.raises(ContractError, match='readiness'):
        training.verify_training_bundle(root)
    (root / 'training.json').write_text(json.dumps(certificate))
    (root / 'public.json').write_text('{}')
    with pytest.raises(ContractError, match='changed'):
        training.verify_training_bundle(root)


def test_training_segments_retain_sampled_tokens_logprobs_and_masks():
    group = {'task_id': 'v2-task', 'policy_version': 3, 'episodes': [{'actions': [
        {'prompt': [10, 11], 'tokens': [12, 13], 'old_logp': [-.2, -.4], 'loss_mask': [1, 1]}]}]}
    attach_contract(group)
    validate_contract(group)
    sampled = group['episodes'][0]['actions'][0]
    assert sampled['tokens'] == [12, 13] and sampled['old_logp'] == [-.2, -.4]
    assert sampled['loss_mask'] == [1, 1]


def test_near_duplicate_family_cannot_cross_train_eval(bundle, tmp_path):
    first, second = direct(), direct()
    first['id'], second['id'] = 'first', 'second'
    first['task'] += ' 版本1'
    second['task'] += ' 版本2'
    a, b = bundle(first), bundle(second)
    training.certify(a)
    training.certify(b)
    manifest = tmp_path / 'tasks.json'
    manifest.write_text(json.dumps({'tasks': [{'id': 'a', 'sandbox': str(a), 'split': 'train'},
                                            {'id': 'b', 'sandbox': str(b), 'split': 'eval'}]}))
    with pytest.raises(ValueError, match='family crosses'):
        load_tasks(manifest=str(manifest))
