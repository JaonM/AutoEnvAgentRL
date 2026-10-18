import copy

import pytest

from env_factory.certify import trajectory
from env_factory.production import contracts


def call(name='lookup'):
    return {'id': name, 'type': 'function', 'function': {'name': name, 'arguments': '{}'}}


def test_declared_policy_rejects_batched_and_mixed_turns_without_rewriting_evidence():
    rules = {'max_tool_calls_per_turn': 1, 'allow_text_with_tool_calls': False}
    messages = [{'role': 'system', 'content': 'Public policy'},
                {'role': 'assistant', 'content': 'Checking.', 'tool_calls': [call(), call('save')]}]
    original = copy.deepcopy(messages)
    result = trajectory.qualify({'success': True, 'reward': 1, 'messages': messages}, rules)
    assert result['success'] is True and result['reward'] == 1
    assert result['policy_aware_success'] is False
    assert {v['rule'] for v in result['trajectory_policy']['violations']} == set(rules)
    view = trajectory.provider_view(messages, rules)
    assert 'at most 1' in view[0]['content']
    assert view[1] == messages[1] and messages == original


def test_unconstrained_domains_keep_existing_behavior_and_serial_turns_pass():
    messages = [{'role': 'assistant', 'content': 'Lookup', 'tool_calls': [call(), call('other')]}]
    assert trajectory.audit(messages, {})['passed']
    serial = [{'role': 'assistant', 'content': '', 'tool_calls': [call()]},
              {'role': 'tool', 'content': 'value'}, {'role': 'assistant', 'content': 'Done.'}]
    rules = {'max_tool_calls_per_turn': 1, 'allow_text_with_tool_calls': False}
    assert trajectory.qualify({'success': True, 'messages': serial}, rules)['policy_aware_success']
    assert not trajectory.qualify({'success': True, 'valid': False, 'messages': serial}, rules)['policy_aware_success']


@pytest.mark.parametrize('rules,valid', [
    ({'max_tool_calls_per_turn': 1, 'allow_text_with_tool_calls': False}, True),
    ({'max_tool_calls_per_turn': 2}, True),
    ({'max_tool_calls_per_turn': True}, False),
    ({'max_tool_calls_per_turn': 0}, False),
    ({'max_tool_calls_per_turn': 1.5}, False),
    ({'allow_text_with_tool_calls': 'false'}, False),
])
def test_contract_declares_typed_dialogue_constraints(rules, valid):
    contract = {'policy': 'Membership policy', 'constraints': rules,
                'tools': [{'name': 'lookup', 'parameters': {'type': 'object', 'properties': {}}}]}
    assert (not contracts.problems(contract)) is valid


def test_policy_player_requests_selected_consent_without_changing_transcript(monkeypatch):
    from env_factory.certify import policy_gate
    resets = []
    def post(opener, url, body):
        if url.endswith('/reset'):
            resets.append(body)
            return {'episode': 'e', 'instructions': {'tools': []}}
        return {}
    monkeypatch.setattr(policy_gate, '_post', post)
    monkeypatch.setattr(policy_gate.agent_view, 'tools', lambda *args, **kwargs: [])
    monkeypatch.setattr(policy_gate.agent_view, 'messages', lambda *args, **kwargs: [])
    monkeypatch.setattr(policy_gate.agent_view, 'speaks', lambda instructions: False)
    for consent in ('token', 'plain'):
        result = policy_gate.play('http://test', None, 'original', 'simple', max_turns=0, consent=consent)
        assert result['success'] is False
        assert resets[-1] == {'world': 'original', 'consent': consent}


def test_plain_consent_requires_natural_user_before_loading_bundle():
    from env_factory.certify.policy_gate import gate
    with pytest.raises(ValueError, match='natural'):
        gate('not-loaded', None, user_mode='structured', consent='plain')


def test_saved_policy_acceptance_is_bound_to_bundle_evaluator_and_consent():
    from env_factory.certify import policy_gate, bundle
    manifest = {'files': {'request.json': 'current-request', 'env.py': 'current-tools'}}
    report = {'k': 4, 'user_mode': 'natural', 'consent': 'plain', 'status': 'passed',
              'platform_identity': bundle.platform_identity(), 'bundle_files': manifest['files'],
              'evaluation_identity': policy_gate.evaluation_identity()}
    def usable(r, m=manifest, consent='plain'):
        return policy_gate.reusable(r, m, k=4, user_mode='natural', consent=consent)
    assert usable(report)
    assert not usable(report, consent='token')
    assert not usable(report, {'files': {'request.json': 'modified-request'}})
    assert not usable({**report, 'evaluation_identity': 'old-player'})
    assert not usable({**report, 'platform_identity': 'old-runtime'})
    assert not usable({'k': 4, 'status': 'passed'})
