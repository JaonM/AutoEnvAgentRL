import json
from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.sandbox_runtime import ContractToolRegistry, ContractUserSimulator, EpisodeStore, SandboxApplication
from tests.generation.test_reused_user_inputs import source_with_reuse


def test_wrong_bound_argument_cannot_be_classified_as_positive_recovery(tmp_path):
    source, request = source_with_reuse()
    reference = next(s for s in source['scenarios'] if s['kind'] == 'goal_success')
    source['reward_pair_probes'] = [{'id': 'wrong_region', 'kind': 'readonly_recovery',
        'reference_id': reference['scenario_id'], 'before_tool': 'read_temperature',
        'calls': [{'tool_name': 'read_temperature', 'arguments': {'id': 821, 'region': '西区'}}]}]
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as judge:
        with pytest.raises(ValueError, match='DETERMINISTIC_PREFLIGHT_FAILED.*positive-probe:wrong_region'):
            compile_source(source, root=tmp_path, request=request)
    judge.assert_not_called()
    report = json.loads((tmp_path / 'deterministic_preflight.json').read_text())
    assert any('USER_INPUT_MISMATCH' in json.dumps(f) for f in report['findings'])


def test_http_bound_arguments_reject_wrong_reads_and_allow_correct_recovery(tmp_path, monkeypatch):
    source, request = source_with_reuse()
    artifacts = compile_source(source, root=tmp_path, request=request, structural_preview=True)
    scripts = json.loads((tmp_path / 'data/user_simulation/user_scripts.json').read_text())
    profiles = json.loads((tmp_path / 'data/user_simulation/user_profiles.json').read_text())
    store = EpisodeStore(tmp_path / 'http.sqlite3')
    user = ContractUserSimulator(store, scripts=scripts, profiles=profiles)
    calls = []
    def read(arguments):
        calls.append(arguments)
        return {'records': [{'id': 821}], 'count': 1}
    registry = ContractToolRegistry(artifacts['tools'], {name: read for name in ['lookup_zone', 'read_temperature']})
    app = SandboxApplication(episode_store=store, tool_registry=registry, observation=lambda: {},
        reward=lambda: {}, user_turn=user.turn, reset_hook=user.reset)
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'scope-test')
    headers = {'Authorization': 'Bearer scope-test'}
    def call(path, body):
        return app.handle('POST', path, body, headers)[:2]
    assert call('/v1/reset', {'seed': 17})[0] == 200
    assert call('/v1/tools/lookup_zone', {'name': '白杨库', 'region': '西区'})[1]['error']['code'] == 'USER_INTERACTION_REQUIRED'
    script = user.scripts[store.get_state(user.STATE_KEY)['script_id']]
    messages = []
    for stage in script['interaction_protocol']['stages']:
        messages.append({'role': 'assistant', 'content': stage['reference_response']})
        status, response = call('/v1/user_simulator', {'messages': messages})
        assert status == 200 and response['fsm_transition_applied']
        messages.append({'role': 'user', 'content': response['user_query']})
    for name, arguments in [('lookup_zone', {'name': '白杨库', 'region': '东区'}),
                             ('read_temperature', {'id': 821, 'region': '西区'})]:
        status, response = call('/v1/tools/' + name, arguments)
        assert status == 409 and response['error']['code'] == 'USER_INPUT_MISMATCH'
    assert calls == []
    assert call('/v1/tools/lookup_zone', {'name': '青松库', 'region': '东区'})[0] == 200
    assert call('/v1/tools/read_temperature', {'id': 821, 'region': '东区'})[0] == 200
    # The unbound ID is governed by ordinary tool/business constraints.
    assert call('/v1/tools/read_temperature', {'id': 547, 'region': '东区'})[0] == 200
    assert len(calls) == 3
    assert call('/v1/reset', {'seed': 17})[0] == 200
    assert call('/v1/tools/read_temperature', {'id': 821, 'region': '东区'})[1]['error']['code'] == 'USER_INTERACTION_REQUIRED'
    public = next(m['content'] for m in artifacts['public_input']['materials'] if m['name'] == 'disclosed_user_input_scope')
    assert 'USER_INPUT_MISMATCH' in public
    assert '青松库' not in public
