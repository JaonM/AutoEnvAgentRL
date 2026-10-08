"""Retry transport correlation must not masquerade as changing business evidence."""
import copy
import json

import pytest

from scripts.sandbox import validate_agentic_training_value as validator
from env_factory.sandbox_runtime import (
    BusinessLifecycle, ContractToolRegistry, EpisodeStore, ManifestDataStore, SandboxApplication,
)


@pytest.mark.parametrize('frozen', [False, True])
def test_native_transient_retry_preserves_private_reward_sensitivity(tmp_path, monkeypatch, frozen):
    monkeypatch.setenv('SANDBOX_TRAINER_API_KEY', 'probe-key')
    (tmp_path / 'schema.json').write_text(json.dumps({'columns': [
        {'name': 'id', 'type': 'INTEGER'}, {'name': 'amount', 'type': 'INTEGER'}]}))
    (tmp_path / 'rows.jsonl').write_text('{"id":1,"amount":2}\n')
    store = EpisodeStore(tmp_path / 'episodes.sqlite3')
    data = ManifestDataStore({'environment_mode': 'reference_data', 'tables': [{'table_name': 'items',
        'schema_file': 'schema.json', 'rows_file': 'rows.jsonl'}]}, tmp_path, store)
    lifecycle = BusinessLifecycle({'version': '1.0', 'roles': [], 'tools': {}, 'dynamics': {
        'faults': [{'id': 'transient', 'tool': 'read', 'call_index': 1, 'phase': 'before', 'status': 503}]}},
        data, store, ['read'])
    registry = ContractToolRegistry([{'type': 'function', 'function': {'name': 'read',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}],
        {'read': lambda args: {'records': data.table('items')}}, event_recorder=store.event, lifecycle=lifecycle)
    app = SandboxApplication(episode_store=store, tool_registry=registry, observation=lambda: {},
        reward=lambda: {'reward': float(frozen or data.table('items')[0]['amount'] == 2)},
        user_turn=lambda messages: {}, business_snapshot=data._all, reset_hook=data.reset)
    app.mutate_business_state = lambda mutation: data.update(mutation['table'], mutation['selector'], mutation['changes'])
    scenario = {'steps': [{'operation': 'reset'},
        {'operation': 'tool_call', 'tool_name': 'read', 'arguments': {}, 'expected_status': 503},
        {'operation': 'tool_call', 'tool_name': 'read', 'arguments': {}},
        {'operation': 'agent_response', 'content': '库存为2'}, {'operation': 'reward'}]}
    result = validator.probe_reference_data_reward_sensitivity(app, {'Authorization': 'Bearer probe-key'},
        scenario, {'table': 'items', 'selector': {'id': 1}, 'changes': {'amount': 3}}, .8)
    assert result['baseline_tool_results_stable']
    assert result['proved'] is (not frozen)
    assert result['tool_results_changed']


@pytest.mark.parametrize('difference', ['status', 'code', 'details', 'success_request_id'])
def test_business_or_error_semantic_changes_remain_visible(difference):
    event = {'operation': 'tool_call', 'status': 503, 'body': {'error': {
        'code': 'TRANSIENT_FAILURE', 'message': 'retry', 'details': {'request_id': 'business-id'},
        'request_id': 'transport-id'}}}
    changed = copy.deepcopy(event)
    if difference == 'status':
        changed['status'] = 504
    elif difference == 'code':
        changed['body']['error']['code'] = 'OTHER'
    elif difference == 'details':
        changed['body']['error']['details']['request_id'] = 'another-business-id'
    else:
        event.update(status=200, body={'request_id': 'business-id'})
        changed.update(status=200, body={'request_id': 'another-business-id'})
    before = copy.deepcopy(event)
    assert validator.tool_evidence_signature({'history': [event]}) != validator.tool_evidence_signature({'history': [changed]})
    assert event == before
