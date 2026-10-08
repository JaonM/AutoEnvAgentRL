import json
import pytest
from env_factory.sandbox_runtime import (ApprovalTerms, BusinessLifecycle, ContractToolRegistry,
    EpisodeStore, ManifestDataStore, SandboxError)


def setup_booking(tmp_path):
    schema = {'columns': [{'name': 'id', 'type': 'INTEGER', 'nullable': False},
                          {'name': 'daily_fee', 'type': 'INTEGER', 'nullable': False}], 'primary_key': ['id']}
    (tmp_path / 'schema.json').write_text(json.dumps(schema))
    (tmp_path / 'rows.jsonl').write_text('{"id":1,"daily_fee":0}\n')
    store = EpisodeStore(tmp_path / 'episode.sqlite3')
    data = ManifestDataStore({'environment_mode': 'stateful', 'tables': [
        {'table_name': 'bookings', 'schema_file': 'schema.json', 'rows_file': 'rows.jsonl'}]}, tmp_path, store)
    data.reset()
    stage = {'kind': 'execution_confirmation', 'before_tool': 'book', 'grant_permissions': ['book'],
        'approval_terms': {'book': {'scope': 'terms', 'arguments': {'id': 1}, 'postconditions': [
            {'table': 'bookings', 'where': {'id': 1}, 'values': {'daily_fee': 900}, 'count': 1}]}}}
    ApprovalTerms.apply(store, stage)
    store.set_state('business_grants', ['book'])
    lifecycle = BusinessLifecycle({'version': '1.0', 'tools': {'book': {'requires_grants': ['book']}}}, data, store, ['book'])
    tool = {'type': 'function', 'function': {'name': 'book', 'parameters': {
        'type': 'object', 'properties': {'id': {'type': 'integer'}}, 'required': ['id']}}}
    return store, data, stage, lifecycle, tool


def test_server_derived_950_cannot_commit_against_user_approved_900(tmp_path):
    store, data, stage, lifecycle, tool = setup_booking(tmp_path)
    actual_fee = [950]
    def book(arguments):
        return {'updated': data.update('bookings', {'id': arguments['id']}, {'daily_fee': actual_fee[0]})}
    registry = ContractToolRegistry([tool], {'book': book}, event_recorder=store.event, lifecycle=lifecycle)
    with pytest.raises(SandboxError) as error:
        registry.execute('book', {'id': 1}, idem_key='request')
    assert error.value.code == 'AUTHORIZATION_TERMS_MISMATCH'
    assert data.table('bookings')[0]['daily_fee'] == 0  # transaction rolled back
    actual_fee[0] = 900
    registry.execute('book', {'id': 1}, idem_key='request')
    assert data.table('bookings')[0]['daily_fee'] == 900  # rejected response was not cached
    registry.execute('book', {'id': 1}, idem_key='request')
    assert data.table('bookings')[0]['daily_fee'] == 900


def test_wrong_entity_and_revocation_are_rejected(tmp_path):
    store, data, stage, lifecycle, tool = setup_booking(tmp_path)
    with pytest.raises(SandboxError, match='approved terms'):
        lifecycle.before('book', {'id': 2}, None)
    ApprovalTerms.apply(store, {'grant_permissions': [], 'revoke_permissions': ['book']})
    store.set_state('business_grants', [])
    assert store.get_state(ApprovalTerms.KEY) == {}
    with pytest.raises(SandboxError) as error:
        lifecycle.before('book', {'id': 1}, None)
    assert error.value.code == 'AUTHORIZATION_REQUIRED'


def test_consent_cannot_recompute_approved_price_from_current_offer():
    stage = {'grant_permissions': ['book'], 'approval_terms': {'book': {'scope': 'terms',
        'arguments': {'fee': {'tool_result': {'name': 'offer', 'path': '$.fee'}}}}}}
    with pytest.raises(ValueError, match='literal scalars'):
        ApprovalTerms.validate(stage)


def test_existing_correct_row_cannot_hide_write_to_an_unapproved_entity(tmp_path):
    store, data, stage, lifecycle, tool = setup_booking(tmp_path)
    data.update('bookings', {'id': 1}, {'daily_fee': 900})
    data.replace_table('bookings', [{'id': 1, 'daily_fee': 900}, {'id': 2, 'daily_fee': 0}])
    stage['approval_terms']['book'].pop('arguments')
    ApprovalTerms.apply(store, stage)
    registry = ContractToolRegistry([tool], {'book': lambda args: {
        'updated': data.update('bookings', {'id': 2}, {'daily_fee': 950})}},
        event_recorder=store.event, lifecycle=lifecycle)
    with pytest.raises(SandboxError) as error:
        registry.execute('book', {'id': 2})
    assert error.value.code == 'AUTHORIZATION_SCOPE_MISMATCH'
    assert data.table('bookings')[1]['daily_fee'] == 0
