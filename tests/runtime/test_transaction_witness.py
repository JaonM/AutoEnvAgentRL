import copy
import json

import pytest

from env_factory.sandbox_runtime import (
    ApprovalTerms, BusinessLifecycle, ContractToolRegistry, EpisodeStore,
    ManifestDataStore, SandboxError, TransactionWitness,
)
from env_factory.generation.difficulty_profiles import requirements, transaction_covered


def runtime(tmp_path, *, fault_phase='before', external=False):
    manifest = {'environment_mode': 'stateful', 'tables': []}
    for name, ids in [('stock', [1]), ('orders', [1, 2])]:
        schema = {'columns': [{'name': 'id', 'type': 'integer', 'nullable': False},
                              {'name': 'value', 'type': 'integer', 'nullable': False}], 'primary_key': ['id']}
        (tmp_path / f'{name}.schema.json').write_text(json.dumps(schema))
        (tmp_path / f'{name}.jsonl').write_text(''.join(json.dumps({'id': i, 'value': 0}) + '\n' for i in ids))
        manifest['tables'].append({'table_name': name, 'schema_file': f'{name}.schema.json', 'rows_file': f'{name}.jsonl'})
    store = EpisodeStore(tmp_path / 'episode.sqlite3')
    data = ManifestDataStore(manifest, tmp_path, store)
    data.reset()
    tools, handlers, policies = [], {}, {}
    for name, table in [('reserve', 'stock'), ('place', 'orders'), ('release', 'stock')]:
        tools.append({'type': 'function', 'function': {'name': name, 'parameters': {
            'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']}}})
        handlers[name] = lambda args, table=table: {'count': data.update(table, {}, {'value': args['value']})}
        policies[name] = {'requires_grants': [name], 'require_idempotency': True}
        ApprovalTerms.apply(store, {'before_tool': name, 'grant_permissions': [name],
            'approval_terms': {name: {'scope': 'terms', 'arguments': {'value': 0 if name == 'release' else 1}}}})
    store.set_state('business_grants', list(policies))
    dynamics = {'faults': [{'id': 'service', 'tool': 'place', 'call_index': 1,
                            'phase': fault_phase, 'status': 503 if fault_phase == 'before' else 504}]}
    if external:
        dynamics['events'] = [{'id': 'outside', 'at_tick': 2,
            'changes': [{'table': 'orders', 'selector': {'id': 2}, 'values': {'value': 1}}]}]
    lifecycle = BusinessLifecycle({'version': '1.0', 'tools': policies, 'dynamics': dynamics}, data, store, policies)
    contract = {'write_tools': list(policies), 'compensation_tools': ['release'],
                'table_primary_keys': {'stock': ['id'], 'orders': ['id']}}
    registry = ContractToolRegistry(tools, handlers, lifecycle=lifecycle, event_recorder=store.event,
                                    transaction_contract=contract)
    goal = {'table_primary_keys': contract['table_primary_keys'], 'row_predicates': [
        {'table': table, 'where': {'id': i}, 'values': {'value': 0 if table == 'stock' else 1}, 'count': 1}
        for table, ids in [('stock', [1]), ('orders', [1, 2])] for i in ids]}
    def observe():
        return TransactionWitness.observe(contract, store.replay()['events'], data.baseline, data._all(), goal)
    return store, data, registry, observe


def execute(registry, *, compensate=True):
    registry.execute('reserve', {'value': 1}, idem_key='reserve')
    with pytest.raises(SandboxError):
        registry.execute('place', {'value': 1}, idem_key='place')
    if compensate:
        registry.execute('release', {'value': 0}, idem_key='release')
    registry.execute('place', {'value': 1}, idem_key='place')
    registry.execute('place', {'value': 1}, idem_key='place')


@pytest.mark.parametrize('phase', ['before', 'after_commit'])
def test_actual_partial_commit_compensation_and_stable_retry(tmp_path, phase):
    store, data, registry, observe = runtime(tmp_path, fault_phase=phase)
    execute(registry)
    proof = observe()
    assert proof['changed_objects'] == 3
    assert proof['changed_tables'] == ['orders', 'stock']
    assert proof['compensated_fault_ids'] == ['service']
    assert proof['idempotent_replay_tools'] == ['place']
    assert proof['audit_valid'] and proof['unrelated_state_preserved']
    assert transaction_covered(requirements('complex_write_transactions'), proof)
    failures = [e['result'] for e in store.replay()['events'] if e['event'] == 'business_write_audit' and e['result']['status'] >= 400]
    assert bool(failures[0]['changes']) == (phase == 'after_commit')
    assert failures[0]['committed'] == (phase == 'after_commit')


def test_external_updates_do_not_count_as_agent_objects_or_rollback_leaks(tmp_path):
    _, _, registry, observe = runtime(tmp_path, external=True)
    execute(registry)
    proof = observe()
    assert proof['audit_valid']
    assert proof['changed_objects'] == 2
    assert not transaction_covered(requirements('complex_write_transactions'), proof)


def test_successful_noop_compensation_and_fault_before_any_write_do_not_qualify(tmp_path):
    _, _, registry, observe = runtime(tmp_path)
    with pytest.raises(SandboxError):
        registry.execute('place', {'value': 1}, idem_key='place')
    registry.execute('release', {'value': 0}, idem_key='noop')
    registry.execute('reserve', {'value': 1}, idem_key='reserve')
    registry.execute('release', {'value': 0}, idem_key='release')
    registry.execute('place', {'value': 1}, idem_key='place')
    registry.execute('place', {'value': 1}, idem_key='place')
    proof = observe()
    assert proof['changed_objects'] == 3
    assert proof['compensated_fault_ids'] == []


def test_postcondition_failure_really_rolls_back_and_does_not_count_write(tmp_path):
    store, data, registry, observe = runtime(tmp_path)
    def broken(args):
        data.update('stock', {}, {'value': 99})
        raise SandboxError('WRITE_FAILED', 'postcondition failed', 409)
    registry.handlers['reserve'] = broken
    with pytest.raises(SandboxError):
        registry.execute('reserve', {'value': 1}, idem_key='reserve')
    proof = observe()
    assert proof['audit_valid'] and proof['changed_objects'] == 0
    assert proof['failed_without_write_tools'] == ['reserve']
    assert data.table('stock') == [{'id': 1, 'value': 0}]


@pytest.mark.parametrize('defect', ['missing_audit', 'leaked_failure', 'mutating_replay', 'unbound_consent'])
def test_corrupted_evidence_fails_closed(tmp_path, defect):
    store, data, registry, _ = runtime(tmp_path)
    execute(registry)
    events = copy.deepcopy(store.replay()['events'])
    audits = [e for e in events if e['event'] == 'business_write_audit']
    if defect == 'missing_audit':
        events.remove(audits[0])
    elif defect == 'leaked_failure':
        audits[1]['result']['changes'] = copy.deepcopy(audits[0]['result']['changes'])
    elif defect == 'mutating_replay':
        audits[-1]['result']['changes'] = copy.deepcopy(audits[0]['result']['changes'])
    else:
        audits[0]['result']['bound_authorization'] = False
    proof = TransactionWitness.observe(registry.transaction_contract, events, data.baseline, data._all(),
        {'row_predicates': [], 'table_primary_keys': registry.transaction_contract['table_primary_keys']})
    assert not proof['audit_valid']


def test_runtime_cannot_obtain_bound_consent_credit_from_legacy_grant(tmp_path):
    store, _, registry, observe = runtime(tmp_path)
    store.set_state(ApprovalTerms.KEY, {})
    execute(registry)
    assert not observe()['audit_valid']


def test_naming_an_unrelated_write_as_compensation_cannot_get_credit(tmp_path):
    _, _, registry, observe = runtime(tmp_path)
    registry.transaction_contract['compensation_tools'] = ['place']
    execute(registry)
    proof = observe()
    assert proof['audit_valid'] and proof['changed_objects'] == 3
    assert proof['compensated_fault_ids'] == []


def test_insert_delete_compensation_tracks_object_identity_and_null_presence(tmp_path):
    _, data, registry, observe = runtime(tmp_path)
    registry.handlers['reserve'] = lambda args: data.insert('stock', {'id': 3, 'value': 1})
    registry.handlers['release'] = lambda args: data.delete('stock', {'id': 3})
    execute(registry)
    proof = observe()
    assert proof['audit_valid'] and proof['changed_objects'] == 3
    assert proof['compensated_fault_ids'] == ['service']
    assert data.table('stock') == [{'id': 1, 'value': 0}]
