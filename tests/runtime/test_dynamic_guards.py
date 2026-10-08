import pytest

from env_factory.sandbox_runtime import BusinessLifecycle, ContractToolRegistry, SandboxError
from tests.runtime.test_nested_transactions import business_data


def test_stock_guard_uses_each_request_amount_and_current_state(tmp_path):
    store, data = business_data(tmp_path)
    policy = {'version': '1.0', 'tools': {'reserve': {'require_idempotency': True, 'guards': [{
        'id': 'enough_stock', 'table': 'inventory', 'selector': {'id': {'argument': 'id'}},
        'conditions': {'quantity': {'gte': {'argument': 'amount'}}}}]}}}
    lifecycle = BusinessLifecycle(policy, data, store, ['reserve'])
    tool = {'type': 'function', 'function': {'name': 'reserve', 'parameters': {
        'type': 'object', 'properties': {'id': {'type': 'integer'}, 'amount': {'type': 'integer', 'minimum': 1}},
        'required': ['id', 'amount'], 'additionalProperties': False}}}
    def reserve(arguments):
        row = next(row for row in data.table('inventory') if row['id'] == arguments['id'])
        return {'updated_count': data.update('inventory', {'id': row['id']},
                                             {'quantity': row['quantity'] - arguments['amount']})}
    registry = ContractToolRegistry([tool], {'reserve': reserve}, event_recorder=store.event, lifecycle=lifecycle)
    for amount, remaining in [(11, 10), (4, 6), (7, 6)]:
        if amount == 4:
            registry.execute('reserve', {'id': 1, 'amount': amount}, idem_key=str(amount))
        else:
            with pytest.raises(SandboxError) as error:
                registry.execute('reserve', {'id': 1, 'amount': amount}, idem_key=str(amount))
            assert error.value.status == 409
            assert error.value.details['failed_fields'] == [
                {'field': 'quantity', 'operator': 'gte', 'expected': amount, 'actual': remaining}]
        assert data.table('inventory')[0]['quantity'] == remaining
    registry.execute('reserve', {'id': 1, 'amount': 4}, idem_key='4')
    assert data.table('inventory')[0]['quantity'] == 6


def test_duplicate_guard_ids_are_rejected(tmp_path):
    store, data = business_data(tmp_path)
    guard = {'id': 'same', 'table': 'inventory', 'selector': {'id': 1}, 'conditions': {'quantity': {'gte': 1}}}
    with pytest.raises(SandboxError, match='declared business fields'):
        BusinessLifecycle({'version': '1.0', 'tools': {'reserve': {'guards': [guard, guard]}}}, data, store, ['reserve'])


def test_observed_identity_rejects_wrong_row_and_missing_snapshot(tmp_path):
    from env_factory.sandbox_runtime import RewardStateSnapshots
    store, data = business_data(tmp_path)
    expression = {'at_tool': {'name': 'read', 'expression': {'lookup': {
        'table': 'inventory', 'where': {'id': 1}, 'field': 'id'}}}}
    policy = {'version': '1.0', 'tools': {'reserve': {'guards': [{
        'id': 'identity', 'table': 'inventory', 'selector': {'id': {'argument': 'id'}},
        'conditions': {'id': {'eq': {'expression': expression}}}}]}}}
    lifecycle = BusinessLifecycle(policy, data, store, ['reserve', 'read'])
    with pytest.raises(SandboxError) as error:
        lifecycle.before('reserve', {'id': 1}, None)
    assert error.value.code == 'GUARD_EVIDENCE_MISSING'
    store.set_state(RewardStateSnapshots.KEY, {'read': data._all()})
    lifecycle.before('reserve', {'id': 1}, None)
    with pytest.raises(SandboxError) as error:
        lifecycle.before('reserve', {'id': 2}, None)
    assert error.value.status == 409


def test_dynamic_quantity_guard_uses_observed_quantity(tmp_path):
    from env_factory.sandbox_runtime import RewardStateSnapshots
    store, data = business_data(tmp_path)
    expression = {'at_tool': {'name': 'read', 'expression': {'lookup': {
        'table': 'inventory', 'where': {'id': 1}, 'field': 'quantity'}}}}
    policy = {'version': '1.0', 'tools': {'reserve': {'guards': [{
        'id': 'quantity', 'table': 'inventory', 'selector': {'id': 1},
        'conditions': {'quantity': {'gte': {'expression': expression}}}}]}}}
    lifecycle = BusinessLifecycle(policy, data, store, ['reserve', 'read'])
    store.set_state(RewardStateSnapshots.KEY, {'read': data._all()})
    lifecycle.before('reserve', {}, None)
    data.update('inventory', {'id': 1}, {'quantity': 9})
    with pytest.raises(SandboxError) as error:
        lifecycle.before('reserve', {}, None)
    assert error.value.details['failed_fields'][0]['expected'] == 10


def test_shared_contract_freshness_prevents_overwriting_external_stock(tmp_path):
    from env_factory.generation.business_contract import expand_business_contract
    from env_factory.sandbox_runtime import RewardStateSnapshots
    store, data = business_data(tmp_path)
    observed = {'at_tool': {'name': 'read', 'expression': {'lookup': {
        'table': 'inventory', 'where': {'id': 1}, 'field': 'quantity'}}}}
    source = {'tables': [{'table_name': 'inventory', 'columns': [{'name': 'id'}, {'name': 'quantity'}]}],
        'business_tools': [{'name': 'reserve', 'parameters': {'type': 'object',
            'properties': {'id': {'type': 'integer'}}, 'required': ['id']}}],
        'business_contract': {'version': '1.0',
            'state_targets': {'stock': {'table': 'inventory', 'where': {'id': 1}, 'values': {'quantity': 6}, 'count': 1}},
            'outcomes': [{'id': 'reserved', 'rubric': 'reserve four', 'targets': ['stock']}],
            'operations': [{'tool': 'reserve', 'target': 'stock', 'selector': {'id': 'id'},
                'bindings': {}, 'derive': ['quantity'], 'observed': {'quantity': observed}}]}}
    compiled = expand_business_contract(source)
    lifecycle = BusinessLifecycle(compiled['business_lifecycle'], data, store, ['reserve', 'read'])
    store.set_state(RewardStateSnapshots.KEY, {'read': data._all()})
    lifecycle.before('reserve', {'id': 1}, 'request')
    data.update('inventory', {'id': 1}, {'quantity': 9})
    with pytest.raises(SandboxError) as error:
        lifecycle.before('reserve', {'id': 1}, 'request')
    assert error.value.code == 'BUSINESS_PRECONDITION_FAILED'
    assert error.value.details['guard_id'] == 'stock:freshness'
    assert data.table('inventory')[0]['quantity'] == 9
