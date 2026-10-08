import pytest

from env_factory.sandbox_runtime import (ContractToolRegistry, DeclarativeToolCompiler,
                                         RewardStateSnapshots, SandboxError)
from tests.runtime.test_nested_transactions import business_data


def test_derived_write_requires_checkpoint_and_retry_never_recomputes(tmp_path):
    store, data = business_data(tmp_path)
    expression = {'op': 'sub', 'args': [{'at_tool': {'name': 'read_stock', 'expression': {
        'lookup': {'table': 'inventory', 'field': 'quantity', 'where': {'id': 1}}}}}, {'literal': 4}]}
    specs = [
        {'tool_name': 'read_stock', 'operation': 'select', 'table': 'inventory',
         'filters': [{'argument': 'id', 'column': 'id', 'operator': 'eq'}]},
        {'tool_name': 'reserve', 'operation': 'update', 'table': 'inventory',
         'selector': {'id': 'id'}, 'changes': {}, 'derived_changes': {'quantity': expression}}]
    tools = [{'type': 'function', 'function': {'name': name, 'parameters': {
        'type': 'object', 'properties': {'id': {'type': 'integer'}}, 'required': ['id'],
        'additionalProperties': False}}} for name in ('read_stock', 'reserve')]
    registry = ContractToolRegistry(tools, DeclarativeToolCompiler(data).compile_all(specs),
        event_recorder=store.event, reward_snapshots=RewardStateSnapshots(data, specs, [expression]))
    with pytest.raises(SandboxError) as error:
        registry.execute('reserve', {'id': 1}, idem_key='one-reservation')
    assert error.value.code == 'DERIVED_CHANGE_UNRESOLVED'
    assert error.value.status == 409
    assert data.table('inventory')[0]['quantity'] == 10
    registry.execute('read_stock', {'id': 1})
    assert registry.execute('reserve', {'id': 1}, idem_key='one-reservation') == {'updated_count': 1}
    assert data.table('inventory')[0]['quantity'] == 6
    # A later business change must survive retrieval of a cached response.
    data.update('inventory', {'id': 1}, {'quantity': 5})
    assert registry.execute('reserve', {'id': 1}, idem_key='one-reservation') == {'updated_count': 1}
    assert data.table('inventory')[0]['quantity'] == 5
    with pytest.raises(SandboxError):
        registry.execute('reserve', {'id': 1, 'quantity': 999}, idem_key='different')
    assert data.table('inventory')[0]['quantity'] == 5


def test_client_and_server_cannot_own_the_same_changed_field(tmp_path):
    _, data = business_data(tmp_path)
    with pytest.raises(SandboxError, match='overlapping'):
        DeclarativeToolCompiler(data).compile({'operation': 'update', 'table': 'inventory',
            'selector': {'id': 'id'}, 'changes': {'value': 'quantity'},
            'derived_changes': {'quantity': 6}})
