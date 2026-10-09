import pytest

from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import long_chain
from env_factory_v2.expressions import ContractError
from env_factory_v2.runtime import Sandbox


def source():
    value = long_chain(3)
    value['tools'] = value['tools'][:2]
    value['tools'][1]['result_bindings'][0].update(tool=['node_0', 'node_1'], recency='latest')
    for scenario in value['scenarios']:
        scenario['steps'][2]['tool'] = 'node_1'
    return value


def test_latest_rejects_old_successful_capture_after_successor_query():
    s = source()
    sandbox = Sandbox(s)
    sandbox.call('node_0', {'node_id': 'N0'}, call_id='start')
    sandbox.call('node_1', {'node_id': {'$result': 'start', 'path': 'items.0.next_id'}}, call_id='second')
    with pytest.raises(ContractError, match='latest successful business result'):
        sandbox.call('node_1', {'node_id': {'$result': 'start', 'path': 'items.0.next_id'}}, call_id='old')
    sandbox.time()
    sandbox.call('node_1', {'node_id': {'$result': 'second', 'path': 'items.0.next_id'}}, call_id='third')
    assert sandbox.trace[-1]['status'] == 'ok'
    sandbox.reset()
    sandbox.call('node_0', {'node_id': 'N0'}, call_id='start')
    sandbox.call('node_1', {'node_id': {'$result': 'start', 'path': 'items.0.next_id'}}, call_id='second')


def test_any_binding_preserves_existing_contracts_and_public_latest_is_explicit():
    s = source()
    compiled = compile_contract(s)
    assert compiled['public']['tools'][1]['result_bindings'][0]['recency'] == 'latest'
    assert 'most recent successful business tool' in compiled['public']['tools'][1]['description']
    del s['tools'][1]['result_bindings'][0]['recency']
    sandbox = Sandbox(s)
    sandbox.call('node_0', {'node_id': 'N0'}, call_id='start')
    for identity in ['second', 'repeat']:
        sandbox.call('node_1', {'node_id': {'$result': 'start', 'path': 'items.0.next_id'}}, call_id=identity)


def test_invalid_recency_is_rejected():
    s = source()
    s['tools'][1]['result_bindings'][0]['recency'] = 'latest_provider_only'
    with pytest.raises(ContractError, match='dependency recency'):
        compile_contract(s)
