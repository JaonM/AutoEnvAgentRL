import pytest

from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import long_chain
from env_factory_v2.expressions import ContractError
from env_factory_v2.runtime import Sandbox


def source():
    value = long_chain(3)
    value['tools'][1]['result_bindings'][0]['path'] = ['items.*.next_id', 'items.*.id']
    return value


@pytest.mark.parametrize('path,expected', [('items.0.next_id', 'N1'), ('items.0.id', 'N0')])
def test_alternative_paths_accept_only_real_observed_values(path, expected):
    sandbox = Sandbox(source())
    sandbox.call('node_0', {'node_id': 'N0'}, call_id='start')
    result = sandbox.call('node_1', {'node_id': {'$result': 'start', 'path': path}}, call_id='selected')
    assert result['items'][0]['id'] == expected


def test_alternative_paths_do_not_allow_literal_or_wrong_provider():
    sandbox = Sandbox(source())
    sandbox.call('node_0', {'node_id': 'N0'}, call_id='start')
    with pytest.raises(ContractError, match='observed-result dependency'):
        sandbox.call('node_1', {'node_id': 'N1'}, call_id='literal')
    sandbox.call('node_1', {'node_id': {'$result': 'start', 'path': 'items.0.next_id'}}, call_id='second')
    with pytest.raises(ContractError, match='observed-result dependency'):
        sandbox.call('node_1', {'node_id': {'$result': 'second', 'path': 'items.0.id'}}, call_id='wrong_provider')


def test_matching_value_from_undeclared_path_does_not_satisfy_binding():
    value = source()
    value['tables'][0]['schema']['properties']['label'] = {'type': 'string'}
    value['tables'][0]['schema']['required'].append('label')
    for row in value['tables'][0]['rows']:
        row['label'] = 'N1'
    value['tools'][0]['columns'].append('label')
    sandbox = Sandbox(value)
    sandbox.call('node_0', {'node_id': 'N0'}, call_id='start')
    with pytest.raises(ContractError, match='observed-result dependency'):
        sandbox.call('node_1', {'node_id': {'$result': 'start', 'path': 'items.0.label'}}, call_id='wrong_path')


@pytest.mark.parametrize('paths', [[], ['x', 'x'], [None], [''], ['x'] * 17])
def test_invalid_alternative_paths_rejected_during_compile(paths):
    value = source()
    value['tools'][1]['result_bindings'][0]['path'] = paths
    with pytest.raises(ContractError, match='dependency paths'):
        compile_contract(value)


def test_multiple_bindings_still_require_all_conditions():
    value = source()
    value['tools'][1]['result_bindings'].append({
        'argument': 'node_id', 'tool': 'node_0', 'path': 'items.*.next_id'})
    sandbox = Sandbox(value)
    sandbox.call('node_0', {'node_id': 'N0'}, call_id='start')
    with pytest.raises(ContractError, match='observed-result dependency'):
        sandbox.call('node_1', {'node_id': {'$result': 'start', 'path': 'items.0.id'}}, call_id='not_all')
