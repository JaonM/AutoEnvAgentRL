import copy

from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.examples import long_chain, lit, ref
from env_factory_v2.validation.verification import run_scenario


def test_post_write_observation_requires_real_current_read_after_commit():
    from env_factory_v2.examples import transaction
    source = transaction()
    source['success'] = [{'id': 'verified_transfer', 'public': 'Read actual balances after transfer.',
        'when': lit(True), 'required_tools': ['accounts', 'transfer'], 'required_stages': ['approve'],
        'required_observations': [{'id': 'final_balances', 'tools': ['accounts'],
            'after_tools': ['transfer'], 'columns': ['id', 'balance'], 'expected': ref('data.accounts')}]}]
    scenario = copy.deepcopy(source['scenarios'][0])
    compiled = compile_contract(source)
    before_only = run_scenario(compiled, scenario)
    assert before_only['outcome']['reward'] == 0
    assert all(c['passed'] for c in before_only['outcome']['components'].values())
    scenario['steps'].append({'kind': 'call', 'tool': 'accounts', 'id': 'final', 'arguments': {}})
    assert run_scenario(compiled, scenario)['outcome']['reward'] == 1
    # Even unchanged ID columns require a new observation after the write.
    source['success'][0]['required_observations'][0].update(columns=['id'],
        expected={'map': {'items': ref('data.accounts'), 'value': {'object': {'id': ref('row.id')}}}})
    scenario['steps'].pop()
    assert run_scenario(compile_contract(source), scenario)['outcome']['reward'] == 0


def linked_source():
    source = long_chain(3)
    source['tools'] = source['tools'][:2]
    source['tools'][1]['result_bindings'][0]['tool'] = ['node_0', 'node_1']
    for scenario in source['scenarios']:
        scenario['steps'][2]['tool'] = 'node_1'
    chain = {'walk': {'items': ref('data.nodes'), 'start': lit('N0'),
                     'key': 'id', 'next': 'next_id', 'max_nodes': 100}}
    source['success'] = [{'id': 'complete_chain', 'public': 'Observe every reachable node.',
        'when': lit(True), 'required_tools': ['node_0', 'node_1'], 'required_stages': [],
        'required_observations': [{'id': 'all_nodes', 'tools': ['node_0', 'node_1'], 'columns': ['id'],
            'expected': {'map': {'items': chain, 'value': {'object': {'id': ref('row.id')}}}}}]}]
    return source


def test_correct_guess_cannot_replace_observation_of_new_terminal():
    source = linked_source()
    source['tables'][0]['rows'][2]['next_id'] = 'N3'
    source['tables'][0]['rows'].append({'id': 'N3', 'next_id': None, 'value': 14})
    compiled = compile_contract(source)
    incomplete = run_scenario(compiled, source['scenarios'][0])
    assert incomplete['outcome']['components']['value']['passed']
    assert incomplete['outcome']['reward'] == 0
    observation = incomplete['outcome']['process'][0]['observations'][0]
    assert observation == {'id': 'all_nodes', 'passed': False, 'required_count': 4}
    complete = copy.deepcopy(source['scenarios'][0])
    complete['steps'].append({'kind': 'call', 'tool': 'node_1', 'id': 'new_terminal',
        'arguments': {'node_id': {'$result': 'node_result_2', 'path': 'items.0.next_id'}}})
    assert run_scenario(compiled, complete)['outcome']['reward'] == 1


def test_current_relation_shortening_changes_required_observation_set():
    source = linked_source()
    source['tables'][0]['rows'][1]['next_id'] = None
    source['answers'][0]['expected'] = ref('data.nodes.1.value')
    scenario = copy.deepcopy(source['scenarios'][0])
    scenario['steps'] = scenario['steps'][:2]
    scenario['answer'] = {'value': ref('evidence.node_result_1.items.0.value')}
    result = run_scenario(compile_contract(source), scenario)
    assert result['outcome']['reward'] == 1
    assert result['outcome']['process'][0]['observations'][0]['required_count'] == 2
