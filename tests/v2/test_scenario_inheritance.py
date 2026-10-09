import copy

import pytest

from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import long_chain, lit
from env_factory_v2.expressions import ContractError
from env_factory_v2.scenarios import expand_scenarios
from env_factory_v2.verification import preflight_recipes


def test_inherited_negative_runs_full_trajectory_with_unchanged_expectation():
    source = long_chain(10)
    before = copy.deepcopy(source)
    source['scenarios'][1] = {'id': 'wrong', 'extends': 'correct', 'expect_success': False,
                              'answer': {'value': lit(-1)}}
    source['tools'][1]['result_bindings'][0]['recency'] = 'latest'
    source['scenarios'].append({'id': 'stale', 'extends': 'correct', 'expect_success': False,
        'expect_error': 'latest successful business result',
        'append_steps': [{'kind': 'call', 'id': 'older', 'tool': 'node_1',
                         'arguments': {'node_id': {'$result': 'node_result_0', 'path': 'items.0.next_id'}}}]})
    compiled = compile_contract(source)
    negative = compiled['source']['scenarios'][1]
    assert negative['steps'] == before['scenarios'][1]['steps']
    assert 'extends' not in negative
    assert len(compiled['source']['scenarios'][-1]['steps']) == 11
    assert preflight_recipes(compiled)['passed']
    assert source['scenarios'][1]['extends'] == 'correct'


def test_inheritance_matches_full_source_identity_and_supports_forward_parent():
    full = long_chain(3)
    compressed = copy.deepcopy(full)
    compressed['scenarios'][1] = {'id': 'wrong', 'extends': 'correct', 'expect_success': False,
                                'answer': {'value': lit(-1)}}
    assert compile_contract(full)['identity'] == compile_contract(compressed)['identity']
    cases = expand_scenarios([{'id': 'child', 'extends': 'parent', 'expect_success': False,
                              'mutations': []}, {'id': 'parent', 'expect_success': True,
                                                'steps': [], 'answer': {}, 'mutations': [1]}])
    assert cases[0]['mutations'] == []
    cases[0]['steps'].append(1)
    assert cases[1]['steps'] == []


@pytest.mark.parametrize('cases', [
    [{'id': 'a', 'extends': 'a', 'expect_success': False}],
    [{'id': 'a', 'extends': 'missing', 'expect_success': False}],
    [{'id': 'a', 'extends': 'b'}, {'id': 'b'}],
    [{'id': 'a', 'append_steps': []}],
    [{'id': 'a'}, {'id': 'a'}],
])
def test_invalid_inheritance_never_discards_a_case(cases):
    with pytest.raises(ContractError):
        expand_scenarios(cases)


@pytest.mark.parametrize('reverse', [False, True])
def test_cached_parents_cannot_bypass_depth_limit(reverse):
    cases = [{'id': 's0', 'steps': [], 'answer': {}, 'expect_success': True}]
    cases += [{'id': f's{i}', 'extends': f's{i-1}', 'expect_success': False} for i in range(1, 34)]
    with pytest.raises(ContractError, match='depth limit'):
        expand_scenarios(list(reversed(cases)) if reverse else cases)
    assert len(expand_scenarios(cases[:33])) == 33
