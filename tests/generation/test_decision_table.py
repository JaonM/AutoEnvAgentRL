from copy import deepcopy
import pytest

from env_factory.generation.decision_table import expand_decision_table
from env_factory.generation.agent_authoring import compile_source
from tests.generation.test_execution_routes import source_with_routes


def decision_source():
    source, request = source_with_routes()
    routes = source.pop('execution_routes')
    probe = source.pop('route_probes')[0]
    branches = []
    for index, route in enumerate(routes):
        branch = {key: deepcopy(route[key]) for key in ('id', 'required_tools', 'required_stages', 'writes')}
        branch['when'] = {'needs_change': index == 0}
        if index == 1:
            branch['probe'] = {key: deepcopy(probe[key]) for key in ('initial_changes', 'reference')}
        branches.append(branch)
    source['decision_table'] = {'version': '1.0', 'decisions': {'needs_change': routes[0]['when']},
                                'branches': branches}
    return source, request


def test_both_decision_outcomes_execute_in_main_compiler(tmp_path):
    source, request = decision_source()
    result = compile_source(source, root=tmp_path, request=request)
    coverage = result['generation_pipeline']['verification']['execution_route_coverage']
    assert coverage['references'] == ['adjust']
    assert coverage['probes'][0]['route_id'] == 'already_set'
    assert coverage['probes'][0]['passed']
    assert source['decision_table']['branches'][0]['when'] == {'needs_change': True}


@pytest.mark.parametrize('change,reason', [('missing','missing'), ('overlap','overlap'), ('truthy','boolean')])
def test_missing_ambiguous_or_nonboolean_branches_rejected(change, reason):
    source, _ = decision_source()
    branches = source['decision_table']['branches']
    if change == 'missing': branches.pop()
    if change == 'overlap': branches[1]['when'] = {}
    if change == 'truthy': branches[1]['when']['needs_change'] = 'false'
    with pytest.raises(ValueError, match=reason):
        expand_decision_table(source)


def test_uncovered_branch_still_requires_executable_probe(tmp_path):
    source, request = decision_source()
    source['decision_table']['branches'][1].pop('probe')
    with pytest.raises(ValueError, match='UNCOVERED'):
        compile_source(source, root=tmp_path, request=request, structural_preview=True)


def test_conditional_targets_lower_to_one_goal_and_reward_definition():
    from env_factory.generation.business_contract import expand_business_contract
    source, _ = decision_source()
    for key in ('semantic_goal', 'outcomes'):
        source.pop(key, None)
    source['business_contract'] = {'version': '1.0',
        'state_targets': {'stock': {'table': 'stock', 'where': {'id': 1}, 'values': {'status': 'done'}, 'count': 1}},
        'outcomes': [{'id': 'result', 'rubric': 'Correct branch', 'targets': ['stock']}]}
    branches = source['decision_table']['branches']
    branches[0]['goal_values'] = {'stock': {'status': {'literal': 'done'}}}
    branches[1]['goal_values'] = {'stock': {'status': {'literal': 'unchanged'}}}
    result = expand_business_contract(expand_decision_table(source))
    target = result['semantic_goal']['row_predicates'][0]
    assert target == result['outcomes'][0]['rule']['expected'][0]
    assert target['value_expressions']['status']['if']['else'] == {'literal': 'unchanged'}
    branches[1].pop('goal_values')
    with pytest.raises(ValueError, match='same conditional goal fields'):
        expand_decision_table(source)


def test_branch_and_target_share_fact_resolution():
    from env_factory.generation.business_contract import expand_business_contract
    source, _ = decision_source()
    source.pop('semantic_goal', None)
    source.pop('outcomes', None)
    expression = source['decision_table']['decisions']['needs_change']
    source['decision_table']['decisions']['needs_change'] = {'fact': 'needs_change'}
    source['business_contract'] = {'version': '1.0', 'facts': {'needs_change': expression},
        'state_targets': {'stock': {'table': 'stock', 'where': {'id': 1},
                                   'values': {'status': 'done'}, 'count': 1}},
        'outcomes': [{'id': 'result', 'rubric': 'Correct branch', 'targets': ['stock']}]}
    result = expand_business_contract(expand_decision_table(source))
    assert result['execution_routes'][0]['when']['args'][0] == expression
    source['business_contract']['facts'] = {}
    with pytest.raises(ValueError, match='unknown or cyclic fact'):
        expand_business_contract(expand_decision_table(source))


def test_empty_conditional_goals_are_rejected():
    source, _ = decision_source()
    source['decision_table']['branches'][0]['goal_values'] = {}
    with pytest.raises(ValueError, match='same conditional goal fields'):
        expand_decision_table(source)
