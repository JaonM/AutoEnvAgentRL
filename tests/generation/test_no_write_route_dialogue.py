from copy import deepcopy

import pytest

from env_factory.generation.interaction_contract import interleave_scenarios


def inputs():
    stages = [{'id': 'choose', 'before_tool': 'commit', 'reference_response': 'Choose or decline'},
              {'id': 'approve', 'before_tool': 'commit', 'reference_response': 'Confirm or refuse'}]
    scripts = [{'script_id': 'script-1', 'interaction_protocol': {'stages': stages}}]
    route = {'id': 'decline', 'writes': 'forbidden', 'required_tools': ['read'],
             'required_stages': ['choose', 'approve']}
    cases = [{'scenario_id': 'declined', 'kind': 'goal_success', 'execution_route': 'decline',
              'steps': [{'operation': 'tool_call', 'tool_name': 'read', 'arguments': {}},
                        {'operation': 'agent_response', 'content': 'No booking made.'}]}]
    return cases, scripts, route


def test_declared_refusal_dialogue_executes_without_calling_its_protected_write():
    cases, scripts, route = inputs()
    original = deepcopy(cases)
    steps = interleave_scenarios(cases, scripts, execution_routes=[route])[0]['steps']
    assert [step['operation'] for step in steps] == [
        'reset', 'tool_call', 'dialogue_turn', 'dialogue_turn', 'agent_response']
    assert [s['expected_stage'] for s in steps if s['operation'] == 'dialogue_turn'] == ['choose', 'approve']
    assert [s['tool_name'] for s in steps if s['operation'] == 'tool_call'] == ['read']
    assert cases == original


@pytest.mark.parametrize('required_write', [False, True])
def test_missing_required_tool_does_not_get_replaced_by_dialogue(required_write):
    cases, scripts, route = inputs()
    route['required_tools'].append('commit')
    if required_write:
        route['writes'] = 'required'
    with pytest.raises(ValueError, match='ordering cannot be reached'):
        interleave_scenarios(cases, scripts, execution_routes=[route])
