import copy
import itertools

from tests.construction.test_build_workflow import load_script


def test_complete_dependency_reference_is_selected_independently_of_list_order():
    validator = load_script('validate_agentic_training_value.py')
    stop = {'kind': 'goal_success', 'scenario_id': 'refusal', 'execution_route': 'refused',
            'steps': [{'operation': 'tool_call', 'tool_name': 'lookup'}] * 5}
    complete = {'kind': 'goal_success', 'scenario_id': 'complete', 'execution_route': 'complete',
                'steps': [{'operation': 'tool_call', 'tool_name': name}
                          for name in ['lookup', 'lookup', 'read', 'read']]}
    task = {'task_spec': {'capability_dag': {'edges': [{'from_tool': 'lookup', 'to_tool': 'read'}]},
                         'execution_routes': [
                             {'id': 'refused', 'required_tools': ['lookup']},
                             {'id': 'complete', 'required_tools': ['lookup', 'read']}]}}
    original = copy.deepcopy([stop, complete, task])
    for scenarios in itertools.permutations([stop, complete]):
        assert validator.select_success_scenario(scenarios, task) is complete
    assert [stop, complete, task] == original


class BranchChangingApp:
    def __init__(self, valid_baseline=True):
        self.amount = 2
        self.valid_baseline = valid_baseline

    def business_snapshot(self):
        return {'inventory': [{'id': 1, 'amount': self.amount}]}

    def mutate_business_state(self, mutation):
        self.amount = mutation['changes']['amount']
        return {'updated_count': 1}

    def handle(self, method, path, body, headers):
        if path == '/v1/reset':
            self.amount = 2
            return 200, {}, {}
        if path == '/v1/tools/read':
            return 200, {'amount': self.amount}, {}
        if path == '/v1/user_simulator':
            branch = 'decline' if self.amount == 2 and self.valid_baseline else 'needs_information'
            return 200, {'interaction_stage': None, 'interaction_branch': branch,
                         'user_query': branch, 'fsm_transition_applied': True}, {}
        if path == '/v1/agent_response':
            return 200, {'accepted': True}, {}
        if path == '/v1/reward':
            return 200, {'reward': float(self.amount == 2)}, {}
        raise AssertionError(path)


def test_private_data_probe_observes_changed_branch_but_keeps_baseline_strict():
    validator = load_script('validate_agentic_training_value.py')
    scenario = {'steps': [
        {'operation': 'reset'},
        {'operation': 'tool_call', 'tool_name': 'read', 'arguments': {}},
        {'operation': 'dialogue_turn', 'content': 'Proceed?', 'expected_stage': None,
         'expected_branch': 'decline'},
        {'operation': 'agent_response', 'content': 'The observed amount was 2.'},
        {'operation': 'reward'}]}
    mutation = {'table': 'inventory', 'selector': {'id': 1}, 'changes': {'amount': 3}}
    original = copy.deepcopy(scenario)
    result = validator.probe_reference_data_reward_sensitivity(
        BranchChangingApp(), {}, scenario, mutation, .8)
    assert result['proved'] is True
    assert result['changed_rewards'] == [0, 0]
    assert result['execution_errors'] == []
    assert scenario == original
    invalid = validator.probe_reference_data_reward_sensitivity(
        BranchChangingApp(valid_baseline=False), {}, scenario, mutation, .8)
    assert invalid['proved'] is False
    assert invalid['reason'] == 'SandboxError'
