import copy

import pytest

from env_factory.generation.reward_pairs import derive_reward_pairs, run_reward_pairs


def artifact():
    return {'tool_implementations': [{'tool_name': 'read', 'operation': 'select'},
                                    {'tool_name': 'write', 'operation': 'update'}],
            'environment_plan': {'mode': 'reference_data'},
            'acceptance_contract': {'executable_scenarios': [{'scenario_id': 'success', 'kind': 'goal_success',
                'steps': [{'operation': 'reset'}, {'operation': 'tool_call', 'tool_name': 'read', 'arguments': {'id': 1}},
                          {'operation': 'agent_response', 'content': '{"answer":1}'}, {'operation': 'reward'}]}]}}


def test_platform_variants_preserve_value_and_do_not_replace_original_reference():
    source = artifact()
    before = copy.deepcopy(source)
    probes = derive_reward_pairs(source, [])
    assert source == before
    assert {p['kind'] for p in probes} == {'equivalent_answer', 'readonly_recovery'}
    report = run_reward_pairs(probes, source['acceptance_contract']['executable_scenarios'],
                              lambda steps: {'reward': 1., 'state': {'value': 1}})
    assert report['passed'] and report['measured'] == 2


def test_state_changes_and_execution_errors_are_inconclusive():
    source = artifact()
    pairs = derive_reward_pairs(source, [])
    def execute(steps):
        return {'reward': 1., 'state': {'calls': len(steps)}}
    report = run_reward_pairs(pairs, source['acceptance_contract']['executable_scenarios'], execute)
    assert report['inconclusive'] == 1 and not report['passed']
    assert report['false_negative_rate'] == 0


def test_recovery_false_negative_is_measured():
    source = artifact()
    pairs = derive_reward_pairs(source, [])
    report = run_reward_pairs(pairs, source['acceptance_contract']['executable_scenarios'],
        lambda steps: {'reward': 0. if len(steps) > 4 else 1., 'state': {}})
    assert report['false_negatives'] == 1 and report['false_negative_rate'] == .5


def test_mutating_or_capture_overriding_recovery_is_rejected():
    definition = {'id': 'recovery', 'kind': 'readonly_recovery', 'reference_id': 'success',
                  'before_tool': 'read', 'calls': [{'tool_name': 'write', 'arguments': {'id': 2}}]}
    with pytest.raises(ValueError, match='read-only'):
        derive_reward_pairs(artifact(), [definition])
    definition['calls'] = [{'tool_name': 'read', 'arguments': {}, 'capture': {'x': '$.id'}}]
    with pytest.raises(ValueError, match='read-only'):
        derive_reward_pairs(artifact(), [definition])


def test_external_events_prevent_automatic_read_repeat():
    source = artifact()
    source['business_lifecycle'] = {'dynamics': {'events': [{'at': 1}]}}
    assert all(p['kind'] != 'readonly_recovery' for p in derive_reward_pairs(source, []))
