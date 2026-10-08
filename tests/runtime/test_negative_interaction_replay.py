from copy import deepcopy

import pytest

from env_factory.generation.negative_scenarios import derive_negatives
from env_factory.sandbox_runtime import AcceptanceScenarioRunner, SandboxError


def scenario():
    reference = {'scenario_id': 'success', 'kind': 'goal_success', 'steps': [
        {'operation': 'tool_call', 'tool_name': 'lookup', 'arguments': {'code': 'valid'}},
        {'operation': 'dialogue_turn', 'content': 'Choose an observed option', 'expected_stage': 'choice'},
        {'operation': 'tool_call', 'tool_name': 'write', 'arguments': {'choice': 'A'}},
        {'operation': 'agent_response', 'content': 'Booked'},
        {'operation': 'reward', 'step_id': 'reward'}]}
    definition = {'scenario_id': 'wrong_lookup', 'kind': 'goal_failure',
        'argument_mutations': [{'tool_name': 'lookup', 'arguments': {'code': 'invalid'}}],
        'assertions': [{'source': 'step:reward', 'path': '$.reward', 'operator': 'lte', 'expected': .2}]}
    return derive_negatives([reference], [definition])[-1]


def service(reward=0, status=200):
    calls = []
    def call(method, path, body, headers):
        calls.append(path)
        if path == '/v1/user_simulator':
            return status, {'user_query': 'No observed options', 'interaction_stage': None,
                'fsm_transition_applied': False, 'reason_code': 'interaction_requirement_unmet',
                'match_status': 'unmatched'}, {}
        if path == '/v1/reward':
            return 200, {'reward': reward}, {}
        return 200, {}, {}
    return call, calls


def test_rejected_downstream_dialogue_measures_reward_without_executing_write():
    call, calls = service()
    result = AcceptanceScenarioRunner(call).run(scenario())
    assert calls == ['/v1/tools/lookup', '/v1/agent_response', '/v1/user_simulator', '/v1/reward']
    assert result['step_results']['reward']['reward'] == 0
    assert result['counterfactual_stop']['observed_mutated_tools'] == ['lookup']
    assert result['history'][-2]['body']['fsm_transition_applied'] is False


def test_blocked_negative_cannot_pass_when_actual_reward_is_high():
    call, _ = service(reward=1)
    with pytest.raises(SandboxError, match='scenario assertion failed'):
        AcceptanceScenarioRunner(call).run(scenario())


@pytest.mark.parametrize('defect', ['positive', 'unmarked', 'not_all_mutations_executed', 'server_failure'])
def test_only_observed_declared_counterfactuals_can_stop(defect):
    case = scenario()
    call, calls = service(status=500 if defect == 'server_failure' else 200)
    if defect == 'positive':
        case['kind'] = 'goal_success'
    elif defect == 'unmarked':
        del case['counterfactual_argument_tools']
    elif defect == 'not_all_mutations_executed':
        case['counterfactual_argument_tools'].append('write')
    with pytest.raises(SandboxError):
        AcceptanceScenarioRunner(call).run(case)
    assert '/v1/reward' not in calls


def test_reward_observed_before_block_is_not_usable_as_termination_reward():
    case = scenario()
    case['steps'].insert(0, case['steps'].pop())
    call, _ = service()
    with pytest.raises(SandboxError, match='subsequent reward observation'):
        AcceptanceScenarioRunner(call).run(case)


def test_unrelated_wrong_stage_is_not_a_known_protocol_rejection():
    call, _ = service()
    def wrong_stage(method, path, body, headers):
        status, result, response_headers = call(method, path, body, headers)
        if path == '/v1/user_simulator':
            result = {**result, 'fsm_transition_applied': True, 'interaction_stage': 'different'}
        return status, result, response_headers
    with pytest.raises(SandboxError, match='interaction branch did not execute'):
        AcceptanceScenarioRunner(wrong_stage).run(scenario())


def test_realism_retry_preserves_declared_mutation_and_measures_reward():
    case = scenario()
    retry = deepcopy(case['steps'][0])
    case['steps'][0]['expected_status'] = 503
    case['steps'].insert(1, retry)
    base, calls = service()
    lookup_count = 0

    def call(method, path, body, headers):
        nonlocal lookup_count
        result = base(method, path, body, headers)
        if path == '/v1/tools/lookup':
            lookup_count += 1
            if lookup_count == 1:
                return 503, {'error': {'code': 'TEMPORARILY_UNAVAILABLE'}}, {}
        return result

    result = AcceptanceScenarioRunner(call).run(case)
    assert lookup_count == 2
    assert result['step_results']['reward']['reward'] == 0
    assert '/v1/tools/write' not in calls


def test_retry_cannot_silently_restore_unmutated_argument():
    case = scenario()
    retry = deepcopy(case['steps'][0])
    retry['arguments']['code'] = 'valid'
    case['steps'].insert(1, retry)
    call, calls = service()
    with pytest.raises(SandboxError, match='counterfactual provenance'):
        AcceptanceScenarioRunner(call).run(case)
    assert calls == []


def test_mutation_metadata_cannot_merge_different_logical_requests():
    case = scenario()
    retry = deepcopy(case['steps'][0])
    retry['arguments']['unmutated_scope'] = 'different'
    case['steps'].insert(1, retry)
    call, calls = service()
    with pytest.raises(SandboxError, match='counterfactual provenance'):
        AcceptanceScenarioRunner(call).run(case)
    assert calls == []


@pytest.mark.parametrize('defect', [None, 'changed_arguments', 'changed_key', 'missing_key'])
def test_existing_counterfactual_allows_only_one_identical_idempotent_request(defect):
    case = scenario()
    del case['counterfactual_argument_mutations']
    case['steps'][0]['idempotency_key'] = 'lookup-request'
    retry = deepcopy(case['steps'][0])
    if defect == 'changed_arguments':
        retry['arguments']['code'] = 'another'
    elif defect == 'changed_key':
        retry['idempotency_key'] = 'different'
    elif defect == 'missing_key':
        retry.pop('idempotency_key')
    case['steps'].insert(1, retry)
    call, calls = service()
    if defect:
        with pytest.raises(SandboxError, match='counterfactual provenance'):
            AcceptanceScenarioRunner(call).run(case)
        assert calls == []
    else:
        result = AcceptanceScenarioRunner(call).run(case)
        assert result['step_results']['reward']['reward'] == 0


def missing_capture_case():
    case = scenario()
    case['steps'][0]['capture'] = {'selected_id': '$.records[0].id'}
    case['steps'].pop(1)
    case['steps'][1]['arguments'] = {'choice': {'$ref': 'selected_id'}}
    return case


def missing_capture_service(reward=0, status=400, code='INVALID_ARGUMENT', message='arguments.choice must be string'):
    calls = []
    def call(method, path, body, headers):
        calls.append((path, body))
        if path == '/v1/tools/lookup':
            return 200, {'records': [], 'count': 0}, {}
        if path == '/v1/tools/write':
            assert body == {'choice': None}
            return status, {'error': {'code': code, 'message': message}}, {}
        if path == '/v1/reward':
            return 200, {'reward': reward}, {}
        return 200, {}, {}
    return call, calls


def test_missing_mutated_query_capture_retains_actual_client_rejection_and_reward():
    call, calls = missing_capture_service()
    result = AcceptanceScenarioRunner(call).run(missing_capture_case())
    assert [path for path, _ in calls] == ['/v1/tools/lookup', '/v1/tools/write', '/v1/reward']
    assert result['counterfactual_stop']['missing_captures'] == ['selected_id']
    assert result['history'][1]['status'] == 400
    assert result['step_results']['reward']['reward'] == 0


@pytest.mark.parametrize('defect', ['high_reward', 'server_error', 'other_error',
                                 'unmarked', 'unrelated_capture', 'positive', 'reward_before_stop', 'other_argument'])
def test_missing_capture_never_turns_unrelated_errors_or_high_reward_into_pass(defect):
    case = missing_capture_case()
    call, calls = missing_capture_service(reward=1 if defect == 'high_reward' else 0,
        status=500 if defect == 'server_error' else 400,
        code='OTHER_ERROR' if defect == 'other_error' else 'INVALID_ARGUMENT',
        message='arguments.other must be integer' if defect == 'other_argument' else 'arguments.choice must be string')
    if defect == 'unmarked':
        del case['counterfactual_argument_tools']
        del case['counterfactual_argument_mutations']
    elif defect == 'unrelated_capture':
        del case['steps'][0]['capture']
    elif defect == 'positive':
        case['kind'] = 'goal_success'
    elif defect == 'reward_before_stop':
        case['steps'].insert(0, case['steps'].pop())
    with pytest.raises(SandboxError):
        AcceptanceScenarioRunner(call).run(case)
