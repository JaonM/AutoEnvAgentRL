from copy import deepcopy
import pytest

from env_factory.contracts.trajectory_probes import bind_success_steps, idempotent_call_groups, logical_trace_matches, swap_logical_calls
from env_factory.generation.agent_authoring import compile_source
from .test_dynamic_state_goal import dynamic_source


def test_runtime_grouping_resolves_captures_at_each_original_position():
    read = {'operation':'tool_call', 'tool_name':'read', 'arguments':{}, 'capture':{'n':'$.n'}}
    write = {'operation':'tool_call', 'tool_name':'write', 'idempotency_key':'request',
             'arguments':{'n':{'$ref':'n'}}}
    steps = [read, write, deepcopy(read), deepcopy(write)]
    history = [{'operation':'tool_call', 'status':200, 'body':body}
               for body in [{'n':4},{'saved':4},{'n':5},{'saved':5}]]
    bound = bind_success_steps(steps, history)
    assert bound[1]['arguments'] == {'n':4}
    assert bound[3]['arguments'] == {'n':5}
    assert idempotent_call_groups(bound) == [[0],[1],[2],[3]]
    history[2]['body']['n'] = 4
    assert idempotent_call_groups(bind_success_steps(steps, history)) == [[0],[1,3],[2]]
    assert steps[1]['arguments'] == {'n':{'$ref':'n'}}
    with pytest.raises(ValueError, match='every authored step'):
        bind_success_steps(steps, history[:-1])
    history[2]['status'] = 500
    with pytest.raises(ValueError, match='operation/status'):
        bind_success_steps(steps, history)


def test_real_reference_can_retry_a_committed_write(tmp_path):
    source, request = dynamic_source()
    steps = source['scenarios'][0]['steps']
    steps[2]['idempotency_key'] = 'same-business-request'
    steps.insert(3, deepcopy(steps[2]))
    result = compile_source(source, request=request, root=tmp_path)
    checks = result['generation_pipeline']['verification']['checks']
    assert checks['reference_0'] and checks['skip_0_2']
    assert 'skip_0_3' not in checks  # one actual request, not two necessity claims


def test_grouping_requires_same_request_and_same_capture_binding():
    call = {'operation': 'tool_call', 'tool_name': 'write', 'idempotency_key': 'request-A',
            'arguments': {'id': {'$ref': 'record.id'}}}
    producer = {'operation': 'tool_call', 'tool_name': 'read', 'arguments': {},
                'capture': {'record': '$.records[0]'}}
    assert idempotent_call_groups([producer, call, deepcopy(call)]) == [[0], [1, 2]]
    # Same textual $ref after recapture is not the same proven payload.
    assert idempotent_call_groups([producer, call, producer, call]) == [[0], [1], [2], [3]]
    for changed in [dict(call, idempotency_key='request-B'), dict(call, arguments={'id': 2}),
                    dict(call, tool_name='other')]:
        assert idempotent_call_groups([call, changed]) == [[0], [1]]
    assert idempotent_call_groups([call, {'operation': 'reset'}, call]) == [[0], [2]]
    plain = {k: v for k, v in call.items() if k != 'idempotency_key'}
    assert idempotent_call_groups([plain, plain]) == [[0], [1]]
    cached_producer = {**producer, 'idempotency_key': 'same-read'}
    assert idempotent_call_groups([cached_producer, call, cached_producer, call]) == [[0, 2], [1, 3]]


def test_real_reference_can_retry_read_used_by_reward_binding(tmp_path):
    source, request = dynamic_source()
    steps = source['scenarios'][0]['steps']
    steps[1]['idempotency_key'] = 'same-read'
    steps.insert(2, deepcopy(steps[1]))
    steps[2]['step_id'] = 'read_temperature_retry'
    result = compile_source(source, request=request, root=tmp_path)
    checks = result['generation_pipeline']['verification']['checks']
    assert checks['reference_0'] and checks['skip_0_1']
    assert 'skip_0_2' not in checks


def test_delayed_retry_trace_preserves_actual_commit_order():
    failed = {'operation': 'tool_call', 'tool_name': 'write', 'arguments': {},
              'idempotency_key': 'request', 'expected_status': 503}
    read = {'operation': 'tool_call', 'tool_name': 'inspect', 'arguments': {}}
    retry = {**failed, 'expected_status': 200}
    steps = [failed, read, retry]
    assert logical_trace_matches(steps, ['inspect', 'write'])
    assert not logical_trace_matches(steps, ['write', 'inspect'])
    lifecycle = {'dynamics': {'faults': [{'tool': 'write', 'phase': 'after_commit', 'status': 503}]}}
    assert logical_trace_matches(steps, ['write', 'inspect'], lifecycle)
    assert not logical_trace_matches(steps, ['write', 'inspect', 'write'], lifecycle)
    assert not logical_trace_matches(steps, ['inspect'], lifecycle)


def test_reordering_inverts_whole_requests_so_retry_cannot_repair_probe():
    read = {'operation': 'tool_call', 'tool_name': 'read', 'arguments': {}, 'idempotency_key': 'r'}
    write = {'operation': 'tool_call', 'tool_name': 'write', 'arguments': {}, 'idempotency_key': 'w'}
    steps = [deepcopy(read), deepcopy(write), deepcopy(read), deepcopy(write)]
    swap_logical_calls(steps, 0, 1)
    assert [step['tool_name'] for step in steps] == ['write', 'write', 'read', 'read']


def test_optional_observations_cannot_exempt_writes_or_first_reads():
    import pytest
    from env_factory.contracts.trajectory_probes import optional_observation_positions
    impl = [{'tool_name': 'read', 'operation': 'select'}, {'tool_name': 'write', 'operation': 'update'}]
    read = {'operation': 'tool_call', 'tool_name': 'read'}
    assert optional_observation_positions([read, {**read, 'optional_observation': True}], impl) == {1}
    with pytest.raises(ValueError, match='repeated read-only'):
        optional_observation_positions([{**read, 'optional_observation': True}], impl)
    with pytest.raises(ValueError, match='repeated read-only'):
        optional_observation_positions([{'operation': 'tool_call', 'tool_name': 'write'},
            {'operation': 'tool_call', 'tool_name': 'write', 'optional_observation': True}], impl)


def test_committed_recovery_does_not_create_resubmission_dependency():
    from env_factory.generation.agent_authoring import expand_source
    def source(key):
        return {'business_tools': [
            {'name': name, 'description': name, 'parameters': {'type': 'object', 'properties': {}},
             'implementation': {'operation': operation, 'table': 'jobs'}}
            for name, operation in [('inspect', 'select'), ('write', 'update'), ('status', 'select')]],
            'outcomes': [{'id': 'done', 'rubric': 'done', 'rule': {
                'source': 'business_state', 'path': '$', 'operator': 'state_predicates', 'expected': []}}],
            'business_lifecycle': {'version': '1.0', 'dynamics': {'faults': [
                {'tool': 'write', 'phase': 'after_commit'}]}},
            'reference': {'calls': [
                {'tool_name': 'inspect', 'arguments': {}, 'capture': {'id': '$.records[0].id'}},
                {'tool_name': 'write', 'arguments': {'id': {'$ref': 'id'}}, 'idempotency_key': 'request', 'expected_status': 504},
                {'tool_name': 'status', 'arguments': {'id': {'$ref': 'id'}}, 'capture': {'confirmed': '$.records[0].id'}},
                {'tool_name': 'write', 'arguments': {'id': {'$ref': 'confirmed'}}, 'idempotency_key': key}], 'answer': 'done'}}
    expanded = expand_source(source('request'), {})
    write = next(c for c in expanded['capability_plan'] if c['action_name'] == 'write')
    assert write['dependencies'] == ['inspect']
    expanded = expand_source(source('new-request'), {})
    write = next(c for c in expanded['capability_plan'] if c['action_name'] == 'write')
    assert write['dependencies'] == ['inspect', 'status']
