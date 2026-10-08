from copy import deepcopy

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.reference_requests import validate_reference_requests
from tests.generation.test_dynamic_state_goal import dynamic_source


def test_fault_status_preflight_handles_commit_retry_and_reset():
    from env_factory.generation.reference_requests import validate_reference_fault_statuses
    source = {'business_lifecycle': {'dynamics': {'faults': [
        {'tool': 'write', 'call_index': 1, 'phase': 'after_commit', 'status': 504}]}},
        'scenarios': [{'kind': 'goal_success', 'steps': [
            {'operation': 'tool_call', 'tool_name': 'write', 'idempotency_key': 'a'},
            {'operation': 'tool_call', 'tool_name': 'write', 'idempotency_key': 'a'}]}]}
    with pytest.raises(ValueError, match='REFERENCE_FAULT_STATUS_CONFLICT') as exc:
        validate_reference_fault_statuses(source)
    assert 'steps[0].expected_status' in str(exc.value)
    assert 'steps[1].expected_status' not in str(exc.value)
    steps = source['scenarios'][0]['steps']
    steps[0]['expected_status'] = 504
    validate_reference_fault_statuses(source)
    steps.extend([{'operation': 'reset'}, {'tool_name': 'write', 'idempotency_key': 'a'}])
    with pytest.raises(ValueError, match=r'steps\[3\]'):
        validate_reference_fault_statuses(source)


def test_before_fault_retry_advances_attempt_and_probe_is_checked():
    from env_factory.generation.reference_requests import validate_reference_fault_statuses
    source = {'business_lifecycle': {'dynamics': {'faults': [
        {'tool': 'read', 'call_index': 2, 'phase': 'before', 'status': 503}]}},
        'route_probes': [{'reference': {'calls': [{'tool_name': 'read'}, {'tool_name': 'read'}]}}]}
    with pytest.raises(ValueError, match=r'route_probes\[0\].reference.calls\[1\]'):
        validate_reference_fault_statuses(source)
    calls = source['route_probes'][0]['reference']['calls']
    calls[1]['expected_status'] = 503
    calls.append({'tool_name': 'read'})
    validate_reference_fault_statuses(source)


def source_with_required_key():
    source, request = dynamic_source()
    source['business_lifecycle'] = {'version': '1.0', 'tools': {
        'update_temperature': {'require_idempotency': True}}}
    return source, request


def test_missing_key_is_rejected_by_structural_preview(tmp_path):
    source, request = source_with_required_key()
    with pytest.raises(ValueError, match=r'REFERENCE_REQUEST_INVALID:.*scenarios\[0\].steps\[2\].idempotency_key'):
        compile_source(source, request=request, root=tmp_path, structural_preview=True)


def test_misplaced_key_reports_both_paths_and_preserves_explicit_negatives():
    source, _ = source_with_required_key()
    step = source['scenarios'][0]['steps'][2]
    step['arguments']['idempotency_key'] = 'request-1'
    with pytest.raises(ValueError) as exc:
        validate_reference_requests(source)
    assert '.arguments.idempotency_key' in str(exc.value)
    assert '.steps[2].idempotency_key' in str(exc.value)
    step['expected_status'] = 400
    validate_reference_requests(source)


def test_route_probe_keys_are_checked_and_captures_are_accepted():
    source, _ = source_with_required_key()
    step = source['scenarios'][0]['steps'][2]
    step['idempotency_key'] = {'$ref': 'request_key'}
    probe_call = deepcopy(step)
    probe_call.pop('idempotency_key')
    source['route_probes'] = [{'reference': {'calls': [probe_call]}}]
    with pytest.raises(ValueError, match=r'route_probes\[0\].reference.calls\[0\].idempotency_key'):
        validate_reference_requests(source)
    probe_call['idempotency_key'] = 'alternate-request'
    validate_reference_requests(source)
