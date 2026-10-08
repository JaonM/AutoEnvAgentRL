from copy import deepcopy

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.realism_requirements import require_realism_coverage
from tests.generation.test_dynamic_state_goal import dynamic_source


def recovery_source():
    source, request = dynamic_source()
    request['realism_requirements'] = {'version': '1.0', 'mechanisms': ['external_change', 'fault_recovery']}
    source['business_lifecycle'] = {'version': '1.0', 'dynamics': {
        'tick_seconds': 1,
        'events': [{'id': 'competition', 'at_tick': 1, 'changes': [
            {'table': 'zones', 'selector': {'name': '白杨库'}, 'values': {'temperature': 24}}]}],
        'faults': [{'id': 'read_unavailable', 'tool': 'lookup_zone', 'call_index': 1, 'phase': 'before', 'status': 503}]}}
    steps = source['scenarios'][0]['steps']
    steps[0]['idempotency_key'] = 'lookup-request'
    failed = deepcopy(steps[0])
    failed.update(step_id='failed_lookup', expected_status=503)
    failed.pop('capture', None)
    steps.insert(0, failed)
    return source, request


def test_recovery_profile_cannot_remove_mechanics_to_pass_preview(tmp_path):
    source, request = recovery_source()
    source['business_lifecycle']['dynamics']['faults'] = []
    with pytest.raises(ValueError, match='REALISM_REQUIREMENT_MISSING'):
        compile_source(source, request=request, root=tmp_path, structural_preview=True)


def test_recovery_reference_has_executed_change_and_recovery_evidence(tmp_path):
    source, request = recovery_source()
    result = compile_source(source, request=request, root=tmp_path)
    evidence = result['generation_pipeline']['verification']['realism_observations']
    assert evidence[0]['changed_event_ids'] == ['competition']
    assert evidence[0]['recovered_fault_ids'] == ['read_unavailable']
    with pytest.raises(ValueError, match='REALISM_EXECUTION_UNCOVERED'):
        require_realism_coverage(result, [])


def test_allocator_preserves_categories_and_assigns_only_eligible_slots():
    import random
    from env_factory.generation.realism_requirements import allocate_realism_requirements
    categories = ['direct_response', 'simple_agentic'] + ['multi_step_agentic'] * 10
    allocation = allocate_realism_requirements(categories, available_modes=['stateful'], rng=random.Random(42))
    assert allocation[:2] == [{'version': '1.0', 'mechanisms': []}] * 2
    names = [tuple(r['mechanisms']) for r in allocation[2:]]
    assert names.count(()) == 5
    assert names.count(('external_change',)) == 3
    assert names.count(('fault_recovery',)) == 2
    assert allocation == allocate_realism_requirements(categories, available_modes=['stateful'], rng=random.Random(42))
    assert all(not row['mechanisms'] for row in allocate_realism_requirements(
        categories, available_modes=['reference_data'], rng=random.Random(42)))


def test_declared_mechanics_cannot_escape_coverage_without_profile(tmp_path):
    source, request = recovery_source()
    request.pop('realism_requirements')
    result = compile_source(source, request=request, root=tmp_path)
    assert result['realism_requirements']['mechanisms'] == ['external_change', 'fault_recovery']
    with pytest.raises(ValueError, match='REALISM_EXECUTION_UNCOVERED'):
        require_realism_coverage(result, [])


def test_normal_task_needs_no_events_or_faults(tmp_path):
    source, request = dynamic_source()
    result = compile_source(source, request=request, root=tmp_path)
    assert result['realism_requirements']['mechanisms'] == []


def test_fault_only_requirement_does_not_demand_unrelated_event(tmp_path):
    source, request = recovery_source()
    request['realism_requirements']['mechanisms'] = ['fault_recovery']
    source['business_lifecycle']['dynamics']['events'] = []
    result = compile_source(source, request=request, root=tmp_path)
    assert result['realism_requirements']['mechanisms'] == ['fault_recovery']


def test_external_change_does_not_force_agent_write_or_modification_intent():
    from env_factory.generation.realism_requirements import validate_realism_request
    source, request = recovery_source()
    request['realism_requirements']['mechanisms'] = ['external_change']
    source['business_lifecycle']['dynamics']['faults'] = []
    source['environment_plan']['mode'] = 'reference_data'
    source['description']['task_intent'] = 'query'
    source['tool_implementations'] = [t for t in source['tool_implementations'] if t['operation'] == 'select']
    result = validate_realism_request(source, request)
    assert result['mechanisms'] == ['external_change']
    assert source['description']['task_intent'] == 'query'
    source['environment_plan']['mode'] = 'stateless'
    with pytest.raises(ValueError, match='require business data'):
        validate_realism_request(source, request)


def test_singleton_runs_can_receive_each_mechanism_with_reproducible_seeds():
    import random
    from env_factory.generation.realism_requirements import allocate_realism_requirements
    observed = {tuple(allocate_realism_requirements(['multi_step_agentic'], available_modes=['stateful'],
                                                  rng=random.Random(seed))[0]['mechanisms']) for seed in range(30)}
    assert observed == {(), ('external_change',), ('fault_recovery',)}
