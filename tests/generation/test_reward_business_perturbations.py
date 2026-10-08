"""Fixed multi-step tasks must reward changed facts, not memorized fixtures."""
import copy

import pytest

from env_factory.generation.agent_authoring import compile_source, verify_execution
from tests.generation.test_agent_authoring import fixture


@pytest.mark.parametrize('temperature', [-4, 0, 37])
@pytest.mark.parametrize('reward_origin', ['business_state', 'observed_tool_result'])
def test_changed_fact_accepts_updated_answer_and_rejects_stale_answer(tmp_path, temperature, reward_origin):
    source, request = fixture('multi_step_agentic')
    if reward_origin == 'observed_tool_result':
        source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
            'tool_result': {'name': 'read_temperature', 'path': '$.records[0].temperature'}}
    artifacts = compile_source(source, root=tmp_path, request=request)
    changes = [{'table': 'zones', 'where': {'id': 821},
                'values': {'temperature': temperature}}]
    # Replay unchanged tools and captures against changed data: the old answer
    # must fail even though it passed in the original fixture.
    with pytest.raises(ValueError, match='AGENT_PREFLIGHT_FAILED:.*reference_0'):
        verify_execution(copy.deepcopy(artifacts), tmp_path,
                         initial_changes=changes, verify_route_probes=False)
    candidate = copy.deepcopy(artifacts)
    for scenario in candidate['acceptance_contract']['executable_scenarios']:
        if scenario.get('kind') == 'goal_success':
            for step in scenario['steps']:
                if step['operation'] == 'agent_response':
                    step['content'] = f'温度：{temperature}摄氏度。'
    # Retain dependency, skipped-step, wrong-answer and authored-failure checks.
    # Existing probes contain the original answer, so recursive probe expansion
    # is disabled; this is a separate fixed-data perturbation regression.
    report = verify_execution(candidate, tmp_path, initial_changes=changes,
                              verify_route_probes=False)
    assert report['passed']
    assert all(report['checks'].values())


def test_unselected_record_change_does_not_penalize_correct_multistep_answer(tmp_path):
    source, request = fixture('multi_step_agentic')
    artifacts = compile_source(source, root=tmp_path, request=request)
    report = verify_execution(artifacts, tmp_path, initial_changes=[{
        'table': 'zones', 'where': {'id': 547}, 'values': {'temperature': 999}}],
        verify_route_probes=False)
    assert report['passed']
    assert all(report['checks'].values())
