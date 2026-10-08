from copy import deepcopy

import pytest

from env_factory.generation.agent_authoring import compile_source
from .test_dynamic_state_goal import dynamic_source


def source_with_routes():
    source, request = dynamic_source()
    source['metrics'][0]['weight'] = .8
    for implementation in source['tool_implementations']:
        name = implementation['tool_name']
        source['metrics'].append({'id': 'process_' + name, 'category': 'process', 'type': 'rule-based',
            'scope': 'step', 'weight': .2 / 3, 'score_range': [0, 1], 'rubric': '调用 ' + name,
            'target_action': name, 'evaluator': {'kind': 'trajectory_rule', 'source': 'runtime_rule',
                'assertion': '调用 ' + name, 'score_mapping': {'pass': 1, 'fail': 0}}})
    source['tool_implementations'][1]['projection'].append('temperature')
    def observed(field):
        return {'at_tool': {'name': 'read_temperature', 'expression': {
            'from_tool': {'name': 'read_temperature', 'field': field}}}}
    same = {'op': 'eq', 'args': [observed('temperature'), observed('configured_temperature')]}
    different = {**deepcopy(same), 'op': 'ne'}
    reads = ['lookup_zone', 'read_temperature']
    source['execution_routes'] = [
        {'id': 'adjust', 'when': different, 'required_tools': [*reads, 'update_temperature'],
         'required_stages': [], 'writes': 'required'},
        {'id': 'already_set', 'when': same, 'required_tools': reads,
         'required_stages': [], 'writes': 'forbidden'}]
    source['scenarios'][0]['execution_route'] = 'adjust'
    source['scenarios'][0]['steps'][-1]['content'] = '温度：25摄氏度。'
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'from_tool': {'name': 'read_temperature', 'field': 'configured_temperature'}}
    source['route_probes'] = [{'route_id': 'already_set', 'initial_changes': [
        {'table': 'zones', 'where': {'name': '青松库'}, 'values': {'temperature': 25}}],
        'reference': {'calls': deepcopy(source['scenarios'][0]['steps'][:2]), 'answer': '温度：25摄氏度。'}}]
    return source, request


def test_all_routes_receive_real_reference_and_ablations(tmp_path):
    source, request = source_with_routes()
    result = compile_source(source, request=request, root=tmp_path)
    coverage = result['generation_pipeline']['verification']['execution_route_coverage']
    assert coverage['references'] == ['adjust']
    probe = coverage['probes'][0]
    assert probe['route_id'] == 'already_set' and probe['passed']
    assert probe['checks']['reference_route_0']
    assert probe['checks']['no_tools_0'] and probe['checks']['wrong_answer_0']


def test_missing_route_probe_is_rejected_in_preview(tmp_path):
    source, request = source_with_routes()
    source['route_probes'] = []
    with pytest.raises(ValueError, match='UNCOVERED'):
        compile_source(source, request=request, root=tmp_path, structural_preview=True)


def test_probe_cannot_claim_a_different_route(tmp_path):
    source, request = source_with_routes()
    # Data still requires a write; the purported no-write reference must fail.
    source['route_probes'][0]['initial_changes'][0]['values']['temperature'] = 19
    with pytest.raises(ValueError, match='SCENARIO_FAILED|PREFLIGHT_FAILED'):
        compile_source(source, request=request, root=tmp_path)


def test_invalid_probe_data_does_not_count_as_a_passing_negative(tmp_path):
    source, request = source_with_routes()
    source['route_probes'][0]['initial_changes'][0]['values']['temperature'] = 'not an integer'
    from env_factory.sandbox_runtime import SandboxError
    with pytest.raises(SandboxError):
        compile_source(source, request=request, root=tmp_path)
