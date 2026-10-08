from copy import deepcopy
import json
import random

import pytest

from env_factory.generation.execution_routes import compile_route_probes
from env_factory.generation.reference_requests import validate_reference_requests


def fixture():
    scripts = [{'script_id': f'script-{i+1}', 'interaction_protocol': {'stages': [
        {'id': 'choose', 'before_tool': 'read', 'reference_response': f'choose {i}'}]}}
        for i in range(3)]
    def ref(i):
        return {'calls': [{'tool_name': 'read', 'arguments': {'scope': i}}],
                'answer': json.dumps({'scope': i}), 'user_script_indices': [i]}
    probe = {'route_id': 'empty', 'initial_changes': [], 'reference': ref(0),
             'reference_variants': [{'id': f'variant{i}', **ref(i)} for i in [1, 2]]}
    source = {'scenarios': [], 'route_probes': [probe]}
    spec = {'execution_routes': [{'id': 'empty', 'required_tools': ['read'],
             'required_stages': ['choose'], 'writes': 'forbidden'}]}
    return source, spec, scripts


def test_each_script_gets_its_own_probe_query_answer_and_seed():
    source, spec, scripts = fixture()
    original = deepcopy(source)
    result = compile_route_probes(source, spec, scripts)[0]
    successes = [s for s in result['scenarios'] if s['kind'] == 'goal_success']
    assert len(successes) == 3
    assert source == original
    for index, case in enumerate(successes):
        assert random.Random(case['steps'][0]['body']['seed']).randrange(3) == index
        assert case['steps'][1]['content'] == f'choose {index}'
        assert case['steps'][2]['arguments'] == {'scope': index}
        assert json.loads(case['steps'][3]['content']) == {'scope': index}
        assert case['assertions'][0]['expected'] == 1
    assert len(result['scenarios']) == 4  # one separate empty-trajectory negative


@pytest.mark.parametrize('defect', ['missing', 'overlap', 'duplicate_id', 'bad_index',
                                 'empty_scope', 'missing_tool', 'no_scripts'])
def test_route_probe_variants_cannot_omit_or_duplicate_required_coverage(defect):
    source, spec, scripts = fixture()
    probe = source['route_probes'][0]
    if defect == 'missing': probe['reference_variants'].pop()
    elif defect == 'overlap': probe['reference'].pop('user_script_indices')
    elif defect == 'duplicate_id': probe['reference_variants'][1]['id'] = 'variant1'
    elif defect == 'bad_index': probe['reference']['user_script_indices'] = [True]
    elif defect == 'empty_scope': probe['reference']['user_script_indices'] = []
    elif defect == 'missing_tool': probe['reference_variants'][1]['calls'][0]['tool_name'] = 'other'
    elif defect == 'no_scripts': scripts = []
    with pytest.raises(ValueError):
        compile_route_probes(source, spec, scripts)


def test_variant_probe_requests_receive_same_idempotency_and_fault_validation():
    source, _, _ = fixture()
    source.update(tools=[{'function': {'name': 'read', 'parameters': {}}}],
        business_lifecycle={'tools': {'read': {'require_idempotency': True}}})
    probe = source['route_probes'][0]
    for reference in [probe['reference'], *probe['reference_variants']]:
        reference['calls'][0]['idempotency_key'] = 'read-key'
    validate_reference_requests(source)
    del probe['reference_variants'][1]['calls'][0]['idempotency_key']
    with pytest.raises(ValueError, match=r'reference_variants\[1\].*idempotency_key'):
        validate_reference_requests(source)
    probe['reference_variants'][1]['calls'][0]['idempotency_key'] = 'read-key'
    source['business_lifecycle']['dynamics'] = {'faults': [{'tool': 'read', 'call_index': 1, 'status': 503}]}
    for reference in [probe['reference'], probe['reference_variants'][0]]:
        reference['calls'][0]['expected_status'] = 503
    with pytest.raises(ValueError, match=r'REFERENCE_FAULT_STATUS_CONFLICT.*reference_variants\[1\]'):
        validate_reference_requests(source)


@pytest.mark.parametrize('location', ['primary', 'probe', 'probe_variant'])
def test_reference_cannot_call_tools_omitted_from_its_route(location):
    source, spec, scripts = fixture()
    extra = {'tool_name': 'read_final_details', 'arguments': {}}
    if location == 'primary':
        source['scenarios'] = [{'scenario_id': 'primary', 'kind': 'goal_success',
            'execution_route': 'empty', 'steps': [
                {'operation': 'tool_call', 'tool_name': 'read', 'arguments': {}},
                {**extra, 'operation': 'tool_call'}]}]
    elif location == 'probe':
        source['route_probes'][0]['reference']['calls'].append(extra)
    else:
        source['route_probes'][0]['reference_variants'][0]['calls'].append(extra)
    with pytest.raises(ValueError, match='EXECUTION_ROUTE_REFERENCE_MISMATCH.*read_final_details'):
        compile_route_probes(source, spec, scripts)
