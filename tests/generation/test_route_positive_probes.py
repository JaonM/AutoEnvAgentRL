import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from .test_readonly_execution_routes import readonly_routes


def source_with_positive_probe():
    source, request = readonly_routes()
    source['reward_pair_probes'] = [{'id': 'faithful_format', 'kind': 'equivalent_answer',
        'reference_id': source['scenarios'][0]['scenario_id'],
        'answer': '{"status":"found", "temperature":18}'}]
    return source, request


def test_route_variations_execute_in_their_own_business_context(tmp_path):
    source, request = source_with_positive_probe()
    artifacts = compile_source(source, request=request, root=tmp_path)
    primary = json.loads((tmp_path / 'reward_pair_validation.json').read_text())
    alternate_root = tmp_path / 'route_probes' / 'unresolved'
    alternate = json.loads((alternate_root / 'reward_pair_validation.json').read_text())
    assert primary['passed'] and alternate['passed']
    assert any(p['provenance'] == 'authored' for p in primary['results'])
    assert all(p['reference_id'].startswith('route_unresolved') for p in alternate['results'])
    assert {p['kind'] for p in alternate['results']} == {'equivalent_answer', 'readonly_recovery'}
    assert all(p['reference_reward'] == p['variant_reward'] == 1 for p in alternate['results'])
    route = artifacts['generation_pipeline']['verification']['execution_route_coverage']['probes'][0]
    assert route['passed'] and route['checks']['equivalent_and_recovery_pairs']
    assert route['checks']['no_tools_0'] and route['checks']['wrong_answer_0']


@pytest.mark.parametrize('location', ['primary_probe', 'alternate_reference'])
def test_wrong_business_values_still_fail_in_each_context(tmp_path, location):
    source, request = source_with_positive_probe()
    if location == 'primary_probe':
        source['reward_pair_probes'][0]['answer'] = '{"temperature":0,"status":"lookup_not_unique"}'
    else:
        source['route_probes'][0]['reference']['answer'] = '{"temperature":18,"status":"found"}'
    with pytest.raises(ValueError, match='DETERMINISTIC_PREFLIGHT_FAILED'):
        compile_source(source, request=request, root=tmp_path)
