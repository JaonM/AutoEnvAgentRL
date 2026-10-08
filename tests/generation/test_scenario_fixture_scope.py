"""Authored acceptance cases must execute through the production interfaces."""
from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source
from tests.generation.test_agent_authoring import fixture


@pytest.mark.parametrize('preview', [False, True])
@pytest.mark.parametrize('field', ['user_variables', 'initial_changes'])
def test_scenario_fixture_overrides_cannot_be_silently_ignored(tmp_path, preview, field):
    source, request = fixture('multi_step_agentic')
    source['scenarios'][1][field] = {'refuse': True} if field == 'user_variables' else []
    with patch('env_factory.generation.agent_authoring.verify_execution') as execute:
        with pytest.raises(ValueError, match='SCENARIO_FIXTURE_SCOPE.*failure.*' + field):
            compile_source(source, root=tmp_path, request=request, structural_preview=preview)
        execute.assert_not_called()
    assert not (tmp_path/'data').exists()


@pytest.mark.parametrize('preview', [False, True])
def test_private_mutation_callback_is_not_a_production_acceptance_operation(tmp_path, preview):
    source, request = fixture('multi_step_agentic')
    source['scenarios'][1]['steps'].insert(0, {'operation': 'mutate_business_state',
        'mutation': {'table': 'zones', 'selector': {'id': 821}, 'changes': {'temperature': 99}}})
    with patch('env_factory.generation.agent_authoring.verify_execution') as execute:
        with pytest.raises(ValueError, match='SCENARIO_FIXTURE_SCOPE.*mutate_business_state'):
            compile_source(source, root=tmp_path, request=request, structural_preview=preview)
        execute.assert_not_called()
    assert not (tmp_path/'data').exists()


def test_readonly_business_snapshot_executes_in_authoring_preflight(tmp_path):
    source, request = fixture('multi_step_agentic')
    case = source['scenarios'][0]
    case['steps'].insert(0, {'operation': 'business_snapshot', 'step_id': 'snapshot'})
    case['assertions'].append({'source': 'step:snapshot', 'path': '$.zones',
                               'operator': 'count_gte', 'expected': 2})
    artifact = compile_source(source, root=tmp_path, request=request)
    assert artifact['generation_pipeline']['verification']['passed']
