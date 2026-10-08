from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.scenario_assertions import validate_reward_assertion_paths
from .test_agent_authoring import fixture


@pytest.mark.parametrize('path', ['$.metrics.temperature_answer', '$.components.missing', '$.score'])
def test_invalid_reward_path_fails_before_execution(tmp_path, path):
    source, request = fixture('multi_step_agentic')
    source['scenarios'][1]['steps'].append({'operation': 'reward', 'step_id': 'score'})
    source['scenarios'][1]['assertions'] = [{'source': 'step:score', 'path': path,
        'operator': 'lte', 'expected': .2}]
    with patch('env_factory.generation.agent_authoring.verify_execution') as execute:
        with pytest.raises(ValueError, match='REWARD_ASSERTION_PATH: scenario=failure'):
            compile_source(source, request=request, root=tmp_path)
        execute.assert_not_called()


@pytest.mark.parametrize('path', ['$', '$.reward', '$.raw_reward', '$.components', '$.components.quality',
                                  '$.execution_route', '$.inactive_components[0]'])
def test_documented_reward_paths_remain_valid(path):
    validate_reward_assertion_paths([{'scenario_id': 'x', 'steps': [
        {'operation': 'reward', 'step_id': 'r'}], 'assertions': [
        {'source': 'step:r', 'path': path}]}], [{'id': 'quality'}], routed=True)


def test_non_reward_step_is_not_validated_against_reward_schema():
    validate_reward_assertion_paths([{'steps': [{'operation': 'reward', 'step_id': 'r'}],
        'assertions': [{'source': 'step:tool', 'path': '$.records[0].id'},
                       {'source': 'variables', 'path': '$.price'}]}], [{'id': 'quality'}])
