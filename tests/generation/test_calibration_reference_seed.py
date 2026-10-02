from unittest.mock import patch

from env_factory.generation.agent_authoring import compile_source, verify_execution
from .test_interaction_contract import interactive_source


def test_calibration_replays_the_reference_variant_seed(tmp_path):
    source, request = interactive_source()
    for index, variant in enumerate(source['interaction_contract']['variants']):
        for stage in variant['stages']:
            stage['id'] += f'_variant_{index}'
    artifacts = compile_source(source, root=tmp_path, request=request)
    # No online metric is needed to test which script is replayed before calibration.
    with patch('env_factory.generation.semantic_reward.calibrate_semantic_outcomes',
               return_value={'passed': True}) as calibrate:
        result = verify_execution(artifacts, tmp_path, calibration={'test_metric': []})
    assert result['passed']
    calibrate.assert_called_once()
