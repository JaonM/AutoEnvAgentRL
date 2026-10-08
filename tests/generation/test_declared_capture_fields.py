"""Diagnose nonexistent return fields before invoking episodes or model judges."""
from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source
from .test_agent_authoring import fixture


@pytest.mark.parametrize('path', ['$.budget', '$.records[0].budget'])
def test_missing_declarative_capture_field_is_rejected_before_execution(tmp_path, path):
    source, request = fixture('multi_step_agentic')
    source['scenarios'][0]['steps'][0]['capture']['zone_id'] = path
    with patch('env_factory.generation.agent_authoring.verify_execution') as execute:
        with pytest.raises(ValueError, match='CAPTURE_PATH_INVALID:.*budget.*absent from closed output schema'):
            compile_source(source, request=request, root=tmp_path)
        execute.assert_not_called()


def test_valid_projected_capture_keeps_normal_runtime_validation(tmp_path):
    source, request = fixture('multi_step_agentic')
    result = compile_source(source, request=request, root=tmp_path)
    assert result['generation_pipeline']['verification']['passed']
