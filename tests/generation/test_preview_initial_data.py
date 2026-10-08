from unittest.mock import patch
import pytest
from tests.generation.test_agent_authoring import fixture
from env_factory.sandbox_runtime import SandboxError

from env_factory.generation import agent_authoring as authoring


@pytest.mark.parametrize('defect',['wrong_type','duplicate_unique'])
def test_initial_data_failure_is_detected_without_execution_or_judge(tmp_path, defect):
    source, request=fixture('multi_step_agentic')
    table=source['tables'][0]
    if defect=='wrong_type':
        for row in table['rows']:row['temperature']='18'
    else:
        table['indexes']=[{'columns':['temperature'],'unique':True}]
        for row in table['rows']:row['temperature']=18
    with patch.object(authoring,'verify_execution',side_effect=AssertionError('must not execute')):
        with pytest.raises(SandboxError):
            authoring.compile_source(source,request=request,root=tmp_path,structural_preview=True)


def test_valid_preview_still_not_certified(tmp_path):
    source,request=fixture('multi_step_agentic')
    with patch.object(authoring,'verify_execution',side_effect=AssertionError('must not execute')):
        result=authoring.compile_source(source,request=request,root=tmp_path,structural_preview=True)
    assert result['task_readiness']['ready'] is False
    assert result['generation_pipeline']['verification']['passed'] is False
