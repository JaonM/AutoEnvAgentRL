import pytest
from tests.generation.test_agent_authoring import fixture
from env_factory.generation.agent_authoring import compile_source


def test_unconsumed_tool_dependency_fails_before_execution(tmp_path):
    source,request=fixture('multi_step_agentic')
    source['tool_implementations'][1]['filters']=[]
    with pytest.raises(ValueError,match='QUERY_DEPENDENCY_UNUSED: read_temperature.id'):
        compile_source(source,root=tmp_path,request=request,structural_preview=True)


def test_real_query_dependency_is_accepted(tmp_path):
    source,request=fixture('multi_step_agentic')
    result=compile_source(source,root=tmp_path,request=request)
    assert result['generation_pipeline']['verification']['passed']
