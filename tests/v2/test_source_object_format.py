from unittest.mock import Mock

import pytest

from env_factory_v2.code_agent import NativeCodeAgent
from env_factory_v2.contract import ContractError, compile_contract, validate_schema
from env_factory_v2.examples import direct, simple
from env_factory_v2.response_format import SOURCE_OBJECT_SCHEMA, validate_source_response


@pytest.mark.parametrize('source_factory', [direct, simple])
def test_direct_objects_preserve_executable_contract_without_inner_json_or_repair(tmp_path, source_factory):
    source = source_factory()
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a' * 24, evidence_root=tmp_path)
    agent.ask = Mock(return_value=({'source': source}, {'native': True}))
    result, provenance = agent.generate({'goal': source['task']}, source_object=True)
    assert result == source
    assert compile_contract(result)['identity'] == compile_contract(source)['identity']
    assert provenance['generation_format'] == 'native_source_object'
    assert provenance['strict_contract_grammar_enforced'] is False
    assert agent.ask.call_count == 1
    assert agent.ask.call_args.kwargs == {'source_object': True}
    assert agent.ask.call_args.args[2] == SOURCE_OBJECT_SCHEMA


@pytest.mark.parametrize('value', [{}, {'source': '{}'}, {'source': []},
                                  {'source': {}, 'extra': True}, {'source': {'amount': float('nan')}}])
def test_envelope_rejects_non_objects_extra_fields_and_nonfinite_content(value):
    with pytest.raises(ValueError):
        validate_source_response(value)


def test_transport_format_cannot_weaken_tool_or_business_schema_validation():
    with pytest.raises(ContractError, match='forbid extra fields'):
        validate_schema(SOURCE_OBJECT_SCHEMA)
    validate_source_response({'source': {'id': 'syntactically-valid-but-not-a-contract'}})
    with pytest.raises(ContractError, match='fields mismatch'):
        compile_contract({'id': 'syntactically-valid-but-not-a-contract'})


def test_source_object_flag_cannot_bypass_arbitrary_response_schema_validation(tmp_path):
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a' * 24, evidence_root=tmp_path)
    with pytest.raises(ContractError, match='unsupported generation response format'):
        agent.ask('role', {}, {'type': 'object'}, source_object=True)


def test_unverified_object_format_is_not_the_default_pipeline_generation(tmp_path):
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a' * 24, evidence_root=tmp_path)
    agent.ask = Mock(return_value=({'source_json': '{}'}, {}))
    assert agent.generate({'goal': 'diagnostic'})[0] == {}
    assert not agent.ask.call_args.kwargs
    assert set(agent.ask.call_args.args[2]['properties']) == {'source_json'}
