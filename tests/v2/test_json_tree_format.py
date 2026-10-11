from scripts.experiments.v2_legacy_generation import generate_legacy
import json
from unittest.mock import Mock

import pytest

from env_factory.code_agent_runtime import cli_schema
from env_factory_v2.native.code_agent import NativeCodeAgent
from env_factory_v2.contracts.contract import ContractError, compile_contract
from env_factory_v2.examples import direct, simple, retrieval, transaction, long_chain, complex_transaction
from env_factory_v2.native.json_tree import TREE_SCHEMA, encode_response, decode_response
from env_factory_v2.validation.verification import preflight_recipes


@pytest.mark.parametrize('factory', [direct, simple, retrieval, transaction, long_chain, complex_transaction])
def test_full_contract_roundtrip_preserves_compilation_and_reference_behavior(factory):
    original = factory()
    encoded = json.loads(json.dumps(encode_response(original), ensure_ascii=False))
    decoded = decode_response(encoded)
    assert decoded == original
    assert compile_contract(decoded)['identity'] == compile_contract(original)['identity']
    assert preflight_recipes(compile_contract(decoded))['passed']


def test_native_cli_retains_closed_recursive_schema_and_all_variants():
    assert cli_schema(TREE_SCHEMA) == TREE_SCHEMA
    def check(node):
        if isinstance(node, dict):
            if node.get('type') == 'object':
                assert node['additionalProperties'] is False
                assert set(node['required']) == set(node['properties'])
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)
    check(TREE_SCHEMA)


def test_dynamic_keys_unicode_empty_values_and_all_json_scalars_are_lossless():
    original = {'': None, 'a/b~key': [True, False, -1, 1.5, 10**500, '中\n"文', {}, []],
                'literal': {'o': 'a normal business key', 'a': 'not a wire node'}}
    assert json.dumps(decode_response(encode_response(original)), ensure_ascii=False) == json.dumps(original, ensure_ascii=False)


@pytest.mark.parametrize('node', [
    {}, {'s': 1}, {'n': True}, {'n': float('inf')}, {'b': 1}, {'z': False}, {'unknown': None},
    {'s': 'x', 'n': 1}, {'a': {}}, {'o': {}},
    {'o': [{'k': 'same', 'v': 1}, {'k': 'same', 'v': 2}]},
    {'o': [{'k': 'id', 'v': 1, 'extra': True}]},
    {'o': [{'k': 1, 'v': 1}]}, float('inf'), float('nan'),
])
def test_schema_or_decoder_cannot_accept_malformed_nodes_or_duplicate_keys(node):
    with pytest.raises(ContractError):
        decode_response({'source': {'o': [{'k': 'field', 'v': node}]}})


def test_duplicate_business_keys_are_rejected_instead_of_last_value_winning():
    with pytest.raises(ContractError, match='duplicate JSON tree object key'):
        decode_response({'source': {'o': [{'k': 'rule', 'v': True}, {'k': 'rule', 'v': False}]}})


def test_wire_validity_does_not_replace_business_validation():
    decoded = decode_response(encode_response({'task': 'missing rules and tools'}))
    with pytest.raises(ContractError, match='fields mismatch'):
        compile_contract(decoded)


def test_generation_uses_exact_tree_schema_and_returns_decoded_contract(tmp_path):
    original = complex_transaction()
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a' * 24, evidence_root=tmp_path)
    agent.ask = Mock(return_value=(encode_response(original), {'native': True}))
    source, provenance = generate_legacy(agent, {'goal': original['task']}, source_tree=True)
    assert source == original
    assert provenance['generation_format'] == 'closed_json_tree'
    assert agent.ask.call_args.args[2] == TREE_SCHEMA
    assert agent.ask.call_args.kwargs == {'source_tree': True}
    assert agent.ask.call_count == 1


def test_excessive_depth_and_conflicting_formats_fail_closed(tmp_path):
    node = 'x'
    for _ in range(130):
        node = [node]
    with pytest.raises(ContractError, match='complexity limit'):
        decode_response({'source': {'o': [{'k': 'field', 'v': node}]}})
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a' * 24, evidence_root=tmp_path)
    with pytest.raises(ContractError):
        generate_legacy(agent, {}, source_tree=True, source_object=True)
