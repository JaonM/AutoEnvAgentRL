"""Replay amendments preserve authority tails and every non-description answer constraint."""
import copy
import importlib
import json
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'experiments'))
replay = importlib.import_module('replay_world_dialogues')


def fixture_request():
    contract = {'policy': 'one call only', 'tools': [
        {'type': 'function', 'function': {'name': 'read', 'parameters': {
            'type': 'object', 'properties': {}, 'required': []}}}],
        'constraints': {'max_tool_calls_per_turn': 1}}
    request = {'goal_schema': {'query': {'outcome': {'properties': {'status': {
        'type': 'string', 'enum': ['completed'], 'description': 'old'}}}}},
        'user_facts': {'goals': [{'process': 'query'}]}, 'tables': {}, 'time': {'start': 0},
        'business_rules': 'Original rule.'}
    replay.contracts.attach(request, contract)
    request['business_rules'] += '\n\nAdditional binding business rule.'
    return request


CHANGES = {'policy_replacements': [{'old': 'one call only', 'new': 'independent calls allowed'}],
           'constraints': {'max_tool_calls_per_turn': None, 'allow_text_with_tool_calls': True},
           'outcome_descriptions': [{'processes': ['query'], 'field': 'status', 'description': 'clarified'}]}


def test_amend_preserves_extra_rules_and_original_input():
    request = fixture_request()
    before = copy.deepcopy(request)
    result = replay.amend(request, CHANGES)
    assert request == before
    assert result['business_rules'].endswith('Additional binding business rule.')
    assert result['agent_policy'] == 'independent calls allowed'
    assert result['public_contract_sha256'] == replay.contracts.digest(result['public_contract'])
    with pytest.raises(ValueError):
        replay.amend(result, CHANGES)


def test_requalification_preserves_partial_credit_metadata(tmp_path, monkeypatch):
    request = fixture_request()
    schema = {'fields': {'goal_1': {'type': 'object', 'properties': {
        'status': {'type': 'string', 'enum': ['completed'], 'partial': False, 'description': 'old'}}}}}
    loaded = {'request': request, 'schema': schema, 'env': Mock(), 'manifest': {'files': {}}, 'rubric': None}
    monkeypatch.setattr(replay.bundle, 'load_bundle', lambda *a, **kw: loaded)
    monkeypatch.setattr(replay.contracts, 'interface_problems', lambda *a: [])
    seen = []
    def qualify(candidate):
        seen.append(candidate['schema'])
        return {'passed': False}  # no export required for this admission check
    monkeypatch.setattr(replay.bundle, 'qualify_loaded', qualify)
    monkeypatch.setattr(replay, 'run_reference', lambda *a, **kw: {'answer': {'goal_1': {'status': 'completed'}}})
    result = replay.prepare_bundle(tmp_path / 'source', tmp_path / 'output/bundle', CHANGES)
    expected = copy.deepcopy(schema)
    expected['fields']['goal_1']['properties']['status']['description'] = 'clarified'
    assert seen == [expected]
    assert schema['fields']['goal_1']['properties']['status']['description'] == 'old'
    assert not result['passed']
    assert not (tmp_path / 'output/bundle').exists()
