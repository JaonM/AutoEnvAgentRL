import copy

import pytest

from env_factory_v2.contracts.contract import compile_contract, ContractError
from env_factory_v2.examples import direct, simple, transaction
from env_factory_v2.requirements.tool_interference import augment
from env_factory_v2.validation.verification import preflight_recipes


def test_executable_partial_views_preserve_task_and_hide_selection_labels():
    original = simple()
    source, report = augment(original)
    assert len(report['added']) == 2
    assert all(c['passed'] for c in report['execution_checks'])
    compiled = compile_contract(source)
    assert preflight_recipes(compiled)['passed']
    assert source['answers'] == original['answers']
    assert source['user'] == original['user']
    assert all('selection_role' not in t for t in compiled['public']['tools'])
    for tool in compiled['public']['tools']:
        assert 'Only returns columns:' in tool['description']
    again, _ = augment(source)
    assert again == source
    assert len(original['tools']) == 1


def test_direct_and_disabled_preserve_tool_set():
    assert augment(direct())[0]['tools'] == []
    assert augment(simple(), count=0)[0] == simple()


def test_paginated_interference_witness_checks_tool_pages_before_collection():
    original = simple()
    original['tables'][0]['rows'].append({'id': 'I3', 'size': 'M', 'display': '绿竹'})
    original['tools'][0]['page_size'] = 1
    for scenario in original['scenarios']:
        for step in scenario['steps']:
            if step['kind'] == 'call':
                step['kind'] = 'traverse'
    source, report = augment(original)
    assert preflight_recipes(compile_contract(source))['passed']
    for check in report['execution_checks']:
        assert check['passed']
        assert check['trace'][-1]['kind'] == 'collect'
        pages = [event for event in check['trace']
                 if event.get('tool') == check['tool']]
        assert len(pages) > 1
        assert all(event['status'] == 'ok' for event in pages)


def test_distractor_cannot_be_a_write():
    source = transaction()
    tool = next(t for t in source['tools'] if t['kind'] == 'transaction')
    tool['selection_role'] = 'distractor'
    with pytest.raises(ContractError, match='read-only'):
        compile_contract(source)
