import copy
import json
from unittest.mock import Mock

import pytest

from env_factory_v2.code_agent import NativeCodeAgent
from env_factory_v2.contract import ContractError, compile_contract
from env_factory_v2.examples import direct, simple
from env_factory_v2.pipeline import build
from env_factory_v2.repair import apply_edits, recover_edit_values


def test_edit_syntax_recovery_preserves_paths_values_and_atomic_application():
    edits = [{'op': 'replace', 'path': '/a', 'value_json': '{"x":1 "y":2}'},
             {'op': 'replace', 'path': '/b', 'value_json': '[3 4]'}]
    original = copy.deepcopy(edits)
    agent = Mock()
    agent.ask.return_value = ({'replacements': [
        {'old': '1 "y"', 'new': '1, "y"'}, {'old': '[3 4]', 'new': '[3,4]'}]}, {'test': True})
    recovered, evidence = recover_edit_values(edits, agent)
    assert agent.ask.call_count == 1 and evidence['provenance'] == {'test': True}
    assert edits == original
    assert [(e['op'], e['path']) for e in recovered] == [('replace', '/a'), ('replace', '/b')]
    assert apply_edits({'a': {}, 'b': []}, recovered) == {'a': {'x': 1, 'y': 2}, 'b': [3, 4]}


def test_edit_recovery_rejects_business_value_change():
    agent = Mock()
    agent.ask.return_value = ({'replacements': [
        {'old': '1 "y":2', 'new': '1,"y":200'}]}, {})
    with pytest.raises(ContractError, match='changed business content'):
        recover_edit_values([{'op': 'replace', 'path': '/a', 'value_json': '{"x":1 "y":2}'}], agent)


def test_valid_edits_need_no_extra_call_and_unquoted_prose_is_not_recovered():
    agent = Mock()
    edits = [edit('replace', '/a', 'text')]
    assert recover_edit_values(edits, agent) == (edits, None)
    with pytest.raises(ContractError, match='unsupported JSON content repair'):
        recover_edit_values([{'op': 'replace', 'path': '/a', 'value_json': 'unquoted prose'}], agent)
    agent.ask.assert_not_called()


def edit(op, path, value=None):
    return {'op': op, 'path': path, 'value_json': json.dumps(value)}


def test_atomic_ordered_edits_and_escaped_paths_preserve_input():
    source = {'items': [1, 2], 'a/b': {'~key': 3}}
    original = copy.deepcopy(source)
    assert apply_edits(source, [edit('add', '/items/1', 7), edit('remove', '/items/0'),
                               edit('add', '/items/-', 9), edit('replace', '/a~1b/~0key', 4)]) == {
                                   'items': [7, 2, 9], 'a/b': {'~key': 4}}
    assert source == original
    with pytest.raises(ContractError, match='does not exist'):
        apply_edits(source, [edit('replace', '/items/0', 5), edit('remove', '/missing')])
    assert source == original


@pytest.mark.parametrize('edits', [[], [edit('replace', '', {})], [edit('replace', '/items/01', 4)],
                                 [edit('replace', '/items/-', 4)], [edit('add', '/items', [])],
                                 [edit('replace', '/a~2b', 0)], [edit('remove', '/items/2')]])
def test_invalid_edits_never_silently_rewrite_other_fields(edits):
    with pytest.raises(ContractError):
        apply_edits({'items': [1, 2]}, edits)


def test_native_repair_requests_edits_and_result_still_requires_compilation(tmp_path):
    source = direct()
    agent = NativeCodeAgent(base_url='http://127.0.0.1:9999/v1', token='test-token-0000000000000000000000',
                            evidence_root=tmp_path)
    agent.ask = Mock(return_value=({'edits': [edit('replace', '/answers/0/expected', {'literal': 99})]},
                                  {'unit_test': True}))
    result, _ = agent.generate({'goal': source['task']}, feedback={'previous_source': source, 'error': 'test'})
    assert result['answers'][0]['expected'] == {'literal': 99}
    assert source['answers'][0]['expected'] == {'literal': 18}
    payload = agent.ask.call_args.args[1]
    assert payload['previous_source'] == source and payload['repair_feedback'] == {'error': 'test'}
    assert set(agent.ask.call_args.args[2]['properties']) == {'edits'}
    invalid = apply_edits(source, [edit('remove', '/answers')])
    with pytest.raises(ContractError):
        compile_contract(invalid)


def test_pipeline_passes_actual_failed_contract_and_current_error_to_repair(tmp_path):
    source = simple()
    source['scenarios'][0]['steps'] = source['scenarios'][0]['steps'][1:]
    agent = Mock()
    agent.generate.side_effect = [(source, {'unit_test': True}), RuntimeError('stop after inspecting repair input')]
    result = build({'goal': source['task']}, agent=agent, output=tmp_path / 'run', repairs=1)
    assert result['qualified'] is False
    feedback = agent.generate.call_args.kwargs['feedback']
    assert feedback['previous_source'] == source
    assert feedback['recipe_preflight']['passed'] is False
    assert any(check['error'] for check in feedback['recipe_preflight']['checks'])
