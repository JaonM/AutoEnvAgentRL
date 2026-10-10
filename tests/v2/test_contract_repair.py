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
    agent.ask.return_value = ({'repairs': [
        {'index': 0, 'json': '{"x":1,"y":2}'}, {'index': 1, 'json': '[3,4]'}]}, {'test': True})
    recovered, evidence = recover_edit_values(edits, agent)
    assert agent.ask.call_count == 1 and evidence['provenance'] == {'test': True}
    assert edits == original
    assert [(e['op'], e['path']) for e in recovered] == [('replace', '/a'), ('replace', '/b')]
    assert apply_edits({'a': {}, 'b': []}, recovered) == {'a': {'x': 1, 'y': 2}, 'b': [3, 4]}


def test_edit_recovery_rejects_business_value_change():
    agent = Mock()
    agent.ask.return_value = ({'repairs': [{'index': 0, 'json': '{"x":1,"y":200}'}]}, {})
    with pytest.raises(ContractError, match='changed business content'):
        recover_edit_values([{'op': 'replace', 'path': '/a', 'value_json': '{"x":1 "y":2}'}], agent)


@pytest.mark.parametrize('repairs', [[], [{'index': 1, 'json': '[1,2]'}],
    [{'index': True, 'json': '[1,2]'}],
    [{'index': 0, 'json': '[1,2]'}, {'index': 0, 'json': '[1,2]'}]])
def test_edit_recovery_cannot_omit_duplicate_or_reassign_values(repairs):
    agent = Mock()
    agent.ask.return_value = ({'repairs': repairs}, {})
    with pytest.raises(ContractError, match='edit'):
        recover_edit_values([{'op': 'replace', 'path': '/a', 'value_json': '[1 2]'}], agent)


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


@pytest.mark.parametrize('same_request', [True, False])
def test_repair_payload_deduplicates_only_identical_request(same_request):
    request = {'goal': '查当前业务状态', 'category': 'direct_response'}
    included = copy.deepcopy(request) if same_request else {'goal': 'distinct review evidence'}
    feedback = {'previous_source': direct(), 'original_request': included,
                'recipe_preflight': {'passed': False, 'checks': [{'id': 'negative', 'error': 'actual error'}]}}
    before = copy.deepcopy(feedback)
    agent = Mock()
    agent.ask.return_value = ({'edits': [edit('replace', '/task', request['goal'])]}, {'test': True})
    NativeCodeAgent.generate(agent, request, feedback=feedback)
    payload = agent.ask.call_args.args[1]
    assert payload['original_request'] == request
    assert ('original_request' not in payload['repair_feedback']) is same_request
    if not same_request:
        assert payload['repair_feedback']['original_request'] == included
    assert payload['repair_feedback']['recipe_preflight'] == feedback['recipe_preflight']
    assert feedback == before


def test_candidate_alignment_failure_retains_source_for_atomic_repair(tmp_path):
    source = simple()
    agent = Mock()
    agent.generate.side_effect = [(source, {'unit_test': True}), RuntimeError('stop after repair input')]
    validator = Mock(side_effect=ContractError('frozen business records changed'))
    result = build({'goal': source['task']}, agent=agent, output=tmp_path / 'alignment',
                   repairs=1, candidate_validator=validator)
    assert not result['qualified']
    validator.assert_called_once()
    feedback = agent.generate.call_args.kwargs['feedback']
    assert feedback['previous_source'] == source
    assert feedback['error'] == 'frozen business records changed'
    assert feedback['failed_phase'] == 'compile'
    assert json.loads((tmp_path / 'alignment/attempt-1/source.json').read_text()) == source
