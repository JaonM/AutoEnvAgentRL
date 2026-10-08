import copy

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.interaction_contract import bind_captured_interaction_arguments
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator
from tests.generation.test_interaction_contract import interactive_source


def captured_choice_source():
    source, request = interactive_source()
    request['require_structured_interactions'] = True
    source['metrics'][0]['weight'] = .8
    source['metrics'].append({**copy.deepcopy(source['metrics'][0]), 'id': 'selected_read',
        'category': 'process', 'scope': 'trajectory', 'target_action': 'read_temperature', 'weight': .2})
    source['description']['public_input']['initial_user_message'] = '先询问库区名称，区域为东区。查询后由我选择最终库区编号，再报告温度。'
    for variant in source['interaction_contract']['variants']:
        variant['stages'][0]['user_reply'] += '编号此前考虑547。'
        stage = variant['stages'][1]
        stage.update(private_fact=821, previous_fact=547, options=[821, 547],
                     reference_response=('请在821和547中选择编号。' if stage['kind'] == 'option_selection'
                                         else '是否仍按此前547号库区查询？'),
                     user_reply='最终选择821。', retry_reply='请先询问最终编号。',
                     before_tool='read_temperature', requires_tools=['lookup_zone'],
                     bind_to={'tool': 'read_temperature', 'argument': 'id'})
    return source, request


@pytest.mark.parametrize('expression', [False, True])
def test_captured_user_choice_passes_real_reference_execution(tmp_path, expression):
    source, request = captured_choice_source()
    if expression:
        source['scenarios'][0]['steps'][1]['arguments']['id'] = {
            '$expr': {'op': 'add', 'args': [{'$ref': 'zone_id'}, 0]}}
    original = copy.deepcopy(source)
    result = compile_source(source, root=tmp_path, request=request)
    assert result['generation_pipeline']['verification']['passed']
    assert source == original
    rule = next(r for r in result['metric_implementations']
                if r.get('expected', {}).get('tool_name') == 'read_temperature')
    for script in ('script-1', 'script-2', 'script-3'):
        option = rule['expected']['by_user_script'][script][0]
        assert option['arguments']['id'] == original['scenarios'][0]['steps'][1]['arguments']['id']
        assert option['disclosed_arguments'] == {'id': 821}


def test_wrong_captured_choice_for_one_script_fails_execution(tmp_path):
    source, request = captured_choice_source()
    stage = source['interaction_contract']['variants'][1]['stages'][1]
    stage.update(private_fact=547, previous_fact=821, user_reply='最终选择547。',
                 reference_response='是否仍按此前821号库区查询？')
    source['interaction_contract']['variants'][1]['stages'][0]['user_reply'] += '编号曾考虑821。'
    with pytest.raises(ValueError, match='DETERMINISTIC_PREFLIGHT_FAILED:.*success-script-2'):
        compile_source(source, root=tmp_path, request=request)


def test_future_capture_cannot_validate_an_earlier_user_choice(tmp_path):
    source, request = captured_choice_source()
    steps = source['scenarios'][0]['steps']
    steps[0].pop('capture')
    steps[1]['capture'] = {'zone_id': '$.records[0].id'}
    with pytest.raises(ValueError, match='unresolved|unavailable|reference'):
        compile_source(source, root=tmp_path, request=request, structural_preview=True)


def test_unenforced_captured_choice_is_rejected():
    with pytest.raises(ValueError, match='INTERACTION_CAPTURE_UNENFORCED'):
        bind_captured_interaction_arguments([], [], [
            {'script_id': 'script-1', 'tool': 'read_temperature', 'argument': 'id'}])


def test_one_call_must_satisfy_both_capture_and_user_choice_and_recovery_remains_possible():
    expected = {'tool_name': 'read', 'arguments': {'id': {'$ref': 'id'}},
        'captures': [{'name': 'id', 'tool_name': 'lookup', 'path': '$.id'}],
        'disclosed_arguments': {'id': 821}}
    def call(name, arguments, result):
        return {'event': 'tool_call', 'payload': {'tool_name': name, 'arguments': arguments}, 'result': result}
    events = [call('lookup', {}, {'id': 547}), call('read', {'id': 547}, {}),
              call('read', {'id': 821}, {})]
    assert not DeclarativeMetricEvaluator._compare(events, 'contains_tool_call', expected)
    events.extend([call('lookup', {}, {'id': 821}), call('read', {'id': 821}, {})])
    assert DeclarativeMetricEvaluator._compare(events, 'contains_tool_call', expected)
    expected['disclosed_arguments'] = {'id': True}
    assert not DeclarativeMetricEvaluator._compare(events, 'contains_tool_call', expected)


def test_script_choice_uses_keyed_row_provenance_even_when_results_reorder():
    rules = [{'operator': 'contains_tool_call', 'expected': {
        'tool_name': 'reserve', 'arguments': {'id': {'$ref': 'chosen'}, 'price': {'$ref': 'price'}},
        'captures': [{'name': 'chosen', 'tool_name': 'offers', 'path': '$.records[0].id'},
                     {'name': 'price', 'tool_name': 'offers', 'path': '$.records[0].price'}]}}]
    scripts = [{'script_id': 'script-1', 'interaction_protocol': {'stages': [
        {'bind_to': {'tool': 'reserve', 'argument': 'id'}, 'private_fact': 'A'}]}}]
    bind_captured_interaction_arguments(rules, scripts, [
        {'script_id': 'script-1', 'tool': 'reserve', 'argument': 'id'}])
    expected = rules[0]['expected']
    selected = expected['by_user_script']['script-1'][0]
    assert selected['captures'][0]['path'] == '$.records[?(@.id=="A")].id'
    assert selected['captures'][1]['path'] == '$.records[?(@.id=="A")].price'
    # Canonical reference metadata remains intact; runtime uses scoped options.
    assert expected['captures'][0]['path'] == '$.records[0].id'
    def evaluate(rows, arguments):
        events = [{'event': 'user_protocol', 'payload': {'script_id': 'script-1'}},
                  {'event': 'tool_call', 'payload': {'tool_name': 'offers', 'arguments': {}},
                   'result': {'records': rows}},
                  {'event': 'tool_call', 'payload': {'tool_name': 'reserve', 'arguments': arguments},
                   'result': {}}]
        return DeclarativeMetricEvaluator._compare(events, 'contains_tool_call', expected)
    rows = [{'id': 'B', 'price': 11}, {'id': 'A', 'price': 29}]
    assert evaluate(rows, {'id': 'A', 'price': 29})
    assert not evaluate(rows, {'id': 'B', 'price': 11})
    assert not evaluate(rows, {'id': 'A', 'price': 11})
    assert not evaluate(rows[:1], {'id': 'A', 'price': 29})
    assert not evaluate(rows + [rows[-1]], {'id': 'A', 'price': 29})
