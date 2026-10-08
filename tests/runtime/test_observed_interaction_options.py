import copy

from env_factory.sandbox_runtime import observed_interaction_options, interaction_request_matches


def test_choices_follow_observed_catalog_and_removed_selection_cannot_advance():
    import json
    stage = {'kind': 'option_selection', 'before_tool': 'count', 'bind_to': {'argument': 'type'},
        'options': ['map', 'text'], 'private_fact': 'map',
        'options_from': {'tool': 'catalog', 'path': '$.records[0].types', 'separator': ','}}
    original = copy.deepcopy(stage)
    events = [{'event': 'tool_call', 'payload': {'tool_name': 'catalog'},
               'result': {'records': [{'types': 'map,text'}]}}]
    selected = observed_interaction_options(stage, events)
    request = {'question': 'Which type?', 'request': {'intent': 'request_input',
        'tool': 'count', 'arguments': ['type'], 'options': ['map', 'text']}}
    assert interaction_request_matches(selected, json.dumps(request), 'structured_v1')
    events[0]['result']['records'][0]['types'] = 'map,image'
    selected = observed_interaction_options(stage, events)
    assert not interaction_request_matches(selected, json.dumps(request), 'structured_v1')
    request['request']['options'] = ['map', 'image']
    assert interaction_request_matches(selected, json.dumps(request), 'structured_v1')
    events[0]['result']['records'][0]['types'] = 'text,image'
    assert observed_interaction_options(stage, events) is None
    assert stage == original


def test_missing_and_failed_responses_do_not_supply_options():
    stage = {'private_fact': 'a', 'options_from': {'tool': 'catalog', 'path': '$.choices'}}
    assert observed_interaction_options(stage, []) is None
    good = {'event': 'tool_call', 'payload': {'tool_name': 'catalog'}, 'result': {'choices': ['a', 'b']}}
    failed = {**good, 'result': {'error': {'code': 'TIMEOUT'}}}
    assert observed_interaction_options(stage, [failed]) is None
    assert observed_interaction_options(stage, [good, failed])['options'] == ['a', 'b']


def test_duplicate_business_identifiers_remain_selectable_without_hiding_records():
    import json
    from env_factory.sandbox_runtime import AcceptanceScenarioRunner
    stage = {'kind': 'option_selection', 'before_tool': 'read',
             'bind_to': {'argument': 'choice'}, 'private_fact': 'a',
             'options_from': {'tool': 'catalog', 'path': '$.records[*].id'}}
    events = [{'event': 'tool_call', 'payload': {'tool_name': 'catalog'},
               'result': {'records': [{'id': 'a', 'owner': 'one'},
                                      {'id': 'a', 'owner': 'two'}, {'id': 'b', 'owner': 'three'}]}}]
    original = copy.deepcopy(events)
    resolved = observed_interaction_options(stage, events)
    assert resolved is not None and resolved['options'] == ['a', 'b']
    content = json.dumps({'question': '{{observed_result}} {{observed_options}}',
                          'request': {'intent': 'request_input', 'tool': 'read',
                                      'arguments': ['choice'], 'options': ['a', 'b']}})
    rendered = AcceptanceScenarioRunner.render_observed_dialogue(content, stage['options_from'], events)
    assert interaction_request_matches(resolved, rendered, 'structured_v1')
    question = json.loads(rendered)['question']
    assert 'one' in question and 'two' in question
    assert events == original  # Business ambiguity must remain available to route selection.
    assert observed_interaction_options({**stage, 'private_fact': 'missing'}, events) is None


def test_duplicate_options_are_deduplicated_after_separator_parsing():
    stage = {'private_fact': 'b', 'options_from': {'tool': 'catalog', 'path': '$.choices', 'separator': ','}}
    event = {'event': 'tool_call', 'payload': {'tool_name': 'catalog'}, 'result': {'choices': 'a,b,a'}}
    assert observed_interaction_options(stage, [event])['options'] == ['a', 'b']
    event['result']['choices'] = 'a,,b'
    assert observed_interaction_options(stage, [event]) is None
