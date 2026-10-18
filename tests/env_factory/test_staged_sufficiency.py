import json
from unittest.mock import patch

from env_factory.certify import sufficiency


def fixture():
    from types import SimpleNamespace
    loaded = {'request': {'tables': {}, 'user_facts': {'approve_create': True},
                          'time': {'start': 0}, 'business_rules': 'Create, then label the returned record.',
                          'user_request': 'Create my record and label it.'},
              'schema': {'fields': {'amount': {'type': 'integer', 'description': 'Amount charged'}}},
              'env': SimpleNamespace(make_tools=lambda: {})}
    trace = [
        {'kind': 'tool', 'status': 'ok', 'tool': 'quote', 'result': {'amount': 5}},
        {'kind': 'tool', 'status': 'ok', 'tool': 'create', 'write': True, 'args': {'amount': 5},
         'result': {'record_id': 'NEW-910234'}},
        {'kind': 'tool', 'status': 'ok', 'tool': 'label', 'write': True, 'args': {'id': 'NEW-910234'},
         'result': {'status': 'done'}},
    ]
    return loaded, {'trace': trace, 'answer': {'amount': 5}}


class Client:
    def __init__(self, replies):
        self.replies, self.prompts = iter(replies), []

    def chat(self, messages, **kwargs):
        self.prompts.append(messages[-1]['content'])
        return {'content': json.dumps(next(self.replies))}


def test_generated_identifier_only_appears_after_its_write():
    loaded, result = fixture()
    client = Client([{'writes': [{'tool': 'create', 'args': {'amount': 5}}]},
                     {'writes': [{'tool': 'label', 'args': {'id': 'NEW-910234'}}]},
                     {'answer': {'amount': 5}}])
    with patch.object(sufficiency, 'run_reference', return_value=result):
        report = sufficiency.check(loaded, client, k=1)
    assert report['passed']
    assert 'NEW-910234' not in client.prompts[0]
    assert 'NEW-910234' in client.prompts[1] and 'NEW-910234' in client.prompts[2]
    assert 'write_receipt' in client.prompts[1]


def test_right_report_cannot_rescue_wrong_decision():
    loaded, result = fixture()
    client = Client([{'writes': []}, {'writes': []}, {'answer': {'amount': 5}}])
    with patch.object(sufficiency, 'run_reference', return_value=result):
        report = sufficiency.check(loaded, client, k=1)
    assert not report['passed'] and report['gaps']['writes']
    assert report['attempts'][0]['fields']['amount']


def test_independent_remaining_actions_may_be_predicted_together_in_reverse_order():
    loaded, result = fixture()
    result['trace'][2]['args'] = {'id': 'existing'}
    loaded['request']['user_facts']['id'] = 'existing'
    create = {'tool': 'create', 'args': {'amount': 5}}
    label = {'tool': 'label', 'args': {'id': 'existing'}}
    client = Client([{'writes': [label, create]}, {'writes': [label]}, {'answer': {'amount': 5}}])
    with patch.object(sufficiency, 'run_reference', return_value=result):
        assert sufficiency.check(loaded, client, k=1)['passed']


def test_read_only_task_still_rejects_invented_writes():
    loaded, result = fixture()
    result['trace'] = result['trace'][:1]
    client = Client([{'writes': [{'tool': 'create', 'args': {'amount': 5}}]}, {'answer': {'amount': 5}}])
    with patch.object(sufficiency, 'run_reference', return_value=result):
        assert not sufficiency.check(loaded, client, k=1)['passed']


def test_unwrapped_report_is_only_a_format_variant_and_wrong_values_still_fail():
    loaded, result = fixture()
    result['trace'] = result['trace'][:1]
    for amount, passed in [(5, True), (6, False)]:
        client = Client([{'writes': []}, {'amount': amount}])
        with patch.object(sufficiency, 'run_reference', return_value=result):
            report = sufficiency.check(loaded, client, k=1)
        assert report['passed'] is passed
        assert report['attempts'][0]['report_format_valid']
        assert json.loads(report['attempts'][0]['raw_report_response']) == {'amount': amount}
    assert sufficiency.reported_answer({'writes': []}, loaded['schema']) is None
