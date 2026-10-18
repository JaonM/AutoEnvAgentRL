import copy

from env_factory.production.dialogue_audit import validate, writes_of


MESSAGES = [
    {'role': 'assistant', 'content': 'Renew loan L1 and L2 for one week?'},
    {'role': 'user', 'content': 'Yes, renew both.'},
    {'role': 'assistant', 'content': 'I will renew them.', 'tool_calls': [
        {'function': {'name': 'renew', 'arguments': '{"loan_id":"L1"}'}},
        {'function': {'name': 'renew', 'arguments': '{"loan_id":"L2"}'}},
    ]},
]


def verdict():
    return {'writes': [{'message_index': 2, 'call_index': i, 'approved': True,
                        'proposal': {'message_index': 0, 'quote': MESSAGES[0]['content']},
                        'confirmation': {'message_index': 1, 'quote': MESSAGES[1]['content']}}
                       for i in (0, 1)], 'issues': [], 'script_events': []}


def check(raw, messages=MESSAGES, elements=()):
    return validate(raw, messages, writes_of(messages, {'renew'}), elements)


def test_independent_batch_with_shared_confirmation_is_allowed():
    assert check(verdict())['passed']


def test_same_turn_proposal_or_later_confirmation_cannot_authorize_write():
    raw = verdict()
    raw['writes'][0]['proposal'] = {'message_index': 2, 'quote': 'I will renew them.'}
    assert not check(raw)['passed']
    raw = verdict()
    messages = [*MESSAGES, {'role': 'user', 'content': 'Yes, renew both.'}]
    raw['writes'][0]['confirmation']['message_index'] = 3
    assert not check(raw, messages)['passed']


def test_private_approval_or_fabricated_quote_is_not_consent_evidence():
    raw = verdict()
    raw['writes'][0]['confirmation']['quote'] = 'approve_first=true'
    assert not check(raw)['passed']
    raw = verdict()
    raw['writes'][0]['confirmation']['message_index'] = 0
    assert not check(raw)['passed']


def test_every_write_and_script_element_must_be_audited():
    raw = verdict()
    raw['writes'].pop()
    assert not check(raw)['passed']
    assert not check(verdict(), elements=['wrong_first'])['passed']
    raw = verdict()
    raw['script_events'] = [{'element': 'wrong_first', 'observed': False, 'evidence': []}]
    assert not check(raw, elements=['wrong_first'])['passed']


def test_grounded_tool_argument_issue_blocks_admission():
    raw = verdict()
    raw['issues'] = [{'kind': 'argument_provenance', 'reason': 'Identifier not established',
                      'evidence': [{'message_index': 2, 'quote': '"loan_id":"L1"'}]}]
    assert not check(raw)['passed']
    raw['issues'][0]['evidence'][0]['quote'] = 'made up'
    assert 'ungrounded issue' in check(raw)['errors']


def test_malformed_response_fails_closed():
    for raw in (None, {}, {'writes': [], 'issues': [], 'script_events': []}):
        assert not check(raw)['passed']
    raw = verdict()
    raw['writes'][0]['call_index'] = []
    assert not check(raw)['passed']
    raw = verdict()
    raw['issues'] = [{'kind': [], 'evidence': []}]
    assert not check(raw)['passed']
