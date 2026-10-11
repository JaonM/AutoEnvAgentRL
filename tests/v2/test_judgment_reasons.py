import copy
from unittest.mock import Mock

import pytest

from env_factory_v2.native.code_agent import NativeCodeAgent
from env_factory_v2.contracts.expressions import ContractError
from env_factory_v2.validation.judgment_reasons import validate_reasons


def payload():
    return {'field': {'criteria': [{'id': 'scope', 'obligation': 'Report the actual deadline.'}]},
            'text': 'Deadline is 50.', 'facts': {'a/b~c': 50}, 'observations': []}


def reasons():
    return {'scope': {'reason': 'The supplied text reports the deadline.',
        'obligation_quote': 'actual deadline',
        'evidence': [{'path': '/text', 'quote': 'Deadline is 50.'},
                     {'path': '/facts/a~1b~0c', 'quote': '50'}]}}


def test_reasons_preserve_exact_scoped_original_evidence_without_aliasing():
    original = reasons(); actual = validate_reasons(original, payload())
    assert actual == original
    actual['scope']['evidence'].clear()
    assert len(original['scope']['evidence']) == 2


@pytest.mark.parametrize('bad', ['wrong_id', 'wrong_obligation', 'invented_path', 'invented_quote',
    'bad_escape', 'bad_index', 'too_long', 'no_evidence', 'too_many'])
def test_unverifiable_repair_evidence_is_rejected(bad):
    value = reasons(); item = value['scope']
    if bad == 'wrong_id': value['other'] = value.pop('scope')
    elif bad == 'wrong_obligation': item['obligation_quote'] = 'Explain all policy rules'
    elif bad == 'invented_path': item['evidence'][0]['path'] = '/unobserved'
    elif bad == 'invented_quote': item['evidence'][0]['quote'] = 'Deadline is 60.'
    elif bad == 'bad_escape': item['evidence'][0]['path'] = '/facts/a~2b'
    elif bad == 'bad_index': item['evidence'][0]['path'] = '/field/criteria/01/obligation'
    elif bad == 'too_long': item['reason'] = 'x'*1601
    elif bad == 'no_evidence': item['evidence'] = []
    else: item['evidence'] *= 5
    with pytest.raises(ContractError): validate_reasons(value, payload())


@pytest.mark.parametrize('valid', [True, False])
def test_rationale_validation_never_changes_boolean_score_or_adds_calls(tmp_path, valid):
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='x'*24, evidence_root=tmp_path)
    value = reasons()
    if not valid: value['scope']['evidence'][0]['quote'] = 'fabricated'
    agent.ask = Mock(return_value=({'criteria': {'scope': False}, 'reasons': value}, {'response_id': 'test'}))
    first = agent.judge(payload()); repeat = agent.judge(payload())
    assert first['criteria'] == repeat['criteria'] == {'scope': False}
    assert agent.ask.call_count == 1 and repeat['provenance']['reused']
    assert first['provenance']['reason_validation']['passed'] is valid
    assert bool(first['provenance']['reasons']) is valid
    assert 'criteria' not in first['provenance']['reasons']


def test_reason_wording_does_not_poison_identical_boolean_judgments(tmp_path):
    agent = NativeCodeAgent(base_url='http://localhost:9999/v1', token='x'*24, evidence_root=tmp_path)
    second = copy.deepcopy(reasons()); second['scope']['reason'] = 'Faithful deadline statement.'
    agent.ask = Mock(side_effect=[({'criteria': {'scope': True}, 'reasons': r}, {'response_id': str(i)})
                                 for i, r in enumerate([reasons(), second])])
    agent.judge(payload()); fresh = agent.independent_judge(payload())
    assert fresh['criteria'] == {'scope': True} and not agent.poisoned
