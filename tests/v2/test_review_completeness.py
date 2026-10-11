from review_fixtures import with_requirement_reviews
import copy
from unittest.mock import Mock

import pytest

from env_factory_v2.challenge import challenge
from env_factory_v2.code_agent import NativeCodeAgent
from env_factory_v2.contract import compile_contract
from env_factory_v2.examples import direct


def case(identifier, positive):
    return {'id': identifier, 'rule_ids': ['fixed_fact'],
            'scenario_id': 'correct' if positive else 'wrong', 'reason': identifier,
            'expect_success': positive, 'mutations_json': '[]', 'answer_json': '{}'}


def test_append_only_recovery_executes_missing_negative(tmp_path):
    native = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a' * 24, evidence_root=tmp_path)
    review = {'business_consistent': True, 'reason': 'original', 'cases': [case('yes', True)]}
    review = with_requirement_reviews(direct(), review)
    before = copy.deepcopy(review)
    native.ask = Mock(side_effect=[(review, {}), ({'cases': [case('no', False)]}, {})])
    result = challenge(compile_contract(direct()), {'goal': direct()['task']}, native)
    assert result['passed'] and len(result['records']) == 2
    assert result['records'][1]['first']['outcome']['reward'] == 0
    assert result['completeness_recovery']['original_review'] == before == review
    assert native.ask.call_count == 2


def test_incomplete_supplement_never_passes_or_retries(tmp_path):
    native = NativeCodeAgent(base_url='http://localhost:9999/v1', token='a' * 24, evidence_root=tmp_path)
    review = {'business_consistent': True, 'reason': 'original', 'cases': [case('yes', True)]}
    native.ask = Mock(side_effect=[(review, {}), ({'cases': []}, {})])
    with pytest.raises(RuntimeError, match='2..32'):
        challenge(compile_contract(direct()), {'goal': direct()['task']}, native)
    assert native.ask.call_count == 2
