from review_fixtures import with_requirement_reviews
import copy
from unittest.mock import Mock

import pytest

from env_factory_v2.validation.challenge import _repair_recipe_json, challenge
from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.examples import direct
from env_factory_v2.native.code_agent import NativeCodeAgent


def review():
    return with_requirement_reviews(direct(), {'business_consistent': True, 'reason': 'fixed fact', 'cases': [
        {'id': 'yes', 'rule_ids': ['fixed_fact'], 'scenario_id': 'correct', 'reason': 'correct',
         'expect_success': True, 'mutations_json': '[]', 'answer_json': '{"temperature":{"literal":18}}}'},
        {'id': 'no', 'rule_ids': ['fixed_fact'], 'scenario_id': 'wrong', 'reason': 'wrong',
         'expect_success': False, 'mutations_json': '[]', 'answer_json': '{}'}]})


def agent(tmp_path):
    return NativeCodeAgent(base_url='http://localhost:9999/v1', token='a' * 24, evidence_root=tmp_path)


def test_format_recovery_preserves_cases_and_still_executes_counterexamples(tmp_path):
    native = agent(tmp_path)
    value = review()
    before = copy.deepcopy(value)
    native.ask = Mock(side_effect=[(value, {'initial': True}),
        ({'repairs': [{'index': 0, 'field': 'answer_json', 'json': '{"temperature":{"literal":18}}'}]}, {'repair': True})])
    result = challenge(compile_contract(direct()), {'goal': direct()['task']}, native)
    assert result['passed']
    assert len(result['records']) == 2
    assert result['records'][0]['first']['outcome']['reward'] == 1
    assert result['records'][1]['first']['outcome']['reward'] == 0
    assert result['format_repair']['original_review'] == before
    assert value == before
    assert native.ask.call_count == 2


@pytest.mark.parametrize('fixed', ['{"temperature":{"literal":19}}', '{}'])
def test_repair_cannot_change_or_remove_business_values(tmp_path, fixed):
    native = agent(tmp_path)
    native.ask = Mock(return_value=({'repairs': [{'index': 0, 'field': 'answer_json', 'json': fixed}]}, {}))
    with pytest.raises(ValueError, match='changed business content'):
        _repair_recipe_json(review(), native)


def test_valid_review_never_calls_format_model(tmp_path):
    native = agent(tmp_path)
    native.ask = Mock()
    value = review()
    value['cases'][0]['answer_json'] = '{}'
    recovered, evidence = _repair_recipe_json(value, native)
    assert recovered is value and evidence is None
    native.ask.assert_not_called()
