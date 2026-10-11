from review_fixtures import with_requirement_reviews
import json
from unittest.mock import Mock

import pytest

from env_factory_v2.validation.challenge import challenge
from env_factory_v2.native.code_agent import NativeCodeAgent
from env_factory_v2.contracts.contract import compile_contract
from env_factory_v2.examples import simple


def review():
    return with_requirement_reviews(simple(), {'business_consistent': True, 'reason': 'Identity independent of display.', 'cases': [
        {'id': 'renamed', 'scenario_id': 'correct', 'rule_ids': ['size_matches'], 'reason': 'Same actual size.',
         'expect_success': True, 'mutations_json': json.dumps([
             {'table': 'items', 'key': 'I1', 'column': 'id', 'value': 'changed'},
             {'table': 'items', 'key': 'I1', 'column': 'display', 'value': 'new display'}]), 'answer_json': '{}'},
        {'id': 'wrong', 'scenario_id': 'wrong', 'rule_ids': ['size_matches'], 'reason': 'Missing actual item.',
         'expect_success': False, 'mutations_json': '[]', 'answer_json': '{}'}]})


def agent(tmp_path, order):
    result = NativeCodeAgent(base_url='http://localhost:9999/v1', token='x' * 24, evidence_root=tmp_path)
    result.ask = Mock(side_effect=[(review(), {}), ({'order': order}, {'native': 'unit-test'})])
    return result


def test_reorder_preserves_changes_and_executes_all_cases_twice(tmp_path):
    native = agent(tmp_path, [1, 0])
    result = challenge(compile_contract(simple()), {'goal': simple()['task']}, native)
    assert result['passed'] and len(result['records']) == 2
    record = result['records'][0]
    assert record['first']['final_state']['items'][0]['id'] == 'changed'
    assert record['first']['final_state']['items'][0]['display'] == 'new display'
    assert record['repeat']['final_state'] == record['first']['final_state']
    assert record['mutation_order_repair']['order'] == [1, 0]
    assert record['case']['expect_success']


@pytest.mark.parametrize('order', [[0], [1, 1], [2, 0], [True, 0]])
def test_reorder_cannot_drop_duplicate_or_invent_mutations(tmp_path, order):
    with pytest.raises(RuntimeError, match='preserve every mutation'):
        challenge(compile_contract(simple()), {'goal': simple()['task']}, agent(tmp_path, order))
