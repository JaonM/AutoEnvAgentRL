import copy
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_mapping_validation import MappingValidationError, validated_mapping
from v2_natural_semantics import schema


def mapping(path):
    return [{'role': 'amount', 'selectors': [{'field': 'ranked', 'path': path}],
             'packing': 'single', 'time_unit': 'native', 'names': 'native'}]


def validate(review, agent, answers=None):
    return validated_mapping(review, agent=agent,
        mapping_schema=schema({'amount': 0}, ['ranked'])['properties']['mapping'],
        role_types={'amount': int}, fields={'ranked': {'meaning': '按金额升序排列的候选'}},
        answers=answers or [{'ranked': [{'amount': 10}, {'amount': 20}]}],
        user_request='最低金额是多少？', public_context={})


def test_mapping_repair_changes_only_projection_and_receives_no_oracle_values():
    review = {'mapping': mapping('*.amount'), 'business_consistent': False, 'reason': 'unrelated business concern'}
    original = copy.deepcopy(review)
    agent = Mock()
    agent.ask.return_value = ({'mapping': mapping('0.amount')}, {'response_id': 'unit'})
    entries, recovery = validate(review, agent)
    assert entries == mapping('0.amount')
    assert review == original  # no override of independent business rejection
    assert recovery['structural_error'] == 'scalar semantic role requires exactly one observed value'
    payload = agent.ask.call_args.args[1]
    assert set(payload) == {'user_request', 'public_context', 'answer_fields', 'observed_answers',
                            'required_roles', 'previous_mapping', 'structural_error'}
    assert payload['required_roles'] == {'amount': 'int'}


def test_valid_but_wrong_value_does_not_trigger_mapping_retry():
    agent = Mock()
    entries, recovery = validate({'mapping': mapping('0.amount')}, agent,
                                 [{'ranked': [{'amount': 999}]}])
    assert entries == mapping('0.amount') and recovery is None
    agent.ask.assert_not_called()  # independent value checks decide correctness


@pytest.mark.parametrize('bad', [[], mapping('*.amount'), mapping('missing')])
def test_invalid_mapping_has_bounded_evaluator_failure(bad):
    agent = Mock()
    agent.ask.return_value = ({'mapping': bad}, {})
    with pytest.raises(MappingValidationError):
        validate({'mapping': []}, agent)
    assert agent.ask.call_count == 1
