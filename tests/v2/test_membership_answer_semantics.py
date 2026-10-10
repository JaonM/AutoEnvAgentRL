import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_natural_semantics import fill_projections, normalize
from v2_library_oracle import expected
from env_factory_v2.expressions import ContractError


ENTRY = {'role': 'eligible', 'selectors': [{'field': 'ids', 'path': ''}],
    'packing': 'list', 'time_unit': 'native', 'names': 'native', 'representation': 'set_membership'}


def truth():
    data = {'books': [{'id': 'B1', 'renewal_limit': 2}], 'holds': [], 'loans': [
        {'id': 'L1', 'book_id': 'B1', 'reader_id': 'R1', 'active': True, 'due_day': 10, 'renewals': 0},
        {'id': 'L2', 'book_id': 'B1', 'reader_id': 'R1', 'active': True, 'due_day': 10, 'renewals': 1},
        {'id': 'L3', 'book_id': 'B1', 'reader_id': 'R2', 'active': True, 'due_day': 10, 'renewals': 0}]}
    return expected({'kind': 'library_renewal_eligibility', 'params': {'current_day': 10}},
        data, {'reader_id': 'R1', 'loan_id': 'L1'})


def project(ids, t=None):
    return normalize({'ids': ids}, [ENTRY], role_types={'eligible': bool},
        set_memberships=(t or truth())['set_memberships'])


def test_complete_set_is_equivalent_to_selected_loan_boolean():
    assert project(['L2', 'L1']) == truth()['answer'] == {'eligible': True}


@pytest.mark.parametrize('ids', [['L1'], ['L1', 'L2', 'L3'], ['L1', 'L2', 'L2'], [True], []])
def test_partial_foreign_duplicate_or_wrong_sets_never_pass(ids):
    with pytest.raises(ContractError):
        project(ids)


def test_nonempty_set_does_not_mean_selected_loan_eligible():
    t = {'answer': {'eligible': False}, 'set_memberships': {
        'eligible': {'member': 'L1', 'complete_set': ['L2']}}}
    assert project(['L2'], t) == {'eligible': False}
    assert fill_projections({'ids': ['L1']}, [ENTRY], t['answer'],
        set_memberships=t['set_memberships']) == {'ids': ['L2']}


def test_model_cannot_invent_conversion_without_independent_scope():
    with pytest.raises(ContractError, match='independently declared'):
        normalize({'ids': ['L1']}, [ENTRY], role_types={'eligible': bool})


def test_inverse_uses_current_independent_set_and_rejects_inconsistent_truth():
    t = truth()
    answer = fill_projections({'ids': []}, [ENTRY], t['answer'], set_memberships=t['set_memberships'])
    assert answer == {'ids': ['L1', 'L2']}
    with pytest.raises(ContractError, match='membership truth'):
        fill_projections(answer, [ENTRY], {'eligible': False}, set_memberships=t['set_memberships'])
