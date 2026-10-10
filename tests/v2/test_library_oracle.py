"""Independent domain truth checks; these do not measure model yield."""
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_library_oracle import expected, renewal_decision


@pytest.fixture
def data():
    return {
        'books': [{'id': 'B2', 'renewal_limit': 1, 'renewal_days': 3},
                  {'id': 'B1', 'renewal_limit': 2, 'renewal_days': 7}],
        'loans': [{'id': 'L1', 'book_id': 'B1', 'reader_id': 'R1', 'active': True,
                   'due_day': 10, 'renewals': 0},
                  {'id': 'L2', 'book_id': 'B2', 'reader_id': 'R1', 'active': True,
                   'due_day': 10, 'renewals': 0}],
        'holds': [{'id': 'H1', 'book_id': 'B1', 'reader_id': 'R1', 'active': True,
                   'requested_day': 2, 'copy_id': 'K0'},
                  {'id': 'H2', 'book_id': 'B2', 'reader_id': 'R2', 'active': True,
                   'requested_day': 1, 'copy_id': None}],
        'copies': [{'id': 'K0', 'book_id': 'B1', 'site_id': 'S1', 'reserved_by': 'H1'},
                   {'id': 'K1', 'book_id': 'B1', 'site_id': 'S2', 'reserved_by': None}],
        'sites': [{'id': 'S1', 'supports_holds': True}, {'id': 'S2', 'supports_holds': True}],
    }


def evaluate(kind, data, facts, **kwargs):
    return expected({'kind': kind, 'params': {'current_day': 10}}, data, facts, **kwargs)


@pytest.mark.parametrize('days,amount', [(0, 0), (1, 0), (2, 2), (3, 4), (5, 8), (20, 8)])
def test_fine_grace_and_cap(days, amount):
    spec = {'kind': 'library_fine', 'params': dict(overdue_days=days, grace_days=1,
                                                 daily_cents=200, cap_cents=800)}
    assert expected(spec, {}, {})['answer'] == {'fine': amount}
    spec['params']['overdue_days'] = True
    with pytest.raises(ValueError):
        expected(spec, {}, {})


def test_renewal_follows_actual_relationship_and_boundary(data):
    assert renewal_decision(data, 'L1', 'R1', 10) == (True, 'eligible')
    assert renewal_decision(data, 'L1', 'R1', 11) == (False, 'overdue')
    assert renewal_decision(data, 'L2', 'R1', 10) == (False, 'competing_hold')
    data['holds'][1]['active'] = False
    assert renewal_decision(data, 'L2', 'R1', 10)[0]
    data['loans'][1]['renewals'] = 1
    assert renewal_decision(data, 'L2', 'R1', 10) == (False, 'renewal_limit')
    assert renewal_decision(data, 'L1', 'R2', 10) == (False, 'different_reader')
    data['books'].reverse()
    assert renewal_decision(data, 'L1', 'R1', 10)[0]


def test_queue_excludes_unrelated_and_cancelled_and_breaks_ties(data):
    data['holds'].extend([
        dict(id='H0', book_id='B1', reader_id='R2', active=True, requested_day=2),
        dict(id='H3', book_id='B1', reader_id='R2', active=False, requested_day=0)])
    facts = dict(hold_id='H1', reader_id='R1')
    assert evaluate('library_queue', data, facts)['answer'] == {'rank': 2, 'ahead': 1}
    data['holds'].reverse()
    assert evaluate('library_queue', data, facts)['answer']['rank'] == 2
    facts['hold_id'] = 'H3'
    assert evaluate('library_queue', data, facts)['answer'] == {'rank': None, 'ahead': None}


@pytest.mark.parametrize('approvals,refuse,renewed', [([True, True], False, ['L1']),
    ([False, True], False, []), ([True, False], False, ['L1']), ([True, True], True, [])])
def test_partial_renewal_preserves_first_success_and_unrelated_state(data, approvals, refuse, renewed):
    original = copy.deepcopy(data)
    truth = evaluate('library_partial_renewal', data,
                     dict(loan_ids=['L1', 'L2'], reader_id='R1'), approvals=approvals, refuse=refuse)
    assert truth['answer']['renewed_ids'] == renewed
    wanted = copy.deepcopy(original)
    if renewed:
        wanted['loans'][0].update(due_day=17, renewals=1)
    assert truth['final_state'] == wanted
    assert data == original == truth['initial']


def test_pickup_change_is_atomic_and_preserves_priority(data):
    original = copy.deepcopy(data)
    truth = evaluate('library_pickup_change', data,
                     dict(hold_id='H1', reader_id='R1', copy_choice='K1', approve_change=True))
    wanted = copy.deepcopy(original)
    wanted['copies'][0]['reserved_by'] = None
    wanted['copies'][1]['reserved_by'] = 'H1'
    wanted['holds'][0]['copy_id'] = 'K1'
    assert truth['answer'] == {'committed': True}
    assert truth['communication']['available_site_ids']['expected'] == ['S2']
    assert truth['final_state'] == wanted
    assert data == original


@pytest.mark.parametrize('fault', ['wrong_book', 'occupied', 'unsupported', 'not_owner',
                                 'invalid_old_owner', 'refusal', 'no_approval'])
def test_pickup_rejection_has_no_partial_writes(data, fault):
    facts = dict(hold_id='H1', reader_id='R1', copy_choice='K1', approve_change=True)
    if fault == 'wrong_book':
        data['copies'][1]['book_id'] = 'B2'
    elif fault == 'occupied':
        data['copies'][1]['reserved_by'] = 'H2'
    elif fault == 'unsupported':
        data['sites'][1]['supports_holds'] = False
    elif fault == 'not_owner':
        facts['reader_id'] = 'R2'
    elif fault == 'invalid_old_owner':
        data['copies'][0]['reserved_by'] = 'H2'
    elif fault == 'no_approval':
        facts['approve_change'] = False
    truth = evaluate('library_pickup_change', data, facts, refuse=fault == 'refusal')
    assert truth['answer']['committed'] is False
    assert truth['final_state'] == data


@pytest.mark.parametrize('approvals', [[], [True, True], [1]])
def test_pickup_approval_shape_is_explicit(data, approvals):
    with pytest.raises(ValueError, match='one boolean approval'):
        evaluate('library_pickup_change', data,
                 dict(hold_id='H1', reader_id='R1', copy_choice='K1'), approvals=approvals)


def test_five_prepared_cases_use_shared_independent_acceptance_entry():
    from prepare_v2_cross_business import library_cases
    from v2_natural_oracle import expected as shared_expected
    rows = library_cases()
    assert [row['stratum'] for row in rows] == ['direct', 'simple', 'ordinary', 'complex', 'complex']
    truths = [shared_expected(row['acceptance'], row['business'], row['user']) for row in rows]
    assert truths[0]['answer'] == {'fine': 4}
    assert truths[1]['answer'] == {'eligible': True}
    assert truths[2]['answer'] == {'rank': 2, 'ahead': 1}
    assert truths[3]['answer'] == {'renewed_ids': ['L1'],
        'blocked_queue_counts': [{'loan_id': 'L2', 'count': 2}]}
    assert truths[4]['answer'] == {'committed': True}
    assert truths[4]['communication']['available_site_ids']['expected'] == ['S2']
    for row, truth in zip(rows, truths):
        refused = shared_expected(row['acceptance'], row['business'], row['user'], refuse=True)
        assert refused['final_state'] == refused['initial']
        if row['stratum'] != 'complex':
            assert truth['initial'] == truth['final_state']
    row = rows[1]
    altered = shared_expected(row['acceptance'], row['business'], row['user'],
        mutation={'table': 'loans', 'key': 'L1', 'column': 'due_day', 'value': 9})
    assert altered['answer'] == {'eligible': False}
