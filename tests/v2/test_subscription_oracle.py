"""Domain truth checks; not model yield."""
import sys
from pathlib import Path
from fractions import Fraction

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_subscription_oracle import refund, effective_plan, bill, cancel_refund, change_date


@pytest.mark.parametrize('used,prior,wanted', [(11, 0, 5700), (0, 0, 9000),
    (30, 0, 0), (11, 8500, 500), (11, 9000, 0)])
def test_refund_respects_paid_balance(used, prior, wanted):
    assert refund(9000, 30, used, prior) == wanted


@pytest.mark.parametrize('args', [(9000, 0, 0), (9000, 30, 31), (9000, 30, 1, 9001), (True, 30, 1)])
def test_invalid_refund_inputs_do_not_become_rewards(args):
    with pytest.raises(ValueError): refund(*args)


def changes():
    return [dict(id='C1', subscription_id='S1', effective_day=10, plan_id='B', cancelled=False),
            dict(id='C2', subscription_id='S1', effective_day=20, plan_id='C', cancelled=True),
            dict(id='C3', subscription_id='S1', effective_day=35, plan_id='C', cancelled=False),
            dict(id='C4', subscription_id='S2', effective_day=0, plan_id='C', cancelled=False)]


def test_current_plan_uses_actual_subscription_and_inclusive_effective_day():
    subscription = dict(id='S1', base_plan_id='A')
    for day, wanted in [(9, 'A'), (10, 'B'), (20, 'B'), (34, 'B'), (35, 'C')]:
        assert effective_plan(subscription, changes(), day) == wanted
        assert effective_plan(subscription, list(reversed(changes())), day) == wanted


def test_interval_partition_rounds_once_and_matches_daywise_independent_sum():
    subscription = dict(id='S1', base_plan_id='A')
    plans = [dict(id='A', period_cents=101, period_days=30),
             dict(id='B', period_cents=103, period_days=30),
             dict(id='C', period_cents=107, period_days=30)]
    rates = {p['id']: Fraction(p['period_cents'], p['period_days']) for p in plans}
    for start, end in [(0, 30), (30, 60), (9, 11), (35, 36)]:
        truth = bill(subscription, changes(), plans, start, end)
        daily = sum((rates[effective_plan(subscription, changes(), day)]
                     for day in range(start, end)), Fraction(0))
        assert truth['amount_cents'] == daily.numerator // daily.denominator
        assert sum(i['end'] - i['start'] for i in truth['intervals']) == end - start
    assert bill(subscription, changes(), plans, 9, 11)['amount_cents'] == 6


def test_conflicting_same_day_changes_are_not_silently_resolved():
    rows = changes()
    rows.append(dict(id='C5', subscription_id='S1', effective_day=10, plan_id='C', cancelled=False))
    with pytest.raises(ValueError, match='explicit public precedence'):
        effective_plan(dict(id='S1', base_plan_id='A'), rows, 10)


def test_refund_transaction_partial_success_and_duplicate_protection():
    data = {'subscriptions': [dict(id='S1', reader_id='R1', active=True, paid_cents=9000,
        period_days=30, refunded_cents=0), dict(id='S2', reader_id='R1', active=True,
        paid_cents=9000, period_days=30, refunded_cents=9000)],
        'refunds': [dict(id='F0', subscription_id='S2', amount_cents=9000)]}
    passed, amount, first = cancel_refund(data, 'S1', 'R1', used_days=11, approved=True, refund_id='F1')
    assert passed and amount == 5700
    assert first['subscriptions'][0]['refunded_cents'] == 5700
    assert not first['subscriptions'][0]['active']
    passed, amount, second = cancel_refund(first, 'S2', 'R1', used_days=11, approved=True, refund_id='F2')
    assert not passed and amount == 0 and second == first
    passed, _, duplicate = cancel_refund(first, 'S1', 'R1', used_days=11, approved=True, refund_id='F1')
    assert not passed and duplicate == first
    assert data['subscriptions'][0]['active']
    passed, _, refused = cancel_refund(data, 'S1', 'R1', used_days=11, approved=False, refund_id='F1')
    assert not passed and refused == data


@pytest.mark.parametrize('authorized_day,authorized_cents,approved,success',
    [(15, 1500, True, True), (10, 1500, True, False), (15, 1600, True, False), (15, 1500, False, False)])
def test_date_correction_invalidates_old_authorization_and_updates_bill_once(
        authorized_day, authorized_cents, approved, success):
    data = {'subscriptions': [dict(id='S1', reader_id='R1', active=True, base_plan_id='A')],
            'changes': [dict(id='C1', subscription_id='S1', plan_id='B', effective_day=10, cancelled=False)],
            'plans': [dict(id='A', period_cents=1000, period_days=30),
                      dict(id='B', period_cents=2000, period_days=30)],
            'bills': [dict(id='I1', subscription_id='S1', amount_cents=1666, period_start=0, period_end=30)]}
    kwargs = dict(change_id='C1', plan_id='B', day=15, start=0, end=30,
        authorized_day=authorized_day, authorized_cents=authorized_cents, approved=approved, bill_id='I1')
    passed, final = change_date(data, 'S1', 'R1', **kwargs)
    assert passed is success
    if success:
        assert final['bills'][0]['amount_cents'] == 1500
        assert len(final['bills']) == 1
        assert final['changes'][0]['effective_day'] == 15
        assert change_date(final, 'S1', 'R1', **kwargs)[1] == final
    else:
        assert final == data


def test_five_subscription_cases_use_shared_acceptance_and_refusal_truth():
    from prepare_v2_cross_business import subscription_cases
    from v2_natural_oracle import expected
    rows = subscription_cases()
    assert [r['id'] for r in rows] == [11, 12, 13, 14, 15]
    truths = [expected(r['acceptance'], r['business'], r['user']) for r in rows]
    assert truths[0]['answer'] == {'refund_cents': 5700}
    assert truths[1]['answer'] == {'plan_id': 'B', 'effective_day': 35}
    assert truths[2]['answer'] == {'amount_cents': 1666, 'intervals': [
        {'start': 0, 'end': 10, 'plan_id': 'A'}, {'start': 10, 'end': 30, 'plan_id': 'B'}]}
    assert truths[3]['answer'] == {'estimates': [{'day': 10, 'amount_cents': 1666},
        {'day': 15, 'amount_cents': 1500}], 'committed': True}
    assert truths[4]['answer']['results'] == [
        {'subscription_id': 'S1', 'refundable_cents': 5700, 'refunded_cents': 5700, 'committed': True},
        {'subscription_id': 'S2', 'refundable_cents': 0, 'refunded_cents': 0, 'committed': False}]
    for row in rows:
        refused = expected(row['acceptance'], row['business'], row['user'], refuse=True)
        assert refused['final_state'] == refused['initial']
