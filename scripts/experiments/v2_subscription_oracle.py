"""Independent subscription arithmetic using exact rational amounts.

Preparation policy: periods are [start, end), cents rounded down once after
aggregation. These conventions must appear in public policy before freezing.
"""
from fractions import Fraction
import copy


def natural(value, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError('Subscription day counts and cents must be nonnegative integers')
    return value


def refund(paid_cents, period_days, used_days, refunded_cents=0):
    natural(paid_cents)
    natural(period_days, 1)
    natural(used_days)
    natural(refunded_cents)
    if used_days > period_days or refunded_cents > paid_cents:
        raise ValueError('Subscription usage or ledger exceeds its period or payment')
    prorated = Fraction(paid_cents * (period_days - used_days), period_days)
    return min(prorated.numerator // prorated.denominator, paid_cents - refunded_cents)


def effective_plan(subscription, changes, day):
    applicable = [row for row in changes if row['subscription_id'] == subscription['id']
                  and not row['cancelled'] and row['effective_day'] <= day]
    if not applicable:
        return subscription['base_plan_id']
    latest_day = max(row['effective_day'] for row in applicable)
    latest = [row for row in applicable if row['effective_day'] == latest_day]
    if len(latest) != 1:
        raise ValueError('Conflicting effective changes require an explicit public precedence rule')
    return latest[0]['plan_id']


def bill(subscription, changes, plans, start, end):
    natural(start)
    natural(end)
    if end <= start:
        raise ValueError('Billing period must have positive duration')
    if len({p['id'] for p in plans}) != len(plans):
        raise ValueError('Plan identities must be unique')
    rates = {p['id']: Fraction(natural(p['period_cents']), natural(p['period_days'], 1)) for p in plans}
    boundaries = sorted({start, end, *(row['effective_day'] for row in changes
        if row['subscription_id'] == subscription['id'] and not row['cancelled']
        and start < row['effective_day'] < end)})
    total = Fraction(0)
    intervals = []
    for left, right in zip(boundaries, boundaries[1:]):
        plan_id = effective_plan(subscription, changes, left)
        if plan_id not in rates:
            raise ValueError('Subscription references a missing plan')
        total += (right - left) * rates[plan_id]
        intervals.append({'start': left, 'end': right, 'plan_id': plan_id})
    return {'amount_cents': total.numerator // total.denominator, 'intervals': intervals}


def unique(rows, identity):
    found = [row for row in rows if row['id'] == identity]
    if len(found) != 1:
        raise ValueError('Subscription business identity must exist uniquely')
    return found[0]


def cancel_refund(data, subscription_id, reader_id, *, used_days, approved, refund_id):
    if type(approved) is not bool:
        raise ValueError('Cancellation authorization must be boolean')
    final = copy.deepcopy(data)
    subscription = unique(final['subscriptions'], subscription_id)
    if not approved or subscription['reader_id'] != reader_id or not subscription['active']:
        return False, 0, final
    # A duplicate request must never write a second ledger entry.
    if any(row['id'] == refund_id for row in final['refunds']):
        return False, 0, final
    ledger_total = sum(row['amount_cents'] for row in final['refunds']
                       if row['subscription_id'] == subscription_id)
    if ledger_total != subscription['refunded_cents']:
        raise ValueError('Refund ledger and subscription balance disagree')
    amount = refund(subscription['paid_cents'], subscription['period_days'], used_days, ledger_total)
    if amount == 0:
        return False, 0, final
    subscription['active'] = False
    subscription['refunded_cents'] += amount
    final['refunds'].append({'id': refund_id, 'subscription_id': subscription_id, 'amount_cents': amount})
    return True, amount, final


def change_date(data, subscription_id, reader_id, *, change_id, plan_id, day,
                start, end, authorized_day, authorized_cents, approved, bill_id):
    """Atomic replacement of one pending change and its existing bill."""
    if type(approved) is not bool:
        raise ValueError('Date-change authorization must be boolean')
    natural(day)
    final = copy.deepcopy(data)
    subscription = unique(final['subscriptions'], subscription_id)
    change = unique(final['changes'], change_id)
    invoice = unique(final['bills'], bill_id)
    if (not approved or authorized_day != day or subscription['reader_id'] != reader_id
            or not subscription['active'] or change['subscription_id'] != subscription_id
            or invoice['subscription_id'] != subscription_id):
        return False, final
    change.update(effective_day=day, plan_id=plan_id, cancelled=False)
    amount = bill(subscription, final['changes'], final['plans'], start, end)['amount_cents']
    if type(authorized_cents) is not int or authorized_cents != amount:
        return False, copy.deepcopy(data)
    invoice.update(amount_cents=amount, period_start=start, period_end=end)
    return True, final
