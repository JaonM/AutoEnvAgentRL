"""Independent synthetic library truth; never import generated reward or DSL.

This evaluator is preparation infrastructure, not measured factory yield.
"""
from __future__ import annotations

import copy


def record(data, table, identity):
    rows = [row for row in data[table] if row['id'] == identity]
    if len(rows) > 1:
        raise ValueError('Library oracle requires unique record identities')
    return rows[0] if rows else None


def active_queue(data, book_id):
    return sorted((row for row in data['holds'] if row['book_id'] == book_id and row['active']),
                  key=lambda row: (row['requested_day'], row['id']))


def renewal_decision(data, loan_id, reader_id, current_day):
    loan = record(data, 'loans', loan_id)
    if loan is None:
        return False, 'missing_loan'
    if loan['reader_id'] != reader_id:
        return False, 'different_reader'
    if not loan['active']:
        return False, 'inactive_loan'
    book = record(data, 'books', loan['book_id'])
    if book is None:
        raise ValueError('Library loan has a dangling book relationship')
    if current_day > loan['due_day']:
        return False, 'overdue'
    if loan['renewals'] >= book['renewal_limit']:
        return False, 'renewal_limit'
    if any(row['reader_id'] != reader_id for row in active_queue(data, loan['book_id'])):
        return False, 'competing_hold'
    return True, 'eligible'


def expected(spec, data, facts, *, refuse=False, approvals=None):
    """Return independent answers and complete expected state, with no grader reads."""
    initial = copy.deepcopy(data)
    final = copy.deepcopy(data)
    params, kind = spec['params'], spec['kind']
    if kind == 'library_fine':
        values = [params[key] for key in ('overdue_days', 'grace_days', 'daily_cents', 'cap_cents')]
        if any(type(value) is not int or value < 0 for value in values):
            raise ValueError('Library fine inputs must be nonnegative integers')
        overdue, grace, rate, cap = values
        answer = {'fine': min(max(0, overdue - grace) * rate, cap) / 100}
    elif kind == 'library_renewal_eligibility':
        eligible, reason = renewal_decision(final, facts['loan_id'], facts['reader_id'], params['current_day'])
        answer = {'eligible': eligible}
    elif kind == 'library_queue':
        hold = record(final, 'holds', facts['hold_id'])
        if hold is None or not hold['active'] or hold['reader_id'] != facts['reader_id']:
            answer = {'rank': None, 'ahead': None}
        else:
            queue = active_queue(final, hold['book_id'])
            index = next(i for i, row in enumerate(queue) if row['id'] == hold['id'])
            answer = {'rank': index + 1, 'ahead': index}
    elif kind == 'library_partial_renewal':
        identifiers = facts['loan_ids']
        if len(identifiers) != 2 or len(set(identifiers)) != 2:
            raise ValueError('Library partial renewal requires two distinct loans')
        approval_values = approvals if approvals is not None else [facts['approve_first'], facts['approve_second']]
        if len(approval_values) != 2 or any(type(value) is not bool for value in approval_values):
            raise ValueError('Library approvals must be two independent booleans')
        renewed = []
        blocked_queue_counts = []
        for identifier, approved in zip(identifiers, approval_values):
            eligible, _ = renewal_decision(final, identifier, facts['reader_id'], params['current_day'])
            if not eligible:
                loan = record(final, 'loans', identifier)
                blocked_queue_counts.append({'loan_id': identifier, 'count':
                    len(active_queue(final, loan['book_id'])) if loan is not None else 0})
            if eligible and approved and not refuse:
                loan = record(final, 'loans', identifier)
                book = record(final, 'books', loan['book_id'])
                loan['due_day'] += book['renewal_days']
                loan['renewals'] += 1
                renewed.append(identifier)
        answer = {'renewed_ids': renewed, 'blocked_queue_counts': blocked_queue_counts}
    elif kind == 'library_pickup_change':
        hold = record(final, 'holds', facts['hold_id'])
        target = record(final, 'copies', facts['copy_choice'])
        old = record(final, 'copies', hold['copy_id']) if hold is not None else None
        site = record(final, 'sites', target['site_id']) if target is not None else None
        available_sites = sorted({row['site_id'] for row in initial['copies']
            if hold is not None and hold['active'] and hold['reader_id'] == facts['reader_id']
            and row['book_id'] == hold['book_id'] and row['reserved_by'] is None
            and record(initial, 'sites', row['site_id']) is not None
            and record(initial, 'sites', row['site_id'])['supports_holds']})
        eligible = (hold is not None and hold['active'] and hold['reader_id'] == facts['reader_id']
                    and target is not None and site is not None and site['supports_holds']
                    and target['book_id'] == hold['book_id'] and target['reserved_by'] is None
                    and old is not None and old['reserved_by'] == hold['id'] and old['id'] != target['id'])
        if approvals is not None and (len(approvals) != 1 or type(approvals[0]) is not bool):
            raise ValueError('Library pickup change requires one boolean approval')
        approved = facts['approve_change'] if approvals is None else approvals[0]
        if type(approved) is not bool:
            raise ValueError('Library change approval must be boolean')
        committed = bool(eligible and approved and not refuse)
        if committed:
            old['reserved_by'] = None
            target['reserved_by'] = hold['id']
            hold['copy_id'] = target['id']
        answer = {'committed': committed, 'available_site_ids': available_sites}
    else:
        raise ValueError('Unknown independent library business kind')
    return {'answer': answer, 'initial': initial, 'final_state': final}
