from env_factory_v3.runtime import BusinessError, write

BUSINESS_DAY = 10


def _find(rows, record_id, label):
    matches = [row for row in rows if row['id'] == record_id]
    if len(matches) != 1:
        raise BusinessError(f'{label} {record_id} not found')
    return matches[0]


def _reasons(db, loan_id, reader_id):
    loan = _find(db['loans'], loan_id, 'loan')
    book = _find(db['books'], loan['book_id'], 'book')
    reasons = []
    if loan['reader_id'] != reader_id:
        reasons.append('not_your_loan')
    if not loan['active']:
        reasons.append('inactive_loan')
    if loan['due_day'] < BUSINESS_DAY:
        reasons.append('overdue')
    if loan['renewals'] >= book['renewal_limit']:
        reasons.append('renewal_limit_reached')
    if any(h['active'] and h['book_id'] == book['id'] and h['reader_id'] != reader_id for h in db['holds']):
        reasons.append('other_reader_has_active_hold')
    return loan, book, reasons


def check_renewal(db, now, loan_id: str, reader_id: str):
    """Check whether a loan can be renewed by this reader today; returns eligibility and book id.

    Args:
        loan_id: loan id, e.g. "L1"
        reader_id: id of the reader asking, e.g. "R1"
    """
    loan, book, reasons = _reasons(db, loan_id, reader_id)
    return {'loan_id': loan['id'], 'book_id': book['id'], 'eligible': not reasons, 'reasons': reasons}


def get_waitlist(db, now, book_id: str):
    """Number of active holds (all readers) on a book.

    Args:
        book_id: book id, as returned by check_renewal
    """
    _find(db['books'], book_id, 'book')
    return {'book_id': book_id, 'active_hold_count': sum(1 for h in db['holds'] if h['active'] and h['book_id'] == book_id)}


@write
def renew_loan(db, now, loan_id: str, reader_id: str):
    """Renew one eligible loan: due day + renewal_days, renewals + 1.

    Args:
        loan_id: loan id to renew
        reader_id: id of the reader who owns the loan
    """
    loan, book, reasons = _reasons(db, loan_id, reader_id)
    if reasons:
        raise BusinessError(', '.join(reasons))
    loan['due_day'] += book['renewal_days']
    loan['renewals'] += 1
    return {'loan_id': loan_id, 'due_day': loan['due_day'], 'renewals': loan['renewals']}


def make_tools():
    return {'check_renewal': check_renewal, 'get_waitlist': get_waitlist, 'renew_loan': renew_loan}


def reference(api):
    reader_id = api.ask('reader_id')
    renewed, blocked = [], []
    for index, loan_id in enumerate(api.ask('loan_ids')):
        status = api.call('check_renewal', loan_id=loan_id, reader_id=reader_id)
        if not status['eligible']:
            waitlist = api.call('get_waitlist', book_id=status['book_id'])
            blocked.append({'loan_id': loan_id, 'active_hold_count': waitlist['active_hold_count']})
            continue
        args = {'loan_id': loan_id, 'reader_id': reader_id}
        token = api.confirm(('approve_first', 'approve_second')[index], {'tool': 'renew_loan', 'args': args})
        if token is not None:
            api.call('renew_loan', consent=token, **args)
            renewed.append(loan_id)
    return {'renewed_loan_ids': renewed, 'blocked': blocked}
