from copy import deepcopy

from env_factory.runtime import InvalidWorld


def expected(data, facts, now):
    state = deepcopy(data)
    books = {b['id']: b for b in state['books']}
    if any(row['book_id'] not in books for table in ('loans', 'holds') for row in state[table]):
        raise InvalidWorld('dangling book reference')
    loans = {l['id']: l for l in state['loans']}
    reader = facts['reader_id']
    renewed, blocked = [], []
    for index, loan_id in enumerate(facts['loan_ids']):
        loan = loans[loan_id]
        book = books[loan['book_id']]
        holds = [h for h in state['holds'] if h['active'] and h['book_id'] == book['id']]
        eligible = (loan['reader_id'] == reader and loan['active'] and loan['due_day'] >= 10
                    and loan['renewals'] < book['renewal_limit']
                    and not any(h['reader_id'] != reader for h in holds))
        if not eligible:
            blocked.append({'loan_id': loan_id, 'active_hold_count': len(holds)})
        elif facts[('approve_first', 'approve_second')[index]]:
            loan['due_day'] += book['renewal_days']
            loan['renewals'] += 1
            renewed.append(loan_id)
    return {'answer': {'renewed_loan_ids': renewed, 'blocked': blocked}, 'final_state': state}
