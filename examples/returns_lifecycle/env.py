"""Return lifecycle tools. Managed writes are committed by the platform state machine."""
from env_factory.runtime import write, BusinessError


def get_return(db, now, return_id: str):
    """Look up the current return status and refund amount.

    Args:
        return_id: The return case identifier supplied by the customer.
    """
    row = next((r for r in db['returns'] if r['id'] == return_id), None)
    if row is None:
        raise BusinessError('return not found')
    return dict(row)


@write
def request_return(db, now, return_id: str, request_id: str):
    """Submit a return request after the customer approves.

    Args:
        return_id: The customer's return case identifier.
        request_id: Stable idempotency key; reuse it only for the same request.
    """
    return get_return(db, now, return_id)


@write
def cancel_return(db, now, return_id: str, request_id: str):
    """Cancel a requested return before the carrier receives the parcel.

    Args:
        return_id: The customer's return case identifier.
        request_id: Stable idempotency key for this cancellation.
    """
    return get_return(db, now, return_id)


@write
def request_refund(db, now, return_id: str, request_id: str):
    """Request payment after warehouse inspection accepted the returned goods.

    Args:
        return_id: The customer's return case identifier.
        request_id: Stable idempotency key for this refund request.
    """
    return get_return(db, now, return_id)


def make_tools():
    return {'get_return': get_return, 'request_return': request_return,
            'cancel_return': cancel_return, 'request_refund': request_refund}


def approval_key(tool, args, facts):
    return 'approve_refund' if tool == 'request_refund' else 'approve_return'


def reference(api):
    identifier = api.ask('return_id')
    scope = api.ask('scope')
    row = api.call('get_return', return_id=identifier)
    if row['status'] == 'none':
        args = {'return_id': identifier, 'request_id': 'return-' + identifier}
        token = api.confirm('approve_return', {'tool': 'request_return', 'args': args})
        if not token:
            return {'status': 'none', 'refund': 0}
        row = api.call('request_return', consent=token, **args)
    if scope == 'request_only':
        return {'status': row['status'], 'refund': row['refund']}
    # An explicit bounded observation window is part of this business's policy.
    for _ in range(10):
        if row['status'] in ('refunded', 'refund_failed', 'rejected', 'cancelled'):
            break
        if row['status'] == 'accepted':
            args = {'return_id': identifier, 'request_id': 'refund-' + identifier}
            token = api.confirm('approve_refund', {'tool': 'request_refund', 'args': args})
            if not token:
                break
            row = api.call('request_refund', consent=token, **args)
        api.wait(10)
        row = api.call('get_return', return_id=identifier)
    return {'status': row['status'], 'refund': row['refund']}
