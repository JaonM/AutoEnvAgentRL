"""Independent business arithmetic and causal-history verifier; no executor imports."""
import copy
import datetime
from env_factory.runtime import InvalidWorld


RULES = {
    'apply': ('agent', 'none', 'requested'),
    'handover': ('customer', 'requested', 'in_transit'),
    'receive': ('carrier', 'in_transit', 'received'),
    'accept': ('warehouse', 'received', 'accepted'),
    'reject': ('warehouse', 'received', 'rejected'),
    'refund': ('agent', 'accepted', 'refund_pending'),
    'settle': ('payment', 'refund_pending', 'refunded'),
    'payment_failure': ('payment', 'refund_pending', 'refund_failed'),
    'cancel': ('agent', 'requested', 'cancelled'),
}
TOOLS = {'apply': 'request_return', 'refund': 'request_refund', 'cancel': 'cancel_return'}


def moment(value):
    if isinstance(value, str):
        return (datetime.datetime.fromisoformat(value) - datetime.datetime(1970, 1, 1)).total_seconds()
    return value


def emissions(action, params):
    if action == 'apply' and params['will_ship']:
        return [('handover', params['ship_delay'])]
    if action == 'handover':
        return [('receive', 20)]
    if action == 'receive':
        return [('accept' if params['inspection_pass'] else 'reject', 10)]
    if action == 'refund':
        return [('settle' if params['payment_success'] else 'payment_failure', 10)]
    return []


def initial(data):
    state = copy.deepcopy(data)
    for row in state['returns']:
        if row['status'] != 'none' or row['refund'] != 0 or row['amount'] <= 0:
            raise InvalidWorld('base return rows require positive amounts, no refund and initial state')
    return state


def change(state, action, entity):
    row = next(r for r in state['returns'] if r['id'] == entity)
    if row['status'] != RULES[action][1]:
        raise ValueError('invalid predecessor')
    row['status'] = RULES[action][2]
    if action == 'settle':
        row['refund'] = row['amount']


def expected(data, facts, now, *, scenario):
    if facts['scope'] != scenario['scope']:
        raise InvalidWorld('customer goal and scenario completion scope disagree')
    state = initial(data)
    params = scenario['parameters']
    pending = []
    def apply(action, entity, at):
        change(state, action, entity)
        pending.extend((at + delay, next_action, entity) for next_action, delay in emissions(action, params))
        pending.sort()
    def advance(until):
        while pending and pending[0][0] <= until:
            at, action, entity = pending.pop(0)
            row = next(r for r in state['returns'] if r['id'] == entity)
            if row['status'] == RULES[action][1]:
                apply(action, entity, at)
    current = moment(now)
    for item in scenario.get('history', []):
        at = moment(now) + item['offset']
        advance(at)
        apply(item['transition'], item['entity'], at)
    advance(current)
    row = next(r for r in state['returns'] if r['id'] == facts['return_id'])
    if row['status'] == 'none' and facts.get('approve_return'):
        apply('apply', row['id'], current)
    if facts['scope'] == 'full' and row['status'] != 'none':
        for _ in range(10):
            if row['status'] in ('refunded','refund_failed','rejected','cancelled'):
                break
            if row['status'] == 'accepted':
                if not facts.get('approve_refund'):
                    break
                apply('refund', row['id'], current)
            current += 10
            advance(current)
    return {'answer': {'status': row['status'], 'refund': row['refund']}, 'final_state': state}


def verify_lifecycle(data, facts, now, result, *, scenario):
    """Check realized actions against independent causal rules, not the reference path."""
    errors, allowed, used = [], [], set()
    state = initial(data)
    last = None
    for entry in result.get('lifecycle', {}).get('log', []):
        if entry['kind'] != 'transition':
            continue
        action, key, entity, at = entry['transition'], entry['id'], entry['entity'], moment(entry['time'])
        if action not in RULES or key in used:
            errors.append('unknown or duplicate transition'); continue
        used.add(key)
        actor, source, target = RULES[action]
        if (entry['actor'], entry['from'], entry['to']) != (actor, source, target):
            errors.append('transition actor or predecessor is invalid')
        if (last is not None and at < last) or at > moment(result['now']):
            errors.append('transition outside chronological episode history')
        last = at
        if entry['cause'] == 'history':
            matched = any('history:' + h['key'] == key and h['transition'] == action and h['entity'] == entity
                          and moment(now) + h['offset'] == at and h['args'] == entry['args']
                          for h in scenario.get('history', []))
            if not matched: errors.append('unapproved historical transition')
        elif actor == 'agent':
            matched = any(t.get('kind') == 'tool' and t.get('status') == 'ok' and t.get('tool') == TOOLS[action]
                          and t['args'].get('return_id') == entity and 'tool:' + t['args'].get('request_id', '') == key
                          and moment(t['time']) == at for t in result['trace'])
            if not matched: errors.append('agent transition lacks a committed tool call')
        else:
            event = (entry['cause'], action, entity, at)
            if event not in allowed:
                errors.append('external transition lacks its causal event')
            else:
                allowed.remove(event)
        try:
            change(state, action, entity)
        except (ValueError, StopIteration):
            errors.append('business predecessor missing'); continue
        allowed.extend((key, next_action, entity, at + delay) for next_action, delay in emissions(action, scenario['parameters']))
    if state != result['final_state']:
        errors.append('realized transition history does not produce the final business state')
    return errors
