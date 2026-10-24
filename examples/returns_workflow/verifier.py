"""Independent return policy oracle. Does not import the workflow executor or schema."""
import copy
import hashlib
import json

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
TOOLS = {'request_return': 'apply', 'request_refund': 'refund', 'cancel_return': 'cancel'}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class Policy:
    """Small policy-specific oracle: shipping, receipt, inspection, then payment."""
    def __init__(self, data, scenario):
        self.state = copy.deepcopy(data)
        self.params = scenario['parameters']
        self.order = scenario.get('dispatch_order', [])
        self.pending, self.log, self.processed, self.receipts = [], [], [], {}
        for row in self.state['returns']:
            if row['status'] != 'none' or row['refund'] != 0 or row['amount'] <= 0:
                raise InvalidWorld('initial returns require positive amounts and no completed work')
        for item in scenario.get('history', []):
            if item['type'] == 'yield':
                if not self.handoff()['progressed']:
                    raise InvalidWorld('historical handoff has no work')
            else:
                if RULES[item['transition']][0] != 'agent':
                    raise InvalidWorld('external work cannot be injected as history')
                self.change(item['transition'], item['entity'], item['args'], 'history:' + item['key'], 'history')

    def row(self, entity):
        return next(row for row in self.state['returns'] if row['id'] == entity)

    def record(self, **entry):
        self.log.append({**entry, 'sequence': len(self.log) + 1})

    def successor(self, action):
        if action == 'apply' and self.params['will_ship']:
            return 'handover', 0
        if action == 'handover':
            return 'receive', 0
        if action == 'receive':
            return ('accept', 0) if self.params['inspection_pass'] else ('reject', 1)
        if action == 'refund':
            return ('settle', 0) if self.params['payment_success'] else ('payment_failure', 1)
        return None

    def change(self, action, entity, args, key, cause):
        actor, source, target = RULES[action]
        signature = canonical([action, entity, args, actor])
        if key in self.receipts:
            if self.receipts[key]['signature'] != signature:
                raise ValueError('conflicting retry')
            return
        row = self.row(entity)
        if row['status'] != source:
            raise ValueError('missing business predecessor')
        before = digest(self.state)
        row['status'] = target
        if action == 'settle':
            row['refund'] = row['amount']
        self.record(kind='transition', id=key, cause=cause, transition=action, entity=entity,
                    actor=actor, args=copy.deepcopy(args), **{'from': source, 'to': target},
                    before=before, after=digest(self.state), dependencies=[])
        next_step = self.successor(action)
        if next_step:
            name, index = next_step
            event = {'id': 'event:' + digest([key, index]), 'transition': name, 'entity': entity,
                     'args': {}, 'cause': key, 'activation': len(self.log) + 1,
                     'ready': True, 'cancel_when': False}
            self.pending.append(event)
            self.record(kind='activated', **event)
        receipt = {'signature': signature, 'transition': action, 'entity': entity, 'state': target}
        if actor == 'agent' and cause.startswith('tool:'):
            receipt['result'] = copy.deepcopy(row)
        self.receipts[key] = receipt
        self.cancel_stale()

    def ordered(self):
        priority = {name: index for index, name in enumerate(self.order)}
        return sorted(self.pending, key=lambda e: (priority.get(e['transition'], len(priority)), e['activation'], e['id']))

    def cancel_stale(self):
        for event in self.ordered():
            if self.row(event['entity'])['status'] != RULES[event['transition']][1]:
                self.pending.remove(event)
                self.processed.append(event['id'])
                self.record(kind='event_cancelled', id=event['id'], cause=event['cause'])

    def handoff(self):
        self.cancel_stale()
        self.record(kind='handoff')
        if not self.pending:
            return {'progressed': False, 'status': 'blocked'}
        event = self.ordered()[0]
        self.pending.remove(event)
        self.change(event['transition'], event['entity'], event['args'], event['id'], event['cause'])
        self.processed.append(event['id'])
        return {'progressed': True, 'status': 'resumed'}


def expected(data, facts, now, *, scenario):
    if facts['scope'] != scenario['scope']:
        raise InvalidWorld('customer goal and completion scope disagree')
    policy = Policy(data, scenario)
    row = policy.row(facts['return_id'])
    if row['status'] == 'none' and facts.get('approve_return'):
        policy.change('apply', row['id'], {}, 'expected:apply', 'oracle')
    if facts['scope'] == 'full' and row['status'] != 'none':
        for _ in range(10):
            if row['status'] in ('refunded', 'refund_failed', 'rejected', 'cancelled'):
                break
            if row['status'] == 'accepted':
                if not facts.get('approve_refund'):
                    break
                policy.change('refund', row['id'], {}, 'expected:refund', 'oracle')
            if not policy.handoff()['progressed']:
                break
    return {'answer': {'status': row['status'], 'refund': row['refund']}, 'final_state': policy.state}


def verify_lifecycle(data, facts, now, result, *, scenario):
    """Rebuild causal work from committed commands/handoffs, never from claimed transitions."""
    errors = []
    try:
        policy = Policy(data, scenario)
        for entry in result['trace']:
            if entry['kind'] == 'yield':
                if policy.handoff() != entry['result']:
                    errors.append('handoff result contradicts executable work')
            elif entry['kind'] == 'tool' and entry.get('status') == 'ok' and entry['tool'] in TOOLS:
                args = entry['args']
                policy.change(TOOLS[entry['tool']], args['return_id'], args,
                              'tool:' + args['request_id'], 'tool:' + entry['tool'])
        realized = result['lifecycle']
        # Cancellation reasons are explanatory text; the causal cancellation itself is checked.
        log = [{k: v for k, v in item.items() if k != 'reason'} for item in realized['log']]
        if log != policy.log:
            errors.append('causal history differs from independently replayed commands and handoffs')
        for name in ('pending', 'processed', 'receipts'):
            if realized[name] != getattr(policy, name):
                errors.append('incorrect causal ' + name)
        if realized['event_count'] != len(policy.processed):
            errors.append('incorrect processed activity count')
        if result['now'] != now:
            errors.append('workflow cannot advance the calendar')
        if policy.state != result['final_state']:
            errors.append('causal actions do not produce the final business state')
    except (KeyError, ValueError, TypeError, StopIteration) as exc:
        errors.append('invalid causal history: ' + str(exc))
    return errors
