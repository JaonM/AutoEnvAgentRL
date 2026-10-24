"""Lifecycle v2: causal activities, explicit handoffs and joins, with no scheduling clock."""
from __future__ import annotations

import copy

from env_factory.lifecycle import Lifecycle, LifecycleError, validate, expr, digest, row_of, seconds
from env_factory.runtime import BusinessError


class Workflow(Lifecycle):
    def __init__(self, schema, scenario, data, start, tools=None):
        problems = validate(schema, scenario, data, tools)
        if problems:
            raise LifecycleError('; '.join(problems))
        self.schema, self.scenario = copy.deepcopy(schema), copy.deepcopy(scenario)
        self.calendar, self.now = isinstance(start, str), seconds(start)
        self.pending, self.processed, self.log, self.receipts = [], [], [], {}
        self.count = 0
        self.check(data)
        for item in scenario.get('history', []):
            if item['type'] == 'yield':
                if not self.handoff(data)['progressed']:
                    raise LifecycleError('historical yield has no executable external activity')
            else:
                self.execute(data, item['transition'], item['entity'], item['args'],
                             key='history:' + item['key'], actor='agent', cause='history')
                self.settle(data)

    def stamp(self):
        return {'sequence': len(self.log) + 1}

    def dependencies(self, rule, context):
        witnesses = []
        for dependency in rule.get('requires', []):
            entity = expr(dependency['entity'], context)
            witness = next((e for e in reversed(self.log) if e.get('kind') == 'transition'
                            and e['transition'] == dependency['transition'] and e['entity'] == entity), None)
            if witness is None:
                return None
            witnesses.append(witness['id'])
        return {'dependencies': witnesses}

    def emit(self, rule, context, key):
        for index, emission in enumerate(rule.get('emit', [])):
            if expr(emission.get('when', True), context) is not True:
                continue
            event = {'id': 'event:' + digest([key, index]), 'transition': emission['transition'],
                     'entity': expr(emission['entity'], context), 'args': expr(emission.get('args', {}), context),
                     'cause': key, 'activation': len(self.log) + 1,
                     'ready': copy.deepcopy(emission.get('ready', True)),
                     'cancel_when': copy.deepcopy(emission.get('cancel_when', False))}
            self.pending.append(event)
            self.log.append({'kind': 'activated', **copy.deepcopy(event), **self.stamp()})

    def _context(self, db, event):
        rule = self.schema['transitions'][event['transition']]
        machine = self.schema['machines'][rule['machine']]
        row = row_of(db, machine['table'], event['entity'])
        return {'db': db, 'row': row, 'args': event['args'], 'params': self.scenario.get('parameters', {}),
                'entity': event['entity'], 'now': self.snapshot_now()}, rule, machine

    def snapshot_now(self):
        from env_factory.lifecycle import clock
        return clock(self.now, self.calendar)

    def _available(self, db, event):
        context, rule, machine = self._context(db, event)
        if expr(event['cancel_when'], context) is True or context['row'][machine['field']] not in rule['from']:
            return 'cancel'
        if expr(event['ready'], context) is not True or self.dependencies(rule, context) is None:
            return 'blocked'
        return 'ready'

    def _consume(self, db, event, *, cancel=False):
        if self.count >= self.schema.get('max_events', 1000):
            raise LifecycleError('episode activity budget exhausted')
        self.count += 1
        self.pending.remove(event)
        candidate, engine = copy.deepcopy(db), copy.deepcopy(self)
        try:
            if cancel:
                raise BusinessError('activity cancelled by source state or cancellation predicate')
            rule = self.schema['transitions'][event['transition']]
            engine.execute(candidate, event['transition'], event['entity'], event['args'],
                           key=event['id'], actor=rule['actor'], cause=event['cause'])
        except BusinessError as exc:
            self.log.append({'kind': 'event_cancelled', 'id': event['id'], 'cause': event['cause'],
                             'reason': str(exc), **self.stamp()})
            success = False
        else:
            db.clear(); db.update(candidate)
            self.__dict__.update(engine.__dict__)
            success = True
        self.processed.append(event['id'])
        return success

    def _ordered(self):
        priority = {name: i for i, name in enumerate(self.scenario.get('dispatch_order', []))}
        return sorted(self.pending, key=lambda e: (priority.get(e['transition'], len(priority)), e['activation'], e['id']))

    def settle(self, db):
        """Drain only enabled system-internal activities; never execute external actors."""
        while True:
            ready = None
            for event in self._ordered():
                availability = self._available(db, event)
                if availability == 'cancel':
                    ready = (event, True); break
                if availability == 'ready' and self.schema['transitions'][event['transition']].get('execution') == 'internal':
                    ready = (event, False); break
            if ready is None:
                return
            self._consume(db, ready[0], cancel=ready[1])

    def handoff(self, db):
        """One external completion, plus its immediate internal consequences, then yield back."""
        self.settle(db)
        self.log.append({'kind': 'handoff', **self.stamp()})
        for event in self._ordered():
            if self._available(db, event) != 'ready':
                continue
            progressed = self._consume(db, event)
            self.settle(db)
            # Public response exposes no private transition, scenario, queue or result choice.
            return {'progressed': progressed, 'status': 'resumed'}
        return {'progressed': False, 'status': 'blocked'}

    def advance(self, db, target):
        raise LifecycleError('workflow v2 has no scheduling clock; use yield_control')

    def snapshot(self):
        return copy.deepcopy({'version': 2, 'schema_sha256': digest(self.schema),
                              'scenario_sha256': digest(self.scenario), 'now': self.snapshot_now(),
                              'pending': self.pending, 'processed': self.processed, 'log': self.log,
                              'receipts': self.receipts, 'event_count': self.count})
