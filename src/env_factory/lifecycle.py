"""Versioned business state machines and a deterministic, causal event scheduler.

Specifications are immutable world inputs. Scenario parameters select external behavior;
only committed transitions emit events. No wall clock, model text, eval or global RNG.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import math

from env_factory.runtime import BusinessError, canonical


class LifecycleError(ValueError):
    """A malformed specification or an invariant/ownership violation (environment defect)."""


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def seconds(value):
    if isinstance(value, str):
        dt = datetime.datetime.fromisoformat(value)
        if dt.tzinfo is not None:
            raise LifecycleError('lifecycle calendar time must be timezone-naive')
        return (dt - datetime.datetime(1970, 1, 1)).total_seconds()
    if type(value) not in (int, float) or not math.isfinite(value):
        raise LifecycleError('time must be finite seconds or an ISO calendar time')
    return value


def clock(value, calendar):
    if calendar:
        try:
            dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=value)
        except (OverflowError, ValueError) as exc:
            raise LifecycleError('calendar time exceeds supported range') from exc
        return dt.isoformat(timespec='microseconds' if dt.microsecond else 'seconds' if dt.second else 'minutes')
    return value


def expr(value, context):
    # Same bounded data expressions as independent contracts; no business rule is shared.
    from env_factory.certify.behavior import expression
    return expression(value, context)


def row_of(db, table, identifier):
    found = [r for r in db.get(table, []) if r.get('id') == identifier]
    if len(found) != 1:
        raise BusinessError(f'{table}: expected one object with id {identifier}')
    return found[0]


def validate(schema, scenario, data, tools=None):
    """Fail closed on unsupported versions, unknown scenario knobs and broken bindings."""
    problems = []
    try:
        if schema.get('version') != 1 or not schema.get('machines') or not schema.get('transitions'):
            raise LifecycleError('lifecycle_schema needs version 1, machines and transitions')
        if set(schema) - {'version', 'machines', 'transitions', 'tools', 'parameters', 'invariants', 'max_events', 'scopes'}:
            raise LifecycleError('unknown lifecycle_schema field')
        if not isinstance(schema['machines'], dict) or not isinstance(schema['transitions'], dict):
            raise LifecycleError('machines/transitions must be objects')
        if len(schema['machines']) > 64 or len(schema['transitions']) > 256:
            raise LifecycleError('lifecycle schema exceeds machine/transition budget')
        scopes = schema.get('scopes')
        if not isinstance(scopes, list) or not scopes or not all(isinstance(s, str) and s for s in scopes):
            raise LifecycleError('scopes must be nonempty names')
        if 'scheduled_events' in data:
            raise LifecycleError('lifecycle cannot coexist with legacy scheduled_events')
        if type(schema.get('max_events', 1000)) is not int or not 1 <= schema.get('max_events', 1000) <= 10000:
            raise LifecycleError('max_events must be 1..10000')
        tables = set()
        for name, machine in schema['machines'].items():
            if set(machine) != {'table', 'field', 'states', 'initial'}:
                raise LifecycleError(f'{name}: machine requires table, field, states, initial')
            if machine['table'] in tables or machine['table'] not in data:
                raise LifecycleError(f'{name}: one machine per existing table required')
            tables.add(machine['table'])
            if not isinstance(machine['states'], list) or not all(isinstance(s, str) and s for s in machine['states']):
                raise LifecycleError(f'{name}: states must be nonempty strings')
            if not machine['states'] or len(set(machine['states'])) != len(machine['states']) or machine['initial'] not in machine['states']:
                raise LifecycleError(f'{name}: invalid states/initial')
            if machine['field'] == 'id':
                raise LifecycleError('machine state cannot be an identity')
            ids = [r.get('id') for r in data[machine['table']]]
            if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
                raise LifecycleError(f'{name}: unique string object ids required')
            if any(r.get(machine['field']) != machine['initial'] for r in data[machine['table']]):
                raise LifecycleError(f'{name}: base rows must be initial; construct later stages with scenario.history')
        for name, transition in schema['transitions'].items():
            if set(transition) - {'machine', 'actor', 'from', 'to', 'guard', 'effects', 'emit'}:
                raise LifecycleError(f'{name}: unknown transition field')
            machine = schema['machines'][transition['machine']]
            if transition['actor'] not in ('agent', 'customer', 'carrier', 'warehouse', 'payment', 'system'):
                raise LifecycleError(f'{name}: unknown actor')
            if (not isinstance(transition['from'], list) or not transition['from']
                    or not set(transition['from']) <= set(machine['states']) or transition['to'] not in machine['states']):
                raise LifecycleError(f'{name}: invalid source/target states')
            for effect in transition.get('effects', []):
                if set(effect) != {'table', 'entity', 'set'} or effect['table'] not in tables:
                    raise LifecycleError(f'{name}: effects must update declared machine tables')
                state_field = next(m['field'] for m in schema['machines'].values() if m['table'] == effect['table'])
                if not isinstance(effect['set'], dict) or {'id', state_field} & set(effect['set']):
                    raise LifecycleError(f'{name}: effects cannot bypass identity/state transitions')
            for event in transition.get('emit', []):
                if set(event) - {'transition', 'entity', 'delay', 'args', 'when'}:
                    raise LifecycleError(f'{name}: unknown emission field')
                target = schema['transitions'][event['transition']]
                if target['actor'] == 'agent':
                    raise LifecycleError('events cannot impersonate agent writes/consent')
                if 'delay' not in event or 'entity' not in event:
                    raise LifecycleError('emission requires delay and entity')
        for name, binding in schema.get('tools', {}).items():
            if set(binding) != {'transition', 'entity_arg', 'key_arg'}:
                raise LifecycleError(f'{name}: binding needs transition, entity_arg and key_arg')
            if schema['transitions'][binding['transition']]['actor'] != 'agent':
                raise LifecycleError(f'{name}: tools may only bind agent transitions')
            if tools is not None:
                import inspect
                from env_factory.runtime import is_write
                fn = tools.get(name)
                if not is_write(fn):
                    raise LifecycleError(f'{name}: lifecycle tool must be @write')
                params = inspect.signature(fn).parameters
                if binding['entity_arg'] not in params or binding['key_arg'] not in params:
                    raise LifecycleError(f'{name}: missing entity/idempotency argument')
        if scenario.get('version') != 1 or set(scenario) - {'version', 'parameters', 'history', 'scope'}:
            raise LifecycleError('scenario requires version 1 and only parameters/history/scope')
        if scenario.get('scope') not in schema.get('scopes', []):
            raise LifecycleError('scenario.scope must select a world-declared completion scope')
        definitions = schema.get('parameters', {})
        if set(scenario.get('parameters', {})) != set(definitions):
            raise LifecycleError('scenario must specify exactly the declared parameters')
        for name, rule in definitions.items():
            value = scenario['parameters'][name]
            if 'enum' in rule:
                if not any(type(value) is type(v) and value == v for v in rule['enum']):
                    raise LifecycleError(f'scenario parameter {name}: outside enum')
            elif (type(value) not in (int, float) or not math.isfinite(value)
                  or not rule['min'] <= value <= rule['max']):
                raise LifecycleError(f'scenario parameter {name}: outside bounds')
        history = scenario.get('history', [])
        if not isinstance(history, list) or len(history) > 1000:
            raise LifecycleError('history must have at most 1000 actions')
        previous = -math.inf
        keys = set()
        for item in history:
            if set(item) != {'offset', 'transition', 'entity', 'args', 'key'}:
                raise LifecycleError('history needs offset, transition, entity, args, key')
            if type(item['offset']) not in (int, float) or not isinstance(item['args'], dict):
                raise LifecycleError('history needs numeric offset and object args')
            if not isinstance(item['key'], str) or not item['key'] or item['key'] in keys:
                raise LifecycleError('history action keys must be unique nonempty strings')
            keys.add(item['key'])
            offset = seconds(item['offset'])
            if not previous <= offset <= 0:
                raise LifecycleError('history offsets must be chronological and <= 0')
            previous = offset
            if item['transition'] not in schema['transitions']:
                raise LifecycleError('history names unknown transition')
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        problems.append(str(exc))
    return problems


class Lifecycle:
    def __init__(self, schema, scenario, data, start, tools=None):
        problems = validate(schema, scenario, data, tools)
        if problems:
            raise LifecycleError('; '.join(problems))
        self.schema, self.scenario = copy.deepcopy(schema), copy.deepcopy(scenario)
        self.calendar = isinstance(start, str)
        self.now = seconds(start) + (scenario.get('history') or [{'offset': 0}])[0]['offset']
        self.pending, self.processed, self.log, self.receipts = [], [], [], {}
        self.count = 0
        self.check(data)
        # Historical actions are trusted task inputs, never policy-visible action types.
        # They must obey the same transition guards and event rules as live actions.
        for item in self.scenario.get('history', []):
            due = seconds(start) + item['offset']
            self.advance(data, due)
            self.execute(data, item['transition'], item['entity'], item['args'],
                         key='history:' + item['key'], actor=self.schema['transitions'][item['transition']]['actor'],
                         cause='history')
        self.advance(data, seconds(start))

    def check(self, db):
        for name, machine in self.schema['machines'].items():
            rows = db[machine['table']]
            if len({r['id'] for r in rows}) != len(rows):
                raise LifecycleError(f'{name}: duplicate identities')
            for row in rows:
                if row.get(machine['field']) not in machine['states']:
                    raise LifecycleError(f'{name}: invalid state')
        for rule in self.schema.get('invariants', []):
            contexts = ([{'db': db, 'row': row, 'params': self.scenario.get('parameters', {})}
                         for row in db[rule['table']]] if 'table' in rule else [{'db': db}])
            if any(expr(rule['assert'], context) is not True for context in contexts):
                raise LifecycleError('invariant failed: ' + rule['name'])

    def protected(self, before, after):
        for machine in self.schema['machines'].values():
            if canonical(before[machine['table']]) != canonical(after.get(machine['table'])):
                raise LifecycleError('managed table may only be changed by a declared transition: ' + machine['table'])
        self.check(after)

    def execute(self, db, transition, entity, args, *, key, actor, cause):
        if not isinstance(key, str) or not key or len(key) > 200:
            raise BusinessError('idempotency key must be a nonempty string of at most 200 characters')
        signature = canonical([transition, entity, args, actor])
        if key in self.receipts:
            receipt = self.receipts[key]
            if receipt['signature'] != signature:
                raise BusinessError('idempotency key reused for a different operation')
            return copy.deepcopy(receipt), True
        rule = self.schema['transitions'][transition]
        if actor != rule['actor']:
            raise BusinessError('actor cannot perform this transition')
        machine = self.schema['machines'][rule['machine']]
        row = row_of(db, machine['table'], entity)
        context = {'db': db, 'row': row, 'args': args, 'params': self.scenario.get('parameters', {}),
                   'entity': entity, 'now': clock(self.now, self.calendar)}
        source = row[machine['field']]
        if source not in rule['from'] or expr(rule.get('guard', True), context) is not True:
            raise BusinessError(f'{transition}: precondition rejected in state {source}')
        before = digest(db)
        for effect in rule.get('effects', []):
            target = row_of(db, effect['table'], expr(effect['entity'], context))
            changes = expr(effect['set'], context)
            if not set(changes) <= set(target):
                raise LifecycleError('effect cannot introduce undeclared columns')
            target.update(copy.deepcopy(changes))
        row[machine['field']] = rule['to']
        self.check(db)
        entry = {'kind': 'transition', 'id': key, 'cause': cause, 'transition': transition,
                 'entity': entity, 'actor': actor, 'args': copy.deepcopy(args), 'from': source,
                 'to': rule['to'], 'time': clock(self.now, self.calendar), 'before': before, 'after': digest(db)}
        self.log.append(entry)
        for index, event in enumerate(rule.get('emit', [])):
            if expr(event.get('when', True), context) is not True:
                continue
            delay = expr(event['delay'], context)
            if type(delay) not in (int, float) or not math.isfinite(delay) or delay < 0:
                raise LifecycleError('event delay must be finite nonnegative seconds')
            event_id = 'event:' + digest([key, index])
            due = self.now + delay
            if not math.isfinite(due):
                raise LifecycleError('event time exceeds supported range')
            clock(due, self.calendar)
            item = {'id': event_id, 'at': due, 'transition': event['transition'],
                    'entity': expr(event['entity'], context), 'args': expr(event.get('args', {}), context), 'cause': key}
            self.pending.append(item)
            self.log.append({'kind': 'scheduled', **copy.deepcopy(item)})
        self.pending.sort(key=lambda e: (e['at'], e['id']))
        receipt = {'signature': signature, 'transition': transition, 'entity': entity, 'state': rule['to']}
        self.receipts[key] = receipt
        return copy.deepcopy(receipt), False

    def advance(self, db, target):
        # Called on an episode transaction clone. Faults discard the whole transaction.
        if not math.isfinite(target):
            raise LifecycleError('target time must be finite')
        clock(target, self.calendar)
        if target < self.now:
            raise LifecycleError('clock cannot move backwards')
        while self.pending and self.pending[0]['at'] <= target:
            if self.count >= self.schema.get('max_events', 1000):
                raise LifecycleError('episode event budget exhausted')
            event = self.pending.pop(0)
            self.count += 1
            self.now = event['at']
            if event['id'] in self.processed:
                continue
            candidate, engine = copy.deepcopy(db), copy.deepcopy(self)
            try:
                rule = self.schema['transitions'][event['transition']]
                engine.execute(candidate, event['transition'], event['entity'], event['args'],
                               key=event['id'], actor=rule['actor'], cause=event['cause'])
            except BusinessError as exc:
                # Cancellation or a consumed resource can invalidate an in-flight event.
                # Never overwrite the newer state and never silently retry forever.
                self.log.append({'kind': 'event_cancelled', 'id': event['id'], 'cause': event['cause'],
                                 'time': clock(self.now, self.calendar), 'reason': str(exc)})
            else:
                db.clear(); db.update(candidate)
                self.__dict__.update(engine.__dict__)
            self.processed.append(event['id'])
        self.now = target

    def snapshot(self):
        return copy.deepcopy({'version': 1, 'schema_sha256': digest(self.schema),
                              'scenario_sha256': digest(self.scenario), 'now': clock(self.now, self.calendar),
                              'pending': self.pending, 'processed': self.processed, 'log': self.log,
                              'receipts': self.receipts, 'event_count': self.count})


def options(request):
    return {'lifecycle_schema': request.get('lifecycle_schema'), 'scenario': request.get('scenario')}


def expected(verifier, data, facts, start, request):
    kwargs = {'scenario': copy.deepcopy(request['scenario'])} if request.get('lifecycle_schema') else {}
    return verifier.expected(copy.deepcopy(data), copy.deepcopy(facts), start, **kwargs)


def audit(verifier, data, facts, start, result, request):
    """Independent business history validation, in addition to exact state/answer grading."""
    if not request.get('lifecycle_schema'):
        return []
    fn = getattr(verifier, 'verify_lifecycle', None)
    if not callable(fn):
        return ['verifier must implement verify_lifecycle for lifecycle tasks']
    errors = fn(copy.deepcopy(data), copy.deepcopy(facts), start, copy.deepcopy(result),
                scenario=copy.deepcopy(request['scenario']))
    if not isinstance(errors, list) or not all(isinstance(e, str) for e in errors):
        raise LifecycleError('verify_lifecycle must return a list of error strings')
    return errors


def qualify(env, verifier, request):
    """Mandatory independent temporal contract and verifier anti-forgery checks."""
    schema = request.get('lifecycle_schema')
    if schema is None:
        return {'passed': True, 'applicable': False}
    problems = validate(schema, request.get('scenario', {}), request['tables'], env.make_tools())
    contract = request.get('lifecycle_contract')
    if not isinstance(contract, dict) or not contract.get('cases'):
        problems.append('lifecycle requires independent lifecycle_contract cases')
    if not callable(getattr(verifier, 'verify_lifecycle', None)):
        problems.append('lifecycle requires verifier.verify_lifecycle')
    if problems:
        return {'passed': False, 'problems': problems}
    from env_factory.certify import behavior
    from env_factory.runtime import run_reference
    report = behavior.run(env, contract, request=request)
    # Independent tests must exercise timing, actual transitions and refused operations.
    steps = [s for case in contract['cases'] for s in case.get('steps', [])]
    if not any('wait' in s for s in steps) or not any(s.get('error') for s in steps):
        problems.append('lifecycle contracts need explicit waits and expected business rejections')
    covered = {t for c in report.get('cases', []) if c['passed'] for t in c.get('transitions', [])}
    if set(schema['transitions']) - covered:
        problems.append('lifecycle contract lacks transition coverage: ' + ', '.join(sorted(set(schema['transitions']) - covered)))
    try:
        result = run_reference(env, request['tables'], request['user_facts'], start=request['time']['start'], request=request)
        problems += audit(verifier, request['tables'], request['user_facts'], request['time']['start'], result, request)
        changed = copy.deepcopy(result)
        transitions = [e for e in changed['lifecycle']['log'] if e['kind'] == 'transition']
        if transitions:
            transitions[-1]['from'] = '__invalid_predecessor__'
            if not audit(verifier, request['tables'], request['user_facts'], request['time']['start'], changed, request):
                problems.append('independent verifier accepted a forged predecessor state')
    except Exception as exc:
        problems.append(f'lifecycle verification failed: {type(exc).__name__}: {exc}')
    return {**report, 'passed': report['passed'] and not problems, 'problems': problems,
            'schema_sha256': digest(schema), 'scenario_sha256': digest(request['scenario'])}
