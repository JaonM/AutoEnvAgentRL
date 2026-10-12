"""Episode runtime shared by authoring checks, gates, the HTTP server and training.

An environment module (env.py) defines:

    make_tools() -> dict[str, callable]
        read tool:  fn(db, now, **args) -> JSON value. Runs on a private copy of db.
        write tool: decorated with @write; fn(db, now, **args) mutates db in place and
                    returns a JSON receipt. Raise BusinessError to reject; the write is
                    rolled back atomically.
    reference(api) -> dict
        a deterministic reference agent using only api.call / api.ask / api.confirm and
        returning an answer that conforms to answer_schema.json.

Every write needs consent: token = api.confirm(approval_key, {"tool": name, "args": args})
(a missing, forged, reused or other-tool token is a safety violation; the right token with
different arguments only refuses that write, and the token stays usable)
and then api.call(name, consent=token, **args). A token is single-use and bound to that
exact proposal. Attempting a write without valid consent is a safety violation.
"""
from __future__ import annotations

import copy
import inspect
import json
import re


EVENTS = 'scheduled_events'


def apply_event(db, event):
    """Set `event['set']` on row `event['row_id']` of `event['table']` (oracles may call this too)."""
    row = next((r for r in db.get(event['table'], []) if r.get('id') == event['row_id']), None)
    if row is not None:
        row.update(copy.deepcopy(event['set']))


class BusinessError(Exception):
    """A business rule rejected the action. Writes roll back atomically."""


class InvalidWorld(Exception):
    """Raised by an oracle when generated world data violates the business's own invariants."""


class EpisodeFinished(Exception):
    """The episode already received its final answer."""


def write(fn):
    fn.__v3_write__ = True
    return fn


def is_write(fn):
    return getattr(fn, '__v3_write__', False) is True


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def jsonable(value):
    json.dumps(value, allow_nan=False)
    return value


JSON_TYPES = {'str': 'string', 'int': 'integer', 'float': 'number', 'bool': 'boolean', 'dict': 'object',
              'list': 'array'}


def json_type(annotation):
    """JSON schema of a parameter annotation: str, int, float, bool, dict, list[...], X | None."""
    if isinstance(annotation, str):
        text = annotation
    elif isinstance(annotation, type):
        text = annotation.__name__
    else:
        text = str(annotation)
    options = text.replace('typing.', '').replace(' ', '').split('|')
    kept = [o for o in options if o not in ('None', 'NoneType')]
    match = re.fullmatch(r'(str|int|float|bool|dict|list)(?:\[(.+)\])?', kept[0]) if len(kept) == 1 else None
    if match is None:
        return None
    schema = {'type': JSON_TYPES[match[1]]}
    if match[1] == 'list' and match[2]:
        items = json_type(match[2])
        if items is None:
            return None
        schema['items'] = items
    if len(kept) < len(options):
        schema['type'] = [schema['type'], 'null']
    return schema


def argument_docs(doc):
    """`name: description` lines of a docstring's Args: section."""
    docs, inside, last = {}, False, None
    for line in (doc or '').splitlines():
        stripped = line.strip()
        if stripped.lower() in ('args:', 'arguments:', 'parameters:'):
            inside = True
        elif inside and not stripped:
            break
        elif inside:
            match = re.match(r'(\w+)\s*(?:\([^)]*\))?\s*:\s*(.+)', stripped)
            if match:
                last = match[1]
                docs[last] = match[2]
            elif last:
                docs[last] += ' ' + stripped
    return docs


def _parameters(fn):
    return [p for p in list(inspect.signature(fn).parameters.values())[2:]
            if p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]


def tool_spec_problems(tools):
    """The public tool description is all an agent sees: every parameter needs a type and a meaning."""
    problems = []
    for name in sorted(tools):
        fn = tools[name]
        if not (inspect.getdoc(fn) or '').strip():
            problems.append(f'tool {name}: missing docstring')
        docs = argument_docs(inspect.getdoc(fn))
        for parameter in _parameters(fn):
            if parameter.annotation is parameter.empty or json_type(parameter.annotation) is None:
                problems.append(f'tool {name}: parameter {parameter.name} needs a type annotation '
                                '(str, int, float, bool, dict, list[...], or X | None)')
            if parameter.name not in docs:
                problems.append(f'tool {name}: parameter {parameter.name} needs a description in the '
                                'docstring Args: section')
    return problems


def tool_specs(tools):
    """Public wire description of tools, derived from signatures, annotations and docstrings."""
    specs = []
    for name in sorted(tools):
        fn = tools[name]
        docs = argument_docs(inspect.getdoc(fn))
        properties, required = {}, []
        for parameter in _parameters(fn):
            typed = json_type(parameter.annotation) if parameter.annotation is not parameter.empty else None
            properties[parameter.name] = {**(typed or {}), 'description': docs.get(parameter.name, parameter.name)}
            if parameter.default is parameter.empty:
                required.append(parameter.name)
        if is_write(fn):
            properties['consent'] = {'type': 'string', 'description': 'token returned by confirm for this exact call'}
            required.append('consent')
        specs.append({'name': name, 'kind': 'write' if is_write(fn) else 'read',
                      'description': inspect.getdoc(fn) or '',
                      'parameters': {'type': 'object', 'properties': properties, 'required': required}})
    return specs


class Episode:
    """One isolated run: data, private user facts, a virtual clock and a trace."""

    BusinessError = BusinessError

    def __init__(self, tools, data, facts, *, start=0, tick=0):
        self.tools = tools
        self.initial = copy.deepcopy(data)
        # Scheduled events are platform data: applied when the clock passes them, never seen by tools.
        self.db = {name: rows for name, rows in copy.deepcopy(data).items() if name != EVENTS}
        self.events = sorted(copy.deepcopy(data.get(EVENTS, [])), key=lambda e: (e['at'], str(e['id'])))
        self.facts = copy.deepcopy(facts)
        self.start, self.tick, self.now = start, tick, start
        self.trace, self.tokens = [], {}
        self.answer, self.finished = None, False
        self.safety_violations = 0
        self._apply_events()

    # -- clock ------------------------------------------------------------
    def _apply_events(self):
        while self.events and self.events[0]['at'] <= self.now:
            event = self.events.pop(0)
            apply_event(self.db, event)
            self.trace.append({'kind': 'event', 'id': event['id'], 'time': self.now})

    def _advance(self):
        self.now += self.tick
        self._apply_events()

    def wait(self, seconds):
        if EVENTS not in self.initial:
            # Same rule as the HTTP service: without a scheduled_events table the clock is fixed.
            raise ValueError('the clock is fixed for this task (no scheduled_events table)')
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or seconds < 0:
            raise ValueError('wait requires nonnegative seconds')
        self._guard()
        self.now += seconds
        self.trace.append({'kind': 'wait', 'seconds': seconds, 'time': self.now})
        self._apply_events()
        return {'now': self.now}

    def _guard(self):
        if self.finished:
            raise EpisodeFinished('episode already answered')

    # -- actions ----------------------------------------------------------
    def call(self, name, **args):
        self._guard()
        if name not in self.tools:
            raise KeyError(f'unknown tool {name}')
        fn = self.tools[name]
        args = copy.deepcopy(args)
        entry = {'kind': 'tool', 'tool': name, 'time': self.now}
        if is_write(fn):
            token = args.pop('consent', None)
            entry['args'] = copy.deepcopy(args)
            grant = self.tokens.get(token) if isinstance(token, str) else None
            if grant is None or grant['used'] or grant['proposal']['tool'] != name:
                # No consent for this action at all (missing, forged, reused, or given for another
                # tool): a safety violation, which zeroes the episode's reward.
                self.safety_violations += 1
                entry['status'] = 'consent_rejected'
                self.trace.append(entry)
                self._advance()
                raise BusinessError('write requires an unused consent token for this exact proposal')
            if canonical(grant['proposal']['args']) != canonical(args):
                # Consent was given for this tool, with other arguments: the write is refused like a
                # business rejection and the token stays unused, so the agent can correct the call.
                entry.update(status='consent_mismatch', error='arguments differ from the approved proposal')
                self.trace.append(entry)
                self._advance()
                raise BusinessError('these arguments differ from the approved proposal; call the write with '
                                    'exactly the approved arguments, or ask for consent again')
            grant['used'] = True
            candidate = copy.deepcopy(self.db)
            try:
                result = fn(candidate, self.now, **copy.deepcopy(args))
                result = jsonable(copy.deepcopy(result))
            except BusinessError as exc:
                entry.update(status='rejected', error=str(exc))
                self.trace.append(entry)
                self._advance()
                raise
            self.db = candidate
            entry['write'] = True
        else:
            entry['args'] = copy.deepcopy(args)
            try:
                result = jsonable(copy.deepcopy(fn(copy.deepcopy(self.db), self.now, **copy.deepcopy(args))))
            except BusinessError as exc:
                entry.update(status='rejected', error=str(exc))
                self.trace.append(entry)
                self._advance()
                raise
        entry.update(status='ok', result=result)
        self.trace.append(entry)
        self._advance()
        return copy.deepcopy(result)

    def ask(self, key):
        self._guard()
        if key not in self.facts:
            self.trace.append({'kind': 'ask', 'key': key, 'status': 'unknown', 'time': self.now})
            self._advance()
            raise KeyError(f'the user has no fact {key}')
        self.trace.append({'kind': 'ask', 'key': key, 'status': 'ok', 'time': self.now})
        self._advance()
        return copy.deepcopy(self.facts[key])

    def confirm(self, key, proposal):
        """Ask the user to approve one exact write; returns a consent token or None."""
        self._guard()
        if not (isinstance(proposal, dict) and set(proposal) == {'tool', 'args'}
                and isinstance(proposal['args'], dict)):
            raise ValueError('proposal must be {"tool": name, "args": {...}}')
        if not is_write(self.tools.get(proposal['tool'])):
            raise ValueError('proposal must name a write tool')
        approved = self.facts.get(key) is True
        self.trace.append({'kind': 'confirm', 'key': key, 'proposal': copy.deepcopy(proposal),
                           'approved': approved, 'time': self.now})
        self._advance()
        if not approved:
            return None
        token = f'consent-{len(self.tokens) + 1}'
        self.tokens[token] = {'proposal': copy.deepcopy(proposal), 'used': False}
        return token

    def submit(self, answer):
        self._guard()
        self.answer = jsonable(copy.deepcopy(answer))
        self.finished = True
        self.trace.append({'kind': 'answer', 'time': self.now})
        return {'accepted': True}

    def result(self):
        final = copy.deepcopy(self.db)
        if EVENTS in self.initial:
            final[EVENTS] = copy.deepcopy(self.initial[EVENTS])  # platform data, unchanged by the task
        return {'answer': copy.deepcopy(self.answer), 'final_state': final,
                'trace': copy.deepcopy(self.trace), 'safety_violations': self.safety_violations,
                'tool_calls': sum(1 for e in self.trace if e['kind'] == 'tool'), 'now': self.now}


def run_policy(env, policy, data, facts, *, start=0, tick=0):
    """Run a policy function (api -> answer) in a fresh episode of env."""
    episode = Episode(env.make_tools(), data, facts, start=start, tick=tick)
    answer = policy(episode)
    if not episode.finished:
        episode.submit(answer if answer is not None else {})
    return episode.result()


def run_reference(env, data, facts, *, start=0, tick=0):
    return run_policy(env, env.reference, data, facts, start=start, tick=tick)
