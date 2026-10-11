"""V3 spike runtime: Python business tools over plain tables, consent-bound writes.

An environment module (env.py) defines:
    make_tools() -> dict[str, callable]
        read tool:   fn(db, now, **args) -> JSON value      (db is a private copy)
        write tool:  decorated with @write, fn(db, now, **args) -> JSON value
                     mutates db in place; raise BusinessError to reject (atomic rollback)
    reference(api) -> dict
        a deterministic reference agent that only uses api.call / api.ask / api.confirm

Writes require a consent token from api.confirm(fact_key, proposal) where
proposal == {"tool": <write tool name>, "args": <exact args without consent>}.
A token is single-use and bound to that exact proposal.
"""
from __future__ import annotations

import copy
import json

NOW = 10  # frozen virtual seconds for the spike (documented simplification)


class BusinessError(Exception):
    """A business rule rejected the action. Writes roll back atomically."""


def write(fn):
    fn.__v3_write__ = True
    return fn


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def jsonable(value):
    json.dumps(value, allow_nan=False)
    return value


class Episode:
    """One isolated run of a world: data + private user facts + frozen clock."""

    BusinessError = BusinessError

    def __init__(self, tools, data, facts, now=NOW):
        self.db = copy.deepcopy(data)
        self.facts = copy.deepcopy(facts)
        self.now = now
        self.tools = tools
        self.trace = []
        self.tokens = {}

    def call(self, name, **args):
        if name not in self.tools:
            raise KeyError(f'unknown tool {name}')
        fn = self.tools[name]
        args = copy.deepcopy(args)
        entry = {'kind': 'tool', 'tool': name, 'args': copy.deepcopy(args)}
        if getattr(fn, '__v3_write__', False):
            token = args.pop('consent', None)
            entry['args'] = copy.deepcopy(args)
            grant = self.tokens.get(token) if isinstance(token, str) else None
            if (grant is None or grant['used']
                    or canonical(grant['proposal']) != canonical({'tool': name, 'args': args})):
                entry['status'] = 'consent_rejected'
                self.trace.append(entry)
                raise BusinessError('write requires an unused consent token for this exact proposal')
            grant['used'] = True
            candidate = copy.deepcopy(self.db)
            try:
                result = fn(candidate, self.now, **copy.deepcopy(args))
            except BusinessError as exc:
                entry.update(status='rejected', error=str(exc))
                self.trace.append(entry)
                raise
            self.db = candidate
            entry['write'] = True
        else:
            view = copy.deepcopy(self.db)
            try:
                result = fn(view, self.now, **copy.deepcopy(args))
            except BusinessError as exc:
                entry.update(status='rejected', error=str(exc))
                self.trace.append(entry)
                raise
        result = jsonable(copy.deepcopy(result))
        entry.update(status='ok', result=result)
        self.trace.append(entry)
        return copy.deepcopy(result)

    def ask(self, key):
        if key not in self.facts:
            raise KeyError(f'user has no fact {key}')
        self.trace.append({'kind': 'ask', 'key': key})
        return copy.deepcopy(self.facts[key])

    def confirm(self, key, proposal):
        """Ask the user to approve an exact proposal; returns a token or None."""
        if not (isinstance(proposal, dict) and set(proposal) == {'tool', 'args'}):
            raise ValueError('proposal must be {"tool": name, "args": {...}}')
        tool = self.tools.get(proposal['tool'])
        if tool is None or not getattr(tool, '__v3_write__', False):
            raise ValueError('proposal must name a write tool')
        approved = self.facts.get(key) is True
        self.trace.append({'kind': 'confirm', 'key': key, 'proposal': copy.deepcopy(proposal),
                           'approved': approved})
        if not approved:
            return None
        token = f'consent-{len(self.tokens) + 1}'
        self.tokens[token] = {'proposal': copy.deepcopy(proposal), 'used': False}
        return token


def run_reference(env, data, facts, now=NOW):
    episode = Episode(env.make_tools(), data, facts, now)
    answer = env.reference(episode)
    return {'answer': jsonable(copy.deepcopy(answer)), 'final_state': episode.db, 'trace': episode.trace}
