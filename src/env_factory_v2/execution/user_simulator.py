"""Hierarchical statechart with independent stage regions and private context."""
from __future__ import annotations

import copy

from env_factory_v2.contracts.contract import require, validate_value
from env_factory_v2.contracts.expressions import canonical, path, typed_equal


# Leaf names encode parents. Stage regions run independently; guards, rather
# than one global conversation cursor, synchronize their dependencies.
TRANSITIONS = {
    'waiting.blocked': {'dependencies_ready': 'waiting.ready'},
    'waiting.ready': {'dependencies_lost': 'waiting.blocked',
                      'disclose': 'resolved.disclosed', 'select': 'resolved.selected',
                      'refuse': 'resolved.refused', 'propose': 'consent.awaiting_confirmation'},
    'resolved': {'invalidate': 'waiting.ready'},
    'resolved.disclosed': {},
    'resolved.selected': {},
    'resolved.refused': {},
    'consent': {'invalidate': 'consent.invalidated'},
    'consent.awaiting_confirmation': {'approve': 'consent.authorized', 'decline': 'consent.declined'},
    'consent.authorized': {'revoke': 'consent.revoked',
                           'consume': 'consent.consumed'},
    'consent.declined': {'propose': 'consent.awaiting_confirmation'},
    'consent.invalidated': {'propose': 'consent.awaiting_confirmation'},
    'consent.revoked': {'propose': 'consent.awaiting_confirmation'},
    'consent.consumed': {},
}


def statechart_spec():
    return {'version': '1.0', 'root': 'user', 'regions': 'one per declared stage',
            'parents': ['waiting', 'resolved', 'consent'],
            'transitions': copy.deepcopy(TRANSITIONS),
            'guards': ['declared dependencies completed', 'actual observed menu',
                       'schema-valid exact proposal', 'private authorization preference'],
            'consumed_is_terminal': True}


class UserSimulator:
    def __init__(self, contract):
        self.contract = copy.deepcopy(contract)
        self.stages = {stage['id']: stage for stage in contract['user']['stages']}
        self.facts = copy.deepcopy(contract['user']['facts'])
        self.disclosed = {}
        self.completed = {}
        self.authorizations = {}
        self.states = {sid: 'waiting.blocked' if stage['depends_on'] else 'waiting.ready'
                       for sid, stage in self.stages.items()}
        self.history = []
        self.disclosure_owners = {}

    def _transition(self, stage_id, event):
        previous = self.states[stage_id]
        # Resolve child handlers first, then inherit common parent handlers.
        # A consumed leaf is terminal even if its parent defines an event.
        handlers = {} if previous == 'consent.consumed' else {
            **TRANSITIONS.get(previous.split('.')[0], {}), **TRANSITIONS[previous]}
        require(event in handlers, f'illegal user transition: {previous}/{event}')
        current = handlers[event]
        self.states[stage_id] = current
        self.history.append({'stage': stage_id, 'event': event, 'from': previous, 'to': current})

    def snapshot(self):
        """Inspect state without exposing private facts, choices or proposals."""
        return {'root': 'user', 'regions': copy.deepcopy(self.states), 'history': copy.deepcopy(self.history)}

    def _refresh(self):
        for sid, stage in self.stages.items():
            ready = set(stage['depends_on']) <= set(self.completed)
            if ready and self.states[sid] == 'waiting.blocked':
                self._transition(sid, 'dependencies_ready')
            elif not ready and self.states[sid] == 'waiting.ready':
                self._transition(sid, 'dependencies_lost')

    def _invalidate(self, stage_id):
        state = self.states[stage_id]
        if state == 'consent.consumed':
            return  # Never enable another write with a consumed approval stage.
        if state.startswith(('consent.', 'resolved.')):
            self._transition(stage_id, 'invalidate')
        self.authorizations.pop(stage_id, None)
        self.completed.pop(stage_id, None)
        for fact, owner in list(self.disclosure_owners.items()):
            if owner == stage_id:
                self.disclosed.pop(fact, None)
                del self.disclosure_owners[fact]

    def _invalidate_dependents(self, roots, *, exclude=()):
        affected = set(roots)
        while True:
            expanded = affected | {sid for sid, stage in self.stages.items()
                                   if set(stage['depends_on']) & affected}
            if expanded == affected:
                break
            affected = expanded
        for sid in self.stages:
            if sid in affected and sid not in exclude:
                self._invalidate(sid)
        self._refresh()

    def _disclose(self, stage_id, fact, value):
        if fact in self.disclosed and not typed_equal(self.disclosed[fact], value):
            previous = self.disclosure_owners[fact]
            # Private consent is bound to the whole user decision context, even
            # where an older contract did not declare a dependency edge.
            approvals = list(self.authorizations)
            descendants = [sid for sid, stage in self.stages.items() if previous in stage['depends_on']]
            if self.stages[previous]['kind'] == 'choice':
                descendants.append(previous)
            self._invalidate_dependents(descendants + approvals, exclude=[stage_id])
        self.disclosed[fact] = copy.deepcopy(value)
        self.disclosure_owners[fact] = stage_id

    def respond(self, stage_id, *, observations, tools, proposal=None, source_id=None):
        require(stage_id in self.stages, 'unknown user stage')
        stage = self.stages[stage_id]
        self._refresh()
        require(set(stage['depends_on']) <= set(self.completed), 'user stage prerequisites unmet')
        kind = stage['kind']
        state = self.states[stage_id]
        if kind == 'authorization':
            require(state != 'consent.consumed', 'authorization already consumed; stage already completed')
        else:
            require(state == 'waiting.ready', 'stage already completed')
        if kind in {'fact', 'correction'}:
            value = copy.deepcopy(self.facts[stage['fact']])
            result = {'value': value, 'refused': False}
            self._transition(stage_id, 'disclose')
            self._disclose(stage_id, stage.get('as_fact', stage['fact']), value)
        elif kind == 'choice':
            require(source_id in observations and observations[source_id]['tool'] == stage['source_tool'],
                    'choice must use an actual observed menu')
            options = path(observations[source_id]['result'], stage['source_path'])
            require(isinstance(options, list), 'menu must be a list')
            values = [path(option, stage['value_field']) for option in options]
            preferred = self.disclosed.get(stage['fact'], self.facts[stage['fact']])
            match = next((i for i, v in enumerate(values) if typed_equal(v, preferred)), None)
            found = match is not None
            value = values[match] if found else None
            if not found and stage['fallback'] == 'first_available' and values:
                value, found = values[0], True
            result = {'value': copy.deepcopy(value), 'refused': not found,
                      'options': copy.deepcopy(values)}
            self._transition(stage_id, 'select' if found else 'refuse')
            self._disclose(stage_id, stage.get('as_fact', stage['fact']), value)
        else:
            require(isinstance(proposal, dict) and set(proposal) == {'tool', 'arguments'}, 'exact proposal required')
            require(proposal['tool'] == stage['tool'], 'authorization tool mismatch')
            validate_value(proposal['arguments'], tools[stage['tool']]['parameters'], 'proposal')
            if state == 'consent.authorized':
                require(self.authorizations[stage_id] != canonical(proposal), 'stage already completed')
                self._invalidate_dependents([stage_id])
            self._transition(stage_id, 'propose')
            # Authorization decisions are never manufactured by a model.
            approved = self.facts.get(f'approve_{stage_id}')
            require(type(approved) is bool, 'authorization preference must be boolean')
            result = {'approved': approved, 'proposal': copy.deepcopy(proposal)}
            if approved:
                self.authorizations[stage_id] = canonical(proposal)
            self._transition(stage_id, 'approve' if approved else 'decline')
        self.completed[stage_id] = copy.deepcopy(result)
        self._refresh()
        return result

    def authorize(self, stage_id, tool, arguments):
        require(stage_id in self.states, 'unknown authorization')
        if self.states[stage_id] == 'consent.authorized' and self.authorizations.get(stage_id) != canonical({'tool': tool, 'arguments': arguments}):
            self._invalidate_dependents([stage_id])
        require(self.authorizations.get(stage_id) == canonical({'tool': tool, 'arguments': arguments}),
                'authorization missing, revoked, consumed or parameters changed')

    def consume(self, stage_id):
        require(stage_id in self.states and self.states[stage_id] == 'consent.authorized', 'authorization not active')
        self._transition(stage_id, 'consume')
        self.authorizations.pop(stage_id, None)

    def revoke(self, stage_id):
        require(stage_id in self.stages and self.stages[stage_id]['kind'] == 'authorization', 'unknown authorization')
        require(self.states[stage_id] == 'consent.authorized', 'authorization not active or already consumed/revoked')
        self._transition(stage_id, 'revoke')
        self.authorizations.pop(stage_id, None)
        self.completed.pop(stage_id, None)
        dependents = [sid for sid, stage in self.stages.items() if stage_id in stage['depends_on']]
        self._invalidate_dependents(dependents)


def validate_language_event(event, simulator):
    """A language interpreter may select a request, never set user facts."""
    require(isinstance(event, dict) and set(event) == {'stage_id', 'question'}, 'language event cannot mutate user state')
    require(event['stage_id'] in simulator.stages and isinstance(event['question'], str)
            and event['question'].strip(), 'invalid language event')
    return event
