"""Deterministic user facts, choices and exact-parameter authorizations."""
from __future__ import annotations

import copy

from .contract import require, validate_value
from .expressions import canonical, path, typed_equal


class UserSimulator:
    def __init__(self, contract):
        self.contract = copy.deepcopy(contract)
        self.stages = {stage['id']: stage for stage in contract['user']['stages']}
        self.facts = copy.deepcopy(contract['user']['facts'])
        self.disclosed = {}
        self.completed = {}
        self.authorizations = {}

    def respond(self, stage_id, *, observations, tools, proposal=None, source_id=None):
        require(stage_id in self.stages, 'unknown user stage')
        stage = self.stages[stage_id]
        require(set(stage['depends_on']) <= set(self.completed), 'user stage prerequisites unmet')
        require(stage_id not in self.completed, 'stage already completed')
        kind = stage['kind']
        if kind in {'fact', 'correction'}:
            value = copy.deepcopy(self.facts[stage['fact']])
            result = {'value': value, 'refused': False}
            self.disclosed[stage.get('as_fact', stage['fact'])] = value
        elif kind == 'choice':
            require(source_id in observations and observations[source_id]['tool'] == stage['source_tool'],
                    'choice must use an actual observed menu')
            options = path(observations[source_id]['result'], stage['source_path'])
            require(isinstance(options, list), 'menu must be a list')
            values = [path(option, stage['value_field']) for option in options]
            preferred = self.facts[stage['fact']]
            match = next((i for i, v in enumerate(values) if typed_equal(v, preferred)), None)
            found = match is not None
            value = values[match] if found else None
            if not found and stage['fallback'] == 'first_available' and values:
                value, found = values[0], True
            result = {'value': copy.deepcopy(value), 'refused': not found,
                      'options': copy.deepcopy(values)}
            self.disclosed[stage.get('as_fact', stage['fact'])] = copy.deepcopy(value)
        else:
            require(isinstance(proposal, dict) and set(proposal) == {'tool', 'arguments'}, 'exact proposal required')
            require(proposal['tool'] == stage['tool'], 'authorization tool mismatch')
            validate_value(proposal['arguments'], tools[stage['tool']]['parameters'], 'proposal')
            # Authorization decisions are never manufactured by a model.
            approved = self.facts.get(f'approve_{stage_id}', True)
            require(type(approved) is bool, 'authorization preference must be boolean')
            result = {'approved': approved, 'proposal': copy.deepcopy(proposal)}
            if approved:
                self.authorizations[stage_id] = canonical(proposal)
        self.completed[stage_id] = copy.deepcopy(result)
        return result

    def authorize(self, stage_id, tool, arguments):
        require(self.authorizations.get(stage_id) == canonical({'tool': tool, 'arguments': arguments}),
                'authorization missing, revoked, consumed or parameters changed')

    def consume(self, stage_id):
        self.authorizations.pop(stage_id, None)

    def revoke(self, stage_id):
        require(stage_id in self.stages and self.stages[stage_id]['kind'] == 'authorization', 'unknown authorization')
        self.authorizations.pop(stage_id, None)


def validate_language_event(event, simulator):
    """A language interpreter may select a request, never set user facts."""
    require(isinstance(event, dict) and set(event) == {'stage_id', 'question'}, 'language event cannot mutate user state')
    require(event['stage_id'] in simulator.stages and isinstance(event['question'], str)
            and event['question'].strip(), 'invalid language event')
    return event
