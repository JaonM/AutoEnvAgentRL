"""Data-only tools for one author session; no oracle, shell or filesystem access."""
from __future__ import annotations

import copy
import json

from .contract import ContractError, compile_contract, require
from .expressions import canonical
from .repair import apply_edits
from .verification import preflight_recipes


SECTIONS = frozenset({'version', 'id', 'category', 'task', 'rules', 'tables', 'tools',
    'user', 'answers', 'explanations', 'invariants', 'goals', 'success', 'scenarios',
    'probes', 'limits', 'time', 'events', 'demand'})


def decode_json(text):
    require(isinstance(text, str) and len(text.encode()) <= 4 * 1024 * 1024,
            'author JSON size limit')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'duplicate author JSON key: {key}')
            result[key] = value
        return result
    try:
        value = json.loads(text, object_pairs_hook=unique)
        canonical(value)
        return value
    except (ValueError, TypeError, RecursionError) as exc:
        raise ContractError(f'author JSON invalid: {exc}') from exc


class AuthorWorkbench:
    def __init__(self, *, source=None, max_operations=128):
        require(type(max_operations) is int and 1 <= max_operations <= 512,
                'invalid author operation budget')
        self.source = {}
        self._replace(source if source is not None else {})
        self.max_operations = max_operations
        self.operations = 0

    def call(self, tool, arguments):
        require(self.operations < self.max_operations, 'author operation budget exceeded')
        self.operations += 1
        try:
            require(isinstance(arguments, dict), 'author arguments must be an object')
            if tool == 'put_section':
                require(set(arguments) == {'section', 'value_json'} and
                        isinstance(arguments['section'], str) and
                        arguments['section'] in SECTIONS, 'unknown author section')
                candidate = {**self.source, arguments['section']: decode_json(arguments['value_json'])}
                self._replace(candidate)
                return {'passed': True, 'sections': sorted(self.source)}
            if tool == 'edit_contract':
                require(set(arguments) == {'edits'}, 'author edits required')
                self._replace(apply_edits(self.source, arguments['edits']))
                return {'passed': True, 'sections': sorted(self.source)}
            require(tool == 'check_contract' and not arguments, 'unknown author tool')
            compiled = compile_contract(copy.deepcopy(self.source))
            recipes = preflight_recipes(compiled)
            return {'passed': recipes['passed'], 'phase': 'recipe_preflight',
                    'checks': recipes['checks'],
                    'scope': 'authored_recipe_validation_not_business_acceptance'}
        except ContractError as exc:
            return {'passed': False, 'error': str(exc)}

    def _replace(self, candidate):
        require(isinstance(candidate, dict) and set(candidate) <= SECTIONS,
                'unknown author contract section')
        require(len(canonical(candidate).encode()) <= 4 * 1024 * 1024,
                'author contract size limit')
        self.source = copy.deepcopy(candidate)

    def snapshot(self):
        return copy.deepcopy(self.source)
