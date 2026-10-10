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


def author_feedback(report):
    """Shorten passing cases only; retain every failing case and its evidence."""
    result = copy.deepcopy(report)
    result['checks'] = [
        {key: check[key] for key in ('id', 'kind', 'passed', 'error') if key in check}
        if check['passed'] else copy.deepcopy(check)
        for check in report['checks']]
    result['check_count'] = len(report['checks'])
    result['failed_count'] = sum(not check['passed'] for check in report['checks'])
    return result


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
    def __init__(self, *, source=None, max_operations=128, request=None):
        require(type(max_operations) is int and 1 <= max_operations <= 512,
                'invalid author operation budget')
        self.source = {}
        self._replace(source if source is not None else {})
        self.max_operations = max_operations
        self.operations = 0
        self.last_check = None
        self.request = copy.deepcopy(request or {})

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
            self.last_check = None
            compiled = compile_contract(copy.deepcopy(self.source))
            from .author_constraints import check_author_inputs
            check_author_inputs(compiled, self.request)
            recipes = preflight_recipes(compiled)
            self.last_check = {'passed': recipes['passed'], 'phase': 'recipe_preflight',
                               'checks': recipes['checks'],
                               'scope': 'authored_recipe_validation_not_business_acceptance'}
            return author_feedback(self.last_check)
        except ContractError as exc:
            return {'passed': False, 'error': str(exc)}

    def _replace(self, candidate):
        require(isinstance(candidate, dict) and set(candidate) <= SECTIONS,
                'unknown author contract section')
        require(len(canonical(candidate).encode()) <= 4 * 1024 * 1024,
                'author contract size limit')
        self.source = copy.deepcopy(candidate)
        self.last_check = None

    def snapshot(self):
        return copy.deepcopy(self.source)
