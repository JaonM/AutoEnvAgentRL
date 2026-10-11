"""Data-only tools for one author session; no oracle, shell or filesystem access."""
from __future__ import annotations

import copy
import json
import re

from env_factory_v2.contracts.contract import ContractError, require
from env_factory_v2.contracts.business_definition import compile_source
from env_factory_v2.contracts.expressions import canonical
from env_factory_v2.authoring.repair import apply_edits
from env_factory_v2.validation.verification import preflight_recipes


SECTIONS = frozenset({'version', 'id', 'category', 'task', 'rules', 'tables', 'tools',
    'user', 'answers', 'explanations', 'invariants', 'goals', 'success', 'scenarios',
    'probes', 'limits', 'time', 'events', 'demand', 'environment'})
DEFINITION_SECTIONS = frozenset({'environment', 'task'})
LEGACY_SECTIONS = SECTIONS - {'environment'}


def read_draft(source, arguments):
    """Bounded JSON text slices from the draft only; paths never address files."""
    require(set(arguments) == {'path', 'offset'}, 'read requires path and offset')
    pointer, offset = arguments['path'], arguments['offset']
    require(isinstance(pointer, str) and len(pointer) <= 2048
            and (pointer == '' or pointer.startswith('/'))
            and not re.search(r'~(?![01])', pointer), 'invalid read JSON pointer')
    require(type(offset) is int and offset >= 0, 'invalid read offset')
    node = source
    for token in ([] if pointer == '' else pointer[1:].split('/')):
        key = token.replace('~1', '/').replace('~0', '~')
        if isinstance(node, list):
            require(key.isdecimal() and str(int(key)) == key and int(key) < len(node), 'read array index invalid')
            node = node[int(key)]
        else:
            require(isinstance(node, dict) and key in node, 'read path does not exist')
            node = node[key]
    text = canonical(node)
    require(offset <= len(text), 'read offset exceeds value')
    end = min(offset + 8192, len(text))
    return {'passed': True, 'path': pointer, 'offset': offset, 'total_chars': len(text),
            'json_fragment': text[offset:end], 'next_offset': end if end < len(text) else None,
            'complete': offset == 0 and end == len(text),
            'sections': sorted(source), 'scope': 'draft_data_only_not_validation'}


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
        # Fresh production drafts have one authored representation. A supplied
        # flat draft is an explicit historical import, never an automatic fallback.
        require(source is None or isinstance(source, dict), 'author draft must be an object')
        self.sections = (DEFINITION_SECTIONS if not source or 'environment' in source
                         or isinstance(source.get('task'), dict)
                         else LEGACY_SECTIONS)
        self._replace(source if source is not None else {})
        self.max_operations = max_operations
        self.operations = 0
        self.last_check = None
        self.request = copy.deepcopy(request or {})
        self.last_index = None

    def call(self, tool, arguments):
        require(self.operations < self.max_operations, 'author operation budget exceeded')
        self.operations += 1
        try:
            require(isinstance(arguments, dict), 'author arguments must be an object')
            if tool == 'read_contract':
                return read_draft(self.source, arguments)
            if tool == 'put_section':
                require(set(arguments) == {'section', 'value_json'} and
                        isinstance(arguments['section'], str) and
                        arguments['section'] in self.sections, 'unknown author section for this draft format')
                candidate = {**self.source, arguments['section']: decode_json(arguments['value_json'])}
                self._replace(candidate)
                return {'passed': True, 'sections': sorted(self.source)}
            if tool == 'edit_contract':
                require(set(arguments) == {'edits'}, 'author edits required')
                self._replace(apply_edits(self.source, arguments['edits']))
                return {'passed': True, 'sections': sorted(self.source)}
            require(tool == 'check_contract' and not arguments, 'unknown author tool')
            self.last_check = None
            compiled, compilation = compile_source(copy.deepcopy(self.source))
            from env_factory_v2.authoring.author_constraints import check_author_inputs
            check_author_inputs(compiled, self.request)
            from env_factory_v2.requirements.obligation_index import obligation_index, reference_changes
            index = obligation_index(compiled)
            changes = reference_changes(self.last_index, index) if self.last_index is not None else None
            self.last_index = index
            recipes = preflight_recipes(compiled)
            self.last_check = {'passed': recipes['passed'], 'phase': 'recipe_preflight',
                               'checks': recipes['checks'],
                               'scope': 'authored_recipe_validation_not_business_acceptance'}
            if changes is not None:
                self.last_check['reference_changes'] = changes
            from env_factory_v2.validation.input_diagnostics import input_diagnostics
            self.last_check['input_diagnostics'] = input_diagnostics(compiled)
            from env_factory_v2.validation.process_diagnostics import process_diagnostics
            self.last_check['process_diagnostics'] = process_diagnostics(compiled)
            from env_factory_v2.validation.temporal_diagnostics import temporal_diagnostics
            self.last_check['temporal_diagnostics'] = temporal_diagnostics(compiled)
            if compilation is not None:
                self.last_check['compilation'] = compilation
            return author_feedback(self.last_check)
        except ContractError as exc:
            return {'passed': False, 'error': str(exc)}

    def _replace(self, candidate):
        require(isinstance(candidate, dict) and set(candidate) <= self.sections,
                'unknown author contract section for this draft format')
        if self.sections == DEFINITION_SECTIONS:
            require(all(isinstance(value, dict) for value in candidate.values()),
                    'authored environment and task must be documents, not flat contract fields')
        require(len(canonical(candidate).encode()) <= 4 * 1024 * 1024,
                'author contract size limit')
        self.source = copy.deepcopy(candidate)
        self.last_check = None

    def snapshot(self):
        return copy.deepcopy(self.source)
