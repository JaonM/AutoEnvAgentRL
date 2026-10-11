"""Data-only tools for one author session; no oracle, shell or filesystem access."""
from __future__ import annotations

import copy
import hashlib
import json
import re

from env_factory_v2.contracts.contract import ContractError, require
from env_factory_v2.contracts.business_definition import compile_source
from env_factory_v2.contracts.expressions import canonical
from env_factory_v2.authoring.repair import apply_edits
from env_factory_v2.authoring.repair_evidence import repair_index
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
    def __init__(self, *, source=None, max_operations=128, request=None, context=None):
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
        self.checked_identity = None
        self.submitted = False
        self.check_cache = {}
        self.recipe_cache = {}
        self.context = copy.deepcopy(context or {'request': self.request, 'feedback': {},
                                                'repair': {'mode': 'create', 'issues': []}})
        self.initial_source = self.snapshot()
        self.repair_claims = {}

    def call(self, tool, arguments):
        require(self.operations < self.max_operations, 'author operation budget exceeded')
        self.operations += 1
        try:
            require(isinstance(arguments, dict), 'author arguments must be an object')
            require(not self.submitted, 'author draft already submitted')
            identity = hashlib.sha256(canonical(self.source).encode()).hexdigest()
            if tool == 'read_context':
                return read_draft(self.context, arguments)
            if tool == 'address_repair':
                require(set(arguments) == {'issue_id', 'paths', 'explanation'}, 'repair evidence fields required')
                require(arguments['issue_id'] in {i['id'] for i in self.context['repair']['issues']}, 'unknown repair issue')
                require(isinstance(arguments['explanation'], str) and 20 <= len(arguments['explanation']) <= 4000,
                        'explain the proposed fix and its original requirement')
                require(isinstance(arguments['paths'], list) and 1 <= len(arguments['paths']) <= 16,
                        'repair requires changed source paths')
                for path in arguments['paths']:
                    require(isinstance(path, str) and path.startswith('/'), 'repair needs specific JSON Pointers')
                    def at(source):
                        try:
                            first = read_draft(source, {'path': path, 'offset': 0})
                            # Comparing the whole value via bounded slices avoids
                            # treating a common first8192 chars as equality.
                            text = first['json_fragment']
                            while first['next_offset'] is not None:
                                first = read_draft(source, {'path': path, 'offset': first['next_offset']})
                                text += first['json_fragment']
                            return text
                        except ContractError:
                            return None
                    require(at(self.initial_source) != at(self.source), 'repair path has not changed')
                self.repair_claims[arguments['issue_id']] = {**copy.deepcopy(arguments), 'source_identity': identity}
                return {'passed': True, 'scope': 'proposed_fix_not_independently_verified', 'issue_id': arguments['issue_id']}
            if tool == 'inspect_contract':
                require(not arguments, 'inspect takes no arguments')
                return {'passed': True, 'sections': {key: {'type': type(value).__name__,
                    'characters': len(canonical(value)), 'keys': sorted(value) if isinstance(value, dict) else []}
                    for key, value in self.source.items()},
                    'checked': self.checked_identity == identity and bool(self.last_check and self.last_check['passed']),
                    'repair': repair_index(self.context),
                    'pending_issues': [i['id'] for i in self.context['repair']['issues']
                                       if self.repair_claims.get(i['id'], {}).get('source_identity') != identity]}
            if tool == 'read_check':
                require(self.last_check is not None, 'no current check')
                return read_draft(self.last_check, arguments)
            if tool == 'submit_contract':
                require(not arguments, 'submit takes no arguments')
                require(self.checked_identity == identity and self.last_check is not None
                        and self.last_check['passed'] is True, 'submit requires a successful check of the current draft')
                if self.context['repair']['mode'] == 'repair':
                    require(self.source != self.initial_source, 'rejected draft cannot be resubmitted unchanged')
                    require(all(self.repair_claims.get(i['id'], {}).get('source_identity') == identity
                                for i in self.context['repair']['issues']), 'unaddressed independent repair findings')
                self.submitted = True
                return {'passed': True, 'submitted': True,
                        'proposed_repairs': copy.deepcopy(self.repair_claims),
                        'scope': 'authored_recipe_validation_not_business_acceptance'}
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
            if identity in self.check_cache:
                from env_factory_v2.requirements.obligation_index import reference_changes
                report, index = self.check_cache[identity]
                self.last_check = copy.deepcopy(report)
                self.last_check['reference_changes'] = reference_changes(self.last_index, index)
                self.last_index = copy.deepcopy(index)
                self.checked_identity = identity
                return {**author_feedback(self.last_check), 'reused': True}
            compiled, compilation = compile_source(copy.deepcopy(self.source))
            from env_factory_v2.authoring.author_constraints import check_author_inputs
            check_author_inputs(compiled, self.request)
            from env_factory_v2.requirements.obligation_index import obligation_index, reference_changes
            index = obligation_index(compiled)
            changes = reference_changes(self.last_index, index) if self.last_index is not None else None
            self.last_index = index
            recipes = preflight_recipes(compiled, cache=self.recipe_cache)
            self.last_check = {'passed': recipes['passed'], 'phase': 'recipe_preflight',
                               'checks': recipes['checks'],
                               'execution_reuse': recipes['execution_reuse'],
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
            self.checked_identity = identity
            self.check_cache[identity] = (copy.deepcopy(self.last_check), copy.deepcopy(index))
            if len(self.check_cache) > 8:
                del self.check_cache[next(iter(self.check_cache))]
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
        self.checked_identity = None

    def snapshot(self):
        return copy.deepcopy(self.source)
