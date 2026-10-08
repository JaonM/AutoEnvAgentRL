"""One authored definition for answer meaning, schema and executable value."""
from __future__ import annotations

import copy

from env_factory.sandbox_runtime import validate_json_schema, SandboxError

KINDS = {'observed', 'derived', 'explanation'}


def field_meanings(fields):
    return {name: {'kind': field['kind'], 'meaning': field['schema']['description'],
                   **({'context_fields': field['context_fields']} if field.get('context_fields') else {})}
            for name, field in fields.items()}


def expand_answer_fields(source, *, required=False):
    fields = source.get('answer_fields')
    if fields is None:
        if required:
            raise ValueError('ANSWER_FIELDS_REQUIRED: define answer_fields once; derive answer schema and outcome targets from them')
        return source
    if (not isinstance(fields, dict) or not 1 <= len(fields) <= 16
            or any(not isinstance(name, str) or not name for name in fields)):
        raise ValueError('ANSWER_FIELDS_INVALID: require 1..16 named fields')
    if 'answer_contract' in source or 'answer_field_semantics' in source:
        raise ValueError('ANSWER_FIELDS_DUPLICATE: answer_contract and field semantics are compiler-owned')
    if 'business_tools' not in source:
        raise ValueError('ANSWER_FIELDS_INVALID: shared fields require compact business_tools/outcomes authoring')
    tools = {tool['name'] for tool in source['business_tools']}
    expressions, public = {}, {}
    for name, field in fields.items():
        if not isinstance(field, dict) or field.get('kind') not in KINDS:
            raise ValueError('ANSWER_FIELDS_INVALID: field kind must be observed, derived or explanation')
        kind = field['kind']
        expected = {'kind', 'schema'} | ({'observation', 'missing'} if kind == 'observed'
            else {'expression'} if kind == 'derived' else set())
        optional = {'context_fields'} if kind == 'explanation' else {'tolerance', 'array_order'}
        if set(field) - expected - optional or not expected <= set(field):
            raise ValueError('ANSWER_FIELDS_INVALID: invalid definition for ' + name)
        schema = field['schema']
        if not isinstance(schema, dict) or not isinstance(schema.get('description'), str) or not schema['description'].strip():
            raise ValueError('ANSWER_FIELDS_INVALID: every field needs a public business meaning: ' + name)
        public[name] = {'kind': kind, 'meaning': schema['description']}
        if kind == 'observed':
            ref = field['observation']
            if (not isinstance(ref, dict) or set(ref) != {'tool', 'path'}
                    or not isinstance(ref['tool'], str) or ref['tool'] not in tools
                    or not isinstance(ref['path'], str) or not ref['path'].startswith('$.')):
                raise ValueError('ANSWER_FIELDS_INVALID: observation requires declared tool and response JSON path')
            try:
                validate_json_schema(schema, field['missing'])
            except SandboxError as exc:
                raise ValueError('ANSWER_FIELDS_INVALID: missing value violates field schema: ' + name) from exc
            expressions[name] = {'observed_value': {
                'value': {'tool_result': {'name': ref['tool'], 'path': ref['path']}},
                'missing': {'literal': copy.deepcopy(field['missing'])}}}
            public[name].update(observation=copy.deepcopy(ref), missing=copy.deepcopy(field['missing']),
                rule='Copy the observed value even when several records exist; use missing only when this path has no value. Uniqueness and eligibility are separate derived fields.')
        elif kind == 'derived':
            expressions[name] = copy.deepcopy(field['expression'])
        else:
            if schema.get('type') != 'string' or 'enum' in schema:
                raise ValueError('ANSWER_FIELDS_INVALID: explanation must be an open string without deterministic comparison options')
            context = field.get('context_fields', [])
            if (not isinstance(context, list) or any(not isinstance(key, str)
                    or not isinstance(fields.get(key), dict)
                    or fields[key].get('kind') not in {'observed', 'derived'} for key in context)
                    or len(set(context)) != len(context)):
                raise ValueError('ANSWER_FIELDS_INVALID: explanation context must uniquely select observed or derived fields')
            if context:
                public[name]['context_fields'] = list(context)

    remaining_nodes = 20000
    def resolve(value, stack=(), depth=0):
        nonlocal remaining_nodes
        remaining_nodes -= 1
        if depth > 40 or remaining_nodes < 0:
            raise ValueError('ANSWER_FIELDS_INVALID: field expression too deep')
        if isinstance(value, dict):
            if set(value) == {'literal'}:
                return copy.deepcopy(value)
            if set(value) == {'answer_field'}:
                name = value['answer_field']
                if not isinstance(name, str) or name not in expressions or name in stack:
                    raise ValueError('ANSWER_FIELDS_INVALID: unknown, explanatory or cyclic field reference')
                return resolve(expressions[name], (*stack, name), depth + 1)
            return {key: resolve(child, stack, depth + 1) for key, child in value.items()}
        if isinstance(value, list):
            return [resolve(child, stack, depth + 1) for child in value]
        return copy.deepcopy(value)

    result = copy.deepcopy(source)
    for outcome in result.get('outcomes', []):
        if 'fields' in outcome:
            names = outcome['fields']
            if (set(outcome) != {'id', 'rubric', 'fields'} or not isinstance(names, list) or not names
                    or any(not isinstance(name, str) or name not in expressions for name in names)
                    or len(set(names)) != len(names)):
                raise ValueError('ANSWER_FIELDS_INVALID: outcome fields must select deterministic field IDs')
            outcome.pop('fields')
            outcome['rule'] = {'source': 'final_agent_response', 'path': '$', 'operator': 'value_targets',
                'expected': {'answer_format': 'json_object', 'targets': [
                    {'key': name, 'expression': resolve(expressions[name], (name,)),
                     **{key: copy.deepcopy(fields[name][key]) for key in ('tolerance', 'array_order') if key in fields[name]}}
                    for name in names]}}
        elif 'semantic' in outcome:
            if any(fields.get(name, {}).get('kind') != 'explanation' for name in outcome['semantic'].get('fields', [])):
                raise ValueError('ANSWER_FIELDS_INVALID: semantic outcomes may own only explanation fields')
            if 'context_fields' in outcome['semantic']:
                raise ValueError('ANSWER_FIELDS_DUPLICATE: explanation context is derived from answer_fields')
            context = list(dict.fromkeys(key for name in outcome['semantic'].get('fields', [])
                for key in fields[name].get('context_fields', [])))
            if context:
                outcome['semantic']['context_fields'] = context
        elif outcome.get('rule', {}).get('source') == 'final_agent_response':
            raise ValueError('ANSWER_FIELDS_DUPLICATE: select field IDs instead of restating answer reward expressions')
    result['answer_contract'] = {'format': 'json_object', 'schema': {'type': 'object',
        'properties': {name: copy.deepcopy(field['schema']) for name, field in fields.items()},
        'required': list(fields), 'additionalProperties': False}}
    result['answer_field_semantics'] = public
    return result
