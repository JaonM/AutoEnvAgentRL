"""Answer schema shared by the environment, the independent oracle and the reward.

answer_schema.json:
    {"fields": {"<name>": {"description": "...", "type": "integer", "nullable": false,
                           "items": {...}, "properties": {...}, "order": "exact" | "any"}}}

Supported types: string, integer, number, boolean, array, object. Arrays declare "items";
objects may declare "properties" (all listed properties are then required, no extras).
"order": "any" on an array compares it as a multiset. Strings may declare "enum".
A top-level field or object property with "graded": false (user-facing prose such as an
explanation) is shape-checked but excluded from the reward and the differential: two correct
implementations never agree on wording. Outcomes must be expressed by graded fields (ids, numbers,
booleans, enums).
"""
from __future__ import annotations

import json

TYPES = {'string', 'integer', 'number', 'boolean', 'array', 'object'}
PROSE_WORDS = ('reason', 'explanation', 'message', 'summary', 'note', 'comment', 'text', 'description', 'detail')


class SchemaError(ValueError):
    pass


def check_schema(schema):
    problems = []
    if not isinstance(schema, dict) or set(schema) != {'fields'} or not isinstance(schema['fields'], dict):
        return ['answer schema must be {"fields": {name: spec}}']
    if not schema['fields']:
        problems.append('answer schema needs at least one field')

    def spec_problems(path, spec, top, named=None):
        named = top if named is None else named
        if not isinstance(spec, dict) or spec.get('type') not in TYPES:
            return [f'{path}: type must be one of {sorted(TYPES)}']
        out = []
        if top and not (isinstance(spec.get('description'), str) and spec['description'].strip()):
            out.append(f'{path}: description required')
        allowed = {'type', 'description', 'nullable', 'items', 'properties', 'order', 'enum'} | \
            ({'graded'} if named else set())
        extra = set(spec) - allowed
        if extra:
            out.append(f'{path}: unknown keys {sorted(extra)}')
        if 'enum' in spec and (spec['type'] != 'string' or not isinstance(spec['enum'], list) or not spec['enum']
                               or not all(isinstance(v, str) for v in spec['enum'])):
            out.append(f'{path}: enum must be a nonempty list of strings on a string field')
        if named and 'graded' in spec and spec['graded'] is not False:
            out.append(f'{path}: graded may only be set to false')
        name = path.rsplit('.', 1)[-1].lower()
        if (named and spec['type'] == 'string' and 'enum' not in spec and spec.get('graded', True)
                and any(word in name for word in PROSE_WORDS)):
            out.append(f'{path}: looks like free text; give it an enum or mark it "graded": false')
        if spec['type'] == 'array':
            if 'items' not in spec:
                out.append(f'{path}: array needs items')
            else:
                out += spec_problems(f'{path}[]', spec['items'], False)
            if spec.get('order', 'exact') not in ('exact', 'any'):
                out.append(f'{path}: order must be exact or any')
        elif 'order' in spec or 'items' in spec:
            out.append(f'{path}: order/items only apply to arrays')
        if spec['type'] == 'object':
            for name, child in (spec.get('properties') or {}).items():
                out += spec_problems(f'{path}.{name}', child, False, named=True)
        elif 'properties' in spec:
            out.append(f'{path}: properties only apply to objects')
        return out

    for name, spec in schema['fields'].items():
        problems += spec_problems(name, spec, True)
    # The rubric reply schema ({"reply": ungraded}) is the one schema with nothing graded exactly.
    if (schema['fields'] and set(schema['fields']) != {'reply'}
            and not any(spec.get('graded', True) for spec in schema['fields'].values() if isinstance(spec, dict))):
        problems.append('at least one field must be graded')
    return problems


def load_schema(path):
    with open(path, encoding='utf-8') as handle:
        schema = json.load(handle)
    problems = check_schema(schema)
    if problems:
        raise SchemaError('; '.join(problems))
    return schema


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def value_problems(spec, value, path='answer'):
    if value is None:
        return [] if spec.get('nullable') else [f'{path}: null not allowed']
    kind = spec['type']
    if kind == 'string' and not isinstance(value, str):
        return [f'{path}: expected string']
    if kind == 'string' and 'enum' in spec and value not in spec['enum']:
        return [f'{path}: {value!r} not in enum']
    if kind == 'boolean' and not isinstance(value, bool):
        return [f'{path}: expected boolean']
    if kind == 'integer' and not (isinstance(value, int) and not isinstance(value, bool)):
        if not (_is_number(value) and float(value).is_integer()):
            return [f'{path}: expected integer']
    if kind == 'number' and not _is_number(value):
        return [f'{path}: expected number']
    if kind == 'array':
        if not isinstance(value, list):
            return [f'{path}: expected array']
        out = []
        for index, item in enumerate(value):
            out += value_problems(spec['items'], item, f'{path}[{index}]')
        return out
    if kind == 'object':
        if not isinstance(value, dict):
            return [f'{path}: expected object']
        properties = spec.get('properties')
        if properties is None:
            return []
        out = []
        if set(value) != set(properties):
            out.append(f'{path}: keys {sorted(value)} != {sorted(properties)}')
        for name, child in properties.items():
            if name in value:
                out += value_problems(child, value[name], f'{path}.{name}')
        return out
    return []


def answer_problems(schema, answer):
    if not isinstance(answer, dict):
        return ['answer must be a JSON object']
    fields = schema['fields']
    problems = []
    if set(answer) != set(fields):
        problems.append(f'answer keys {sorted(answer)} != schema fields {sorted(fields)}')
    for name, spec in fields.items():
        if name in answer:
            problems += value_problems(spec, answer[name], name)
    return problems


def normalize(spec, value):
    """Comparable, hashable form that respects numeric equality and order='any'."""
    if value is None:
        return ('null',)
    if _is_number(value):
        return ('number', round(float(value), 9))
    if isinstance(value, bool) or isinstance(value, str):
        return (type(value).__name__, value)
    if isinstance(value, list):
        item_spec = (spec or {}).get('items') or {}
        items = [normalize(item_spec, item) for item in value]
        if (spec or {}).get('order') == 'any':
            items = sorted(items, key=repr)
        return ('list', tuple(items))
    if isinstance(value, dict):
        properties = (spec or {}).get('properties') or {}
        return ('dict', tuple(sorted((key, normalize(properties.get(key), item)) for key, item in value.items()
                                     if (properties.get(key) or {}).get('graded', True))))
    return ('other', repr(value))


def field_equal(spec, expected, actual):
    return normalize(spec, expected) == normalize(spec, actual)


def graded_fields(schema):
    return {name: spec for name, spec in schema['fields'].items() if spec.get('graded', True)}


def compare(schema, expected, actual):
    """{graded field: bool}. A non-object answer fails every field."""
    fields = graded_fields(schema)
    if not isinstance(actual, dict):
        return {name: False for name in fields}
    return {name: name in actual and name in expected and field_equal(spec, expected[name], actual[name])
            for name, spec in fields.items()}
