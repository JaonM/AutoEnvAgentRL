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

Comparison is about what was communicated, not its typography: strings compare after Unicode NFKC,
case folding and removing whitespace ("尾号 6628" equals "尾号6628"). A field's score is the share of its
graded leaves that match (objects average their properties, equal-length exact-order arrays their
items), so one wrong sub-field no longer zeroes a whole object; a field counts as correct only when
its score is 1. With `known` (strings the user knows: the request and their facts), a string leaf
that looks like a system-internal identifier ("i1", "return-ord-102") and that the user never saw
is not graded: an agent delivering by talking has no reason to read it out.
"""
from __future__ import annotations

import json
import re
import unicodedata

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
    if isinstance(value, bool):
        return ('bool', value)
    if isinstance(value, str):
        if NUMERIC_TEXT.match(value.strip()):  # "12.50", "12.5" and "$12.50" are one amount
            return ('number', round(float(value.strip().lstrip('$¥￥').replace(',', '')), 9))
        return ('str', text_key(value))
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


NUMERIC_TEXT = re.compile(r'^[$¥￥]?-?\d{1,3}(,\d{3})*(\.\d+)?$|^[$¥￥]?-?\d+(\.\d+)?$')


def text_key(value):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', value).casefold())


MACHINE_ID = re.compile(r'^(?:[a-z]{1,3}\d{1,4}|[a-z][a-z0-9]*(?:[-_][a-z0-9]+)+)$')


def internal(spec, value, known):
    """A system-internal identifier the user never saw (graded only when known is None)."""
    if known is None or not isinstance(value, str) or 'enum' in (spec or {}):
        return False
    return bool(MACHINE_ID.match(value)) and not any(value in text for text in known)


def score(spec, expected, actual, known=None):
    """Share of graded leaves of `expected` that `actual` matches (1.0 when nothing is graded)."""
    spec = spec or {}
    if internal(spec, expected, known) and actual in (None, ''):
        return 1.0  # an internal id the agent did not state is not required; naming another record is wrong
    if isinstance(expected, dict) and isinstance(actual, dict) and spec.get('properties'):
        graded = [(name, child) for name, child in spec['properties'].items() if child.get('graded', True)]
        parts = {name: score(child, expected.get(name), actual.get(name), known) for name, child in graded}
        if not parts:
            return 1.0
        if all(value == 1.0 for value in parts.values()):
            return 1.0
        # Partial credit only for substance: an expected null (nothing to report) and an enum (a status, a
        # decision) earn nothing on their own, so saying "refused" or nothing at all does not collect half
        # an object; both are still required for the full score.
        enums = {name for name, child in graded if child.get('enum')}
        substance = [value for name, value in parts.items() if name not in enums and expected.get(name) is not None]
        if not substance:
            return 0.5 * sum(parts.values()) / len(parts) if all(parts[name] == 1.0 for name in enums) else 0.0
        share = sum(substance) / len(substance)
        return share if all(parts[name] == 1.0 for name in enums) else 0.5 * share
    if (isinstance(expected, list) and isinstance(actual, list) and spec.get('order', 'exact') == 'exact'
            and len(expected) == len(actual) and expected):
        return sum(score(spec.get('items'), e, a, known) for e, a in zip(expected, actual)) / len(expected)
    if known is not None and isinstance(expected, (dict, list)):
        if normalize(spec, expected) == normalize(spec, actual):
            return 1.0
        if isinstance(expected, list) and isinstance(actual, list) and spec.get('order') == 'any' \
                and len(expected) == len(actual):
            # multiset: each expected item matched by a distinct actual item, internal ids ignored
            left = list(actual)
            for item in expected:
                hit = next((a for a in left if score(spec.get('items'), item, a, known) == 1.0), None)
                if hit is None:
                    return 0.0
                left.remove(hit)
            return 1.0
        return 0.0
    return 1.0 if normalize(spec, expected) == normalize(spec, actual) else 0.0


def field_equal(spec, expected, actual):
    return normalize(spec, expected) == normalize(spec, actual)


def graded_fields(schema):
    return {name: spec for name, spec in schema['fields'].items() if spec.get('graded', True)}


def scores(schema, expected, actual, known=None):
    """{graded field: share of its graded leaves matched}. A non-object answer scores 0 everywhere."""
    fields = graded_fields(schema)
    if not isinstance(actual, dict):
        return {name: 0.0 for name in fields}
    return {name: (score(spec, expected[name], actual[name], known) if name in actual and name in expected else 0.0)
            for name, spec in fields.items()}


def compare(schema, expected, actual, known=None):
    """{graded field: bool}: True when the field's score is 1. A non-object answer fails every field."""
    return {name: value == 1.0 for name, value in scores(schema, expected, actual, known).items()}


def known_strings(request, facts=None):
    """What the user knows: the opening request and every string among their facts."""
    out = [request.get('user_request') or '']

    def walk(value):
        if isinstance(value, str):
            out.append(value)
        elif isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(facts if facts is not None else request.get('user_facts') or {})
    return out
