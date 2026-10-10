"""Closed answer projections for post-round measurement-error diagnosis.

This module is NOT used by the frozen first pilot. It cannot change business
values, reorder answers, infer missing information or execute expressions.
"""
from __future__ import annotations

import copy

from env_factory_v2.code_agent import object_schema
from env_factory_v2.contract import require


def schema(roles, fields):
    if not roles:
        return object_schema({'business_consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
            'mapping': {'type': 'array', 'items': object_schema({})}})
    return object_schema({'business_consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
        'mapping': {'type': 'array', 'items': object_schema({
            'role': {'type': 'string', 'enum': list(roles)},
            'selectors': {'type': 'array', 'items': object_schema({
                'field': {'type': 'string', 'enum': list(fields)}, 'path': {'type': 'string'}})},
            'packing': {'type': 'string', 'enum': ['single', 'list']},
            'time_unit': {'type': 'string', 'enum': ['native', 'minutes', 'seconds']},
            'names': {'type': 'string', 'enum': ['native', 'public_id_to_name']}})}})


def select(value, segments):
    if not segments:
        return copy.deepcopy(value)
    token, *remaining = segments
    if token == '*':
        require(isinstance(value, list), 'wildcard projection requires an actual array')
        return [select(v, remaining) for v in value]
    if isinstance(value, dict):
        require(token in value, 'answer projection field absent')
        return select(value[token], remaining)
    require(isinstance(value, list) and token.isdecimal() and int(token) < len(value), 'answer projection path invalid')
    return select(value[int(token)], remaining)


def path_segments(path):
    """Accept root notation only; never evaluate JSONPath expressions."""
    require(isinstance(path, str), 'closed answer projection path required')
    if path == '$':
        path = ''
    elif path.startswith('$.'):
        path = path[2:]
    segments = path.split('.') if path else []
    require(len(segments) <= 16 and all(
        s and (s == '*' or s.isidentifier() or s.isdecimal()) for s in segments),
        'closed answer projection path required')
    return segments


def normalize(answer, entries, *, name_aliases=None, role_types=None):
    output = {}
    for entry in entries:
        role = entry['role']
        require(role not in output, 'duplicate semantic role')
        selectors = entry['selectors']
        require(1 <= len(selectors) <= 16, 'bounded answer selectors required')
        values = []
        for item in selectors:
            require(item['field'] in answer, 'declared answer field missing')
            segments = path_segments(item['path'])
            values.append(select(answer[item['field']], segments))
        if entry['packing'] == 'single':
            require(len(values) == 1, 'single projection requires exactly one selector')
            value = values[0]
        else:
            value = [part for v in values for part in (v if isinstance(v, list) else [v])]
        if role_types and role_types.get(role) is not list and isinstance(value, list):
            require(len(value) == 1, 'scalar semantic role requires exactly one observed value')
            value = value[0]
        if entry['time_unit'] != 'native':
            require(role == 'hours' and type(value) in {int, float}, 'time conversion applies only to numeric duration')
            value /= 60 if entry['time_unit'] == 'minutes' else 3600
        if entry['names'] != 'native':
            require(role == 'names' and isinstance(value, list) and name_aliases, 'name conversion needs public aliases')
            require(all(v in name_aliases for v in value), 'unrecognized public name alias')
            value = [name_aliases[v] for v in value]
        output[role] = value
    return output


def rename_observations(scenario, prefix='independent_'):
    """Rename caller-owned IDs and fixture references, never reward targets."""
    identifiers = [s['id'] for s in scenario['steps'] if s['kind'] in {'call', 'traverse'}]
    def identifier(value):
        for old in sorted(identifiers, key=len, reverse=True):
            if value == old or value.startswith(old + '_page'):
                return prefix + value
        return value
    def rewrite(value):
        if isinstance(value, list):
            return [rewrite(v) for v in value]
        if not isinstance(value, dict):
            return value
        if set(value) == {'literal'}:
            return copy.deepcopy(value)
        result = {k: rewrite(v) for k, v in value.items()}
        for key in ('id', '$result', 'source_id'):
            if key in result and isinstance(result[key], str):
                result[key] = identifier(result[key])
        if set(result) == {'ref'} and result['ref'].startswith('evidence.'):
            parts = result['ref'].split('.')
            parts[1] = identifier(parts[1])
            result['ref'] = '.'.join(parts)
        return result
    return rewrite(scenario)


def wrong_projection(answer, entry):
    """Change one actual projected value, preserving the surrounding layout."""
    from v2_natural_oracle import wrong_value
    result = copy.deepcopy(answer)
    selected = entry['selectors'][0]
    segments = path_segments(selected['path'])
    def change(value, path):
        if not path:
            if isinstance(value, list):
                require(value, 'negative projection needs a nonempty array')
                value[0] = change(value[0], [])
                return value
            return wrong_value(value)
        token, *rest = path
        key = (0 if token == '*' else int(token)) if isinstance(value, list) else token
        value[key] = change(value[key], rest)
        return value
    result[selected['field']] = change(result[selected['field']], segments)
    return result


def fill_projections(answer, entries, truth, *, name_aliases=None):
    """Insert independently computed answers, keeping the declared layout."""
    result = copy.deepcopy(answer)
    def replace(current, segments, value):
        if not segments:
            if type(current) is int and type(value) is float and value.is_integer():
                return int(value)
            return copy.deepcopy(value)
        token, *rest = segments
        if token == '*':
            require(isinstance(current, list), 'inverse wildcard needs an actual array')
            values = value if isinstance(value, list) else [value]
            require(len(current) == len(values), 'inverse projection cannot add or omit records')
            return [replace(v, rest, replacement) for v, replacement in zip(current, values)]
        key = int(token) if isinstance(current, list) else token
        current[key] = replace(current[key], rest, value)
        return current
    for e in entries:
        value = copy.deepcopy(truth[e['role']])
        if e['names'] != 'native':
            inverse = {}
            for key, name in (name_aliases or {}).items():
                require(name not in inverse, 'inverse aliases must be unambiguous')
                inverse[name] = key
            value = [inverse[v] for v in value]
        if e['time_unit'] != 'native':
            value *= 60 if e['time_unit'] == 'minutes' else 3600
        selectors = e['selectors']
        if len(selectors) == 1:
            replacements = [value]
        else:
            require(e['packing'] == 'list' and isinstance(value, list) and len(value) == len(selectors),
                    'inverse split projection shape mismatch')
            replacements = value
        for selector, replacement in zip(selectors, replacements):
            field = selector['field']
            segments = path_segments(selector['path'])
            result[field] = replace(result[field], segments, replacement)
    return result
