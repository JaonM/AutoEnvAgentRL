"""Closed answer projections governed by independently declared truth.

Membership equivalence requires an exact independently computed complete set;
no model-supplied conversion code, guessed identity or partial set is accepted.
Frozen historical measurements remain unchanged by subsequent revisions.
"""
from __future__ import annotations

import copy

from env_factory_v2.code_agent import object_schema
from env_factory_v2.contract import require


COLLECTION_MAPPING_INSTRUCTIONS = (
    'For native collections, exclude_selectors may select complete unique primitive subsets '
    'to subtract from selectors; exclusions must be contained in the selected collection. '
    'Alternatively columns maps each selected object to canonical names via record-relative '
    'paths. Use only relationships justified by answer meanings and original requirements. '
    'Do not combine exclusions and columns, or add constants, calculations or expected values. '
    'Use empty arrays for unused exclude_selectors and columns. ')


def value_shape(value):
    """Describe canonical field structure without revealing oracle answer values."""
    from env_factory_v2.expressions import canonical
    if isinstance(value, dict):
        return {'type': 'object', 'properties': {k: value_shape(v) for k, v in value.items()}}
    if isinstance(value, list):
        shapes = {canonical(value_shape(item)): value_shape(item) for item in value}
        return {'type': 'array', 'item_shapes': list(shapes.values())}
    return {'type': type(value).__name__}


def schema(roles, fields, *, set_memberships=None, quantity_offsets=None, record_roles=None):
    if not roles:
        return object_schema({'business_consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
            'mapping': {'type': 'array', 'items': object_schema({})}})
    projection = {
            'role': {'type': 'string', 'enum': list(roles)},
            'selectors': {'type': 'array', 'items': object_schema({
                'field': {'type': 'string', 'enum': list(fields)}, 'path': {'type': 'string'}})},
            'packing': {'type': 'string', 'enum': ['single', 'list']},
            'time_unit': {'type': 'string', 'enum': ['native', 'minutes', 'seconds']},
            'names': {'type': 'string', 'enum': ['native', 'public_id_to_name']}}
    projection['exclude_selectors'] = {'type': 'array', 'items': object_schema({
        'field': {'type': 'string', 'enum': list(fields)}, 'path': {'type': 'string'}})}
    projection['columns'] = {'type': 'array', 'items': object_schema({
        'name': {'type': 'string'}, 'path': {'type': 'string'}})}
    if record_roles:
        projection['record_layout'] = object_schema({
            'identity_path': {'type': 'string'}, 'details_path': {'type': 'string'},
            'scope_field': {'type': 'string'}, 'columns': {'type': 'array', 'items': object_schema({
                'name': {'type': 'string'}, 'path': {'type': 'string'}})}})
    if set_memberships or quantity_offsets or record_roles:
        projection['representation'] = {'type': 'string', 'enum': ['native']
            + (['set_membership'] if set_memberships else [])
            + (['total_quantity'] if quantity_offsets else []) + (['records'] if record_roles else [])}
    return object_schema({'business_consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
        'mapping': {'type': 'array', 'items': object_schema(projection)}})


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


def project(answer, entry):
    """One closed selector implementation for preflight and value checking."""
    selectors = entry['selectors']
    require(1 <= len(selectors) <= 16, 'bounded answer selectors required')
    require(entry['packing'] in {'single', 'list'}, 'invalid packing')
    values = []
    for item in selectors:
        require(item['field'] in answer, 'declared answer field missing')
        values.append(select(answer[item['field']], path_segments(item['path'])))
    if entry['packing'] == 'single':
        require(len(values) == 1, 'single projection requires exactly one selector')
        value = values[0]
    else:
        value = [part for value in values for part in (value if isinstance(value, list) else [value])]
    exclusions, columns = entry.get('exclude_selectors', []), entry.get('columns', [])
    require(isinstance(exclusions, list) and len(exclusions) <= 16
            and isinstance(columns, list) and len(columns) <= 32, 'bounded collection transforms required')
    if exclusions or columns:
        require(isinstance(value, list) and entry.get('representation', 'native') == 'native'
                and entry['names'] == entry['time_unit'] == 'native',
                'collection transforms require a native collection')
        require(not (exclusions and columns), 'collection transforms cannot be combined')
    if exclusions:
        from env_factory_v2.expressions import canonical
        excluded = []
        for selector in exclusions:
            require(selector['field'] in answer, 'exclusion answer field missing')
            part = select(answer[selector['field']], path_segments(selector['path']))
            require(isinstance(part, list), 'exclusion selector requires a collection')
            excluded.extend(part)
        require(all(type(v) in {str, int, float, bool} for v in value + excluded),
                'difference requires primitive collection elements')
        require(len({canonical(v) for v in value}) == len(value)
                and len({canonical(v) for v in excluded}) == len(excluded),
                'difference requires unique collections')
        require({canonical(v) for v in excluded} <= {canonical(v) for v in value},
                'excluded collection must be a subset')
        value = [v for v in value if canonical(v) not in {canonical(v) for v in excluded}]
    if columns:
        require(len({c['name'] for c in columns}) == len(columns)
                and all(isinstance(c['name'], str) and c['name'] for c in columns),
                'unique mapped column names required')
        require(all(isinstance(row, dict) for row in value), 'column mapping requires record objects')
        value = [{c['name']: select(row, path_segments(c['path'])) for c in columns} for row in value]
    return value


def normalize(answer, entries, *, name_aliases=None, role_types=None, set_memberships=None, quantity_offsets=None, record_roles=None):
    output = {}
    for entry in entries:
        role = entry['role']
        require(role not in output, 'duplicate semantic role')
        value = project(answer, entry)
        if entry.get('representation', 'native') == 'set_membership':
            descriptor = (set_memberships or {}).get(role)
            require(descriptor is not None and role_types and role_types.get(role) is bool,
                    'set membership requires independently declared boolean role')
            require(entry['time_unit'] == entry['names'] == 'native'
                    and isinstance(value, list) and all(type(v) is str for v in value)
                    and len(value) == len(set(value)), 'unique identifier set required')
            require(set(value) == set(descriptor['complete_set']), 'independent complete qualification set mismatch')
            value = descriptor['member'] in value
        elif entry.get('representation', 'native') == 'total_quantity':
            offset = (quantity_offsets or {}).get(role)
            require(type(offset) is int and offset >= 0 and role_types
                    and role_types.get(role) in (int, type(None)),
                    'total quantity requires independently declared integer quantity role')
            require(entry['time_unit'] == entry['names'] == 'native', 'quantity conversion cannot combine aliases or time units')
            if isinstance(value, list):
                require(len(value) == 1, 'quantity role requires exactly one value')
                value = value[0]
            require(value is None or type(value) is int, 'total quantity must be integer or null')
            value = None if value is None else value - offset
        elif entry.get('representation', 'native') == 'records':
            from v2_record_semantics import record_projection
            descriptor = (record_roles or {}).get(role)
            boolean_records = descriptor and descriptor.get('kind') == 'selected_boolean'
            require(descriptor and role_types and role_types.get(role) is (bool if boolean_records else list)
                    and entry['time_unit'] == entry['names'] == 'native', 'independent record role required')
            value = record_projection(value, entry['record_layout'], descriptor)
            if boolean_records:
                from v2_record_semantics import selected_boolean
                value = selected_boolean(value, descriptor)
        elif entry.get('representation', 'native') != 'native':
            require(False, 'unknown closed answer representation')
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


def wrong_projection(answer, entry, *, record_roles=None):
    """Change one actual projected value, preserving the surrounding layout."""
    from v2_natural_oracle import wrong_value
    if entry.get('exclude_selectors') or entry.get('columns'):
        from env_factory_v2.expressions import canonical
        original = canonical(project(answer, entry))
        # Alter an actual primitive leaf, never a computed expected answer. Try
        # all selected records so an excluded first item cannot hide the mutation.
        def leaves(value, path=()):
            if isinstance(value, dict):
                for key, item in value.items():
                    yield from leaves(item, (*path, key))
            elif isinstance(value, list):
                for key, item in enumerate(value):
                    yield from leaves(item, (*path, key))
            else:
                yield path, value
        for selector in entry['selectors']:
            for path, value in leaves(answer[selector['field']]):
                candidate = copy.deepcopy(answer)
                current = candidate
                full_path = (selector['field'], *path)
                for key in full_path[:-1]:
                    current = current[key]
                current[full_path[-1]] = wrong_value(value)
                try:
                    changed = canonical(project(candidate, entry)) != original
                except ValueError:
                    continue
                if changed:
                    return candidate
        require(False, 'no observable negative collection projection')
    if entry.get('representation', 'native') == 'records':
        descriptor = (record_roles or {})[entry['role']]
        if descriptor.get('kind') == 'selected_boolean':
            canonical = normalize(answer, [entry], role_types={entry['role']: bool}, record_roles=record_roles)
            canonical[entry['role']] = not canonical[entry['role']]
            return fill_projections(answer, [entry], canonical, record_roles=record_roles)
        roles = {entry['role']: list}
        canonical = normalize(answer, [entry], role_types=roles, record_roles=record_roles)
        rows = canonical[entry['role']]
        require(rows, 'negative record projection requires a nonempty array')
        descriptor = (record_roles or {})[entry['role']]
        field = descriptor['fields'][0]
        rows[0][field] = wrong_value(rows[0][field])
        return fill_projections(answer, [entry], canonical, record_roles=record_roles)
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


def fill_projections(answer, entries, truth, *, name_aliases=None, set_memberships=None, quantity_offsets=None, record_roles=None):
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
        if e.get('exclude_selectors') or e.get('columns'):
            from env_factory_v2.expressions import canonical
            observed = project(result, e)
            require(isinstance(value, list) and canonical(sorted(observed, key=canonical))
                    == canonical(sorted(value, key=canonical)),
                    'derived collection differs from independent truth; inverse is ambiguous')
            # Preserve an independently correct real answer. Do not manufacture
            # records or choose one of several possible inverse representations.
            continue
        if e.get('representation', 'native') == 'set_membership':
            descriptor = (set_memberships or {}).get(e['role'])
            require(descriptor is not None and type(value) is bool
                    and value == (descriptor['member'] in descriptor['complete_set']),
                    'independent membership truth required for inverse projection')
            value = copy.deepcopy(descriptor['complete_set'])
        elif e.get('representation', 'native') == 'total_quantity':
            offset = (quantity_offsets or {}).get(e['role'])
            require(type(offset) is int and offset >= 0 and (value is None or type(value) is int),
                    'independent quantity offset required for inverse projection')
            value = None if value is None else value + offset
        elif e.get('representation', 'native') == 'records':
            from v2_record_semantics import record_projection
            require(len(e['selectors']) == 1 and e['packing'] in {'single', 'list'}, 'inverse records require one array selector')
            selector = e['selectors'][0]
            current = select(result[selector['field']], path_segments(selector['path']))
            descriptor = (record_roles or {}).get(e['role'])
            require(descriptor, 'independent record role required')
            if descriptor.get('kind') == 'selected_boolean':
                from v2_record_semantics import boolean_replacement
                value = boolean_replacement(value, descriptor)
            value = record_projection(current, e['record_layout'], descriptor, replacement=value)
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
