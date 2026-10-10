"""Closed record layouts: paths only, with host-owned scope and enum aliases."""
import copy

from env_factory_v2.contract import require


def record_projection(rows, layout, descriptor, *, replacement=None):
    from v2_natural_semantics import path_segments, select
    require(isinstance(rows, list) and len(rows) <= 10000, 'bounded record array required')
    columns = layout['columns']
    require(len(columns) == len(descriptor['fields'])
            and {c['name'] for c in columns} == set(descriptor['fields']), 'complete unique record columns required')
    require(len({c['path'] for c in columns}) == len(columns), 'record columns cannot share one value')
    def segments(path):
        parts = path_segments(path)
        require('*' not in parts, 'record scalar paths cannot contain wildcards')
        return parts
    identity_path = segments(layout['identity_path'])
    require(identity_path, 'record identity must select a field')
    detail_path = segments(layout['details_path'])
    def assign(obj, path, value):
        require(path, 'record columns must select a field')
        for token in path[:-1]:
            obj = obj[int(token)] if isinstance(obj, list) else obj[token]
        key = int(path[-1]) if isinstance(obj, list) else path[-1]
        obj[key] = copy.deepcopy(value)
    results, identities = [], set()
    originals = copy.deepcopy(rows)
    replacements = None
    if replacement is not None:
        require(isinstance(replacement, list), 'replacement records required')
        replacements = {}
        for row in replacement:
            identity = row[descriptor['identity']]
            require(type(identity) is str and identity not in replacements, 'unique replacement identity required')
            replacements[identity] = row
    for row in originals:
        identity = select(row, identity_path)
        require(type(identity) is str and identity not in identities, 'unique record identity required')
        identities.add(identity)
        details = select(row, detail_path)
        if layout['scope_field']:
            require(isinstance(details, list), 'scoped record details require an array')
            scope_path = segments(layout['scope_field'])
            candidates = [i for i, item in enumerate(details)
                          if select(item, scope_path) == descriptor['scope']]
            require(len(candidates) == 1, 'exactly one independently scoped detail required')
            details = details[candidates[0]]
            location = detail_path + [str(candidates[0])]
        else:
            require(isinstance(details, dict), 'flat record details require an object')
            location = detail_path
        result = {descriptor['identity']: identity}
        for column in columns:
            path = segments(column['path'])
            require(path, 'record columns must select a field')
            value = select(details, path)
            aliases = descriptor.get('aliases', {}).get(column['name'], {})
            if type(value) is str:
                value = aliases.get(value, value)
            result[column['name']] = value
            if replacements is not None:
                require(identity in replacements, 'replacement cannot change record identities')
                desired = replacements[identity][column['name']]
                if type(desired) is str and aliases:
                    # Preserve the current public spelling when possible; never guess ambiguous aliases.
                    raw = select(details, path)
                    if aliases.get(raw, raw) == desired:
                        desired = raw
                    else:
                        possible = [k for k, v in aliases.items() if v == desired]
                        require(len(possible) <= 1, 'ambiguous inverse record alias')
                        desired = possible[0] if possible else desired
                assign(row, location + path, desired)
        results.append(result)
    if replacements is not None:
        require(identities == set(replacements), 'replacement cannot add or omit records')
        return originals
    return results


def selected_boolean(rows, descriptor):
    """Validate the entire independently scoped record set before selection."""
    from env_factory_v2.expressions import canonical
    identity, field = descriptor['identity'], descriptor['fields'][0]
    expected = descriptor['complete_records']
    require(all(type(row[field]) is bool for row in rows), 'record qualification requires strict booleans')
    require(sorted(canonical(row) for row in rows) == sorted(canonical(row) for row in expected),
            'independent complete qualification records mismatch')
    selected = [row[field] for row in rows if row[identity] == descriptor['member']]
    require(len(selected) <= 1, 'selected qualification record must be unique')
    return selected[0] if selected else False


def boolean_replacement(value, descriptor):
    require(type(value) is bool, 'selected qualification replacement must be boolean')
    rows = copy.deepcopy(descriptor['complete_records'])
    selected = [row for row in rows if row[descriptor['identity']] == descriptor['member']]
    require(len(selected) <= 1 and (selected or value is False),
            'cannot invent an eligible record outside the independent scope')
    if selected:
        selected[0][descriptor['fields'][0]] = value
    return rows
