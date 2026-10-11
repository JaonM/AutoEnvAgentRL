"""Structural member scopes; JSON-encoded paths use null for each array element."""
import copy
import json

from env_factory_v2.contract import require
from env_factory_v2.expressions import canonical


def member_paths(shapes):
    def record(node):
        return node['type'] == 'object' or (node['type'] == 'array'
            and any(record(item) for item in node['item_shapes']))
    def leaves(node, path):
        if node['type'] == 'object' and node['properties']:
            for name, value in node['properties'].items():
                yield from leaves(value, path + [name])
        elif node['type'] == 'array' and any(record(v) for v in node['item_shapes']):
            for variant in node['item_shapes']:
                yield from leaves(variant, path + [None])
        else:
            yield path
    return sorted({canonical(path) for role, shape in shapes.items()
                   if record(shape) for path in leaves(shape, [role]) if len(path) > 1})


def validate_members(catalog, reviews, roles, validate_citation):
    require(isinstance(catalog, list) and len(set(catalog)) == len(catalog), 'invalid member scope catalog')
    decoded = {}
    role_kinds = {r['role']: r['kind'] for r in roles}
    for path in catalog:
        try:
            parts = json.loads(path)
        except (TypeError, ValueError):
            raise ValueError('invalid encoded member scope path') from None
        require(isinstance(parts, list) and len(parts) > 1 and parts[0] in role_kinds
                and all(p is None or isinstance(p, str) for p in parts)
                and canonical(parts) == path, 'invalid member scope path')
        decoded[path] = parts
    require(isinstance(reviews, list) and len(reviews) == len(catalog)
            and {r['path'] for r in reviews} == set(catalog), 'every nested member needs exactly one review')
    for review in reviews:
        require(set(review) == {'path', 'kind', 'reason', 'citations'}
                and review['kind'] in {'answer', 'state_only', 'optional', 'unsupported'}
                and isinstance(review['reason'], str) and review['reason'].strip()
                and review['citations'], 'nested member requires grounded interpretation')
        for citation in review['citations']:
            validate_citation(citation)
        if review['kind'] == 'answer':
            require(role_kinds[decoded[review['path']][0]] == 'answer',
                    'required member must belong to a required answer role')
    for role, kind in role_kinds.items():
        children = [r for r in reviews if decoded[r['path']][0] == role]
        if children and kind == 'answer':
            require(any(r['kind'] in {'answer', 'unsupported'} for r in children),
                    'required record role has no required answer members')


def project_members(answer, scope):
    """Filter only pre-reviewed members; retain every row and all unprojected roles."""
    if not scope or 'member_reviews' not in scope:
        return answer
    reviews = scope['member_reviews']
    paths = {tuple(json.loads(r['path'])) for r in reviews}
    selected = {tuple(json.loads(r['path'])) for r in reviews if r['kind'] == 'answer'}
    def project(value, prefix):
        if prefix in selected:
            require(not any(p[:len(prefix)] == prefix and p not in selected for p in paths),
                    'overlapping member scopes require explicit interpretation')
            return copy.deepcopy(value)
        relevant = [p for p in selected if p[:len(prefix)] == prefix]
        require(relevant, 'missing required member scope')
        if value is None:
            return None
        if isinstance(value, list):
            require(all(p[len(prefix)] is None for p in relevant), 'member scope expects an array')
            return [project(item, prefix + (None,)) for item in value]
        require(isinstance(value, dict), 'member scope expects an object')
        keys = {p[len(prefix)] for p in relevant}
        require(None not in keys and keys <= value.keys(), 'runtime answer lacks a required member')
        return {key: project(value[key], prefix + (key,)) for key in sorted(keys)}
    return {role: project(value, (role,)) if any(p[0] == role for p in paths) else value
            for role, value in answer.items()}


def project_record_roles(descriptors, answer, scope):
    """Keep record adapters aligned with reviewed output; native projection stays available."""
    result = copy.deepcopy({k: v for k, v in descriptors.items() if k in answer})
    if not scope or 'member_reviews' not in scope:
        return result
    reviewed = {tuple(json.loads(r['path'])) for r in scope['member_reviews']}
    selected = {tuple(json.loads(r['path'])) for r in scope['member_reviews'] if r['kind'] == 'answer'}
    for role, descriptor in list(result.items()):
        paths = {p for p in reviewed if p[0] == role}
        if not paths:
            continue  # Scalar selected_boolean roles retain their complete independent record evidence.
        required = {p for p in selected if p[0] == role}
        if (any(len(p) != 3 or p[1] is not None for p in required)
                or (role, None, descriptor['identity']) not in required):
            del result[role]  # This flat identity-based adapter cannot represent the reviewed structure.
            continue
        fields = {p[2] for p in required} - {descriptor['identity']}
        if not fields <= set(descriptor['fields']):
            del result[role]
            continue
        descriptor['fields'] = [field for field in descriptor['fields'] if field in fields]
        if 'aliases' in descriptor:
            descriptor['aliases'] = {k: v for k, v in descriptor['aliases'].items() if k in fields}
    return result
