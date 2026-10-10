"""Trace independent business interpretations to public clauses and executed cases."""
from .code_agent import object_schema
from .contract import require
from .demand import cited_requirements


def obligations(compiled):
    demand = compiled['source'].get('demand')
    return cited_requirements(demand['agent_context']) if demand else []


def implementation_paths(source):
    """A finite catalog prevents response paths from referring to payload wrappers."""
    paths = []
    for root in ('rules', 'tools', 'answers', 'goals', 'invariants', 'explanations', 'success'):
        if root not in source:
            continue
        value = source[root]
        if not value or root == 'success':
            paths.append('/' + root)
        if isinstance(value, list):
            for index, item in enumerate(value):
                prefix = f'/{root}/{index}'
                paths.append(prefix)
                if isinstance(item, dict):
                    paths.extend(prefix + '/' + key.replace('~', '~0').replace('/', '~1') for key in item)
    stages = source.get('user', {}).get('stages', [])
    if not stages:
        paths.append('/user/stages')
    for index, stage in enumerate(stages):
        prefix = f'/user/stages/{index}'
        paths.append(prefix)
        paths.extend(prefix + '/' + key.replace('~', '~0').replace('/', '~1') for key in stage)
    return paths


def review_schema(base_schema, requirements, source):
    if not requirements:
        return base_schema
    return object_schema({**base_schema['properties'], 'requirement_reviews': {
        'type': 'array', 'items': object_schema({
            'requirement_id': {'type': 'string', 'enum': [r['id'] for r in requirements]},
            'consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
            'implementation_paths': {'type': 'array', 'items': {
                'type': 'string', 'enum': implementation_paths(source)}},
            'case_ids': {'type': 'array', 'items': {'type': 'string'}},
        })}})


def implementation_at(source, path):
    require(isinstance(path, str) and path in implementation_paths(source), 'invalid implementation citation')
    parts = path[1:].split('/')
    value = source
    for part in parts:
        require('~' not in part.replace('~0', '').replace('~1', ''), 'invalid JSON pointer escape')
        key = part.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            require(key.isdigit() and str(int(key)) == key and int(key) < len(value),
                    'implementation citation index does not exist')
            value = value[int(key)]
        else:
            require(isinstance(value, dict) and key in value, 'implementation citation does not exist')
            value = value[key]
    return value


def validate_review(compiled, requirements, value):
    if not requirements:
        return None
    reviews = value.get('requirement_reviews', [])
    require(len(reviews) == len(requirements), 'independent review must cover every public clause')
    expected, seen = {r['id'] for r in requirements}, set()
    cases = {c['id'] for c in value['cases']}
    for item in reviews:
        identity = item['requirement_id']
        require(identity in expected and identity not in seen, 'invalid or duplicate requirement review')
        seen.add(identity)
        require(type(item['consistent']) is bool and item['reason'].strip(), 'requirement review verdict and reason required')
        paths, case_ids = item['implementation_paths'], item['case_ids']
        require(1 <= len(paths) <= 16 and len(set(paths)) == len(paths), 'requirement implementation evidence required')
        require(case_ids and len(set(case_ids)) == len(case_ids) and set(case_ids) <= cases,
                'requirement must cite actual independent executable cases')
        for path in paths:
            implementation_at(compiled['source'], path)
    return {'scope': 'independent_model_interpretation_with_validated_text_and_execution_references',
            'requirements': requirements, 'reviews': reviews,
            'passed': all(item['consistent'] for item in reviews)}
