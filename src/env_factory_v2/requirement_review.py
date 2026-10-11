"""Trace independent business interpretations to public clauses and executed cases."""
from .code_agent import object_schema
from .contract import require
from .demand import cited_requirements


def obligations(compiled, original_request=None):
    demand = compiled['source'].get('demand')
    if demand:
        return cited_requirements(demand['agent_context'])
    # Legacy/plain contracts still require original-demand review. Generated
    # public rules cannot substitute for a missing original business dossier.
    request = original_request or {}
    text = request if isinstance(request, str) else next((request[key]
        for key in ('user_request', 'raw_request', 'goal')
        if isinstance(request, dict) and isinstance(request.get(key), str) and request[key].strip()), None)
    if not isinstance(text, str) or not text.strip():
        text = compiled['source']['task']
    return cited_requirements({'user_request': text})


def mandatory_obligations(source):
    """Reverse-review catalog of authored outputs and enforced success conditions."""
    result = []
    def add(path, kind, value):
        result.append({'path': path, 'kind': kind, 'definition': value})
    def fields(schema, path):
        if schema.get('type') == 'array':
            fields(schema['items'], path + '/items')
        elif schema.get('type') == 'object':
            for key, child in schema['properties'].items():
                escaped = key.replace('~', '~0').replace('/', '~1')
                child_path = path + '/properties/' + escaped
                add(child_path, 'answer_property', child)
                fields(child, child_path)
    for index, answer in enumerate(source['answers']):
        path = f'/answers/{index}'
        add(path, 'answer_field', {key: answer[key] for key in ('id', 'meaning', 'weight', 'comparison')})
        fields(answer['schema'], path + '/schema')
    for index, explanation in enumerate(source.get('explanations', [])):
        for position, criterion in enumerate(explanation['criteria']):
            add(f'/explanations/{index}/criteria/{position}', 'communication', criterion)
    for root in ('goals', 'invariants'):
        for index, item in enumerate(source.get(root, [])):
            add(f'/{root}/{index}', 'world_constraint', item)
    for index, branch in enumerate(source.get('success', [])):
        prefix = f'/success/{index}'
        add(prefix + '/when', 'completion_branch', {'public': branch['public'], 'when': branch['when']})
        for key in ('required_tools', 'required_stages', 'required_platform_observations'):
            for position, item in enumerate(branch.get(key, [])):
                add(f'{prefix}/{key}/{position}', key, item)
        for position, observation in enumerate(branch.get('required_observations', [])):
            for column_index, column in enumerate(observation['columns']):
                add(f'{prefix}/required_observations/{position}/columns/{column_index}',
                    'required_observation_column', {'column': column, 'tools': observation['tools']})
        for key in ('min_business_calls', 'max_business_calls'):
            if key in branch:
                add(prefix + '/' + key, 'call_constraint', branch[key])
    return result


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
    # Declared integrity constraints are executable evidence. Fixture rows and
    # generators are deliberately excluded: example values cannot prove rules.
    for index, table in enumerate(source.get('tables', [])):
        paths.extend(f'/tables/{index}/{key}' for key in ('schema', 'primary_key', 'foreign_keys')
                     if key in table)
    return paths


def review_schema(base_schema, requirements, source):
    if not requirements:
        return base_schema
    return object_schema({**base_schema['properties'], 'requirement_reviews': {
        'type': 'array', 'items': object_schema({
            'requirement_id': {'type': 'string', 'enum': [r['id'] for r in requirements]},
            'consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
            'applicability': {'type': 'string', 'enum': ['applicable', 'conditional', 'outside_request']},
            'implementation_paths': {'type': 'array', 'items': {
                'type': 'string', 'enum': implementation_paths(source)}},
            'case_ids': {'type': 'array', 'items': {'type': 'string'}},
        })}, 'obligation_reviews': {'type': 'array', 'items': object_schema({
            'paths': {'type': 'array', 'items': {'type': 'string',
                'enum': [item['path'] for item in mandatory_obligations(source)]}},
            'consistent': {'type': 'boolean'}, 'reason': {'type': 'string'},
            'requirement_ids': {'type': 'array', 'items': {'type': 'string',
                'enum': [r['id'] for r in requirements]}},
            'case_ids': {'type': 'array', 'items': {'type': 'string'}}})}})


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
    by_id = {r['id']: r for r in requirements}
    cases = {c['id'] for c in value['cases']}
    for item in reviews:
        identity = item['requirement_id']
        require(identity in expected and identity not in seen, 'invalid or duplicate requirement review')
        seen.add(identity)
        applicability = item.get('applicability', 'applicable')
        require(applicability in {'applicable', 'conditional', 'outside_request'},
                'invalid requirement applicability')
        require(applicability != 'outside_request' or by_id[identity]['source'] == 'business_policy',
                'user request obligations cannot be excluded from scope')
        require(type(item['consistent']) is bool and item['reason'].strip(), 'requirement review verdict and reason required')
        paths, case_ids = item['implementation_paths'], item['case_ids']
        require(1 <= len(paths) <= 16 and len(set(paths)) == len(paths), 'requirement implementation evidence required')
        require(case_ids and len(set(case_ids)) == len(case_ids) and set(case_ids) <= cases,
                'requirement must cite actual independent executable cases')
        for path in paths:
            implementation_at(compiled['source'], path)
    reverse = []
    for item in value.get('obligation_reviews', []):
        if 'paths' in item:
            require('path' not in item and isinstance(item['paths'], list) and item['paths'],
                    'obligation group requires nonempty paths without a second path field')
            reverse.extend({**{k:v for k,v in item.items() if k != 'paths'}, 'path': path}
                           for path in item['paths'])
        else:
            reverse.append(item)  # Historical evidence remains reproducible.
    required_paths = {item['path'] for item in mandatory_obligations(compiled['source'])}
    require(len(reverse) == len(required_paths), 'reverse review must cover every mandatory obligation')
    reviewed_paths = set()
    for item in reverse:
        path = item['path']
        require(path in required_paths and path not in reviewed_paths, 'invalid or duplicate obligation review')
        reviewed_paths.add(path)
        require(type(item['consistent']) is bool and isinstance(item['reason'], str)
                and item['reason'].strip(), 'obligation review verdict and reason required')
        references, cited_cases = item['requirement_ids'], item['case_ids']
        require(isinstance(references, list) and len(references) == len(set(references))
                and set(references) <= expected, 'obligation cites unknown or duplicate requirement')
        require(not item['consistent'] or references, 'consistent obligation requires original requirement evidence')
        require(cited_cases and len(cited_cases) == len(set(cited_cases)) and set(cited_cases) <= cases,
                'obligation must cite actual independent executable cases')
    return {'scope': 'independent_model_interpretation_with_validated_text_and_execution_references',
            'requirements': requirements, 'reviews': reviews,
            'obligation_catalog': mandatory_obligations(compiled['source']), 'obligation_reviews': reverse,
            'passed': all(item['consistent'] for item in reviews + reverse)}
