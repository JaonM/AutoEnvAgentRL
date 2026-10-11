"""Trace independent business interpretations to public clauses and executed cases."""
from env_factory_v2.native.code_agent import object_schema
from env_factory_v2.contracts.contract import require
from env_factory_v2.requirements.demand import cited_requirements
from env_factory_v2.requirements.obligation_index import mandatory_obligations, implementation_paths


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
            'communication_basis': {'type': 'string', 'enum': ['explicit_request', 'explicit_policy',
                'necessary_derived', 'behavior_only', 'mixed_overconstraint', 'not_communication']},
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
    catalog = {item['path']: item for item in mandatory_obligations(compiled['source'])}
    required_paths = set(catalog)
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
        # Historical reports omit this field; fresh native responses are required
        # to supply it by review_schema. It records semantic grounds, not keywords.
        if 'communication_basis' in item and catalog[path]['kind'] == 'communication':
            basis = item['communication_basis']
            require(basis in {'explicit_request', 'explicit_policy', 'necessary_derived',
                             'behavior_only', 'mixed_overconstraint'}, 'communication needs a scope interpretation')
            require(not item['consistent'] or basis in {'explicit_request', 'explicit_policy', 'necessary_derived'},
                    'behavior rules alone cannot justify mandatory speech')
        require(cited_cases and len(cited_cases) == len(set(cited_cases)) and set(cited_cases) <= cases,
                'obligation must cite actual independent executable cases')
    return {'scope': 'independent_model_interpretation_with_validated_text_and_execution_references',
            'requirements': requirements, 'reviews': reviews,
            'obligation_catalog': mandatory_obligations(compiled['source']), 'obligation_reviews': reverse,
            'passed': all(item['consistent'] for item in reviews + reverse)}
