"""Review answer obligations before generation, independent of candidate layouts."""
import copy
import hashlib
from env_factory_v2.contracts.expressions import canonical
from env_factory_v2.native.code_agent import object_schema
from env_factory_v2.contracts.contract import require
from v2_natural_semantics import value_shape
from v2_answer_members import member_paths, validate_members


def public_scope_identity(user_request, business):
    return hashlib.sha256(canonical({'user_request': user_request,
                                    'business_policy': business['policy']}).encode()).hexdigest()


def validate_public_scope(scope, *, user_request, business):
    """Validate provenance and allowed data, not the semantic correctness of a model interpretation."""
    split_fields = {'communication_requirements', 'behavior_requirements'}
    require(not (set(scope) & split_fields) or split_fields <= set(scope), 'incomplete requirement scope split')
    require(set(scope) - split_fields in ({'source_identity', 'roles', 'process_requirements'},
                           {'source_identity', 'roles', 'process_requirements', 'member_catalog', 'member_reviews'}),
            'invalid public output scope fields')
    require(scope['source_identity'] == public_scope_identity(user_request, business),
            'output scope belongs to different original sources')
    public = {'user_request': user_request, 'business_policy': business['policy']}
    ids = []
    for role in scope['roles']:
        require(set(role) == {'role', 'kind', 'reason', 'citations'}, 'invalid public output role fields')
        require(isinstance(role['role'], str) and role['role'].strip()
                and role['kind'] in {'answer', 'state_only', 'optional'}
                and isinstance(role['reason'], str) and role['reason'].strip()
                and role['citations'], 'unresolved public output scope')
        ids.append(role['role'])
    require(len(ids) == len(set(ids)), 'duplicate public output roles')
    for c in [c for role in scope['roles'] for c in role['citations']] + scope['process_requirements']:
        require(set(c) == {'source', 'text'} and c['source'] in public
                and isinstance(c['text'], str) and c['text'].strip()
                and c['text'] in public[c['source']], 'output scope citation must match original text')
    for key in split_fields:
        require(isinstance(scope.get(key, []), list), 'scope meanings must be a list')
        for item in scope.get(key, []):
            require(isinstance(item, dict) and set(item) == {'meaning', 'citations'}
                    and isinstance(item['meaning'], str) and item['meaning'].strip()
                    and isinstance(item['citations'], list) and item['citations'], 'grounded scope meaning required')
            for c in item['citations']:
                require(set(c) == {'source', 'text'} and c['source'] in public
                        and isinstance(c['text'], str) and c['text'].strip()
                        and c['text'] in public[c['source']], 'scope meaning citation must match original text')
    if 'member_reviews' in scope:
        def citation_check(c):
            require(set(c) == {'source', 'text'} and c['source'] in public
                    and isinstance(c['text'], str) and c['text'].strip()
                    and c['text'] in public[c['source']], 'member citation must match original text')
        validate_members(scope['member_catalog'], scope['member_reviews'], scope['roles'], citation_check)
        require(all(r['kind'] != 'unsupported' for r in scope['member_reviews']), 'unresolved member scope')
    return copy.deepcopy(scope)


def generation_output_scope(spec, *, user_request, business):
    """Historical assets remain readable; never derive public hints from oracle values."""
    if 'generation_output_scope' not in spec:
        return None
    scope = validate_public_scope(spec['generation_output_scope'], user_request=user_request, business=business)
    require({r['role'] for r in scope['roles'] if r['kind'] == 'answer'} ==
            {r['role'] for r in spec['answer_obligations']}, 'generation and acceptance output scopes differ')
    if 'answer_role_shapes' in spec and 'member_catalog' in scope:
        require(scope['member_catalog'] == member_paths(validate_role_shapes(spec['answer_role_shapes'])),
                'member scope differs from declared answer structure')
    return scope


def validate_role_shapes(shapes):
    def check(node):
        require(isinstance(node, dict), 'answer role shape must be structural')
        kind = node.get('type')
        if kind == 'object':
            require(set(node) == {'type', 'properties'} and isinstance(node['properties'], dict),
                    'object role shape needs only properties')
            for value in node['properties'].values():
                check(value)
        elif kind == 'array':
            require(set(node) == {'type', 'item_shapes'} and isinstance(node['item_shapes'], list),
                    'array role shape needs only item shapes')
            for value in node['item_shapes']:
                check(value)
        else:
            require(set(node) == {'type'} and kind in {'str', 'bool', 'int', 'float', 'NoneType'},
                    'role shape must not contain expected values or constraints')
    require(isinstance(shapes, dict) and all(isinstance(k, str) and k.strip() for k in shapes),
            'answer role shapes must be named')
    for shape in shapes.values():
        check(shape)
    return copy.deepcopy(shapes)


def review_case_answer_obligations(agent, *, user_request, business, user, spec):
    if 'answer_role_shapes' in spec:
        return review_answer_obligations(agent, user_request=user_request, business=business,
            role_shapes=spec['answer_role_shapes'], state_tables=[table['id'] for table in business['tables']])
    from audit_v2_natural import canonical_truth
    return review_answer_obligations(agent, user_request=user_request, business=business,
                                    truth=canonical_truth(spec, business, user))


def review_answer_obligations(agent, *, user_request, business, truth=None, role_shapes=None, state_tables=None):
    """No generated contract, scores, trajectories or candidate answer enter this review."""
    public = {'user_request': user_request, 'business_policy': business['policy']}
    require((truth is None) != (role_shapes is None), 'supply truth or declared role shapes, not both')
    if truth is not None:
        role_shapes = {k: value_shape(v) for k, v in truth['answer'].items()}
        state_tables = list(truth['final_state'])
    role_shapes = validate_role_shapes(role_shapes)
    require(isinstance(state_tables, list) and all(isinstance(t, str) for t in state_tables),
            'state table names required for output scope review')
    roles = list(role_shapes)
    citation = object_schema({'source': {'type': 'string', 'enum': list(public)},
                             'text': {'type': 'string'}})
    item = object_schema({'role': {'type': 'string', **({'enum': roles} if roles else {})},
        'kind': {'type': 'string', 'enum': ['answer', 'state_only', 'optional', 'unsupported']},
        'reason': {'type': 'string'}, 'citations': {'type': 'array', 'items': citation}})
    members = member_paths(role_shapes)
    response_fields = {'roles': {'type': 'array', 'items': item},
                       'process_requirements': {'type': 'array', 'items': citation},
                       'uncovered_answers': {'type': 'array', 'items': citation}}
    meaning = object_schema({'meaning': {'type': 'string'}, 'citations': {'type': 'array', 'items': citation}})
    for key in ('communication_requirements', 'behavior_requirements'):
        response_fields[key] = {'type': 'array', 'items': meaning}
    if members:
        response_fields['member_reviews'] = {'type': 'array', 'items': object_schema({
            # Encoded JSON paths contain quotes, unsupported in strict-output enum literals.
            # Exact catalog membership and completeness are enforced by validate_members.
            'path': {'type': 'string'},
            'kind': {'type': 'string', 'enum': ['answer', 'state_only', 'optional', 'unsupported']},
            'reason': {'type': 'string'}, 'citations': {'type': 'array', 'items': citation}})}
    value, provenance = agent.ask(
        'Before any sandbox generation, independently classify EVERY proposed semantic role against '
        'the original request and business policy. An answer role requires communicating its value; '
        'a state-only role describes an effect or preservation obligation, not an extra final answer field. '
        'Optional truthful information must not become a mandatory answer. Do not classify by business '
        'name, role name alone, or a fixed list of transaction types. Distinguish eligibility, consent and '
        'execution results across refusal and partial-success branches. Cite exact original text for '
        'each decision. List only required FINAL ANSWER meanings absent from the proposed roles in uncovered_answers. '
        'Confirmation questions, disclosure before choice, tool use, authorization and operation ordering are '
        'process_requirements, not missing final answer fields; separate execution/communication verifiers '
        'enforce those requirements. Never turn required user interaction into an extra final answer field. '
        'Separately list communication_requirements: meanings the user must be told, with branch applicability '
        'and interaction ordering where required. List behavior_requirements: rules governing calculation, '
        'eligibility, state changes and safety. A policy describing an algorithm or internal fields does not '
        'automatically require reciting it to the user. Necessary explanations of actual restrictions remain '
        'required; distinguish them from a generic tutorial. Neither list waives any original business rule. '
        'Ground both lists in exact original citations, without assuming private facts have already been disclosed. '
        'Use unsupported for an unresolved role; never guess a scope to obtain acceptance. '
        'When nested_member_paths are supplied, classify EVERY listed member separately in member_reviews. '
        'Paths encode JSON arrays: strings name properties and null means every array element. '
        'A required record does not make all its members mandatory answers. Preserve identifiers needed '
        'to associate requested values with objects, but distinguish final answers from state and process '
        'facts for each member. If any member is required, its parent role must be answer. '
        'Do not remove real execution/state obligations when classifying a member as state_only. '
        'No candidate contract, expected values or run outcomes are available. No tools.',
        {'original_sources': public,
         'proposed_role_shapes': role_shapes,
         'state_tables': state_tables, **({'nested_member_paths': members} if members else {})},
        object_schema(response_fields))
    require(isinstance(value['roles'], list) and len(value['roles']) == len(roles)
            and {r['role'] for r in value['roles']} == set(roles), 'answer scope must review each role exactly once')
    def validate_citation(c):
        require(c['source'] in public and isinstance(c['text'], str) and c['text'].strip()
                and c['text'] in public[c['source']], 'answer scope citation must match original text')
    obligations = []
    for role in value['roles']:
        require(role['kind'] in {'answer', 'state_only', 'optional', 'unsupported'}
                and isinstance(role['reason'], str) and role['reason'].strip()
                and role['citations'], 'answer scope requires grounded interpretation')
        for c in role['citations']:
            validate_citation(c)
        if role['kind'] == 'answer':
            obligations.append({'role': role['role'], **role['citations'][0]})
    for c in value['uncovered_answers'] + value['process_requirements']:
        validate_citation(c)
    if members:
        validate_members(members, value['member_reviews'], value['roles'], validate_citation)
    require(('communication_requirements' in value) == ('behavior_requirements' in value),
            'incomplete requirement scope interpretation')
    passed = (not value['uncovered_answers']
              and all(r['kind'] != 'unsupported' for r in value['roles'] + value.get('member_reviews', [])))
    scope = None
    if passed:
        scope = validate_public_scope({'source_identity': public_scope_identity(user_request, business),
            'roles': value['roles'], 'process_requirements': value['process_requirements'],
            **({key: value[key] for key in ('communication_requirements', 'behavior_requirements')}
               if 'communication_requirements' in value or 'behavior_requirements' in value else {}),
            **({'member_catalog': members, 'member_reviews': value['member_reviews']} if members else {})},
            user_request=user_request, business=business)
    return {'passed': passed, 'answer_obligations': obligations, 'generation_output_scope': scope,
            'review': value, 'provenance': provenance,
            'scope': 'pre_generation_original_requirement_interpretation_not_sandbox_qualification'}
