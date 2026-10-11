"""Review answer obligations before generation, independent of candidate layouts."""
import copy
import hashlib
from env_factory_v2.expressions import canonical
from env_factory_v2.code_agent import object_schema
from env_factory_v2.contract import require
from v2_natural_semantics import value_shape


def public_scope_identity(user_request, business):
    return hashlib.sha256(canonical({'user_request': user_request,
                                    'business_policy': business['policy']}).encode()).hexdigest()


def validate_public_scope(scope, *, user_request, business):
    """Validate provenance and allowed data, not the semantic correctness of a model interpretation."""
    require(set(scope) == {'source_identity', 'roles', 'process_requirements'}, 'invalid public output scope fields')
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
    return copy.deepcopy(scope)


def generation_output_scope(spec, *, user_request, business):
    """Historical assets remain readable; never derive public hints from oracle values."""
    if 'generation_output_scope' not in spec:
        return None
    scope = validate_public_scope(spec['generation_output_scope'], user_request=user_request, business=business)
    require({r['role'] for r in scope['roles'] if r['kind'] == 'answer'} ==
            {r['role'] for r in spec['answer_obligations']}, 'generation and acceptance output scopes differ')
    return scope


def review_answer_obligations(agent, *, user_request, business, truth):
    """No generated contract, scores, trajectories or candidate answer enter this review."""
    public = {'user_request': user_request, 'business_policy': business['policy']}
    roles = list(truth['answer'])
    citation = object_schema({'source': {'type': 'string', 'enum': list(public)},
                             'text': {'type': 'string'}})
    item = object_schema({'role': {'type': 'string', **({'enum': roles} if roles else {})},
        'kind': {'type': 'string', 'enum': ['answer', 'state_only', 'optional', 'unsupported']},
        'reason': {'type': 'string'}, 'citations': {'type': 'array', 'items': citation}})
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
        'Use unsupported for an unresolved role; never guess a scope to obtain acceptance. '
        'No candidate contract, expected values or run outcomes are available. No tools.',
        {'original_sources': public,
         'proposed_role_shapes': {k: value_shape(v) for k,v in truth['answer'].items()},
         'state_tables': list(truth['final_state'])},
        object_schema({'roles': {'type': 'array', 'items': item},
                       'process_requirements': {'type': 'array', 'items': citation},
                       'uncovered_answers': {'type': 'array', 'items': citation}}))
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
    passed = not value['uncovered_answers'] and all(r['kind'] != 'unsupported' for r in value['roles'])
    scope = None
    if passed:
        scope = validate_public_scope({'source_identity': public_scope_identity(user_request, business),
            'roles': value['roles'], 'process_requirements': value['process_requirements']},
            user_request=user_request, business=business)
    return {'passed': passed, 'answer_obligations': obligations, 'generation_output_scope': scope,
            'review': value, 'provenance': provenance,
            'scope': 'pre_generation_original_requirement_interpretation_not_sandbox_qualification'}
