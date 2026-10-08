"""Validate structured user consent against actual protected business tools."""
import json

from ..sandbox_runtime import ApprovalTerms, validate_json_schema


def _reference_value(value):
    """Bound reference containers, leaving expression validation to the runner."""
    remaining = 2048

    def lower(item, depth):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 8:
            raise ValueError('reference proposal exceeds JSON bounds')
        if isinstance(item, dict) and set(item) in ({'$ref'}, {'$expr'}):
            return 0
        if isinstance(item, list):
            if len(item) > 1024:
                raise ValueError('reference proposal array exceeds bounds')
            return [lower(child, depth + 1) for child in item]
        if isinstance(item, dict):
            if len(item) > 64:
                raise ValueError('reference proposal object exceeds bounds')
            return {key: lower(child, depth + 1) for key, child in item.items()}
        return item

    try:
        return ApprovalTerms.literal(lower(value, 0))
    except ValueError:
        return False


def validate_authorizations(source, request):
    rules = source.get('business_lifecycle', {}).get('tools', {})
    tools = {t['function']['name']: t['function'] for t in source.get('tools', [])}
    tables = {t['table_name']: {c['name'] for c in t['columns']} for t in source.get('tables', [])}
    implementations = {t['tool_name']: t for t in source.get('tool_implementations', [])}
    proposal_modes = {}
    for variant in source.get('interaction_contract', {}).get('variants', []):
        for original in variant.get('stages', []):
            for stage in [original, *[{**original, **branch} for branch in original.get('branches', [])]]:
                strict = request.get('require_bound_authorization', False)
                if strict and stage.get('kind') == 'execution_confirmation' and not stage.get('grant_permissions') and not stage.get('revoke_permissions'):
                    raise ValueError('AUTHORIZATION_TERMS_MISSING: confirmation requires an explicit grant or revocation')
                terms = ApprovalTerms.validate(stage, required=strict)
                anchor = stage.get('before_tool')
                if terms and (stage.get('kind') != 'execution_confirmation' or anchor not in implementations
                              or implementations[anchor]['operation'] not in {'insert', 'update', 'delete'}):
                    raise ValueError('AUTHORIZATION_TERMS_INVALID: approval must protect a business write')
                shape = ApprovalTerms.proposal_shape(stage)
                grouped = ApprovalTerms.grouped(stage)
                if terms:
                    modes = proposal_modes.setdefault(anchor, set())
                    modes.add(json.dumps(shape, sort_keys=True))
                    if len(modes) > 1:
                        raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: a tool must use one public confirmation format across grant branches')
                proposal = stage.get('reference_proposal')
                if shape:
                    if not request.get('require_structured_interactions'):
                        raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: proposal consent requires structured interactions')
                    if not isinstance(proposal, dict) or set(proposal) != set(shape):
                        raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: reference_proposal must supply every bound parameter')
                    if grouped and any(not isinstance(proposal[name], dict) or set(proposal[name]) != set(fields)
                                       for name, fields in shape.items()):
                        raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: group parameters under their protected operation')
                    if not _reference_value(proposal):
                        raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: reference values must be bounded JSON or captured expressions')
                for grant, term in terms.items():
                    name = term.get('tool', anchor)
                    if name not in implementations or implementations[name]['operation'] not in {'insert', 'update', 'delete'}:
                        raise ValueError('AUTHORIZATION_TERMS_INVALID: approval must protect a business write')
                    if grant not in rules.get(name, {}).get('requires_grants', []):
                        raise ValueError('AUTHORIZATION_TERMS_UNENFORCED: tool must require the bound grant')
                    # A grant is an operation-scoped capability, never shared by
                    # distinct writes whose terms can overwrite one another.
                    if any(other != name and grant in rule.get('requires_grants', [])
                           for other, rule in rules.items()):
                        raise ValueError('AUTHORIZATION_TERMS_INVALID: a bound grant must protect only its declared operation')
                    parameters = tools.get(name, {}).get('parameters', {}).get('properties', {})
                    bound = set(term.get('arguments', {})) | set(term.get('proposal_arguments', []))
                    if bound - set(parameters):
                        raise ValueError('AUTHORIZATION_TERMS_INVALID: unknown approved argument')
                    for key, value in term.get('arguments', {}).items():
                        validate_json_schema(parameters[key], value, f'approval.{name}.{key}')
                    if term.get('proposal_arguments') or grouped:
                        if bound != set(parameters):
                            raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: bind every write parameter in the displayed proposal')
                        values = proposal[name] if grouped else proposal
                        for key, value in values.items():
                            if ApprovalTerms.literal(value):
                                validate_json_schema(parameters[key], value, f'proposal.{name}.{key}')
                    for predicate in term.get('postconditions', []):
                        columns = tables.get(predicate['table'], set())
                        if (not columns or (set(predicate['where']) | set(predicate['values'])) - columns):
                            raise ValueError('AUTHORIZATION_TERMS_INVALID: unknown approved business field')
