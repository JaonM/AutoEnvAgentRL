"""Validate structured user consent against actual protected business tools."""
from ..sandbox_runtime import ApprovalTerms


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
                name = stage.get('before_tool')
                if terms and (stage.get('kind') != 'execution_confirmation' or name not in implementations
                              or implementations[name]['operation'] not in {'insert', 'update', 'delete'}):
                    raise ValueError('AUTHORIZATION_TERMS_INVALID: approval must protect a business write')
                for grant, term in terms.items():
                    modes = proposal_modes.setdefault(name, set())
                    modes.add(bool(term.get('proposal_arguments')))
                    if len(modes) > 1:
                        raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: a tool must use one public confirmation format across grant branches')
                    if grant not in rules.get(name, {}).get('requires_grants', []):
                        raise ValueError('AUTHORIZATION_TERMS_UNENFORCED: tool must require the bound grant')
                    parameters = tools.get(name, {}).get('parameters', {}).get('properties', {})
                    bound = set(term.get('arguments', {})) | set(term.get('proposal_arguments', []))
                    if bound - set(parameters):
                        raise ValueError('AUTHORIZATION_TERMS_INVALID: unknown approved argument')
                    if term.get('proposal_arguments'):
                        if not request.get('require_structured_interactions'):
                            raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: proposal consent requires structured interactions')
                        if bound != set(parameters):
                            raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: bind every write parameter in the displayed proposal')
                        proposal = stage.get('reference_proposal')
                        if not isinstance(proposal, dict) or set(proposal) != ApprovalTerms.proposal_fields(stage):
                            raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: reference_proposal must supply every bound parameter')
                        for value in proposal.values():
                            if not (ApprovalTerms.scalar(value) or isinstance(value, dict) and set(value) in ({'$ref'}, {'$expr'})):
                                raise ValueError('AUTHORIZATION_PROPOSAL_INVALID: reference values must be scalars or captured expressions')
                    for predicate in term.get('postconditions', []):
                        columns = tables.get(predicate['table'], set())
                        if (not columns or (set(predicate['where']) | set(predicate['values'])) - columns):
                            raise ValueError('AUTHORIZATION_TERMS_INVALID: unknown approved business field')
