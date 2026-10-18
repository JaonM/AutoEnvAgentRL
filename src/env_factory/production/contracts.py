"""Optional public business contracts for generation; no domain-specific rules.

The supplied policy is agent-facing authority. Tool contracts constrain the existing
runtime's signature-derived interface, while implementation notes define effects.
This check verifies interface structure, not semantic equivalence of implementations.
"""
from __future__ import annotations

import copy
import hashlib
import json


def digest(contract):
    return hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def problems(contract):
    if not isinstance(contract, dict):
        return ['public_contract must be an object']
    errors = []
    if not isinstance(contract.get('policy'), str) or not contract['policy'].strip():
        errors.append('public_contract.policy must be nonempty public policy text')
    constraints = contract.get('constraints', {})
    if not isinstance(constraints, dict):
        errors.append('public_contract.constraints must be an object')
    elif set(constraints) - {'allow_unauthenticated_inquiry', 'max_tool_calls_per_turn', 'allow_text_with_tool_calls'}:
        errors.append('unknown public contract constraint')
    else:
        for key, value in constraints.items():
            if key == 'max_tool_calls_per_turn':
                if type(value) is not int or value < 1:
                    errors.append('max_tool_calls_per_turn must be a positive integer')
            elif not isinstance(value, bool):
                errors.append(f'{key} must be a boolean')
    tools = contract.get('tools')
    if not isinstance(tools, list) or not tools:
        return errors + ['public_contract.tools must be a nonempty function-schema list']
    names = []
    for tool in tools:
        fn = tool.get('function', tool) if isinstance(tool, dict) else {}
        if not isinstance(fn, dict):
            errors.append('public contract function must be an object')
            continue
        params = fn.get('parameters', {})
        if not isinstance(fn.get('name'), str) or not fn['name']:
            errors.append('public contract tool needs a name')
        else:
            names.append(fn['name'])
        if not isinstance(params, dict) or params.get('type') != 'object' or not isinstance(params.get('properties'), dict):
            errors.append(f'public contract tool {fn.get("name")}: parameters must declare object properties')
            continue
        required = params.get('required', [])
        if not isinstance(required, list) or not all(isinstance(k, str) and k in params['properties'] for k in required):
            errors.append(f'public contract tool {fn.get("name")}: invalid required parameters')
    if len(names) != len(set(names)):
        errors.append('public contract tool names must be unique')
    return errors


def attach(request, contract):
    """Attach caller-owned authority after authoring, so an author cannot rewrite it."""
    errors = problems(contract)
    if errors:
        raise ValueError('; '.join(errors))
    request['public_contract'] = copy.deepcopy(contract)
    request['public_contract_sha256'] = digest(contract)
    request['agent_policy'] = contract['policy']
    request['business_rules'] += ('\n\nPUBLIC BUSINESS CONTRACT (authoritative; preserve policy and API units):\n'
                                  + json.dumps(contract, ensure_ascii=False, indent=1))


def review(client, request, contract):
    """Independent semantic check before spending implementation sessions.

    Public API effects and agent policy are separate obligations. A permissive API
    never authorizes the agent to violate policy; storage adapters are not new rules.
    """
    deterministic = specification_problems(request, contract)
    if deterministic:
        return {'passed': False, 'problems': deterministic, 'contract_sha256': digest(contract),
                'stage': 'declared_constraints'}
    authority = {k: v for k, v in contract.items() if k in ('policy', 'tools', 'notes', 'implementation_notes', 'constraints')}
    candidate = {k: request.get(k) for k in ('business_rules', 'goal_schema', 'time')}
    prompt = ('Audit a generated business specification against its caller-supplied public contract. '
              'Report concrete contradictions, omitted outcome-affecting rules, unsupported eligibility/identity '
              'exceptions, invented fees/deadlines/services, changed units, and extra public API parameters. '
              'Also inspect goal kinds: an inquiry performs NO identification or tool lookup, so it cannot '
              'represent a service that the policy requires authentication for. Private row layout, original '
              'customers, and platform-side consent metadata are allowed adaptations, not new public parameters. '
              'The public policy governs agent conduct; implementation notes describe observable tool effects '
              'and explicitly acknowledged discrepancies. Do not treat a permissive tool as permission for an '
              'agent to violate policy. Do not demand that all tools enforce every agent-side policy rule. '
              'A contradiction remains a failure even if the correct source text is also quoted elsewhere. '
              'Distinguish raw tool return values from goal outcome fields: outcomes grade only the '
              'customer-requested facts and need not reproduce the full nested API response. '
              'A sample task need not exercise every public capability; judge coverage at the world-process level. '
              'Return only JSON {"problems": ["specific finding with the two conflicting requirements"]}. '
              'Return an empty list only when no concrete problem is found. This is an audit, not permission '
              'to rewrite the source or invent missing policy.\n\nPUBLIC CONTRACT:\n'
              + json.dumps(authority, ensure_ascii=False) + '\n\nGENERATED SPECIFICATION:\n'
              + json.dumps(candidate, ensure_ascii=False))
    content = client.chat([{'role': 'user', 'content': prompt}], temperature=0).get('content') or ''
    try:
        found = json.loads(content[content.index('{'):content.rindex('}') + 1])['problems']
        if not isinstance(found, list) or not all(isinstance(p, str) and p.strip() for p in found):
            raise ValueError('invalid problems')
    except (ValueError, KeyError, TypeError):
        return {'passed': False, 'problems': ['public contract audit returned no valid verdict'],
                'contract_sha256': digest(contract), 'stage': 'audit_invalid', 'response_chars': len(content)}
    return {'passed': not found, 'problems': found, 'contract_sha256': digest(contract)}


def specification_problems(request, contract):
    """Caller-declared invariants, without guessing requirements from a domain name.

    Inquiry has a platform-defined meaning: no identity or tool lookup. A public
    policy requiring authentication for every service therefore cannot offer one.
    """
    errors = problems(contract)
    if errors:
        return errors
    if contract.get('constraints', {}).get('allow_unauthenticated_inquiry') is False:
        inquiries = sorted(name for name, process in (request.get('goal_schema') or {}).items()
                           if isinstance(process, dict) and process.get('kind') == 'inquiry')
        if inquiries:
            errors.append(f'public contract prohibits unauthenticated inquiries, but these processes skip '
                          f'identity and lookup: {inquiries}')
    return errors


def interface_problems(tools, request):
    """Exact names/parameters/requiredness and representable JSON types.

    Enums and nested object properties must be documented and checked by behavioral
    conformance cases: the runtime does not currently expose those annotation details.
    """
    contract = request.get('public_contract')
    if contract is None:
        return []
    errors = specification_problems(request, contract)
    if errors:
        return errors
    if request.get('public_contract_sha256') != digest(contract):
        errors.append('public contract content hash mismatch')
    if request.get('agent_policy') != contract['policy']:
        errors.append('agent policy differs from supplied public policy')
    from env_factory.runtime import tool_specs
    expected = {f['name']: f for t in contract['tools'] for f in [t.get('function', t)]}
    actual = {t['name']: t for t in tool_specs(tools, consent=False)}
    if set(expected) != set(actual):
        errors.append(f'public tool names differ: missing={sorted(set(expected)-set(actual))}, '
                      f'extra={sorted(set(actual)-set(expected))}')

    def types(wanted, found, path):
        # Only compare types explicitly present in the source schema. References,
        # enums and object property schemas are not implied to have been verified.
        if 'type' in wanted:
            a, b = wanted['type'], found.get('type')
            a = set(a if isinstance(a, list) else [a])
            b = set(b if isinstance(b, list) else [b])
            if a != b:
                errors.append(f'{path}: JSON types differ: expected {sorted(a)}, got {sorted(b, key=str)}')
        if 'items' in wanted and isinstance(wanted['items'], dict):
            types(wanted['items'], found.get('items', {}), path + '[]')

    for name in sorted(set(expected) & set(actual)):
        want, got = expected[name]['parameters'], actual[name]['parameters']
        if set(want['properties']) != set(got['properties']):
            errors.append(f'{name}: public parameter names differ')
        if set(want.get('required', [])) != set(got.get('required', [])):
            errors.append(f'{name}: required parameters differ')
        for key in set(want['properties']) & set(got['properties']):
            types(want['properties'][key], got['properties'][key], f'{name}.{key}')
    return errors
