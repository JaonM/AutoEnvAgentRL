"""Caller-owned task difficulty requirements, measured on execution rather than prose.

These checks accept/reject generated exercises; they never replace the business
verifier or add requirements to student evaluation. No domain names are special.
"""
from __future__ import annotations

import copy

from env_factory.certify.gates import trace_structure
from env_factory.runtime import canonical, run_reference


COUNTS = ('min_goals', 'min_tool_calls', 'max_tool_calls', 'min_writes', 'max_writes')
KEYS = {*COUNTS, 'state_effect', 'required_tools', 'result_dependency', 'sop_method'}


def validate(value):
    if not isinstance(value, dict):
        raise ValueError('task requirements must be an object')
    unknown = set(value) - KEYS
    if 'sop_method' in value and (not isinstance(value['sop_method'], str) or not value['sop_method']):
        raise ValueError('sop_method must be a nonempty string')
    if unknown:
        raise ValueError(f'unknown task requirements: {sorted(unknown)}')
    for key in COUNTS:
        if key in value and (type(value[key]) is not int or value[key] < 0):
            raise ValueError(f'{key} must be a nonnegative integer')
    for quantity in ('tool_calls', 'writes'):
        if value.get('min_' + quantity, 0) > value.get('max_' + quantity, float('inf')):
            raise ValueError(f'inconsistent bounds for {quantity}')
    if value.get('state_effect', 'any') not in ('any', 'changed', 'unchanged'):
        raise ValueError('state_effect must be any, changed or unchanged')
    if 'result_dependency' in value and type(value['result_dependency']) is not bool:
        raise ValueError('result_dependency must be boolean')
    required = value.get('required_tools', [])
    if not isinstance(required, list) or any(not isinstance(n, str) or not n for n in required):
        raise ValueError('required_tools must be a list of tool names')
    return copy.deepcopy(value)


def profile(value):
    """Validate the whole configuration before any author call (including unused shapes)."""
    if not isinstance(value, dict) or set(value) - {'default', 'by_shape'}:
        raise ValueError('task profile supports only default and by_shape')
    validate(value.get('default', {}))
    shapes = value.get('by_shape', {})
    if not isinstance(shapes, dict) or set(shapes) - {
            'direct', 'simple', 'ordinary', 'single_write', 'complex', 'subjective'}:
        raise ValueError('invalid task profile shapes')
    for requirements in shapes.values():
        validate({**value.get('default', {}), **validate(requirements)})
    return copy.deepcopy(value)


def select(value, shape):
    return validate({**value.get('default', {}), **value.get('by_shape', {}).get(shape, {})})


def measure(result, request):
    # Attempted/rejected writes must not satisfy a successful-write requirement.
    successful = [e for e in result['trace'] if e.get('kind') == 'tool' and e.get('status') == 'ok']
    structure = trace_structure(successful, request['user_facts'])
    return {**structure, 'goals': len(request['user_facts'].get('goals') or []),
            'state_changed': canonical(result['final_state']) != canonical(request['tables']),
            'tools': sorted({e['tool'] for e in successful})}


def evaluate(env, request, requirements):
    requirements = validate(requirements)
    result = run_reference(env, request['tables'], request['user_facts'], start=request['time']['start'], request=request)
    actual = measure(result, request)
    problems = []
    if requirements.get('sop_method') and request.get('sop_method') != requirements['sop_method']:
        problems.append('assigned SOP method differs from the executed task')
    for key in COUNTS:
        if key not in requirements:
            continue
        bound, quantity = key.split('_', 1)
        if (bound == 'min' and actual[quantity] < requirements[key]) or \
                (bound == 'max' and actual[quantity] > requirements[key]):
            problems.append(f'{key}={requirements[key]}, observed {actual[quantity]}')
    effect = requirements.get('state_effect', 'any')
    if effect != 'any' and actual['state_changed'] != (effect == 'changed'):
        problems.append(f'state_effect={effect}, observed state_changed={actual["state_changed"]}')
    missing = set(requirements.get('required_tools', [])) - set(actual['tools'])
    if missing:
        problems.append(f'missing successful required tools: {sorted(missing)}')
    if 'result_dependency' in requirements and actual['result_dependency'] != requirements['result_dependency']:
        problems.append('reference result dependency differs from requirement')
    return {'passed': not problems, 'requirements': requirements, 'actual': actual, 'problems': problems}


def coverage(rows, policies=None):
    """Report achieved coverage separately from requested labels and model acceptance."""
    groups = {}
    for row in rows:
        name = row.get('requested_shape') or 'probe_or_unspecified'
        group = groups.setdefault(name, {'attempted': 0, 'certified': 0, 'natural_passed': 0,
                                         'measured': 0, 'state_changes': 0, 'successful_writes': [],
                                         'assigned_processes': set()})
        group['attempted'] += 1
        group['certified'] += bool(row.get('passed'))
        group['natural_passed'] += bool(row.get('passed') and (policies or {}).get(row['id']) == 'passed')
        group['assigned_processes'].update(row.get('assigned_processes', []))
        actual = (row.get('task_quality') or {}).get('actual')
        if actual is not None:
            group['measured'] += 1
            group['state_changes'] += actual['state_changed']
            group['successful_writes'].append(actual['writes'])
    for group in groups.values():
        group['assigned_processes'] = sorted(group['assigned_processes'])
    return {'by_requested_shape': groups, 'natural_acceptance_run': policies is not None,
            'limits': ['Reference execution complexity is not measured model difficulty.',
                       'Result dependency is value-flow evidence, not a proof of causal goal dependency.',
                       'Policy-boundary coverage needs independently defined domain cases.']}
