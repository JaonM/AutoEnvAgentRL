"""Static request-envelope checks before model-authored reference execution."""
import json


def _probe_references(source):
    for index, probe in enumerate(source.get('route_probes', [])):
        yield f'route_probes[{index}].reference', probe.get('reference', {})
        variants = probe.get('reference_variants', [])
        if isinstance(variants, list):
            for number, variant in enumerate(variants):
                if isinstance(variant, dict):
                    yield f'route_probes[{index}].reference_variants[{number}]', variant


def validate_reference_requests(source):
    validate_reference_fault_statuses(source)
    tools = {item['function']['name']: item['function'] for item in source['tools']}
    lifecycle = source.get('business_lifecycle', {}).get('tools', {})
    calls = [(f'scenarios[{i}].steps[{j}]', step)
             for i, scenario in enumerate(source['scenarios'])
             for j, step in enumerate(scenario['steps']) if step.get('operation') == 'tool_call']
    calls += [(f'{path}.calls[{j}]', step)
              for path, reference in _probe_references(source)
              for j, step in enumerate(reference.get('calls', []))]
    errors = []
    for path, step in calls:
        # Explicit negative requests retain their expected client rejection.
        status = step.get('expected_status', 200)
        if type(status) is int and 400 <= status < 500:
            continue
        name = step.get('tool_name')
        schema = tools.get(name, {}).get('parameters', {})
        arguments = step.get('arguments', {})
        if ('idempotency_key' in arguments and 'idempotency_key' not in schema.get('properties', {})
                and schema.get('additionalProperties') is False):
            errors.append({'path': path + '.arguments.idempotency_key',
                           'reason': 'Put the transport idempotency_key beside arguments, not inside business arguments.'})
        key = step.get('idempotency_key')
        required = lifecycle.get(name, {}).get('require_idempotency') is True
        if key is None and not required:
            continue
        valid = isinstance(key, str) and bool(key)
        if isinstance(key, dict):
            valid = ((set(key) == {'$ref'} and isinstance(key['$ref'], str) and bool(key['$ref']))
                     or (set(key) == {'$expr'} and isinstance(key['$expr'], dict)))
        if not valid:
            errors.append({'path': path + '.idempotency_key',
                           'reason': 'A nonempty string or captured key is required; reuse the same key only for retries of the same request.'})
    if errors:
        raise ValueError('REFERENCE_REQUEST_INVALID: ' + json.dumps(errors, ensure_ascii=False))


def validate_reference_fault_statuses(source):
    """Check statically known fault attempts; execution remains authoritative.

    Client rejections and captured idempotency keys make later attempt counts
    uncertain, so those paths are left to the runtime rather than guessed.
    """
    faults = source.get('business_lifecycle', {}).get('dynamics', {}).get('faults', [])
    if not faults:
        return
    schedules = {(f.get('tool'), f.get('call_index')): (i, f)
                 for i, f in enumerate(faults)}
    paths = [(f'scenarios[{i}].steps', case['steps'])
             for i, case in enumerate(source.get('scenarios', []))
             if case.get('kind') == 'goal_success']
    paths += [(f'{path}.calls', reference.get('calls', []))
              for path, reference in _probe_references(source)]
    errors = []
    for path, steps in paths:
        counts, committed, uncertain = {}, set(), set()
        for index, step in enumerate(steps):
            if step.get('operation') == 'reset':
                counts.clear(); committed.clear(); uncertain.clear()
                continue
            name = step.get('tool_name')
            if not name:
                continue
            key = step.get('idempotency_key')
            expected = step.get('expected_status', 200)
            if isinstance(key, dict) or not isinstance(expected, int) or 400 <= expected < 500:
                uncertain.add(name)
                continue
            if name in uncertain:
                continue
            if key and key in committed:
                continue
            counts[name] = counts.get(name, 0) + 1
            scheduled = schedules.get((name, counts[name]))
            fault = scheduled[1] if scheduled else None
            if fault and 200 <= expected < 300:
                errors.append({'path': f'{path}[{index}].expected_status',
                    'fault_path': f'business_lifecycle.dynamics.faults[{scheduled[0]}]',
                    'tool': name, 'call_index': counts[name], 'expected_status': expected,
                    'injected_status': fault.get('status'),
                    'reason': 'This reference expects success on a declared fault attempt. '
                              'Declare the intended fault response and retain an executable recovery path.'})
            if key and ((fault and fault.get('phase') == 'after_commit')
                        or (not fault and 200 <= expected < 300)):
                committed.add(key)
    if errors:
        raise ValueError('REFERENCE_FAULT_STATUS_CONFLICT: ' + json.dumps(errors, ensure_ascii=False))
