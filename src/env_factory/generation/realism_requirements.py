"""Runner-owned mechanism allocation and executable coverage requirements."""
import copy
import math

MECHANISMS = {'clarification', 'authorization', 'external_change', 'fault_recovery',
              'idempotency', 'compensation'}


def normalize_requirements(value=None):
    value = copy.deepcopy({'version': '1.0', 'mechanisms': []} if value is None else value)
    if (not isinstance(value, dict) or value.get('version') != '1.0'
            or set(value) - {'version', 'mechanisms', 'compensation_tools'}):
        raise ValueError('REALISM_REQUIREMENT_INVALID: expected version/mechanisms')
    names = value.get('mechanisms')
    if (not isinstance(names, list) or any(not isinstance(n, str) for n in names)
            or len(names) != len(set(names)) or set(names) - MECHANISMS):
        raise ValueError('REALISM_REQUIREMENT_INVALID: unknown or duplicate mechanism')
    tools = value.get('compensation_tools', [])
    if (not isinstance(tools, list) or len(tools) > 8
            or any(not isinstance(n, str) or not n for n in tools)
            or len(tools) != len(set(tools))
            or bool(tools) != ('compensation' in names)):
        raise ValueError('REALISM_REQUIREMENT_INVALID: compensation requires named compensation_tools')
    value['mechanisms'] = sorted(names)
    return value


def allocate_realism_requirements(categories, *, available_modes, rng):
    """Allocate 50/30/20 among eligible multi-step slots; do not change task intent.

    Largest-remainder rounding is per invocation and reproducible with its seed.
    Single eligible tasks use weighted sampling so daily runs do not always stay normal.
    Only slots permitting stateful tasks can receive change/recovery requirements.
    """
    result = [normalize_requirements() for _ in categories]
    if 'stateful' not in available_modes:
        return result
    slots = [i for i, category in enumerate(categories) if category == 'multi_step_agentic']
    rng.shuffle(slots)
    raw = [len(slots) * rate for rate in (.5, .3, .2)]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)[:len(slots)-sum(counts)]:
        counts[index] += 1
    if len(slots) == 1:
        counts = [0, 0, 0]
        counts[rng.choices(range(3), weights=(.5, .3, .2), k=1)[0]] = 1
    labels = ([[]] * counts[0] + [['external_change']] * counts[1]
              + [['fault_recovery']] * counts[2])
    for slot, names in zip(slots, labels):
        result[slot] = normalize_requirements({'version': '1.0', 'mechanisms': names})
    return result


def validate_realism_request(source, request):
    if 'realism_profile' in request:
        raise ValueError('REALISM_PROFILE_REMOVED: use realism_requirements')
    required = normalize_requirements(request.get('realism_requirements'))
    declared = normalize_requirements(source.get('realism_requirements'))
    mechanisms = set(required['mechanisms']) | set(declared['mechanisms'])
    dynamics = source.get('business_lifecycle', {}).get('dynamics', {})
    if dynamics.get('events'):
        mechanisms.add('external_change')
    if dynamics.get('faults'):
        mechanisms.add('fault_recovery')
    mode = source['environment_plan']['mode']
    if 'compensation' in mechanisms and mode != 'stateful':
        raise ValueError('REALISM_REQUIREMENT_INAPPLICABLE: compensation requires a stateful environment')
    if 'external_change' in mechanisms and mode not in {'reference_data', 'stateful'}:
        raise ValueError('REALISM_REQUIREMENT_INAPPLICABLE: external changes require business data')
    for mechanism, field in [('external_change', 'events'), ('fault_recovery', 'faults')]:
        if mechanism in mechanisms and not dynamics.get(field):
            raise ValueError('REALISM_REQUIREMENT_MISSING: ' + mechanism + ' must survive repairs')
    stages = [s for v in source.get('interaction_contract', {}).get('variants', []) for s in v.get('stages', [])]
    if 'clarification' in mechanisms and not any(s.get('kind') in {'information_required', 'correction', 'constraint_update'} for s in stages):
        raise ValueError('REALISM_REQUIREMENT_MISSING: clarification requires grounded user stages')
    if 'authorization' in mechanisms and not any(s.get('kind') == 'execution_confirmation' and s.get('approval_terms') for s in stages):
        raise ValueError('REALISM_REQUIREMENT_MISSING: authorization requires bound approval terms')
    rules = source.get('business_lifecycle', {}).get('tools', {})
    if 'idempotency' in mechanisms and not any(r.get('require_idempotency') for r in rules.values()):
        raise ValueError('REALISM_REQUIREMENT_MISSING: idempotency policy required')
    compensation_tools = sorted(set(required.get('compensation_tools', [])) | set(declared.get('compensation_tools', [])))
    writes = {t['tool_name'] for t in source.get('tool_implementations', []) if t['operation'] in {'insert', 'update', 'delete'}}
    if set(compensation_tools) - writes:
        raise ValueError('REALISM_REQUIREMENT_INVALID: compensation tools must be actual writes')
    result = {'version': '1.0', 'mechanisms': sorted(mechanisms)}
    if compensation_tools:
        result['compensation_tools'] = compensation_tools
    return result


def observe_realism(replay, steps, history, *, scripts=()):
    """Record executed changes and a successful tool response after an injected fault.

    Terminal goal correctness is checked separately by the reference reward gate.
    This witnesses recovery execution, not independent semantic correctness.
    """
    changed, recovered = set(), set()
    for event in replay.get('events', []):
        payload, result = event.get('payload', {}), event.get('result', {})
        if (event.get('event') == 'external_business_change' and result.get('updated_count', 0) > 0
                and result.get('before') != result.get('after')):
            changed.add(payload['event_id'])
        if event.get('event') == 'injected_fault':
            for index, (step, response) in enumerate(zip(steps, history)):
                if (step.get('operation') == 'tool_call' and step.get('tool_name') == payload.get('tool')
                        and response.get('status') == payload.get('status')
                        and any(later.get('operation') == 'tool_call' and 200 <= later.get('status', 0) < 300
                                for later in history[index + 1:])):
                    recovered.add(payload['id'])
                    break
    successful_tools = sorted({step.get('tool_name') for step, response in zip(steps, history)
                               if step.get('operation') == 'tool_call' and 200 <= response.get('status', 0) < 300})
    stages = {(script['script_id'], stage['id']): stage
              for script in scripts for stage in script.get('interaction_protocol', {}).get('stages', [])}
    completed = [stages.get((entry.get('body', {}).get('fsm_script_id'), entry.get('body', {}).get('interaction_stage')), {})
                 for entry in history if entry.get('status') == 200 and entry.get('body', {}).get('fsm_transition_applied')]
    return {'clarification_completed': any(stage.get('kind') in {'information_required', 'constraint_update', 'correction'} for stage in completed),
            'authorization_completed': any(stage.get('kind') == 'execution_confirmation' and stage.get('approval_terms') for stage in completed),
            'changed_event_ids': sorted(changed), 'recovered_fault_ids': sorted(recovered),
            'successful_tools': successful_tools,
            'idempotent_replay': any(e.get('event') == 'tool_call_replay' and e.get('payload', {}).get('idempotent_replay') is True for e in replay.get('events', []))}


def require_realism_coverage(artifacts, observations):
    dynamics = artifacts.get('business_lifecycle', {}).get('dynamics', {})
    covered_events = {v for row in observations for v in row['changed_event_ids']}
    covered_faults = {v for row in observations for v in row['recovered_fault_ids']}
    missing_events = {e['id'] for e in dynamics.get('events', [])} - covered_events
    missing_faults = {f['id'] for f in dynamics.get('faults', [])} - covered_faults
    if missing_events or missing_faults:
        raise ValueError('REALISM_EXECUTION_UNCOVERED: declared mechanisms require successful replay; '
                         f'missing_events={sorted(missing_events)}, missing_faults={sorted(missing_faults)}')
    requirements = normalize_requirements(artifacts.get('realism_requirements'))
    executed = {tool for row in observations for tool in row.get('successful_tools', [])}
    if set(requirements.get('compensation_tools', [])) - executed:
        raise ValueError('REALISM_EXECUTION_UNCOVERED: compensation tool never executed successfully')
    if 'idempotency' in requirements['mechanisms'] and not any(row.get('idempotent_replay') for row in observations):
        raise ValueError('REALISM_EXECUTION_UNCOVERED: no executed idempotent replay')

    for mechanism in ('clarification', 'authorization'):
        if mechanism in requirements['mechanisms'] and not any(row.get(mechanism + '_completed') for row in observations):
            raise ValueError('REALISM_EXECUTION_UNCOVERED: required interaction did not complete: ' + mechanism)
