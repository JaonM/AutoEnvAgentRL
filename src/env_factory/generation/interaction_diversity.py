"""Describe user-script behavior separately from wording and execution coverage."""
import hashlib
import json


def _signature(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def executed_behavior_signature(events, scripts):
    """Fingerprint executed reference actions, excluding wording and arbitrary IDs."""
    protocols = [event for event in events if event.get('event') == 'user_protocol']
    identity = protocols[-1].get('payload', {}).get('script_id') if protocols else None
    script = next((item for item in scripts if item['script_id'] == identity), None)
    if script is None:
        return None
    stages = script['interaction_protocol']['stages']
    sequence = []
    for event in events:
        if event.get('event') == 'tool_call':
            sequence.append({'tool': event.get('payload', {}).get('tool_name'),
                             'arguments': event.get('payload', {}).get('arguments')})
        elif event.get('event') == 'user_turn' and event.get('result', {}).get('fsm_transition_applied'):
            result = event['result']
            stage = next((item for item in stages if item['id'] == result.get('interaction_stage')), None)
            before = result.get('fsm_state_before', '')
            if stage is None and isinstance(before, str) and before.startswith('stage-'):
                position = before.removeprefix('stage-')
                if position.isdigit() and int(position) < len(stages):
                    stage = stages[int(position)]
            if stage is None:
                continue
            behavior = {key: stage[key] for key in ('kind', 'before_tool', 'bind_to', 'private_fact',
                'previous_fact', 'approval_terms', 'approval_scope', 'grant_permissions', 'revoke_permissions') if key in stage}
            branch = next((item for item in stage.get('branches', []) if item['id'] == result.get('interaction_branch')), None)
            if branch:
                behavior['executed_branch'] = {key: branch[key] for key in (
                    'when', 'advance', 'outcome', 'grant_permissions', 'revoke_permissions', 'approval_terms') if key in branch}
            sequence.append(behavior)
    return _signature(sequence)


def interaction_diversity(scripts, scenarios):
    """Report design/compiled-reference diversity, not successful rollout coverage."""
    variants = []
    for script in scripts:
        stages = script.get('interaction_protocol', {}).get('stages', [])
        if not stages:
            continue
        decisions = []
        behaviors = []
        for stage in stages:
            decisions.append({key: stage[key] for key in (
                'before_tool', 'bind_to', 'private_fact', 'approval_terms',
                'approval_scope', 'grant_permissions', 'revoke_permissions') if key in stage})
            behavior = {key: stage[key] for key in (
                'kind', 'before_tool', 'bind_to', 'private_fact', 'previous_fact',
                'requires_tools', 'approval_terms', 'approval_scope',
                'grant_permissions', 'revoke_permissions') if key in stage}
            behavior['branches'] = [{key: branch[key] for key in (
                'when', 'advance', 'outcome', 'grant_permissions', 'revoke_permissions',
                'approval_terms') if key in branch} for branch in stage.get('branches', [])]
            behaviors.append(behavior)
        reference_signatures = []
        reference_ids = []
        for case in scenarios:
            if case.get('kind') != 'goal_success' or not case['scenario_id'].endswith('-' + script['script_id']):
                continue
            calls = [{key: step[key] for key in ('tool_name', 'arguments', 'expected_status') if key in step}
                     for step in case['steps'] if step.get('operation') == 'tool_call']
            reference_signatures.append(_signature(calls))
            reference_ids.append(case['scenario_id'])
        variants.append({'script_id': script['script_id'],
            'decision_signature': _signature(decisions), 'behavior_signature': _signature(behaviors),
            'compiled_reference_signatures': sorted(set(reference_signatures)),
            'reference_ids': reference_ids})
    decision_count = len({v['decision_signature'] for v in variants})
    behavior_count = len({v['behavior_signature'] for v in variants})
    reference_count = len({signature for v in variants for signature in v['compiled_reference_signatures']})
    return {'version': '1.0', 'scope': 'declared_behavior_and_compiled_references_not_rollout_coverage',
        'variant_count': len(variants), 'distinct_decisions': decision_count,
        'distinct_behaviors': behavior_count, 'distinct_tool_call_sequences': reference_count,
        'wording_only_variants': len(variants) > 1 and behavior_count == 1,
        'same_terminal_decisions': len(variants) > 1 and decision_count == 1,
        'variants_without_reference': [v['script_id'] for v in variants if not v['reference_ids']],
        'variants': variants}
