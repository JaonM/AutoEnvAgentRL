"""Declared dialogue constraints for demonstration quality, independent of business reward.

Never repair a transcript after generation: preserve the actual assistant turns and
report violations separately from the environment's grade.
"""


def constraints(request):
    return (request.get('public_contract') or {}).get('constraints', {})


def provider_view(messages, rules):
    """Reinforce caller-declared public policy for teachers without changing saved turns."""
    reminders = []
    if rules.get('max_tool_calls_per_turn') is not None:
        reminders.append(f"Make at most {rules['max_tool_calls_per_turn']} tool call(s) in each assistant turn; "
                         'wait for their results before making another call.')
    if rules.get('allow_text_with_tool_calls') is False:
        reminders.append('A turn with tool calls must have empty content. Send user-facing speech in a separate turn.')
    if not reminders:
        return messages
    return [{**messages[0], 'content': messages[0]['content'] + '\n\nPublic dialogue policy reminder: '
             + ' '.join(reminders)}, *messages[1:]]


def audit(messages, rules):
    violations = []
    for index, message in enumerate(messages):
        if message.get('role') != 'assistant':
            continue
        calls = message.get('tool_calls') or []
        maximum = rules.get('max_tool_calls_per_turn')
        if maximum is not None and len(calls) > maximum:
            violations.append({'message_index': index, 'rule': 'max_tool_calls_per_turn',
                               'actual': len(calls), 'limit': maximum})
        if calls and rules.get('allow_text_with_tool_calls') is False and (message.get('content') or '').strip():
            violations.append({'message_index': index, 'rule': 'allow_text_with_tool_calls'})
    return {'passed': not violations, 'constraints': dict(rules), 'violations': violations}


def qualify(result, rules):
    audit_result = audit(result['messages'], rules)
    return {**result, 'trajectory_policy': audit_result,
            'policy_aware_success': result.get('success') is True and audit_result['passed']
            and result.get('valid', True) is not False}
