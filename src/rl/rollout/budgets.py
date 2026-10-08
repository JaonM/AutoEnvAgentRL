"""Exact admission bounds plus tokenizer-specific reference recommendations.

Reference text is used only by encode(), never by sample()/iter_sample()/decoder.
The recommendation is not a proof that a shorter batched policy cannot succeed.
"""
import copy
import json


class TaskBudgetError(ValueError):
    """Non-retryable configuration error, not a scored policy failure."""


def execution_budget(environment):
    task = getattr(environment, 'task', {})
    spec = task.get('task_spec', {})
    if spec.get('difficulty_requirements', {}).get('profile', 'standard') == 'standard':
        return None
    pipeline = task.get('artifacts', {}).get('generation_pipeline', task.get('generation_pipeline', {}))
    return {'contract': spec.get('execution_budget'),
        'references': pipeline.get('verification', {}).get('policy_budget_references', []),
        'dialogue_tool': (getattr(environment, 'user_interaction_tool', None) or {}).get('function', {}).get('name'),
        'retry_tool': (getattr(environment, 'retry_tool', None) or {}).get('function', {}).get('name')}


def _reference_actions(reference, budget):
    seen = set()
    for index, row in enumerate(reference['records']):
        native_id = f'call-{index:032x}'
        if row['kind'] == 'answer':
            yield {'role': 'assistant', 'content': row['content']}, []
            continue
        if row['kind'] == 'dialogue':
            try:
                envelope = json.loads(row['content'])
            except (ValueError, TypeError):
                envelope = None
            if not budget.get('dialogue_tool') or not isinstance(envelope, dict) or 'request' not in envelope:
                yield {'role': 'assistant', 'content': row['content']}, [{'role': 'user', 'content': row['user_query']}]
                continue
            request = envelope['request']
            name = budget['dialogue_tool']
            arguments = {'question': envelope['question'], 'business_tool': request['tool'],
                         'requested_fields': request['arguments'], 'intent': request.get('intent', 'request_input')}
            arguments.update({key: request[key] for key in ('options', 'context', 'proposal') if key in request})
            response = {'status': 200, 'tool_result': {'user_query': row['user_query'], 'should_end': row['should_end']}}
        elif row['kind'] == 'tool':
            name, arguments = row['name'], row['arguments']
            group = row['request_group']
            handle = f'request-{group:032x}'
            response = {'status': row['status'], 'tool_result': row['result']}
            if budget.get('retry_tool'):
                response['retry_request_id'] = handle
                if group in seen:
                    name, arguments = budget['retry_tool'], {'request_id': handle}
                    response['retried_tool'] = row['name']
                seen.add(group)
        else:
            raise TaskBudgetError('unsupported reference sizing operation')
        yield {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'id': native_id,
            'function': {'name': name, 'arguments': json.dumps(arguments, ensure_ascii=False)}}]}, [
                {'role': 'tool', 'name': name, 'tool_call_id': native_id, 'content': json.dumps(response, ensure_ascii=False)}]


def admit(policy, messages, tool_options, config, budget):
    if budget is None:
        return None
    contract = budget.get('contract')
    references = budget.get('references', [])
    if (not isinstance(contract, dict) or contract.get('version') != '1.0' or not references
            or contract.get('reference_count') != len(references)
            or type(contract.get('minimum_model_actions')) is not int or contract['minimum_model_actions'] < 1):
        raise TaskBudgetError('TASK_BUDGET_EVIDENCE_MISSING: rebuild the hard-profile task with verified sizing evidence')
    configured = {key: config[key] for key in ('max_steps', 'max_tokens', 'max_context')}
    if config['max_steps'] < contract['minimum_model_actions']:
        raise TaskBudgetError(f"TASK_ACTION_BUDGET_INCOMPATIBLE: {contract['profile']} needs capacity for at least "
                             f"{contract['minimum_model_actions']} model actions; max_steps={config['max_steps']}")
    current_tokens = len(policy.encode(messages, **tool_options))
    if current_tokens + config['max_tokens'] > config['max_context']:
        raise TaskBudgetError(f"TASK_CONTEXT_BUDGET_INCOMPATIBLE: current prompt={current_tokens} tokens + "
                             f"max_tokens={config['max_tokens']} exceeds max_context={config['max_context']}")
    initial = []
    for message in messages:
        if message.get('role') == 'assistant':
            break
        initial.append(copy.deepcopy(message))
    peak, action_size, actions = 0, 0, 0
    for reference in references:
        projected = copy.deepcopy(initial)
        count = 0
        for action, replies in _reference_actions(reference, budget):
            before = len(policy.encode(projected, **tool_options))
            peak = max(peak, before)
            projected.append(action)
            after = len(policy.encode(projected, **tool_options))
            action_size = max(action_size, after - before)
            projected.extend(replies)
            count += 1
        actions = max(actions, count)
    # Template delta includes a fresh generation prefix. Extra margin accounts
    # for transport identifiers; thinking tokens are policy-dependent and not
    # supplied by a declarative business reference.
    suggested_output = max(1, action_size) + 32
    return {'version': '1.0', 'profile': contract['profile'], 'configured': configured,
        'minimum_model_actions': contract['minimum_model_actions'], 'current_prompt_tokens': current_tokens,
        'reference_count': len(references), 'reference_serial_actions': actions,
        'reference_peak_prompt_tokens': peak, 'reference_formatted_action_token_delta': action_size,
        'recommended_max_steps': max(actions, contract['minimum_model_actions']),
        'recommended_max_tokens': suggested_output,
        'recommended_max_context': peak + max(config['max_tokens'], suggested_output),
        'reference_capacity_fits': config['max_steps'] >= actions and config['max_tokens'] >= suggested_output
            and config['max_context'] >= peak + config['max_tokens'],
        'scope': 'Tokenizer-specific projection of executed references with serial calls; recommendations, '
                 'not universal minimums or an allowance for unbounded reasoning. Reference text is never sampled.'}
