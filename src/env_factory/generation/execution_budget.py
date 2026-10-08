"""Trainer-only sizing evidence from actually executed successful references."""
import copy

from env_factory.sandbox_runtime import AcceptanceScenarioRunner, DeclarativeMetricEvaluator


def reference_records(scenario, history):
    if len(scenario['steps']) != len(history):
        raise ValueError('BUDGET_EVIDENCE_INVALID: incomplete reference history')
    records, variables, tools, groups = [], {}, [], {}
    for step, response in zip(scenario['steps'], history):
        operation = step['operation']
        if response['operation'] != operation or response['status'] != step.get('expected_status', 200):
            raise ValueError('BUDGET_EVIDENCE_INVALID: reference response mismatch')
        if operation == 'reset':
            records, variables, tools, groups = [], {}, [], {}
        elif operation == 'tool_call':
            arguments = DeclarativeMetricEvaluator.resolve_capture_argument(step.get('arguments', {}), variables)
            key = DeclarativeMetricEvaluator.resolve_capture_argument(step.get('idempotency_key'), variables)
            group = groups.setdefault(key, len(records)) if key else len(records)
            records.append({'kind': 'tool', 'name': step['tool_name'], 'arguments': arguments,
                'request_group': group, 'status': response['status'], 'result': copy.deepcopy(response['body'])})
            if 200 <= response['status'] < 300:
                tools.append({'event': 'tool_call', 'payload': {'tool_name': step['tool_name'], 'arguments': arguments},
                              'result': response['body']})
        elif operation == 'dialogue_turn':
            content = AcceptanceScenarioRunner.resolve_dialogue_content(step['content'], variables)
            if 'observed_options_from' in step or 'observed_context_from' in step:
                content = AcceptanceScenarioRunner.render_observed_dialogue(content, step.get('observed_options_from'), tools,
                    context_binding=step.get('observed_context_from'))
            records.append({'kind': 'dialogue', 'content': content,
                'user_query': response['body']['user_query'], 'should_end': bool(response['body'].get('should_end'))})
        elif operation == 'agent_response':
            records.append({'kind': 'answer', 'content': str(DeclarativeMetricEvaluator.resolve_capture_argument(
                step.get('content', ''), variables))})
        for name, path in step.get('capture', {}).items():
            variables[name] = DeclarativeMetricEvaluator._resolve(response.get('body'), path)
    return {'scenario_id': scenario['scenario_id'], 'records': records}


def compile_budget(task, references):
    profile = task['task_spec']['difficulty_requirements']['profile']
    if profile == 'standard':
        return
    if not references:
        raise ValueError('BUDGET_EVIDENCE_INVALID: hard profile needs executed reference sizing')
    required = task['task_spec']['difficulty_requirements']
    # Native actions may contain several ordered calls. Neither tool DAG depth
    # nor serial reference length proves a minimum number of model actions.
    task['task_spec']['execution_budget'] = {
        'version': '1.0', 'profile': profile, 'minimum_model_actions': 1,
        'business_dependency_depth': required.get('min_dependency_depth'),
        'complete_pages': required.get('min_complete_pages'),
        'reference_serial_actions': max(len(row['records']) for row in references),
        'reference_count': len(references),
        'basis': 'Native actions can contain multiple ordered calls. Dependency depth, page count and '
                 'serial reference capacity are sizing evidence, not proven minimum model-action counts.',
        'reference_visibility': 'trainer_only_generation_verification'}
