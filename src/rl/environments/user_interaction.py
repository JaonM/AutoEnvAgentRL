"""Native policy tool for the sandbox's public structured dialogue protocol."""
import json
import copy


def interaction_tool(public, tools):
    if not any(isinstance(item, dict) and item.get('name') == 'user_interaction_format'
               for item in public.get('materials', [])):
        return None
    names = {tool['function']['name'] for tool in tools}
    name = 'request_user_input'
    while name in names:
        name = '_' + name
    return {'type': 'function', 'function': {
        'name': name,
        'description': (
            '向用户询问缺失信息、展示选项或请求执行授权。中途询问优先使用此工具，'
            '无需手写交互 JSON。business_tool 填后续业务工具名；requested_fields 只列需要向用户获取的'
            '参数名，不包含已知参数或参数值。选择题的 options 必须来自实际工具返回。'
            '此工具不会执行业务操作；获得用户回复后再调用业务工具。'),
        'parameters': {'type': 'object', 'additionalProperties': False,
            'properties': {
                'question': {'type': 'string', 'minLength': 1},
                'business_tool': {'type': 'string', 'enum': sorted(names)},
                'requested_fields': {'type': 'array', 'items': {'type': 'string'}, 'uniqueItems': True},
                'intent': {'type': 'string', 'enum': ['request_input', 'confirm_execution']},
                'options': {'type': 'array', 'items': {}},
            }, 'required': ['question', 'business_tool', 'requested_fields']}}}


def policy_public_input(public, native):
    """Describe the native interface without exposing or changing hidden task state."""
    result = copy.deepcopy(public)
    if native is not None:
        name = native['function']['name']
        instruction = (
            f'For ALL intermediate questions or permission requests, call the {name} tool. '
            'Do not send a plain-text question or hand-written interaction JSON as assistant content. '
            'The tool sends the question and returns the user reply. Supply only missing parameter names '
            'in requested_fields, never known parameters or guessed values. After the reply, continue with '
            'business tools. Direct assistant content is for the final task answer.')
        for item in result.get('materials', []):
            if isinstance(item, dict) and item.get('name') == 'user_interaction_format':
                item['content'] = instruction
    return result


def interaction_content(arguments, schema):
    from env_factory.sandbox_runtime import validate_json_schema
    validate_json_schema(schema, arguments, 'user_interaction')
    # Preserve restored trajectories created with the original adapter schema.
    modern = 'business_tool' in schema.get('properties', {})
    request = {'tool': arguments['business_tool' if modern else 'tool'],
               'arguments': arguments['requested_fields' if modern else 'arguments']}
    if 'options' in arguments:
        request['options'] = arguments['options']
    request['intent'] = arguments.get('intent', 'request_input')
    return json.dumps({'question': arguments['question'], 'request': request}, ensure_ascii=False)
