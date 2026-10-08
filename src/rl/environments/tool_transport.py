"""Classify authored, bounded tool faults without hiding infrastructure failures."""


def retry_tool(tools):
    """Native call IDs stay unique; a separate handle identifies a replay."""
    occupied = {tool['function']['name'] for tool in tools}
    name = 'retry_tool_call'
    while name in occupied:
        name = '_' + name
    return {'type': 'function', 'function': {'name': name,
        'description': '使用业务工具响应中的 retry_request_id 重试同一请求。沿用原工具、原参数和幂等键；'
                       '已提交的请求返回原结果，不重复写入。未提交的请求重新尝试，仍检查当前权限和业务约束。'
                       '只能重试本回合已发出的业务请求；修改参数请重新调用原业务工具。',
        'parameters': {'type': 'object', 'properties': {'request_id': {'type': 'string', 'minLength': 1,
            'description': '业务工具响应中的 retry_request_id'}},
            'required': ['request_id'], 'additionalProperties': False}}}

def is_authored_tool_fault(task, path, status, response):
    prefix='/v1/tools/'
    if not path.startswith(prefix) or not isinstance(response, dict):
        return False
    error=response.get('error')
    if not isinstance(error, dict):
        return False
    code=error.get('code')
    expected_phase={'TRANSIENT_FAILURE':'before','RESPONSE_LOST':'after_commit'}.get(code)
    if expected_phase is None:
        return False
    faults=task.get('business_lifecycle', {}).get('dynamics', {}).get('faults', [])
    return any(f.get('tool')==path[len(prefix):] and f.get('status')==status
               and f.get('phase')==expected_phase for f in faults)


def is_business_conflict(path, status, response):
    return (path.startswith('/v1/tools/') and status==409 and isinstance(response, dict)
            and isinstance(response.get('error'), dict)
            and response['error'].get('code') in {'INVALID_TRANSITION','TRANSITION_TARGET_INVALID',
                'USER_INTERACTION_REQUIRED','USER_INPUT_MISMATCH','USER_REQUEST_STOPPED',
                'BUSINESS_PRECONDITION_FAILED','IDEMPOTENCY_CONFLICT','PAGINATION_CURSOR_STALE'})
