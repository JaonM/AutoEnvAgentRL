"""Classify authored, bounded tool faults without hiding infrastructure failures."""

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
                'USER_INTERACTION_REQUIRED','IDEMPOTENCY_CONFLICT'})
