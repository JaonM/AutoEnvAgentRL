"""Policy-visible sandbox clock; never substitutes host time."""
from datetime import datetime
from zoneinfo import ZoneInfo

from rl.errors import InfrastructureError


def format_current_time(value):
    try:
        current = datetime.fromisoformat(value['virtual_datetime'])
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError('timezone required')
    except (KeyError, ValueError, TypeError) as exc:
        raise InfrastructureError('sandbox returned an invalid virtual datetime') from exc
    return current.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')


def clock_tool(tools):
    occupied = {tool['function']['name'] for tool in tools}
    name = 'get_current_time'
    while name in occupied:
        name = '_' + name
    return {'type': 'function', 'function': {'name': name,
        'description': '读取沙箱虚拟日期时间（ISO 8601 UTC）、tick 和累计秒数。只读，不推进时间；不读取真实世界时间。',
        'parameters': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': False}}}
