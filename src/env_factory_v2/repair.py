"""Atomic data-only contract edits; every result still requires full qualification."""
from __future__ import annotations

import copy
import json
import re

from .contract import require
from .expressions import canonical


def recover_edit_values(edits, agent):
    """Recover delimiters in all edit values with at most one native call."""
    require(isinstance(edits, list) and 1 <= len(edits) <= 64, 'repair requires 1..64 edits')
    for edit in edits:
        require(set(edit) == {'op', 'path', 'value_json'} and isinstance(edit['value_json'], str),
                'invalid repair edit fields')
    broken = []
    from .json_syntax import content_tokens
    for index, edit in enumerate(edits):
        try:
            json.loads(edit['value_json'])
        except json.JSONDecodeError as exc:
            content_tokens(edit['value_json'])
            broken.append({'index': index, 'json': edit['value_json'], 'error': str(exc)})
    if not broken:
        return edits, None
    from .code_agent import object_schema
    value, provenance = agent.ask(
        'Repair ONLY JSON structural punctuation in each malformed edit value. Return the complete '
        'corrected JSON string for each requested index exactly once. Preserve every key, scalar token '
        'and their order, intended nesting and business meaning. Do not change rules, values or labels. '
        'Do not execute tools.', {'broken_values': broken}, object_schema({'repairs': {
            'type': 'array', 'items': object_schema({'index': {'type': 'integer'}, 'json': {'type': 'string'}})}}))
    expected = {entry['index']: entry['json'] for entry in broken}
    result = copy.deepcopy(edits)
    seen = set()
    for entry in value['repairs']:
        index = entry['index']
        require(type(index) is int and index in expected and index not in seen,
                'JSON syntax repair must preserve requested edit indices exactly once')
        seen.add(index)
        require(content_tokens(entry['json']) == content_tokens(expected[index]),
                'JSON syntax repair changed business content')
        canonical(json.loads(entry['json']))
        result[index]['value_json'] = entry['json']
    require(seen == set(expected), 'JSON syntax repair omitted edit values')
    return result, {'provenance': provenance, 'original_values': broken, 'repairs': value}


def apply_edits(source, edits):
    require(isinstance(edits, list) and 1 <= len(edits) <= 64, 'repair requires 1..64 edits')
    result = copy.deepcopy(source)
    for edit in edits:
        require(set(edit) == {'op', 'path', 'value_json'}, 'invalid repair edit fields')
        op, pointer = edit['op'], edit['path']
        require(op in {'add', 'replace', 'remove'} and isinstance(pointer, str)
                and pointer.startswith('/') and len(pointer) > 1, 'invalid repair operation/path')
        require(not re.search(r'~(?![01])', pointer), 'invalid JSON pointer escape')
        keys = [part.replace('~1', '/').replace('~0', '~') for part in pointer[1:].split('/')]
        node = result
        for key in keys[:-1]:
            require(isinstance(node, (list, dict)), 'repair path traverses scalar')
            if isinstance(node, list):
                require(key.isdecimal() and str(int(key)) == key and int(key) < len(node), 'repair array index invalid')
                node = node[int(key)]
            else:
                require(key in node, 'repair parent does not exist')
                node = node[key]
        require(isinstance(node, (list, dict)), 'repair parent is scalar')
        key = keys[-1]
        value = json.loads(edit['value_json'])
        canonical(value)  # Reject non-finite values even where the JSON decoder accepts them.
        if isinstance(node, list):
            if key == '-':
                require(op == 'add', 'append requires add')
                index = len(node)
            else:
                require(key.isdecimal() and str(int(key)) == key, 'repair array index invalid')
                index = int(key)
                require(index < len(node) or (op == 'add' and index == len(node)), 'repair array index out of bounds')
            if op == 'add':
                node.insert(index, value)
            elif op == 'replace':
                node[index] = value
            else:
                node.pop(index)
        elif op == 'add':
            require(key not in node, 'add cannot overwrite existing field; use replace')
            node[key] = value
        else:
            require(key in node, 'repair field does not exist')
            if op == 'replace':
                node[key] = value
            else:
                del node[key]
    require(isinstance(result, dict), 'repair must preserve contract object')
    return result
