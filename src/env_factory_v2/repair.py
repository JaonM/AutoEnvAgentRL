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
    try:
        for edit in edits:
            json.loads(edit['value_json'])
        return edits, None
    except json.JSONDecodeError:
        from .json_syntax import recover_json
        # Paths and operations stay outside recovery. All scalar tokens in all
        # values must survive, including values after the first broken edit.
        values, evidence = recover_json('[' + ','.join(e['value_json'] for e in edits) + ']', agent)
        require(isinstance(values, list) and len(values) == len(edits),
                'JSON syntax repair changed edit value count')
        return [{**edit, 'value_json': canonical(value)} for edit, value in zip(edits, values)], evidence


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
