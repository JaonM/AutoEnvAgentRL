"""Bounded JSON delimiter recovery, preserving every business content token."""
from __future__ import annotations

import json
import re

from env_factory_v2.contracts.contract import require


def content_tokens(raw):
    """Allow delimiter repairs only; retain every key, literal and their order."""
    pattern = r'"(?:[^"\\\x00-\x1f]|\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4}))*"|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?|true|false|null'
    tokens = []
    end = 0
    for match in re.finditer(pattern, raw):
        require(not raw[end:match.start()].strip(' \t\r\n{}[]:,'), 'unsupported JSON content repair')
        tokens.append(match.group())
        end = match.end()
    require(not raw[end:].strip(' \t\r\n{}[]:,'), 'unsupported JSON content repair')
    return tokens


def recover_json(raw, agent):
    """One model call with short unique replacements instead of regeneration."""
    from env_factory_v2.native.code_agent import object_schema
    try:
        return json.loads(raw), None
    except json.JSONDecodeError as error:
        diagnostic = str(error)
    original_tokens = content_tokens(raw)
    schema = object_schema({'replacements': {'type': 'array', 'items': object_schema({
        'old': {'type': 'string'}, 'new': {'type': 'string'}})}})
    value, provenance = agent.ask(
        'Repair ONLY JSON structural punctuation in this malformed generated contract. Preserve every '
        'key, scalar value, literal token and their order exactly. Preserve intended nesting and business '
        'meaning. Return 1..16 short replacements, not the full document. Each old substring must occur '
        'exactly once in the document after preceding replacements; include surrounding unchanged text '
        'as a unique anchor. Fix all syntax errors in this single response. Do not change expectations, '
        'business rules, fields, data, scale or authorization. Do not execute tools.',
        {'malformed_json': raw, 'parse_error': diagnostic}, schema)
    require(1 <= len(value['replacements']) <= 16, 'JSON syntax replacement limit exceeded')
    repaired = raw
    for replacement in value['replacements']:
        old, new = replacement['old'], replacement['new']
        require(old and repaired.count(old) == 1, 'JSON syntax repair anchor must be unique')
        require(content_tokens(old) == content_tokens(new), 'JSON syntax repair changed business content')
        repaired = repaired.replace(old, new, 1)
    require(content_tokens(repaired) == original_tokens, 'JSON syntax repair changed business content')
    parsed = json.loads(repaired)
    return parsed, {'provenance': provenance, 'replacements': value, 'original_parse_error': diagnostic}


