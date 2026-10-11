"""Closed recursive wire grammar for arbitrary JSON, independent of business DSL.

Objects use key/value entries so dynamic business keys never require an open
JSON-schema object. Decoding preserves values and rejects duplicate keys.
"""
import json
import math

from env_factory_v2.contracts.contract import require


def closed(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


NODE_REF = {'$ref': '#/$defs/node'}
TREE_SCHEMA = {**closed({'source': {'$ref': '#/$defs/object'}}), '$defs': {
    'object': closed({'o': {'type': 'array', 'items': closed({'k': {'type': 'string'}, 'v': NODE_REF})}}),
    'node': {'anyOf': [
        {'$ref': '#/$defs/object'}, {'type': 'array', 'items': NODE_REF},
        {'type': 'string'}, {'type': 'number'}, {'type': 'boolean'}, {'type': 'null'},
    ]},
}}


def decode_response(value):
    require(isinstance(value, dict) and set(value) == {'source'}, 'JSON tree envelope required')
    nodes = [0]
    def decode(node, depth=0):
        nodes[0] += 1
        require(depth <= 128 and nodes[0] <= 2_000_000, 'JSON tree complexity limit')
        if isinstance(node, dict):
            require(set(node) == {'o'}, 'JSON tree object must use the object tag')
            payload = node['o']
            require(isinstance(payload, list), 'JSON tree object entries must be a list')
            result = {}
            for entry in payload:
                require(isinstance(entry, dict) and set(entry) == {'k', 'v'}
                        and isinstance(entry['k'], str), 'invalid JSON tree object entry')
                require(entry['k'] not in result, 'duplicate JSON tree object key')
                result[entry['k']] = decode(entry['v'], depth + 1)
            return result
        if isinstance(node, list):
            return [decode(child, depth + 1) for child in node]
        valid = (isinstance(node, str) or type(node) in {int, bool} or node is None
                 or type(node) is float and math.isfinite(node))
        require(valid, 'invalid JSON tree scalar')
        return node
    require(isinstance(value['source'], dict) and set(value['source']) == {'o'},
            'JSON tree source must be an object')
    source = decode(value['source'])
    require(len(json.dumps(source, ensure_ascii=False, allow_nan=False).encode()) <= 4 * 1024 * 1024,
            'decoded contract size limit')
    return source


def encode_response(source):
    """Encode fixtures for diagnostics and prove lossless transport round trips."""
    def encode(value, depth=0):
        require(depth <= 128, 'JSON tree complexity limit')
        if isinstance(value, dict):
            require(all(isinstance(key, str) for key in value), 'JSON object keys must be strings')
            return {'o': [{'k': key, 'v': encode(item, depth + 1)} for key, item in value.items()]}
        if isinstance(value, list):
            return [encode(item, depth + 1) for item in value]
        if isinstance(value, str):
            return value
        if type(value) is bool:
            return value
        if type(value) in {int, float}:
            require(type(value) is int or math.isfinite(value), 'JSON tree numbers must be finite')
            return value
        require(value is None, 'unsupported JSON tree value')
        return None
    result = {'source': encode(source)}
    decode_response(result)
    return result
