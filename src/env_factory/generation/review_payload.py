"""Lossless review-only sharing of repeated JSON subtrees.

The actual task and executable contract remain unchanged. References in this
transport envelope are distinct from business DSL references.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def encode_shared_values(value, *, minimum_size=64):
    counts = Counter()
    info = {}
    keys = set()

    def inspect(node):
        if isinstance(node, dict):
            keys.update(node)
            children = [(key, inspect(child)) for key, child in node.items()]
            size = 2 + max(0, len(children) - 1) + sum(len(_json(k)) + 1 + item[1] for k, item in children)
            fingerprint = 'object:' + _json([(key, item[0]) for key, item in children])
        elif isinstance(node, list):
            children = [inspect(child) for child in node]
            size = 2 + max(0, len(children) - 1) + sum(item[1] for item in children)
            fingerprint = 'array:' + _json([item[0] for item in children])
        else:
            text = _json(node)
            return hashlib.sha256(('scalar:' + text).encode()).hexdigest(), len(text)
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()
        counts[digest] += 1
        info[id(node)] = (digest, size)
        return digest, size

    inspect(value)
    marker = '__review_ref__'
    while marker in keys:
        marker = '_' + marker
    names, definitions = {}, {}

    def content(node):
        if isinstance(node, dict):
            return {key: encode(child) for key, child in node.items()}
        return [encode(child) for child in node]

    def encode(node):
        if not isinstance(node, (dict, list)):
            return node
        digest, size = info[id(node)]
        if counts[digest] > 1 and size >= minimum_size:
            if digest not in names:
                name = 'v' + str(len(names) + 1)
                names[digest] = name
                definitions[name] = content(node)
            return {marker: names[digest]}
        return content(node)

    root = encode(value)
    return {'encoding': 'lossless_json_dag_v1', 'reference_key': marker,
            'root': root, 'shared_values': definitions}


def decode_shared_values(envelope):
    if envelope.get('encoding') != 'lossless_json_dag_v1':
        raise ValueError('Unknown review envelope encoding')
    marker, definitions = envelope['reference_key'], envelope['shared_values']
    cache, active = {}, set()

    def decode(node):
        if isinstance(node, dict):
            if set(node) == {marker}:
                name = node[marker]
                if name not in definitions or name in active:
                    raise ValueError('Missing or cyclic review reference')
                if name not in cache:
                    active.add(name)
                    cache[name] = decode(definitions[name])
                    active.remove(name)
                return copy.deepcopy(cache[name])
            return {key: decode(child) for key, child in node.items()}
        if isinstance(node, list):
            return [decode(child) for child in node]
        return node

    return decode(envelope['root'])


def review_payload(bundle, *, inline_limit=900_000):
    # Whitespace carries no review evidence. Use the same lossless compact
    # representation for ordinary candidates and oversized shared envelopes.
    compact = _json(bundle)
    if len(compact) <= inline_limit:
        return compact, 'json'
    encoded = encode_shared_values(bundle)
    # Review display must preserve every value; never truncate evidence to fit.
    if decode_shared_values(encoded) != bundle:
        raise ValueError('Review transport did not preserve the candidate')
    return _json(encoded), 'lossless_json_dag_v1'
