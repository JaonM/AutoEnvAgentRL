"""Stateless sampling keys, independent of worker interleaving and batch shape."""
import hashlib
import json


def seed_for(*identity):
    return int.from_bytes(hashlib.sha256(json.dumps(identity).encode()).digest()[:4], 'big')
