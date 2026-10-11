"""Generation transport shape, deliberately separate from business schemas.

Dynamic contract fields are checked by compile_contract, not this envelope.
This format does not claim strict decoder enforcement for open-ended objects.
"""
import json

from env_factory_v2.contracts.contract import require


SOURCE_OBJECT_SCHEMA = {'type': 'object', 'properties': {
    'source': {'type': 'object', 'properties': {}, 'required': [], 'additionalProperties': True}},
    'required': ['source'], 'additionalProperties': False}


def validate_source_response(value):
    require(isinstance(value, dict) and set(value) == {'source'}
            and isinstance(value['source'], dict), 'native generation must return a source object')
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError('native generation must contain finite JSON') from exc
    require(len(encoded.encode()) <= 4 * 1024 * 1024, 'generation response size limit')
