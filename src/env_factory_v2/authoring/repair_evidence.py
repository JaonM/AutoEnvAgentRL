"""Reduce repeated passing execution detail in author-only repair feedback."""
from __future__ import annotations

import copy
import hashlib

from env_factory_v2.contracts.expressions import canonical
from env_factory_v2.validation.verification import compact_evidence


def compact_repair_feedback(feedback):
    """Keep all failed executions intact; never alter requests or candidate source."""
    count = 0
    def visit(value):
        nonlocal count
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        is_record = (type(value.get('passed')) is bool
            and all(isinstance(value.get(k), dict)
                    and {'outcome', 'trace', 'final_state', 'time'} <= value[k].keys()
                    for k in ('first', 'repeat')))
        if is_record:
            if not value['passed']:
                return copy.deepcopy(value)
            result = copy.deepcopy(value)
            for key in ('first', 'repeat'):
                result[key] = compact_evidence(value[key])
                result[key]['archived_execution_identity'] = hashlib.sha256(canonical(value[key]).encode()).hexdigest()
            count += 1
            result['repair_evidence_detail'] = 'passing_verdicts_and_answers_full_execution_archived'
            return result
        return {key: copy.deepcopy(item) if key in {
                    'previous_source', 'original_request', 'contract', 'source', 'demand', 'literal'}
                else visit(item) for key, item in value.items()}
    result = visit(feedback)
    original, compact = canonical(feedback).encode(), canonical(result).encode()
    if len(compact) >= len(original):
        result, compact, count = copy.deepcopy(feedback), original, 0
    return result, {'compacted_passing_records': count, 'original_bytes': len(original),
                    'compact_bytes': len(compact), 'original_identity': hashlib.sha256(original).hexdigest(),
                    'compact_identity': hashlib.sha256(compact).hexdigest(),
                    'scope': 'author_feedback_only_not_qualification_evidence'}
