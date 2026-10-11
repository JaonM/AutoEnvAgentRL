"""Explicit inputs for independent cross-business preparation, never task generation."""
import hashlib
import json
from collections import Counter

from env_factory_v2.business_graph import validate_relations
from env_factory_v2.expressions import canonical


def exact_ids(rows, expected=None):
    ids = [row['id'] for row in rows]
    if (any(type(i) is not int or i <= 0 for i in ids)
            or len(ids) != len(set(ids))
            or (expected is not None and set(ids) != set(expected))):
        raise ValueError('Preparation identities must cover the sample set exactly once')
    return set(ids)


def load_pack(path=None):
    if path is None:
        from prepare_v2_cross_business import cases
        from prepare_v2_cross_openings import scenes
        from prepare_v2_natural_pilot import cases as historical_cases
        rows = cases()
        scene_by_id = {row['id']: row for row in scenes()}
        for row in rows:
            row['user_scene'] = {k: v for k, v in scene_by_id[row['id']].items() if k != 'id'}
        pack = {'cases': rows, 'historical_business_rules': [
            {'title': row['title'], 'policy': row['business']['policy']} for row in historical_cases()]}
    else:
        pack = json.loads(path.read_text())
    rows = pack['cases']
    exact_ids(rows)
    count = len(rows)
    mix = {'direct': count // 5, 'simple': count // 5,
           'ordinary': count // 5, 'complex': count * 2 // 5}
    if count < 10 or count % 10 or Counter(row['stratum'] for row in rows) != mix:
        raise ValueError('Preparation must preserve the 2:2:2:4 difficulty mix')
    if not pack['historical_business_rules']:
        raise ValueError('Explicit historical business rules are required for novelty review')
    for row in rows:
        for key in ('title', 'business_family'):
            if not isinstance(row[key], str) or not row[key].strip():
                raise ValueError('Preparation metadata must be nonempty')
        scene = row['user_scene']
        if (set(scene) != {'situation_and_desired_outcome', 'known_information'}
                or not isinstance(scene['situation_and_desired_outcome'], str)
                or not scene['situation_and_desired_outcome'].strip()
                or not isinstance(scene['known_information'], dict)):
            raise ValueError('User scene must contain only the desired outcome and known information')
        validate_relations(row['business'])
        if not isinstance(row['user']['facts'], dict) or not isinstance(row['acceptance'], dict):
            raise ValueError('Independent user facts and acceptance specification required')
    return pack


def identity(pack):
    return {'sha256': hashlib.sha256(canonical(pack).encode()).hexdigest(), 'count': len(pack['cases'])}


def user_scenes(pack):
    return [dict(id=row['id'], **row['user_scene']) for row in pack['cases']]


def business_review_input(pack, openings):
    exact_ids(openings, exact_ids(pack['cases']))
    requests = {row['id']: row['user_request'] for row in openings}
    return {'scope': 'synthetic business novelty and public fidelity review; not yield',
        'historical_business_rules': pack['historical_business_rules'],
        'new_cases': [{'id': row['id'], 'business_family': row['business_family'],
            'user_request': requests[row['id']], 'public_policy': row['business']['policy'],
            'clarifiable_private_user_information': row['user']['facts'],
            'business_tables': row['business']['tables'],
            'business_relations': row['business'].get('relations', [])} for row in pack['cases']]}
