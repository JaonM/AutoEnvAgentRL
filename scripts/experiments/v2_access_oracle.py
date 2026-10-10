"""Independent inherited-access truth. Expiry is exclusive, not wall-clock time."""
import copy
import math


def indexed(rows):
    result = {row['id']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Access identities must be unique')
    return result


def ancestry(data, node_id):
    nodes = indexed(data['nodes'])
    chain = []
    while node_id is not None:
        if node_id in chain:
            raise ValueError('Access ancestry contains a cycle')
        if node_id not in nodes:
            raise ValueError('Access ancestry references a missing node')
        chain.append(node_id)
        node_id = nodes[node_id]['parent_id']
    return chain


def live(row, now):
    return row['active'] and row['starts_at'] <= now and (row['expires_at'] is None or now < row['expires_at'])


def permissions(data, node_id, user_id, permission, now):
    if permission not in ('read', 'write'):
        raise ValueError('Unknown access permission')
    indexed(data['grants'])
    chain = ancestry(data, node_id)
    groups = {row['group_id'] for row in data['memberships']
              if row['user_id'] == user_id and live(row, now)}
    grants = [row for row in data['grants'] if row['node_id'] in chain
              and row['permission'] == permission and live(row, now)
              and ((row['principal_kind'] == 'user' and row['principal_id'] == user_id)
                   or (row['principal_kind'] == 'group' and row['principal_id'] in groups))]
    if any(row['effect'] not in ('allow', 'deny') for row in grants):
        raise ValueError('Unknown access effect')
    denies = sorted(row['id'] for row in grants if row['effect'] == 'deny')
    allows = sorted(row['id'] for row in grants if row['effect'] == 'allow')
    return {'allowed': bool(allows) and not denies, 'allow_sources': allows,
            'deny_sources': denies, 'reason': 'explicit_deny' if denies else
            'effective_allow' if allows else 'no_effective_allow'}


def revoke(data, grant_id, *, approved):
    if type(approved) is not bool:
        raise ValueError('Revocation authorization must be boolean')
    final = copy.deepcopy(data)
    grants = indexed(final['grants'])
    if grant_id not in grants:
        raise ValueError('Revocation target must exist')
    if approved:
        grants[grant_id]['active'] = False
    return final


def temporary_grant(data, grant_id, *, node_id, user_id, permission, now, expires_at, approved):
    """Update only an existing designated record, subject to effective denies."""
    if type(approved) is not bool:
        raise ValueError('Grant authorization must be boolean')
    if (type(now) not in (int, float) or type(expires_at) not in (int, float)
            or not math.isfinite(now) or not math.isfinite(expires_at) or now < 0 or expires_at <= now):
        raise ValueError('Temporary access requires a future virtual expiry')
    final = copy.deepcopy(data)
    grants = indexed(final['grants'])
    if grant_id not in grants:
        raise ValueError('Temporary access target must already exist')
    target = grants[grant_id]
    if (target['node_id'] != node_id or target['principal_kind'] != 'user'
            or target['principal_id'] != user_id or target['permission'] != permission):
        raise ValueError('Grant target does not match the exact authorization')
    current = permissions(data, node_id, user_id, permission, now)
    if not approved or current['deny_sources']:
        return False, final
    target.update(active=True, effect='allow', starts_at=now, expires_at=expires_at)
    return True, final


def expected(spec, data, facts, *, now=None, refuse=False):
    initial, final = copy.deepcopy(data), copy.deepcopy(data)
    kind = spec['kind']
    if kind == 'access_direct':
        rows = spec['params']['effects']
        answer = {p: any(r['permission'] == p and r['effect'] == 'allow' for r in rows)
                     and not any(r['permission'] == p and r['effect'] == 'deny' for r in rows)
                  for p in ('read', 'write')}
        return {'answer': answer, 'initial': initial, 'final_state': final, 'observations': {}}
    if type(now) not in (int, float) or not math.isfinite(now) or now < 0:
        raise ValueError('Access truth requires observed virtual time')
    if kind == 'access_query':
        answer = permissions(data, facts['node_id'], facts['user_id'], 'read', now)
    elif kind == 'access_compare':
        answer = {'files': [{'node_id': identity, **permissions(data, identity, facts['user_id'], 'write', now)}
                            for identity in facts['node_ids']]}
    elif kind == 'access_temporary':
        if (len(facts['node_ids']) != len(facts['grant_ids'])
                or type(facts['approve_grants']) is not bool):
            raise ValueError('Temporary grant targets require exact boolean authorization')
        results = []
        for node_id, grant_id in zip(facts['node_ids'], facts['grant_ids']):
            committed, final = temporary_grant(final, grant_id, node_id=node_id, user_id=facts['user_id'],
                permission='write', now=now, expires_at=facts['expires_at'],
                approved=facts['approve_grants'] and not refuse)
            results.append({'node_id': node_id, 'committed': committed,
                'allowed': permissions(final, node_id, facts['user_id'], 'write', now)['allowed'],
                'allowed_at_expiry': permissions(final, node_id, facts['user_id'], 'write', facts['expires_at'])['allowed']})
        answer = {'results': results}
    elif kind == 'access_revoke':
        targets, approvals = facts['grant_ids'], facts['approvals']
        if len(targets) != len(approvals) or any(type(a) is not bool for a in approvals):
            raise ValueError('Every revocation requires independent boolean authorization')
        before = permissions(data, facts['node_id'], facts['user_id'], 'write', now)
        for identity, approved in zip(targets, approvals):
            if identity not in before['allow_sources']:
                raise ValueError('Revocation target is not an effective source for this user and object')
            final = revoke(final, identity, approved=approved and not refuse)
        after = permissions(final, facts['node_id'], facts['user_id'], 'write', now)
        answer = {'original_sources': before['allow_sources'], 'remaining_sources': after['allow_sources'],
                  'allowed': after['allowed']}
    else:
        raise ValueError('Unknown independent access requirement')
    return {'answer': answer, 'initial': initial, 'final_state': final, 'observations': {}}
