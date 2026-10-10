"""Access oracle checks, not measured agent success."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_access_oracle import ancestry, permissions, revoke, temporary_grant


def grant(key, node, principal='R1', kind='user', effect='allow', permission='write', end=None):
    return dict(id=key, node_id=node, principal_id=principal, principal_kind=kind, effect=effect,
                permission=permission, active=True, starts_at=0, expires_at=end)


def data():
    return {'nodes': [dict(id='N0', parent_id=None), dict(id='N1', parent_id='N0'),
                     dict(id='F1', parent_id='N1'), dict(id='F2', parent_id='N1')],
            'grants': [grant('G1', 'N0', principal='T1', kind='group'),
                       grant('G2', 'F2', effect='deny'), grant('G3', 'F1', permission='read', end=10)],
            'memberships': [dict(id='M1', user_id='R1', group_id='T1', active=True,
                                 starts_at=0, expires_at=20)]}


def test_same_folder_files_have_different_permissions_and_independent_read_write():
    world = data()
    assert permissions(world, 'F1', 'R1', 'write', 10)['allowed']
    assert permissions(world, 'F2', 'R1', 'write', 10)['deny_sources'] == ['G2']
    assert not permissions(world, 'F1', 'R1', 'read', 10)['allowed']
    assert permissions(world, 'F1', 'R1', 'read', 9)['allowed']
    assert not permissions(world, 'F1', 'R1', 'write', 20)['allowed']


def test_long_chain_and_cycle_missing_parent_failures():
    world = data()
    world['nodes'] = [dict(id=f'N{i}', parent_id=f'N{i-1}' if i else None) for i in range(120)]
    assert len(ancestry(world, 'N119')) == 120
    assert permissions(world, 'N119', 'R1', 'write', 10)['allow_sources'] == ['G1']
    world['nodes'][0]['parent_id'] = 'N119'
    with pytest.raises(ValueError, match='cycle'): ancestry(world, 'N119')
    world['nodes'][0]['parent_id'] = 'missing'
    with pytest.raises(ValueError, match='missing'): ancestry(world, 'N119')


def test_independent_revocations_preserve_unrelated_grants_and_remaining_access():
    world = data()
    world['grants'].append(grant('G4', 'F1'))
    first = revoke(world, 'G4', approved=True)
    assert permissions(first, 'F1', 'R1', 'write', 10)['allowed']
    assert revoke(first, 'G1', approved=False) == first
    second = revoke(first, 'G1', approved=True)
    assert not permissions(second, 'F1', 'R1', 'write', 10)['allowed']
    assert second['grants'][1:3] == world['grants'][1:3]
    assert world['grants'][-1]['active']


@pytest.mark.parametrize('blocked,approved', [(False, True), (True, True), (False, False)])
def test_temporary_grant_respects_deny_and_exclusive_expiry(blocked, approved):
    world = data()
    target = grant('T0', 'F1', principal='R2')
    target['active'] = False
    world['grants'].append(target)
    if blocked: world['grants'].append(grant('D0', 'N0', principal='R2', effect='deny'))
    committed, final = temporary_grant(world, 'T0', node_id='F1', user_id='R2',
        permission='write', now=10, expires_at=20, approved=approved)
    assert committed is (approved and not blocked)
    if committed:
        assert permissions(final, 'F1', 'R2', 'write', 19)['allowed']
        assert not permissions(final, 'F1', 'R2', 'write', 20)['allowed']
        assert final['grants'][:-1] == world['grants'][:-1]
    else:
        assert final == world


def test_five_access_fixtures_and_twenty_case_mix():
    from prepare_v2_cross_business import access_cases, library_cases, procurement_cases, subscription_cases
    from v2_natural_oracle import expand
    from v2_access_oracle import expected
    rows = access_cases()
    assert [r['id'] for r in rows] == [16, 17, 18, 19, 20]
    truths = [expected(r['acceptance'], expand(r['business']), r['user']['facts'], now=10) for r in rows]
    assert truths[0]['answer'] == {'read': False, 'write': True}
    assert truths[1]['answer']['deny_sources'] == ['G4']
    assert [f['allowed'] for f in truths[2]['answer']['files']] == [True, False]
    assert truths[3]['answer']['results'] == [
        {'node_id': 'F1', 'committed': True, 'allowed': True, 'allowed_at_expiry': False},
        {'node_id': 'F2', 'committed': False, 'allowed': False, 'allowed_at_expiry': False}]
    assert truths[4]['answer'] == {'original_sources': ['G1', 'G5'], 'remaining_sources': ['G5'], 'allowed': True}
    for row in rows:
        truth = expected(row['acceptance'], expand(row['business']), row['user']['facts'], now=10, refuse=True)
        assert truth['initial'] == truth['final_state']
    all_rows = library_cases() + procurement_cases() + subscription_cases() + rows
    from collections import Counter
    assert [r['id'] for r in all_rows] == list(range(1, 21))
    assert Counter(r['stratum'] for r in all_rows) == {'direct': 4, 'simple': 4, 'ordinary': 4, 'complex': 8}


def test_separate_temporary_grants_bind_their_actual_submission_times():
    from prepare_v2_cross_business import access_cases
    from v2_natural_oracle import expected
    row = access_cases()[3]
    grants = next(t['rows'] for t in row['business']['tables'] if t['id'] == 'grants')
    next(g for g in grants if g['id'] == 'D1')['active'] = False
    truth = expected(row['acceptance'], row['business'], row['user'], now=10,
                     submission_times={'T0': 10, 'T1': 11.5})
    final = {g['id']: g for g in truth['final_state']['grants']}
    assert final['T0']['starts_at'] == 10
    assert final['T1']['starts_at'] == 11.5
    assert all(r['committed'] for r in truth['answer']['results'])
    expired = expected(row['acceptance'], row['business'], row['user'], now=10,
                       submission_times={'T0': 10, 'T1': 50})
    assert [r['committed'] for r in expired['answer']['results']] == [True, False]
    assert not next(g for g in expired['final_state']['grants'] if g['id'] == 'T1')['active']
