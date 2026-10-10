"""Experiment harness checks, not measured model yield."""
import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts/experiments'
sys.path.insert(0, str(SCRIPTS))
from prepare_v2_natural_pilot import cases
from v2_natural_oracle import expected, equivalent
from measure_v2_natural import fixture_alignment, answer_checks, observation_checks
from env_factory_v2.examples import simple, transaction
from env_factory_v2.contract import ContractError
from env_factory_v2.runtime import Sandbox


def correction_observation_check(truth, ids):
    world = SimpleNamespace(version=0, tools={'menu': {'table': 'venues', 'kind': 'query'}},
        user=SimpleNamespace(stages={'changed': {'kind': 'correction'}}),
        trace=[{'kind': 'user', 'stage': 'changed'}, {'kind': 'tool', 'tool': 'menu',
            'status': 'ok', 'result': {'items': [{'id': key} for key in ids], 'state_version': 0}}])
    return next(c for c in observation_checks(world, truth, 'correction')
                if c['id'] == 'observed_venues')['passed']


def test_correction_requires_complete_final_menu_not_obsolete_region():
    row = next(r for r in cases() if r['acceptance']['kind'] == 'correction')
    truth = expected(row['acceptance'], row['business'], row['user'])
    final_ids = [v['id'] for v in truth['initial']['venues']
                 if v['region'] == row['user']['facts']['region_new'] and v['available']]
    assert truth['observations']['venues'] == final_ids
    assert correction_observation_check(truth, final_ids)
    assert not correction_observation_check(truth, final_ids[:-1])
    obsolete_ids = [v['id'] for v in truth['initial']['venues']
                    if v['region'] != row['user']['facts']['region_new']]
    assert not correction_observation_check(truth, obsolete_ids)


def test_correction_menu_excludes_unavailable_records_and_follows_changed_preferences():
    row = copy.deepcopy(next(r for r in cases() if r['acceptance']['kind'] == 'correction'))
    venues = next(t['rows'] for t in row['business']['tables'] if t['id'] == 'venues')
    excluded = next(v for v in venues if v['region'] == 'west' and v['id'] != row['user']['facts']['venue_choice'])
    excluded['available'] = False
    truth = expected(row['acceptance'], row['business'], row['user'])
    assert excluded['id'] not in truth['observations']['venues']
    row['user']['facts'].update(region_new='east', venue_choice=venues[0]['id'])
    changed = expected(row['acceptance'], row['business'], row['user'])
    assert set(changed['observations']['venues']) == {v['id'] for v in venues if v['region'] == 'east' and v['available']}
    assert set(truth['observations']['venues']).isdisjoint(changed['observations']['venues'])


def test_frozen_business_oracle_covers_twenty_natural_requests():
    rows = cases()
    assert len(rows) == 20
    for row in rows:
        truth = expected(row['acceptance'], row['business'], row['user'])
        assert truth['answer']
        if row['acceptance']['mutation']:
            changed = expected(row['acceptance'], row['business'], row['user'], mutation=row['acceptance']['mutation'])
            assert changed['answer'] != truth['answer']
    assert expected(rows[0]['acceptance'], rows[0]['business'], rows[0]['user'])['answer']['energy_kwh'] == pytest.approx(3.625)
    assert expected(rows[13]['acceptance'], rows[13]['business'], rows[13]['user'])['answer']['count'] == 183
    assert expected(rows[14]['acceptance'], rows[14]['business'], rows[14]['user'])['answer']['count'] == 2400


def test_business_data_and_private_user_cannot_be_reinvented():
    source = simple()
    row = {'raw_request': source['task'], 'stratum': 'simple'}
    business = {'policy': '按用户尺码筛选', 'tables': [{'id': 'items', 'rows': source['tables'][0]['rows']}]}
    user = {'facts': source['user']['facts']}
    fixture_alignment(source, row, business, user)
    changed = copy.deepcopy(source)
    changed['tables'][0]['rows'][0]['size'] = 'XL'
    with pytest.raises(ContractError, match='business records differ'):
        fixture_alignment(changed, row, business, user)
    changed = copy.deepcopy(source)
    changed['user']['facts']['size'] = 'XL'
    with pytest.raises(ContractError, match='user facts differ'):
        fixture_alignment(changed, row, business, user)
    changed = copy.deepcopy(source)
    changed['task'] += '补充工程要求'
    with pytest.raises(ContractError, match='natural request verbatim'):
        fixture_alignment(changed, row, business, user)


def test_independent_comparison_rejects_true_as_one_and_missing_ids():
    assert not equivalent(True, 1)
    assert equivalent(0.1 + 0.2, 0.3)
    assert not equivalent(['I1'], ['I1', 'I2'], unordered=True)
    assert not answer_checks({'n': 0}, {'answer': {'count': 1}}, {'count': 'n'})[0]['passed']


def test_post_write_read_must_include_actual_changed_fields():
    world = Sandbox(transaction())
    world.call('accounts', {}, call_id='before')
    args = {'from_id': {'$result': 'before', 'path': 'items.0.id'},
            'to_id': {'$result': 'before', 'path': 'items.1.id'}, 'amount': 20}
    world.interact('approve', question='批准？', proposal={'tool': 'transfer', 'arguments': args})
    world.call('transfer', args, call_id='write')
    truth = {'initial': world.initial, 'final_state': world.data, 'observations': {'accounts': ['A', 'B']}}
    assert not all(c['passed'] for c in observation_checks(world, truth, 'transfer'))
    world.call('accounts', {}, call_id='after')
    assert all(c['passed'] for c in observation_checks(world, truth, 'transfer'))
    world.trace[-1]['result']['items'] = [{'id': 'A'}, {'id': 'B'}]
    assert not all(c['passed'] for c in observation_checks(world, truth, 'transfer'))


def test_refusal_keeps_full_external_state_and_separate_commits_preserve_first():
    row = cases()[-1]
    spec, business, user = row['acceptance'], row['business'], row['user']
    refused = expected(spec, business, user, refuse=True)
    assert refused['initial'] == refused['final_state']
    one = expected(spec, business, user, mutation={'table': 'accounts', 'key': 'AC1', 'column': 'quota', 'value': 1})
    assert one['answer'] == {'first_active': True, 'second_active': False, 'quota': 0}
    assert one['final_state']['logs'][0]['completed'] is True
    assert one['final_state']['logs'][1]['completed'] is False
