"""Complete fixture preparation and explicit evidence plumbing, not measured yield."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from prepare_v2_cross_business import cases, export_preparation
from v2_natural_oracle import expected, expand


def test_all_twenty_reach_shared_truth_without_generated_score_reads():
    rows = cases()
    assert len(rows) == 20
    for row in rows:
        user = copy.deepcopy(row['user'])
        options = {'now': 10}
        if row['acceptance']['kind'] == 'procurement_large':
            user['facts']['proposal_time'] = 10
            data = expand(row['business'])
            options['authorized_snapshot'] = {'quotes': data['quotes'],
                'budget_cents': data['budgets'][0]['remaining_cents']}
        truth = expected(row['acceptance'], row['business'], user, **options)
        assert truth['answer']
        assert truth['initial'] == expand(row['business'])
        fields = {'answer', 'initial', 'final_state', 'observations'}
        if row['acceptance']['kind'] == 'library_renewal_eligibility':
            fields.add('set_memberships')
            fields.add('record_roles')
            assert truth['record_roles']['eligible']['kind'] == 'selected_boolean'
            assert truth['record_roles']['eligible']['complete_records'] == [
                {'loan_id': 'L1', 'eligible': True}, {'loan_id': 'L2', 'eligible': False}]
            assert truth['set_memberships'] == {'eligible': {
                'member': user['facts']['loan_id'], 'complete_set': ['L1']}}
        if row['acceptance']['kind'] in {'procurement_combination', 'procurement_large'}:
            fields.add('quantity_offsets')
            assert truth['quantity_offsets'] == {'overage': user['facts']['demand']}
        if row['acceptance']['kind'] == 'access_compare':
            fields.add('record_roles')
            assert truth['record_roles']['files']['scope'] == 'write'
        if row['acceptance']['kind'] == 'library_pickup_change':
            fields.add('communication')
            assert truth['communication']['available_site_ids']['kind'] == 'location_options_before_choice'
        assert set(truth) == fields
        refused = expected(row['acceptance'], row['business'], user, refuse=True, **options)
        assert refused['final_state'] == refused['initial']


@pytest.mark.parametrize('identity', [7, 8, 9, 10, 17, 18, 19, 20])
def test_missing_runtime_time_is_not_a_fixed_time_fallback(identity):
    row = cases()[identity - 1]
    with pytest.raises(ValueError, match='observed virtual time'):
        expected(row['acceptance'], row['business'], row['user'])


@pytest.mark.parametrize('identity', [7, 17, 19])
def test_fractional_runtime_time_is_valid(identity):
    row = cases()[identity - 1]
    assert expected(row['acceptance'], row['business'], row['user'], now=10.5)['answer']


@pytest.mark.parametrize('now', [True, float('inf'), float('nan'), -1])
@pytest.mark.parametrize('identity', [7, 17])
def test_invalid_virtual_time_is_not_accepted(identity, now):
    row = cases()[identity - 1]
    with pytest.raises(ValueError):
        expected(row['acceptance'], row['business'], row['user'], now=now)


def test_export_is_reviewable_and_cannot_be_mistaken_for_frozen_measurement(tmp_path):
    destination = tmp_path / 'prepared'
    export_preparation(destination)
    manifest = json.loads((destination / 'manifest.json').read_text())
    assert manifest['measurement_ready'] is False
    assert manifest['status'] == 'preparation_only_not_measurement_ready'
    assert len(manifest['samples']) == 20
    checksums = json.loads((destination / 'preparation_checksums.json').read_text())
    assert checksums['is_measurement_freeze'] is False
    assert len(checksums['files']) == 81
    assert all(hashlib.sha256((destination / path).read_bytes()).hexdigest() == digest
               for path, digest in checksums['files'].items())
    assert not (destination / 'frozen.json').exists()
    with pytest.raises(FileExistsError):
        export_preparation(destination)


def test_explicit_graph_relations_are_public_and_reject_broken_business_links():
    from env_factory_v2.business_graph import validate_relations
    from env_factory_v2.contract import ContractError
    rows = cases()
    count = 0
    for row in rows:
        relations = validate_relations(row['business'])
        count += len(relations)
        for relation in relations:
            assert relation['policy_text'] in row['business']['policy']
    assert count >= 20
    broken = copy.deepcopy(rows[1]['business'])
    loans = next(table['rows'] for table in broken['tables'] if table['id'] == 'loans')
    loans[0]['book_id'] = 'not_a_book'
    with pytest.raises(ContractError, match='missing record'):
        validate_relations(broken)


def test_graph_install_handles_empty_tables_and_preserves_declared_schema():
    from env_factory_v2.business_graph import install
    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def run(self, *args, **kwargs): return []
        def execute_write(self, callback, records): self.records = records
    class Driver:
        def session(self, **kwargs): return Session()
    business = {'policy': '订单初始为空。', 'provenance': 'test', 'tables': [
        {'id': 'orders', 'rows': [], 'schema': {'properties': {'id': {}, 'amount': {}}}},
        {'id': 'empty', 'rows': []}]}
    records = install(Driver(), 'test', [{'document_id': 'business', 'title': 'empty', 'business': business}])
    assert records[0]['tables'][0]['columns'] == ['id', 'amount']
    assert records[0]['tables'][1]['columns'] == []


@pytest.mark.parametrize('identity', [9, 15, 20])
def test_independent_consent_preferences_match_user_simulator_fact_interface(identity):
    from env_factory_v2.user_simulator import UserSimulator
    from env_factory_v2.examples import obj
    row = cases()[identity - 1]
    facts = row['user']['facts']
    assert 'approvals' not in facts
    source = {'user': {'facts': facts, 'stages': [
        {'id': name, 'kind': 'authorization', 'tool': 'write_' + name, 'depends_on': []}
        for name in ('first', 'second')]}}
    simulator = UserSimulator(source)
    for name in ('first', 'second'):
        proposal = {'tool': 'write_' + name, 'arguments': {}}
        result = simulator.respond(name, observations={}, tools={'write_' + name: {'parameters': obj({})}},
                                   proposal=proposal)
        assert result['approved'] is facts['approve_' + name]
