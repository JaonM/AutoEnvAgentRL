import copy

import pytest

from env_factory.production.data_scope import check_references, partition


TABLES = {'readers': [{'id': 'r1', 'reader': 'A', 'loans': ['l1']},
                      {'id': 'r2', 'reader': 'B', 'loans': ['l2']}],
          'loans': [{'id': 'l1', 'reader': 'A'}, {'id': 'l2', 'reader': 'B'}],
          'catalog': [{'id': 'book1'}]}
OWNERSHIP = {'owner_table': 'readers', 'owner_key': 'reader', 'tables': {'readers': 'reader', 'loans': 'reader'}}
RULES = [{'table': 'readers', 'field': 'loans', 'target_table': 'loans', 'target_field': 'id'}]


def test_private_rows_are_disjoint_and_catalog_shared_without_mutating_input():
    original = copy.deepcopy(TABLES)
    result = partition(TABLES, OWNERSHIP, seed=123, heldout_count=1)
    assert result == partition(TABLES, OWNERSHIP, seed=123, heldout_count=1)
    train, dev = [result['partitions'][k] for k in ('train', 'dev')]
    assert set(train['owners']).isdisjoint(dev['owners'])
    assert train['tables']['catalog'] == dev['tables']['catalog'] == TABLES['catalog']
    assert not check_references(train['tables'], RULES)
    assert not check_references(dev['tables'], RULES)
    train['tables']['readers'][0]['loans'].append('changed')
    assert TABLES == original


def test_cross_partition_reference_is_rejected_instead_of_erased():
    tables = copy.deepcopy(TABLES)
    tables['readers'][0]['loans'].append('l2')
    result = partition(tables, OWNERSHIP, seed=123, heldout_count=1)
    assert any(check_references(p['tables'], RULES) for p in result['partitions'].values())


def test_unknown_owner_duplicate_id_and_empty_partition_fail():
    tables = copy.deepcopy(TABLES)
    tables['loans'][0]['reader'] = 'unknown'
    with pytest.raises(ValueError):
        partition(tables, OWNERSHIP, seed=123, heldout_count=1)
    tables = copy.deepcopy(TABLES)
    tables['loans'][0]['id'] = 'l2'
    with pytest.raises(ValueError):
        partition(tables, OWNERSHIP, seed=123, heldout_count=1)
    with pytest.raises(ValueError):
        partition(TABLES, OWNERSHIP, seed=123, heldout_count=0)
