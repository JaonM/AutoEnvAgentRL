import pytest

from env_factory.sandbox_runtime import ManifestDataStore, SandboxError


def test_relational_validation_indexes_parent_keys_without_quadratic_scans():
    class Key(int):
        comparisons = 0
        __hash__ = int.__hash__
        def __eq__(self, other):
            Key.comparisons += 1
            return super().__eq__(other)

    count = 2000
    store = object.__new__(ManifestDataStore)
    store.schemas = {
        'parent': {'columns': [{'name': 'id', 'type': 'integer', 'nullable': False}], 'primary_key': ['id']},
        'child': {'columns': [{'name': 'id', 'type': 'integer', 'nullable': False},
                              {'name': 'parent_id', 'type': 'integer', 'nullable': True}], 'primary_key': ['id'],
                  'foreign_keys': [{'column': 'parent_id', 'ref_table': 'parent', 'ref_column': 'id'}]}}
    tables = {'parent': [{'id': Key(i)} for i in range(count)],
              'child': [{'id': i, 'parent_id': Key(count-i-1)} for i in range(count)]}
    store._validate_tables(tables)
    assert Key.comparisons < 4 * count
    tables['child'][-1]['parent_id'] = None
    store._validate_tables(tables)
    tables['child'][-1]['parent_id'] = count
    with pytest.raises(SandboxError, match='foreign key violation'):
        store._validate_tables(tables)
