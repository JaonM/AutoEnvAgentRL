import json

import pytest

from env_factory.sandbox_runtime import EpisodeStore, ManifestDataStore


def business_data(tmp_path):
    tables = []
    for name, quantity in [('inventory', 10), ('orders', 0)]:
        schema = {'columns': [
            {'name': 'id', 'type': 'INTEGER', 'nullable': False},
            {'name': 'quantity', 'type': 'INTEGER', 'nullable': False}], 'primary_key': ['id']}
        (tmp_path / f'{name}.schema.json').write_text(json.dumps(schema))
        (tmp_path / f'{name}.jsonl').write_text(json.dumps({'id': 1, 'quantity': quantity}) + '\n')
        tables.append({'table_name': name, 'schema_file': f'{name}.schema.json',
                       'rows_file': f'{name}.jsonl'})
    store = EpisodeStore(tmp_path / 'episodes.sqlite3')
    data = ManifestDataStore({'environment_mode': 'stateful', 'tables': tables}, tmp_path, store)
    data.reset()
    return store, data


def test_caught_inner_failure_rolls_back_both_tables_events_and_cached_response(tmp_path):
    store, data = business_data(tmp_path)
    with store.transaction():
        store.set_state('outer_progress', 'started')
        try:
            with store.transaction():
                data.update('inventory', {'id': 1}, {'quantity': 6})
                data.update('orders', {'id': 1}, {'quantity': 4})
                store.execute_idempotent('reservation', 'commit', {'id': 1}, lambda: {'committed': True})
                raise ValueError('final validation failed')
        except ValueError:
            store.set_state('outer_progress', 'handled')
        assert data.table('inventory')[0]['quantity'] == 10
        assert data.table('orders')[0]['quantity'] == 0
        assert store.replay()['events'] == []
    reopened = EpisodeStore(tmp_path / 'episodes.sqlite3')
    assert reopened.get_state('outer_progress') == 'handled'
    assert reopened.get_state('business_data')['inventory'][0]['quantity'] == 10
    assert reopened.execute_idempotent('reservation', 'commit', {'id': 1},
                                       lambda: {'committed': False}) == {'committed': False}


def test_outer_failure_rolls_back_successful_inner_commit(tmp_path):
    store, data = business_data(tmp_path)
    with pytest.raises(RuntimeError):
        with store.transaction():
            with store.transaction():
                data.update('inventory', {'id': 1}, {'quantity': 6})
                data.update('orders', {'id': 1}, {'quantity': 4})
                store.event('reservation', {'id': 1}, {'committed': True})
            raise RuntimeError('outer commit failed')
    assert data.table('inventory')[0]['quantity'] == 10
    assert data.table('orders')[0]['quantity'] == 0
    assert store.replay()['events'] == []


def test_nested_success_commits_both_tables(tmp_path):
    store, data = business_data(tmp_path)
    with store.transaction():
        with store.transaction():
            data.update('inventory', {'id': 1}, {'quantity': 6})
            data.update('orders', {'id': 1}, {'quantity': 4})
    reopened = EpisodeStore(tmp_path / 'episodes.sqlite3')
    assert reopened.get_state('business_data')['inventory'][0]['quantity'] == 6
    assert reopened.get_state('business_data')['orders'][0]['quantity'] == 4
