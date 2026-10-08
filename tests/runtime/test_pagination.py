import copy
import json

import pytest

from env_factory.generation.contract_compiler import ContractCompiler
from env_factory.generation.data_recipes import expand_data_recipes
from env_factory.sandbox_runtime import DeclarativeToolCompiler, EpisodeStore, ManifestDataStore, SandboxError
from env_factory.tasks.task_spec import _declarative_output_schema


def dataset(tmp_path, count=205, page_size=5):
    tables = [{'table_name': 'groups', 'description': 'Groups', 'primary_key': ['id'],
        'columns': [{'name': 'id', 'type': 'integer', 'nullable': False},
                    {'name': 'name', 'type': 'text', 'nullable': False}],
        'foreign_keys': [], 'indexes': [], 'constraints': [],
        'rows': [{'id': i, 'name': 'group' + str(i)} for i in range(1, 21)]},
        {'table_name': 'items', 'description': 'Search records', 'primary_key': ['id'],
        'columns': [{'name': 'id', 'type': 'integer', 'nullable': False},
                    {'name': 'group_id', 'type': 'integer', 'nullable': False},
                    {'name': 'value', 'type': 'integer', 'nullable': False}],
        'foreign_keys': [{'column': 'group_id', 'ref_table': 'groups', 'ref_column': 'id'}],
        'indexes': [], 'constraints': [], 'rows': [],
        'row_recipe': {'count': count, 'columns': {
            'id': {'sequence': {'start': 1, 'step': 1}},
            'group_id': {'cycle': list(range(1, 21))}, 'value': {'integer': {'min': 0, 'max': 99}}}}}]
    expanded = expand_data_recipes({'tables': tables}, seed=17)
    tables = expanded['tables']
    ContractCompiler._validate_relational_data(tables)
    manifest = ContractCompiler._materialize_business_data(tables, '', tmp_path / 'data')
    store = EpisodeStore(tmp_path / 'state.sqlite3'); store.reset(seed=17)
    data = ManifestDataStore(manifest, tmp_path / 'data', store)
    data.reset()
    spec = {'tool_name': 'search', 'operation': 'select', 'table': 'items', 'result_field': 'records',
        'filters': [{'argument': 'group', 'column': 'group_id', 'operator': 'eq',
                     'resolve': {'table': 'groups', 'match_column': 'name', 'value_column': 'id'}}],
        'projection': ['id', 'value'], 'order_by': [{'field': 'value', 'direction': 'desc'}, 'id'],
        'pagination': {'cursor_argument': 'cursor', 'page_size': page_size}}
    return data, spec, DeclarativeToolCompiler(data).compile(spec)


def test_ten_thousand_rows_cross_table_query_is_bounded_complete_and_stably_ordered(tmp_path):
    data, spec, search = dataset(tmp_path, count=10000, page_size=100)
    rows, pages, arguments = [], [], {'group': 'group1'}
    while True:
        result = search(arguments)
        assert result['count'] == len(result['records']) <= 100
        assert result['total_count'] == 500
        assert result['has_more'] == bool(result['next_cursor'])
        pages.append(result)
        rows.extend(result['records'])
        if not result['has_more']: break
        arguments['cursor'] = result['next_cursor']
    wanted = sorted((r for r in data.baseline['items'] if r['group_id'] == 1), key=lambda r: (-r['value'], r['id']))
    assert rows == [{'id': r['id'], 'value': r['value']} for r in wanted]
    assert len(pages) == 5 and len({r['id'] for r in rows}) == 500
    assert all(len(json.dumps(page)) < 6000 for page in pages)
    schema = _declarative_output_schema(spec, data.schemas['items'])
    assert schema['properties']['records']['maxItems'] == 100
    from env_factory.sandbox_runtime import validate_json_schema
    for page in pages: validate_json_schema(schema, page)


@pytest.mark.parametrize('count', [1, 5, 6, 10, 11])
def test_exact_page_boundaries_and_retries_have_no_duplicates(tmp_path, count):
    _, _, search = dataset(tmp_path, count=count)
    page, rows = search({}), []
    while True:
        rows.extend(page['records'])
        if not page['has_more']: break
        args = {'cursor': page['next_cursor']}
        page = search(args)
        assert search(args) == page
    assert len(rows) == count and len({r['id'] for r in rows}) == count
    empty = search({'group': 'absent'})
    assert empty == {'records': [], 'count': 0, 'total_count': 0, 'has_more': False, 'next_cursor': ''}


@pytest.mark.parametrize('change', ['forged', 'query', 'tool', 'reset', 'other_episode'])
def test_cursor_is_bound_to_issued_query_and_episode(tmp_path, change):
    data, spec, search = dataset(tmp_path)
    args = {'group': 'group1'}
    cursor = search(args)['next_cursor']; assert cursor
    args['cursor'] = cursor
    if change == 'forged': args['cursor'] = '0' * 64
    if change == 'query': args['group'] = 'group2'
    if change == 'tool': search = DeclarativeToolCompiler(data).compile({**spec, 'tool_name': 'other'})
    if change in {'reset', 'other_episode'}:
        old_id = data.episode_store.current().episode_id
        data.episode_store.reset(episode_id=old_id if change == 'reset' else 'other', seed=17)
        data.reset()
    with pytest.raises(SandboxError, match='cursor') as error: search(args)
    assert error.value.code == 'PAGINATION_CURSOR_INVALID'
    restarted = search({'group': 'group1'})
    assert restarted['count'] == 5
    if change in {'reset', 'other_episode'}:
        assert restarted['next_cursor'] != cursor
        with pytest.raises(SandboxError, match='cursor'): search(args)


def test_cursor_issuance_rolls_back_with_failed_transaction(tmp_path):
    data, _, search = dataset(tmp_path)
    with pytest.raises(RuntimeError):
        with data.episode_store.transaction():
            cursor = search({})['next_cursor']
            raise RuntimeError('rollback')
    with pytest.raises(SandboxError, match='cursor'): search({'cursor': cursor})
    assert search({})['next_cursor'] != cursor


@pytest.mark.parametrize('table,field,value', [('items', 'value', 100), ('groups', 'name', 'renamed')])
def test_table_or_resolver_change_invalidates_cursor_and_allows_restart(tmp_path, table, field, value):
    data, _, search = dataset(tmp_path)
    cursor = search({'group': 'group1'})['next_cursor']
    rows = data.table(table); rows[0][field] = value; data.replace_table(table, rows)
    with pytest.raises(SandboxError) as error: search({'group': 'group1', 'cursor': cursor})
    assert error.value.code == 'PAGINATION_CURSOR_STALE' and error.value.status == 409
    assert search({'group': 'group1'})['count'] <= 5


@pytest.mark.parametrize('change', ['limit', 'zero', 'noninteger', 'no_order', 'nonunique', 'mutation', 'filter_cursor', 'envelope', 'schema'])
def test_invalid_pagination_contract_fails_before_execution(tmp_path, change):
    data, original, _ = dataset(tmp_path)
    spec = copy.deepcopy(original)
    params = {'properties': {'cursor': {'type': 'string'}}, 'required': []}
    if change == 'limit': spec['pagination']['page_size'] = 101
    if change == 'zero': spec['pagination']['page_size'] = 0
    if change == 'noninteger': spec['pagination']['page_size'] = True
    if change == 'no_order': spec.pop('order_by')
    if change == 'nonunique': spec['order_by'] = ['value']
    if change == 'mutation': spec['operation'] = 'update'
    if change == 'filter_cursor': spec['pagination']['cursor_argument'] = 'group'
    if change == 'envelope': spec['result_field'] = 'next_cursor'
    if change == 'schema': params['properties']['cursor']['type'] = 'integer'
    with pytest.raises(SandboxError, match='pagination'):
        DeclarativeToolCompiler.validate_pagination(spec, data.schemas['items'], params)
