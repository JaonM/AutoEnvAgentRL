"""Planning paths and delivered schemas agree with executed primitive responses."""
import pytest

from env_factory.sandbox_runtime import (
    DeclarativeToolCompiler as Compiler, DeclarativeMetricEvaluator as Evaluator,
    SandboxError, validate_json_schema,
)
from env_factory.tasks.task_spec import _declarative_output_schema
from tests.runtime.test_nested_transactions import business_data


@pytest.mark.parametrize('operation,custom,path,expected', [
    ('select', None, '$.records[0].quantity', 10),
    ('select', 'items', '$.items[*].quantity', 10),
    ('aggregate_count', None, '$.count', 1),
    ('aggregate_count', 'total', '$.total', 1),
    ('insert', None, '$.record.id', 2),
    ('update', None, '$.updated_count', 1),
    ('delete', None, '$.deleted_count', 1),
])
def test_planned_paths_resolve_actual_primitive_results(tmp_path, operation, custom, path, expected):
    store, data = business_data(tmp_path)
    spec = {'tool_name': 'business', 'operation': operation, 'table': 'inventory',
            'selector': {'id': 'id'}, 'changes': {'quantity': 'quantity'},
            'values': {'id': 'id', 'quantity': 'quantity'}}
    if custom:
        spec['result_field'] = custom
    Compiler.validate_result_path(spec, path)
    result = Compiler(data).compile(spec)({'id': 2 if operation == 'insert' else 1, 'quantity': 4})
    schema = _declarative_output_schema(spec, data.schemas['inventory'])
    validate_json_schema(schema, result)
    assert Evaluator._resolve(result, path) == expected
    Compiler.validate_result_path(spec, '$')
    assert Evaluator._resolve(result, '$') == result


@pytest.mark.parametrize('operation,path', [
    ('update', '$.reservation_id'), ('delete', '$.record'), ('insert', '$.id'),
    ('update', '$.updated_count.id'), ('delete', '$.deleted_count[0]'),
    ('insert', '$.count[0]'), ('insert', '$.record[0]'),
    ('aggregate_count', '$.count.id'), ('select', '$.id'),
    ('select', '$[0]'), ('select', '$.count[*]'),
])
def test_impossible_envelopes_fail_without_execution(operation, path):
    with pytest.raises(SandboxError, match='output path.*impossible'):
        Compiler.validate_result_path({'operation': operation}, path)


def test_paginated_envelope_keeps_cursor_and_collection_paths():
    spec = {'operation': 'select', 'pagination': {'cursor_argument': 'cursor', 'page_size': 5}}
    for path in ('$.next_cursor', '$.has_more', '$.total_count', '$.records[0].id',
                 '$.records[?(@.active == true)].id', '$.records[*].id'):
        Compiler.validate_result_path(spec, path)
    with pytest.raises(SandboxError, match='impossible'):
        Compiler.validate_result_path({'operation': 'select'}, '$.next_cursor')


def test_blueprint_shorthand_preserves_record_columns_but_not_invented_write_outputs():
    for operation in ('select', 'insert'):
        Compiler.validate_result_path({'operation': operation}, 'id', shorthand=True)
    for operation in ('update', 'delete', 'aggregate_count'):
        with pytest.raises(SandboxError, match='impossible'):
            Compiler.validate_result_path({'operation': operation}, 'id', shorthand=True)
