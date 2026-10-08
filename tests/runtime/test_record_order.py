import copy
import json

import pytest

from env_factory.sandbox_runtime import (
    DeclarativeMetricEvaluator, DeclarativeToolCompiler, SandboxError, normalize_record_order,
)


def test_query_sorts_actual_updated_values_in_mixed_directions_with_nulls_last():
    class Data:
        schemas = {'parts': {'columns': [{'name': name} for name in ('percent', 'name')]}}
        rows = [{'percent': 72, 'name': 'B'}, {'percent': 20, 'name': 'A'},
                {'percent': 72, 'name': 'A'}, {'percent': None, 'name': 'C'}]

        def table(self, name):
            return copy.deepcopy(self.rows)

    data = Data()
    query = DeclarativeToolCompiler(data).compile({'operation': 'select', 'table': 'parts',
        'order_by': [{'field': 'percent', 'direction': 'desc'}, 'name']})
    assert [(r['percent'], r['name']) for r in query({})['records']] == [(72, 'A'), (72, 'B'), (20, 'A'), (None, 'C')]
    data.rows[1]['percent'] = 80
    assert [r['percent'] for r in query({})['records']] == [80, 72, 72, None]
    ascending = DeclarativeToolCompiler(data).compile({'operation': 'select', 'table': 'parts', 'order_by': ['percent']})
    assert [r['percent'] for r in ascending({})['records']] == [72, 72, 80, None]


@pytest.mark.parametrize('order', [['missing'], ['code', 'code'], [{'field': 'code', 'direction': 'sideways'}],
                                 [{'field': 'code'}], [{'field': 'code', 'direction': 'asc', 'extra': 1}], None])
def test_order_rejects_unknown_ambiguous_and_malformed_fields(order):
    with pytest.raises(SandboxError):
        normalize_record_order(order, columns={'code'})


def score(wanted, actual, order):
    expected = {'answer_format': 'json_object', 'targets': [
        {'key': 'records', 'expression': {'literal': wanted}, 'array_order': order}]}
    return DeclarativeMetricEvaluator._value_targets_match(json.dumps({'records': actual}), expected, {})


def test_equal_key_permutations_pass_but_wrong_primary_order_and_membership_fail():
    records = [{'code': 'A', 'id': 'a1'}, {'code': 'A', 'id': 'a2'}, {'code': 'B', 'id': 'b1'}]
    assert score(records, [records[1], records[0], records[2]], ['code'])
    assert not score(records, list(reversed(records)), ['code'])
    assert not score(records, [records[0], records[0], records[2]], ['code'])
    assert not score(records, records[:-1], ['code'])
    assert not score(records, [{**records[0], 'id': 'invented'}, *records[1:]], ['code'])
    assert not score(records, [records[1], records[0], records[2]], ['code', 'id'])
    assert score([], [], ['code'])


def test_unordered_arrays_keep_duplicate_counts_and_types():
    assert score([1, 1, True, 'x'], ['x', True, 1, 1], [])
    assert not score([1, 1, True, 'x'], ['x', True, True, 1], [])
    assert not score([1], ['1'], [])
    assert not score([1], [float('nan')], [])


def test_declared_directions_apply_to_answer_even_if_expected_tool_sequence_differs():
    records = [{'n': 2, 'id': 'b'}, {'n': 1, 'id': 'z'}, {'n': 2, 'id': 'a'}]
    order = [{'field': 'n', 'direction': 'desc'}, 'id']
    assert score(records, [records[2], records[0], records[1]], order)
    assert not score(records, records, order)
    assert not score(records, [{'n': 2}, *records[1:]], order)

