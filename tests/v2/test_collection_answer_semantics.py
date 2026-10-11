import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from env_factory_v2.expressions import ContractError
from v2_natural_semantics import project, fill_projections, wrong_projection, value_shape
from v2_mapping_validation import check_mapping


def entries():
    base = {'packing': 'list', 'names': 'native', 'time_unit': 'native'}
    return [{**base, 'role': 'done', 'selectors': [{'field': 'all', 'path': '*.key'}],
             'exclude_selectors': [{'field': 'pending', 'path': '*.key'}], 'columns': []},
            {**base, 'role': 'pending_counts', 'selectors': [{'field': 'pending', 'path': ''}],
             'exclude_selectors': [], 'columns': [{'name': 'identity', 'path': 'key'},
                                                {'name': 'count', 'path': 'details.number'}]}]


def answer():
    return {'all': [{'key': 'b'}, {'key': 'a'}],
            'pending': [{'key': 'b', 'details': {'number': 3}, 'extra': 'preserved'}]}


def test_difference_and_record_columns_are_answer_only_and_nonmutating():
    actual = answer()
    before = copy.deepcopy(actual)
    first, second = entries()
    assert project(actual, first) == ['a']
    assert project(actual, second) == [{'identity': 'b', 'count': 3}]
    check_mapping(entries(), {'done': list, 'pending_counts': list}, actual, [actual])
    assert actual == before
    truth = {'done': ['a'], 'pending_counts': [{'identity': 'b', 'count': 3}]}
    assert fill_projections(actual, entries(), truth) == actual
    truth['done'] = ['b']
    with pytest.raises(ContractError, match='inverse is ambiguous'):
        fill_projections(actual, entries(), truth)


@pytest.mark.parametrize('index', [0, 1])
def test_negative_changes_observed_projection_even_with_excluded_first_item(index):
    actual = answer()
    entry = entries()[index]
    changed = wrong_projection(actual, entry)
    assert changed != actual
    assert project(changed, entry) != project(actual, entry)


@pytest.mark.parametrize('kind', ['duplicate', 'outside', 'unknown', 'computed'])
def test_difference_cannot_hide_invalid_or_computed_subsets(kind):
    actual = answer()
    entry = entries()[0]
    if kind == 'duplicate':
        actual['pending'] *= 2
    elif kind == 'outside':
        actual['pending'][0]['key'] = 'absent'
    elif kind == 'unknown':
        entry['exclude_selectors'][0]['field'] = 'oracle'
    else:
        entry['exclude_selectors'][0]['path'] = '*[computed]'
    with pytest.raises(ContractError):
        project(actual, entry)


def test_empty_collections_and_shape_metadata_do_not_disclose_values():
    first, second = entries()
    assert project({'all': [], 'pending': []}, first) == []
    assert project({'pending': []}, second) == []
    shape = value_shape([{'identity': 'secret-value', 'count': 1234567}])
    assert shape == {'type': 'array', 'item_shapes': [{'type': 'object', 'properties': {
        'identity': {'type': 'str'}, 'count': {'type': 'int'}}}]}
