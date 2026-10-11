import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from env_factory_v2.expressions import ContractError
from v2_natural_semantics import project, fill_projections, wrong_projection, value_shape
from v2_mapping_validation import check_mapping
from v2_business_acceptance import prepare_branch_answer


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


def test_branch_scope_mismatch_is_inconclusive_with_comparable_evidence():
    observed = answer()
    before = copy.deepcopy(observed)
    truth = {'answer': {'done': [], 'pending_counts': [{'identity': 'b', 'count': 3}]}}
    prepared, report = prepare_branch_answer(observed, truth, entries(), {})
    assert prepared is None
    assert report['status'] == 'evaluator_projection_unresolved'
    assert report['proven_business_defect'] is False
    assert report['projected_answer']['done'] == ['a']
    assert report['independent_answer']['done'] == []
    assert observed == before


def test_correct_branch_keeps_real_answer_and_reversible_layouts_still_prepare():
    actual = answer()
    truth = {'answer': {'done': ['a'], 'pending_counts': [{'identity': 'b', 'count': 3}]}}
    prepared, report = prepare_branch_answer(actual, truth, entries(), {})
    assert prepared == actual and report['status'] == 'ready'
    entry = {'role': 'x', 'selectors': [{'field': 'value', 'path': ''}],
             'packing': 'single', 'names': 'native', 'time_unit': 'native'}
    prepared, report = prepare_branch_answer({'value': 1}, {'answer': {'x': 2}}, [entry], {})
    assert prepared == {'value': 2} and report['status'] == 'ready'


def test_bijective_columns_can_prepare_changed_record_population_without_guessing():
    entry = entries()[1]
    actual = {'pending': [{'key': 'a', 'details': {'number': 1}},
                          {'key': 'b', 'details': {'number': 3}}], 'unrelated': 'keep'}
    truth = {'pending_counts': [{'identity': 'b', 'count': 3}]}
    changed = fill_projections(actual, [entry], truth)
    assert changed == {'pending': [{'key': 'b', 'details': {'number': 3}}], 'unrelated': 'keep'}
    assert project(changed, entry) == truth['pending_counts']
    assert len(actual['pending']) == 2
    assert fill_projections(actual, [entry], {'pending_counts': []})['pending'] == []


def test_partial_columns_cannot_drop_unselected_fields_during_inverse():
    with pytest.raises(ContractError, match='unselected record fields'):
        fill_projections(answer(), [entries()[1]], {'pending_counts': []})
