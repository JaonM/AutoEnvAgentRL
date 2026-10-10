import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_natural_semantics import normalize, rename_observations, wrong_projection
from env_factory_v2.contract import ContractError


def entry(role, selectors, **kwargs):
    return {'role': role, 'selectors': [{'field': f, 'path': p} for f, p in selectors],
            'packing': 'single', 'time_unit': 'native', 'names': 'native', **kwargs}


def test_minutes_and_nested_fields_preserve_business_values():
    answer = {'duration': 165, 'ticket': {'status': 'repairing', 'eta': '2026-10-12'}}
    entries = [entry('hours', [('duration', '')], time_unit='minutes'),
               entry('status', [('ticket', 'status')]), entry('eta', [('ticket', 'eta')])]
    assert normalize(answer, entries) == {'hours': 2.75, 'status': 'repairing', 'eta': '2026-10-12'}


def test_split_fields_and_array_records_keep_order_and_duplicates():
    entries = [entry('names', [('first', ''), ('second', ''), ('third', '')], packing='list')]
    assert normalize({'first': '丙', 'second': '乙', 'third': '甲'}, entries)['names'] == ['丙', '乙', '甲']
    entries = [entry('ids', [('records', '*.id')])]
    assert normalize({'records': [{'id': 'B'}, {'id': 'A'}, {'id': 'B'}]}, entries)['ids'] == ['B', 'A', 'B']


def test_projection_cannot_fill_missing_answers_or_compute_arbitrary_values():
    with pytest.raises(ContractError, match='absent'):
        normalize({'ticket': {'status': 'repairing'}}, [entry('eta', [('ticket', 'eta')])])
    with pytest.raises(ContractError, match='closed'):
        normalize({'ticket': {}}, [entry('status', [('ticket', '__import__(os)')])])
    with pytest.raises(ContractError, match='time conversion'):
        normalize({'fee': 165}, [entry('fee', [('fee', '')], time_unit='minutes')])


def test_aliases_use_only_public_mapping_and_do_not_reorder():
    entries = [entry('names', [('ids', '')], names='public_id_to_name')]
    assert normalize({'ids': ['C', 'B', 'A']}, entries, name_aliases={'A': '甲', 'B': '乙', 'C': '丙'})['names'] == ['丙', '乙', '甲']
    with pytest.raises(ContractError, match='unrecognized'):
        normalize({'ids': ['X']}, entries, name_aliases={'A': '甲'})


def test_id_renaming_changes_fixture_references_but_preserves_literal_values():
    scenario = {'id': 'case', 'steps': [{'kind': 'call', 'id': 'read', 'tool': 'query', 'arguments': {}},
        {'kind': 'call', 'id': 'next', 'tool': 'query', 'arguments': {'id': {'$result': 'read', 'path': 'items.0.id'}}}],
        'answer': {'x': {'ref': 'evidence.next.items.0.id'}, 'y': {'literal': {'id': 'read'}}}}
    changed = rename_observations(scenario)
    assert changed['steps'][1]['arguments']['id']['$result'] == 'independent_read'
    assert changed['answer']['x']['ref'] == 'evidence.independent_next.items.0.id'
    assert changed['answer']['y'] == scenario['answer']['y']
    assert scenario['steps'][0]['id'] == 'read'


def test_wrong_value_changes_only_one_projected_leaf():
    answer = {'ticket': {'active': False, 'id': 'T1'}, 'fee': 20}
    changed = wrong_projection(answer, entry('active', [('ticket', 'active')]))
    assert changed == {'ticket': {'active': True, 'id': 'T1'}, 'fee': 20}
    assert answer['ticket']['active'] is False
