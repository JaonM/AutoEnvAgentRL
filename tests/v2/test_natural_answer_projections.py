import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_natural_semantics import normalize, rename_observations, wrong_projection, fill_projections
from env_factory_v2.contract import ContractError


def entry(role, selectors, **kwargs):
    return {'role': role, 'selectors': [{'field': f, 'path': p} for f, p in selectors],
            'packing': 'single', 'time_unit': 'native', 'names': 'native', **kwargs}


def test_minutes_and_nested_fields_preserve_business_values():
    answer = {'duration': 165, 'ticket': {'status': 'repairing', 'eta': '2026-10-12'}}
    entries = [entry('hours', [('duration', '')], time_unit='minutes'),
               entry('status', [('ticket', 'status')]), entry('eta', [('ticket', 'eta')])]
    assert normalize(answer, entries) == {'hours': 2.75, 'status': 'repairing', 'eta': '2026-10-12'}


def test_root_notation_is_consistent_for_read_write_and_negative_projections():
    answer = {'cost': 3.5, 'ticket': {'active': True}}
    entries = [entry('cost', [('cost', '$')]),
               entry('active', [('ticket', '$.active')])]
    assert normalize(answer, entries) == {'cost': 3.5, 'active': True}
    assert fill_projections(answer, entries, {'cost': 7, 'active': False}) == {
        'cost': 7, 'ticket': {'active': False}}
    assert wrong_projection(answer, entries[1])['ticket']['active'] is False
    for path in ['$..active', '$[0]', '$.active+1']:
        with pytest.raises(ContractError, match='closed'):
            normalize(answer, [entry('active', [('ticket', path)])])


def test_conditional_scalar_alternatives_remain_ambiguous():
    with pytest.raises(ContractError, match='exactly one selector'):
        normalize({'member': 10, 'nonmember': 20},
                  [entry('payable', [('member', '$'), ('nonmember', '$')])])


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


def test_scalar_projection_requires_one_record_and_never_drops_extra_records():
    mapping = [entry('active', [('rows', '*.active')], packing='list')]
    assert normalize({'rows': [{'active': False}]}, mapping, role_types={'active': bool}) == {'active': False}
    with pytest.raises(ContractError, match='exactly one'):
        normalize({'rows': [{'active': False}, {'active': True}]}, mapping, role_types={'active': bool})


def test_correct_changed_answer_preserves_nested_layout_and_integer_schema():
    mapping = [entry('covered', [('assessment', 'covered')]), entry('fee', [('assessment', 'fee')]),
               entry('hours', [('minutes', '')], time_unit='minutes')]
    answer = {'assessment': {'covered': True, 'fee': 0}, 'minutes': 100}
    result = fill_projections(answer, mapping, {'covered': False, 'fee': 120, 'hours': 2.75})
    assert result == {'assessment': {'covered': False, 'fee': 120}, 'minutes': 165}
    assert type(result['minutes']) is int
    assert answer['assessment']['covered'] is True


def test_receipt_evidence_does_not_override_explicit_postquery_requirement():
    from audit_v2_natural import audited_observations
    from env_factory_v2.examples import transaction
    from env_factory_v2.runtime import Sandbox
    world = Sandbox(transaction())
    world.call('accounts', {}, call_id='before')
    args = {'from_id': {'$result': 'before', 'path': 'items.0.id'},
            'to_id': {'$result': 'before', 'path': 'items.1.id'}, 'amount': 20}
    world.interact('approve', question='批准？', proposal={'tool': 'transfer', 'arguments': args})
    world.call('transfer', args, call_id='write')
    truth = {'initial': world.initial, 'final_state': world.data, 'observations': {'accounts': ['A', 'B']}}
    assert all(c['passed'] for c in audited_observations(world, truth, 'transfer', {'policy': '原子提交并报告结果。'}))
    assert not all(c['passed'] for c in audited_observations(world, truth, 'transfer', {'policy': '提交后查询核验。'}))
