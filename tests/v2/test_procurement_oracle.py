"""Independent procurement checks, not native rollout or measured yield."""
import itertools
import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_procurement_oracle import minimum_order, valid_quote, optimum, commit, expected


def quote(key, pack=6, minimum=2, price=100, stock=24):
    return dict(id=key, pack=pack, minimum_boxes=minimum, box_cents=price, stock=stock,
                starts_at=0, ends_at=10, delivery_day=12, quality=4)


def test_minimum_is_constraint_not_extra_purchase():
    assert minimum_order(23, 6, 2) == {'boxes': 4, 'overage': 1}
    assert minimum_order(1, 6, 2) == {'boxes': 2, 'overage': 11}
    with pytest.raises(ValueError):
        minimum_order(True, 6, 2)


@pytest.mark.parametrize('now,valid', [(-1, False), (0, True), (10, True), (11, False)])
def test_inclusive_quote_boundary(now, valid):
    assert valid_quote(quote('A'), now) is valid


def test_combination_can_beat_cheapest_unit_price():
    rows = [quote('A', pack=10, minimum=2, price=90, stock=20),
            quote('B', pack=6, minimum=1, price=60, stock=12)]
    assert optimum(rows, 12, 10) == {'cost_cents': 120, 'overage': 0, 'plans': [(('B', 2),)]}


def test_exact_optimizer_agrees_with_independent_exhaustive_enumeration():
    rows = [quote('A', pack=3, minimum=2, price=70, stock=12),
            quote('B', pack=5, minimum=1, price=100, stock=15),
            quote('C', pack=4, minimum=2, price=85, stock=12)]
    for demand in range(1, 41):
        alternatives = []
        for counts in itertools.product([0, 2, 3, 4], [0, 1, 2, 3], [0, 2, 3]):
            qty = sum(n * q['pack'] for n, q in zip(counts, rows))
            if qty >= demand:
                alternatives.append((sum(n * q['box_cents'] for n, q in zip(counts, rows)), qty - demand))
        actual = optimum(rows, demand, 10)
        if alternatives:
            assert (actual['cost_cents'], actual['overage']) == min(alternatives)
        else:
            assert actual is None
        assert actual == optimum(list(reversed(rows)), demand, 10)


def test_equal_optima_are_preserved_and_quality_filters_apply():
    rows = [quote('A', pack=6, minimum=1, stock=6), quote('B', pack=6, minimum=1, stock=6)]
    assert optimum(rows, 6, 10)['plans'] == [(('A', 1),), (('B', 1),)]
    rows[0]['quality'] = 3
    assert optimum(rows, 6, 10, quality=4)['plans'] == [(('B', 1),)]
    assert optimum(rows, 6, 10, deadline=11) is None


@pytest.mark.parametrize('fault', ['budget', 'stock', 'expiry', 'minimum', 'refusal', 'duplicate_order'])
def test_atomic_rejection_never_leaks_partial_writes(fault):
    data = dict(quotes=[quote('A'), quote('B')], budget_cents=1000, orders=[])
    now, approved, plan = 10, True, [('A', 2), ('B', 2)]
    if fault == 'budget': data['budget_cents'] = 399
    if fault == 'stock': data['quotes'][1]['stock'] = 6
    if fault == 'expiry': now = 11
    if fault == 'minimum': plan[1] = ('B', 1)
    if fault == 'refusal': approved = False
    if fault == 'duplicate_order': data['orders'] = [{'id': 'O1'}]
    passed, final = commit(data, plan, now, approved=approved, order_id='O1')
    assert not passed and final == data


def test_second_failure_preserves_first_success():
    data = dict(quotes=[quote('A'), quote('B')], budget_cents=300, orders=[])
    passed, first = commit(data, [('A', 2)], 10, approved=True, order_id='O1')
    assert passed and first['budget_cents'] == 100
    passed, second = commit(first, [('B', 2)], 10, approved=True, order_id='O2')
    assert not passed and second == first
    assert data['quotes'][0]['stock'] == 24
    assert first['quotes'][0]['stock'] == 12


def test_timed_truth_cannot_guess_virtual_time():
    spec = {'kind': 'procurement_quote', 'params': {}}
    data = {'quotes': [quote('A')]}
    with pytest.raises(ValueError, match='observed virtual time'):
        expected(spec, data, {'quote_id': 'A'})
    assert expected(spec, data, {'quote_id': 'A'}, now=10)['answer'] == {
        'valid': True, 'minimum_units': 12}
    assert expected(spec, data, {'quote_id': 'A'}, now=11)['answer']['valid'] is False


def test_partial_truth_keeps_success_and_reports_remaining_demand():
    data = {'quotes': [quote('A'), quote('B')],
            'budgets': [{'id': 'P1', 'remaining_cents': 300}], 'orders': []}
    facts = {'purchase_plans': [[('A', 2)], [('B', 2)]], 'approvals': [True, True],
             'order_ids': ['O1', 'O2'], 'demand': 23}
    spec = {'kind': 'procurement_partial', 'params': {}}
    truth = expected(spec, data, facts, now=10)
    assert truth['answer'] == {'committed_ids': ['O1'], 'shortfall': 11}
    assert truth['final_state']['budgets'][0]['remaining_cents'] == 100
    assert truth['initial'] == data
    refused = expected(spec, data, facts, now=10, refuse=True)
    assert refused['final_state'] == data
    assert refused['answer'] == {'committed_ids': [], 'shortfall': 23}


@pytest.mark.parametrize('change', ['none', 'price', 'stock', 'budget', 'expired', 'refused'])
def test_large_purchase_respects_exact_authorization_and_current_validity(change):
    original = {'quotes': [quote('A', minimum=1), quote('B', minimum=1, price=200)],
                'budgets': [{'id': 'P1', 'remaining_cents': 1000}], 'orders': []}
    snapshot = {'quotes': copy.deepcopy(original['quotes']), 'budget_cents': 1000}
    current = copy.deepcopy(original)
    facts = dict(demand=12, deadline=12, minimum_quality=4, proposal_time=10,
                 authorized_plan=[['A', 2]], approve_purchase=change != 'refused', order_id='O1')
    if change == 'price': current['quotes'][0]['box_cents'] = 110
    if change == 'stock': current['quotes'][0]['stock'] = 18
    if change == 'budget': current['budgets'][0]['remaining_cents'] = 900
    truth = expected({'kind': 'procurement_large', 'params': {}}, current, facts,
                     now=11 if change == 'expired' else 10, authorized_snapshot=snapshot)
    assert truth['answer'] == {'cost_cents': 200, 'overage': 0, 'committed': change == 'none'}
    if change != 'none':
        assert truth['final_state'] == current
    else:
        assert truth['final_state']['budgets'][0]['remaining_cents'] == 800
        assert truth['final_state']['quotes'][0]['stock'] == 12
    assert snapshot['quotes'] == original['quotes']


def test_large_purchase_accepts_equivalent_optimum_without_hidden_supplier_tie_break():
    data = {'quotes': [quote('A', minimum=1), quote('B', minimum=1)],
            'budgets': [{'id': 'P1', 'remaining_cents': 1000}], 'orders': []}
    snapshot = {'quotes': copy.deepcopy(data['quotes']), 'budget_cents': 1000}
    for key in ['A', 'B']:
        facts = dict(demand=12, deadline=12, minimum_quality=4, proposal_time=10,
                     authorized_plan=[[key, 2]], approve_purchase=True, order_id='O1')
        truth = expected({'kind': 'procurement_large', 'params': {}}, data, facts,
                         now=10, authorized_snapshot=snapshot)
        assert truth['answer']['committed']
        assert truth['final_state']['orders'][0]['lines'][0]['quote_id'] == key


def test_five_procurement_fixtures_have_independent_truth():
    from prepare_v2_cross_business import procurement_cases
    from v2_natural_oracle import expand
    rows = procurement_cases()
    assert [row['id'] for row in rows] == [6, 7, 8, 9, 10]
    truths = []
    for row in rows:
        data = expand(row['business'])
        facts = copy.deepcopy(row['user']['facts'])
        options = {'now': 10}
        if row['acceptance']['kind'] == 'procurement_large':
            facts['proposal_time'] = 10
            options['authorized_snapshot'] = {'quotes': copy.deepcopy(data['quotes']),
                                               'budget_cents': 600}
        truth = expected(row['acceptance'], data, facts, **options)
        truths.append(truth)
        assert data == truth['initial']
        assert 'not frozen' in row['business']['provenance']
    assert truths[0]['answer'] == {'boxes': 4, 'overage': 1}
    assert truths[1]['answer'] == {'valid': True, 'minimum_units': 12}
    assert truths[2]['answer'] == {'cost_cents': 400, 'overage': 1}
    assert truths[3]['answer'] == {'committed_ids': ['O1'], 'shortfall': 11}
    assert len(truths[4]['initial']['quotes']) == 242
    assert truths[4]['answer'] == {'cost_cents': 440, 'overage': 4, 'committed': True}
    assert truths[4]['final_state']['budgets'][0]['remaining_cents'] == 160
