from itertools import product
import random

import pytest

from env_factory_v2.contracts.expressions import ContractError, evaluate, validate_expr
from env_factory_v2.contracts.integer_cover import bounded_cover


def solve(groups, target, **budgets):
    return bounded_cover(groups, target, **{'max_states': 10000,
        'max_transitions': 1000000, 'max_plans': 10000, **budgets})


def group(key, units, cost, minimum=1, maximum=4):
    return dict(key=key, units=units, cost=cost, minimum=minimum, maximum=maximum)


def brute(groups, target):
    feasible = []
    for counts in product(*[[0, *range(g['minimum'], g['maximum'] + 1)] for g in groups]):
        quantity = sum(n * g['units'] for n, g in zip(counts, groups))
        cost = sum(n * g['cost'] for n, g in zip(counts, groups))
        if quantity >= target:
            plan = tuple(sorted((g['key'], n) for n, g in zip(counts, groups) if n))
            feasible.append((cost, quantity, plan))
    if not feasible:
        return None, set()
    objective = min((c, q) for c, q, _ in feasible)
    return objective, {p for c, q, p in feasible if (c, q) == objective}


def test_exact_solver_matches_independent_exhaustive_search():
    rng = random.Random(713)
    for _ in range(150):
        groups = [group(str(i), rng.randint(1, 6), rng.randint(0, 9),
            rng.randint(1, 4), rng.randint(0, 4)) for i in range(rng.randint(0, 4))]
        target = rng.randint(0, 25)
        objective, plans = brute(groups, target)
        actual = solve(groups, target)
        assert actual['feasible'] == (objective is not None)
        if objective is not None:
            assert (actual['cost'], actual['quantity']) == objective
        assert {tuple((x['key'], x['count']) for x in p) for p in actual['plans']} == plans
        assert solve(list(reversed(groups)), target) == actual


def test_large_minimum_and_all_equal_optima():
    assert solve([group('a', 6, 100, 100, 100)], 23)['quantity'] == 600
    result = solve([group('a', 1, 1, maximum=1), group('b', 1, 1, maximum=1)], 1)
    assert len(result['plans']) == 2


def test_dynamic_stock_changes_optimum_and_expression_integration():
    groups = [group('a', 6, 100, 2, 4), group('b', 5, 110, 1, 3)]
    assert solve(groups, 30)['cost'] == 620
    groups[0]['maximum'] = 5
    expression = {'bounded_cover': {'items': {'ref': 'data.groups'}, 'target': {'literal': 30},
        'max_states': 10000, 'max_transitions': 1000000, 'max_plans': 10000}}
    validate_expr(expression, roots={'data'})
    assert evaluate(expression, {'data': {'groups': groups}})['cost'] == 500


@pytest.mark.parametrize('budget', [{'max_states': 1}, {'max_transitions': 1}, {'max_plans': 1}])
def test_budget_failure_is_not_infeasibility_or_partial_optimum(budget):
    with pytest.raises(ContractError, match='exceeds max_'):
        solve([group('a', 1, 1), group('b', 1, 1)], 2, **budget)


@pytest.mark.parametrize('change', [{'units': 0}, {'cost': -1}, {'minimum': True},
    {'maximum': 2.0}, {'minimum': 0}])
def test_invalid_coefficients_rejected(change):
    with pytest.raises(ContractError):
        solve([{**group('a', 1, 1), **change}], 1)


def test_duplicate_group_identity_rejected():
    with pytest.raises(ContractError, match='unique'):
        solve([group('a', 1, 1), group('a', 1, 2)], 2)


def test_two_hundred_candidates_do_not_require_cartesian_enumeration():
    groups = [group(f'g{i:03d}', 1, i + 1, maximum=2) for i in range(200)]
    result = solve(groups, 50, max_transitions=100000)
    assert result['cost'] == 650 and result['quantity'] == 50
    assert len(result['plans']) == 1 and len(result['plans'][0]) == 25
