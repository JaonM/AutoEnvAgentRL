import pytest

from env_factory_v2.expressions import ContractError, evaluate, validate_expr


def run(operator, *values):
    expression = {operator: [{'literal': value} for value in values]}
    validate_expr(expression, roots=set())
    return evaluate(expression, {})


def test_exact_money_rounds_after_combining_segments():
    # Two thirds must round to one cent, not zero from rounding each segment.
    assert run('rational_sum_floor', [[1, 3], [1, 3]]) == 0
    assert run('rational_sum_floor', [[2, 3], [2, 3]]) == 1
    assert run('rational_sum_floor', [[1000 * 10, 30], [2000 * 20, 30]]) == 1666
    assert run('floor_div', 9000 * 19, 30) == 5700
    assert type(run('floor_div', 9000 * 19, 30)) is int


def test_large_integer_boundary_never_uses_float():
    numerator = 10**100 + 1
    assert run('floor_div', numerator, 3) == numerator // 3
    assert run('ceil_div', numerator, 3) == numerator // 3 + 1
    assert run('rational_sum_floor', [[numerator, 3], [1, 3]]) == (numerator + 1) // 3


@pytest.mark.parametrize('a,b,floor,ceiling', [(23, 6, 3, 4), (24, 6, 4, 4),
    (-23, 6, -4, -3), (23, -6, -4, -3), (-23, -6, 3, 4)])
def test_signed_quotients(a, b, floor, ceiling):
    assert run('floor_div', a, b) == floor
    assert run('ceil_div', a, b) == ceiling


@pytest.mark.parametrize('values', [(1, 0), (True, 3), (3, False), (3.0, 2)])
def test_invalid_quotients_rejected(values):
    for operator in ['floor_div', 'ceil_div']:
        with pytest.raises(ContractError):
            run(operator, *values)


@pytest.mark.parametrize('terms', [[[1, 0]], [[True, 3]], [[1.0, 3]], [[1]],
    [[1, 3, 4]], 'not a list', [[1, 1]] * 10001])
def test_invalid_rational_terms_rejected(terms):
    with pytest.raises(ContractError):
        run('rational_sum_floor', terms)


def test_integer_guard_and_empty_sum():
    assert run('is_integer', 0)
    assert not run('is_integer', True)
    assert not run('is_integer', 2.0)
    assert run('rational_sum_floor', []) == 0


def test_exact_billing_contract_rewards_correct_integer_and_rejects_early_rounding():
    from env_factory_v2.examples import base, answer, lit, scenario
    from env_factory_v2.runtime import Sandbox
    from env_factory_v2.reward import grade
    source = base('exact_segment_billing', 'direct_response',
        '前10天按1000分/30天计费，后20天按2000分/30天计费；精确累加后统一向下取整。')
    source['rules'] = [{'id': 'billing', 'public': source['task'], 'predicate': lit(True)}]
    source['answers'] = [answer('cents', '统一取整后的费用（整数分）',
        {'rational_sum_floor': [lit([[10000, 30], [40000, 30]])]}, {'type': 'integer'})]
    source['scenarios'] = [scenario('correct', [], {'cents': lit(1666)}),
        scenario('early_rounding', [], {'cents': lit(1665)}, False)]
    world = Sandbox(source)
    assert grade(world, {'cents': 1666})['success']
    assert not grade(Sandbox(source), {'cents': 1665})['success']


@pytest.mark.parametrize('operator', ['is_integer', 'rational_sum_floor'])
def test_unary_shape_error_gives_exact_operand_array_requirement(operator):
    with pytest.raises(ContractError, match='expected a JSON array of 1 expression operand'):
        validate_expr({operator: {'ref': 'args.value'}}, roots={'args'})


def test_published_rational_map_grammar_executes_exactly():
    import json
    from pathlib import Path
    import re
    grammar = Path('src/env_factory_v2/CONTRACT.md').read_text()
    text = re.search(r'An inline map must also be inside the operand array:\n(.*?)\.\nThe outer array',
        grammar, re.S).group(1)
    expression = json.loads(text)
    validate_expr(expression, roots={'vars'})
    assert evaluate(expression, {'vars': {'segments': [
        {'price_cents': 1000, 'days': 10, 'period_days': 30},
        {'price_cents': 2000, 'days': 20, 'period_days': 30}]}}) == 1666
