import math
import pytest
from env_factory.sandbox_runtime import AcceptanceScenarioRunner, SandboxError, DeclarativeMetricEvaluator


def run_assertion(actual, expected, operator='lte', operation='reward', path='$.reward'):
    runner = AcceptanceScenarioRunner(lambda *args: (200, {'reward': actual, 'count': actual}, {}))
    return runner.run({'steps': [{'operation': operation, 'step_id': 'result', 'tool_name': 'lookup'}],
                       'assertions': [{'source': 'step:result', 'path': path,
                                       'operator': operator, 'expected': expected}]})


@pytest.mark.parametrize('actual,operator', [(math.fsum([.4,.1,.1]), 'lte'),
    (math.nextafter(.6, 0), 'gte'), (math.nextafter(.6, 1), 'eq')])
def test_reward_assertions_accept_only_machine_roundoff(actual, operator):
    run_assertion(actual, .6, operator)


@pytest.mark.parametrize('actual', [.60000000001, .61, float('nan'), float('inf')])
def test_actual_reward_excess_or_nonfinite_still_fails(actual):
    with pytest.raises(SandboxError):
        run_assertion(actual, .6)


@pytest.mark.parametrize('operation,path', [('tool_call','$.reward'), ('reward','$.count')])
def test_business_values_do_not_get_reward_tolerance(operation, path):
    with pytest.raises(SandboxError):
        run_assertion(math.nextafter(.6, 1), .6, operation=operation, path=path)
    assert not DeclarativeMetricEvaluator._compare(math.nextafter(.6,1),'lte',.6)


@pytest.mark.parametrize('operator,direction', [('lt', 0), ('gt', 1)])
def test_strict_reward_comparison_preserves_adjacent_float_boundary(operator, direction):
    run_assertion(math.nextafter(.6, direction), .6, operator)
    for actual in [.6, math.nextafter(.6, 1 - direction), None, True,
                   '0.5', float('nan'), float('inf'), float('-inf')]:
        with pytest.raises(SandboxError, match='scenario assertion failed'):
            run_assertion(actual, .6, operator)


@pytest.mark.parametrize('operator', ['lt', 'gt'])
def test_strict_comparison_rejects_invalid_expected(operator):
    for expected in [None, False, '1', float('nan'), float('inf')]:
        assert not DeclarativeMetricEvaluator._compare(.6, operator, expected)


@pytest.mark.parametrize('operator,actual', [('lt', .5), ('gt', .7)])
def test_compiler_accepts_strict_metric_executed_by_runtime(operator, actual):
    from env_factory.generation.contract_compiler import ContractCompiler
    metric = {'id': 'capacity', 'type': 'rule-based', 'category': 'outcome', 'score_range': [0, 1]}
    rule = {'metric_id': 'capacity', 'source': 'business_state', 'path': '$.available',
            'operator': operator, 'expected': .6, 'score_mapping': {'pass': 1, 'fail': 0}}
    ContractCompiler._validate_metric_implementations([rule], [metric])
    evaluator = DeclarativeMetricEvaluator()
    assert evaluator.evaluate(rule, {'business_state': {'available': actual}}) == 1
    assert evaluator.evaluate(rule, {'business_state': {'available': .6}}) == 0
