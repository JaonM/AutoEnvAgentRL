import copy
import json

import pytest

from env_factory.sandbox_runtime import DeclarativeMetricEvaluator as Evaluator, SandboxError
from env_factory.generation.answer_contract import bind_answer_contract
from tests.generation.test_answer_contract import answer_source


def expected(tolerance=None):
    target = {'key': 'ratio', 'expression': {'op': 'div', 'args': [{'literal': 48}, {'literal': 76}]}}
    if tolerance is not None:
        target['tolerance'] = tolerance
    return {'answer_format': 'json_object', 'targets': [target]}


def test_repeating_ratio_requires_declared_tolerance_and_rejects_wrong_math():
    good = json.dumps({'ratio': 48 / 76})
    assert not Evaluator._value_targets_match(good, expected(), {})
    assert Evaluator._value_targets_match(good, expected(1e-6), {})
    assert Evaluator._value_targets_match('{"ratio":0.631579}', expected(1e-6), {})
    assert not Evaluator._value_targets_match('{"ratio":0.6316}', expected(1e-6), {})
    assert not Evaluator._value_targets_match('{"ratio":0.65}', expected(1e-6), {})
    assert not Evaluator._value_targets_match(good, expected(0), {})


@pytest.mark.parametrize('value', [True, '0.631579', None, [], {}, float('nan'), float('inf')])
def test_tolerance_does_not_coerce_nonfinite_or_nonnumeric_answers(value):
    assert not Evaluator._value_targets_match(json.dumps({'ratio': value}), expected(1e-6), {})


@pytest.mark.parametrize('value', [True, '0.01', -1, 1.01, float('nan'), float('inf')])
def test_invalid_tolerances_are_rejected(value):
    with pytest.raises(SandboxError, match='tolerance'):
        Evaluator.validate_value_targets(expected(value))


def test_other_fields_remain_exact_and_tolerance_is_not_recursive():
    rule = expected(1e-6)
    rule['targets'].append({'key': 'evidence', 'expression': {'literal': [3, 'A']}})
    assert Evaluator._value_targets_match('{"ratio":0.631579,"evidence":[3,"A"]}', rule, {})
    assert not Evaluator._value_targets_match('{"ratio":0.631579,"evidence":[3.000001,"A"]}', rule, {})
    for wanted in [True, '0.63', [0.63]]:
        invalid = copy.deepcopy(rule)
        invalid['targets'][0]['expression'] = {'literal': wanted}
        assert not Evaluator._value_targets_match(json.dumps({'ratio': wanted, 'evidence': [3, 'A']}), invalid, {})


def test_public_contract_discloses_allowed_error_without_gold_answer():
    source, _ = answer_source()
    source['answer_contract']['schema']['properties']['voice_types'] = {'type': 'number'}
    target = source['metric_implementations'][0]['expected']['targets'][0]
    target.update(expression={'op': 'div', 'args': [{'literal': 48}, {'literal': 76}]}, tolerance=1e-6)
    source['scenarios'][0]['steps'][0]['content'] = '{"voice_types":0.631579}'
    bind_answer_contract(source)
    public = source['description']['public_input']
    assert public['numeric_answer_tolerances'] == {'voice_types': 1e-6}
    assert 'absolute errors' in public['initial_user_message']
    assert '0.631579' not in public['initial_user_message']


@pytest.mark.parametrize('kind', ['integer', 'string', 'array'])
def test_tolerance_cannot_loosen_integer_counts_or_text(kind):
    source, _ = answer_source()
    source['answer_contract']['schema']['properties']['voice_types'] = (
        {'type': 'array', 'items': {'type': 'number'}} if kind == 'array' else {'type': kind})
    target = source['metric_implementations'][0]['expected']['targets'][0]
    target['tolerance'] = 0.1
    with pytest.raises(ValueError, match='ANSWER_CONTRACT_TYPES'):
        bind_answer_contract(source)
