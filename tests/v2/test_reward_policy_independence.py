import copy

import pytest

from env_factory_v2.contract import ContractError, compile_contract
from env_factory_v2.examples import direct, simple, lit, ref
from env_factory_v2.reward import equivalent, grade
from env_factory_v2.runtime import Sandbox


def test_numeric_representation_cleanup_never_accepts_material_errors_or_bool_numbers():
    assert equivalent(37.4, 224 - (224 * 0.9 - 15), 'numeric')
    assert equivalent(3.625, 1100 * 3 * 0.001 + 65 * 5 * 0.001, 'numeric')
    assert not equivalent(37.41, 37.4, 'numeric')
    assert not equivalent(True, 1, 'numeric')
    assert not equivalent(0.0, 1e-16, 'numeric')
    assert not equivalent(2**53 + 1, float(2**53), 'numeric')
    assert not equivalent(10**30 + 1, 10**30, 'numeric')
    assert not equivalent(1e12 + 1, 1e12, 'numeric')
    assert not equivalent(37.4, 224 - (224 * 0.9 - 15), 'exact')


def test_decimal_answer_gets_full_reward_with_explicit_numeric_contract():
    source = direct()
    source['answers'][0].update(schema={'type': 'number'}, comparison='numeric',
        expected={'sub': [lit(224), {'sub': [{'mul': [lit(224), lit(0.9)]}, lit(15)]}]})
    assert grade(Sandbox(source), {'temperature': 37.4})['reward'] == 1
    assert grade(Sandbox(source), {'temperature': 37.41})['reward'] == 0


@pytest.mark.parametrize('target', ['answer', 'observation', 'branch'])
def test_grader_cannot_depend_on_a_reference_observation_name(target):
    source = simple()
    if target == 'answer':
        source['answers'][0]['expected'] = ref('evidence.found.items')
    else:
        source['success'] = [{'id': 'observe', 'public': '读取实际款号。', 'when': lit(True),
            'required_tools': ['find_items'], 'required_stages': ['size']}]
        if target == 'branch':
            source['success'][0]['when'] = {'eq': [ref('evidence.found.total'), lit(1)]}
        else:
            source['success'][0]['required_observations'] = [{'id': 'items', 'tools': ['find_items'],
                'columns': ['id'], 'expected': ref('evidence.found.items')}]
    with pytest.raises(ContractError, match='policy-chosen observation IDs'):
        compile_contract(source)


def test_reference_recipes_may_still_use_actual_observation_ids():
    compiled = compile_contract(simple())
    assert compiled['source']['scenarios'][0]['answer']['ids']['map']['items'] == ref('evidence.found.items')
    assert compiled['public']['business_call_limit'] is None


def test_branch_call_bounds_must_be_nonnegative_consistent_integers():
    for bounds in [{'max_business_calls': True}, {'min_business_calls': -1},
                   {'min_business_calls': 2, 'max_business_calls': 1}]:
        source = simple()
        source['success'] = [{'id': 'bounded', 'public': '查询预算。', 'when': lit(True),
            'required_tools': ['find_items'], 'required_stages': ['size'], **bounds}]
        with pytest.raises(ContractError, match='business call bound'):
            compile_contract(source)
