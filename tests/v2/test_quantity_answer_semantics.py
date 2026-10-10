import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/experiments'))
from v2_natural_semantics import normalize, fill_projections, wrong_projection
from env_factory_v2.expressions import ContractError

ENTRY = {'role': 'overage', 'selectors': [{'field': 'optimization', 'path': 'quantity'}],
    'packing': 'single', 'time_unit': 'native', 'names': 'native', 'representation': 'total_quantity'}


def project(value, offset=11):
    return normalize({'optimization': {'quantity': value}}, [ENTRY],
        role_types={'overage': type(None) if value is None else int},
        quantity_offsets={'overage': offset})


def test_total_quantity_round_trip_and_changed_demand():
    answer = {'optimization': {'quantity': 12, 'cost': 500}}
    assert project(12) == {'overage': 1}
    changed = fill_projections(answer, [ENTRY], {'overage': 2}, quantity_offsets={'overage': 17})
    assert changed == {'optimization': {'quantity': 19, 'cost': 500}}
    assert project(changed['optimization']['quantity'], 17) == {'overage': 2}
    assert project(wrong_projection(answer, ENTRY)['optimization']['quantity']) != {'overage': 1}


@pytest.mark.parametrize('value', [True, 12.0, '12', [], [12, 13]])
def test_non_integer_or_ambiguous_quantity_rejected(value):
    with pytest.raises(ContractError):
        project(value)


@pytest.mark.parametrize('offset', [True, -1, 11.0, None])
def test_offset_is_strict_independent_integer(offset):
    with pytest.raises(ContractError):
        project(12, offset)


def test_infeasible_null_and_singleton():
    assert project(None) == {'overage': None}
    assert project([12]) == {'overage': 1}
    assert fill_projections({'optimization': {'quantity': 12}}, [ENTRY], {'overage': None},
        quantity_offsets={'overage': 11}) == {'optimization': {'quantity': None}}


def test_no_model_supplied_arithmetic_or_unoffered_role():
    with pytest.raises(ContractError):
        normalize({'optimization': {'quantity': 12}}, [ENTRY], role_types={'overage': int})
    with pytest.raises(ContractError):
        normalize({'optimization': {'quantity': 12}}, [ENTRY], role_types={'overage': bool},
            quantity_offsets={'overage': 11})
