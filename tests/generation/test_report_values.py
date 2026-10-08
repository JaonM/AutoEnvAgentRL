import json
from decimal import Decimal

import pytest

from env_factory.generation.report_values import report_value
from env_factory.generation.review_replay import baseline_reward_observations


def test_computed_preview_evidence_serializes_without_changing_evaluator_values():
    bundle = {'source': {'tables': []}, 'compiled_reward_rules': [
        {'metric_id': 'cost', 'operator': 'value_targets', 'expected': {'targets': [
            {'key': 'total', 'expression': {'op': 'add', 'args': [
                {'literal': 0.1}, {'literal': 0.2}]}}]}}]}
    observations = baseline_reward_observations(bundle)
    safe = report_value(observations)
    restored = json.loads(json.dumps(safe, allow_nan=False))
    assert restored['targets'][0]['value'] == {'decimal': '0.3'}
    assert observations['targets'][0]['value'] == {'decimal': '0.3'}
    # The report boundary itself must serialize, without caller-specific defaults.
    json.dumps(observations, allow_nan=False)


def test_large_integer_and_fractional_evidence_keep_exact_values():
    values = {'large': Decimal('123456789012345678901234567890'),
              'fraction': [Decimal('0.12345678901234567890123456789')], 'flag': True}
    restored = json.loads(json.dumps(report_value(values)))
    assert restored['large'] == 123456789012345678901234567890
    assert restored['fraction'] == [{'decimal': '0.12345678901234567890123456789'}]
    assert restored['flag'] is True


@pytest.mark.parametrize('value', ['NaN', 'Infinity', '-Infinity'])
def test_nonfinite_decimal_fails_explicitly(value):
    with pytest.raises(ValueError, match='REPORT_VALUE_INVALID'):
        report_value({'value': Decimal(value)})
