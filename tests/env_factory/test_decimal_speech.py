import pytest

from env_factory.core import extract
from env_factory.core.numerals import said


@pytest.mark.parametrize('speech,value', [
    ('The price difference is $4.50 total.', 4.5),
    ('价格差为4.500元。', 4.5),
    ('The amount is 1,234.500.', 1234.5),
    ('The amount is 4.00.', 4),
    ('The amount is 4.', 4.0),
    ('Your refund is $3.50.', 350),  # retain existing minor-unit support
    ('You receive a refund of 4.50.', -4.5),  # sign may be expressed in words
])
def test_equivalent_decimal_spellings_are_preserved(speech, value):
    assert said(speech, value)
    assert extract.verify(value, {'type': 'number'}, speech) == value


@pytest.mark.parametrize('speech,value', [
    ('The price difference is 4.51.', 4.5),
    ('The price difference is 4.5001.', 4.5),
    ('The price difference is 14.50.', 4.5),
    ('The price difference is 4.50.', 4),
    ('The amount is 1,234.501.', 1234.5),
    ('The price was not stated.', 4.5),
])
def test_other_values_and_missing_speech_still_fail(speech, value):
    assert not said(speech, value)
    assert extract.verify(value, {'type': 'number'}, speech) is None
