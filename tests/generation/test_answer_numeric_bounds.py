import pytest

from env_factory.generation.answer_contract import _schema, bind_answer_contract
from env_factory.sandbox_runtime import SandboxError, validate_json_schema
from tests.generation.test_answer_contract import answer_source


@pytest.mark.parametrize('kind', ['integer', 'number', ['integer', 'null'], ['number', 'null']])
def test_nested_numeric_bounds_are_published_and_enforced(kind):
    source, _ = answer_source()
    item = {'type': kind, 'minimum': 0, 'maximum': 10}
    source['answer_contract']['schema']['properties']['voice_types']['items'] = item
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {'literal': [4]}
    source['scenarios'][0]['steps'][0]['content'] = '{"voice_types":[4]}'
    contract = bind_answer_contract(source)
    assert contract['schema']['properties']['voice_types']['items'] == item
    assert '"minimum": 0' in source['description']['public_input']['initial_user_message']
    for values in ([0, 4, 10], *([[None]] if isinstance(kind, list) else [])):
        validate_json_schema(contract['schema'], {'voice_types': values})
    for value in (-1, 11, True, float('nan'), float('inf')):
        with pytest.raises(SandboxError):
            validate_json_schema(contract['schema'], {'voice_types': [value]})


@pytest.mark.parametrize('bounds', [
    {'minimum': True}, {'maximum': '10'}, {'minimum': None},
    {'minimum': float('nan')}, {'maximum': float('inf')},
    {'minimum': 2, 'maximum': 1}])
def test_invalid_bounds_fail_before_publishing(bounds):
    with pytest.raises(ValueError, match='ANSWER_CONTRACT_INVALID:.*bounds'):
        _schema({'type': 'number', **bounds})


def test_fractional_bounds_apply_to_integer_values_and_enum_members():
    schema = {'type': 'integer', 'minimum': 0.5, 'maximum': 2.5, 'enum': [1, 2]}
    _schema(schema)
    for value in (1, 2): validate_json_schema(schema, value)
    with pytest.raises(SandboxError): validate_json_schema(schema, 0)
    with pytest.raises(SandboxError): _schema({**schema, 'enum': [0, 1]})
    with pytest.raises(ValueError, match='unsupported schema keywords'):
        _schema({'type': 'string', 'minimum': 0})
