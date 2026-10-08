import copy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.answer_contract import _schema, bind_answer_contract
from env_factory.generation.answer_fields import expand_answer_fields
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator as E, ExpressionBusinessState, validate_json_schema, SandboxError
from .test_answer_fields import source
from .test_answer_contract import answer_source


def snapshot(result):
    return ExpressionBusinessState({}, {}, {'$tool_results': {'read': result}})


def match(expression, answer, state, **options):
    expected = {'answer_format': 'json_object', 'targets': [{'key': 'answer', 'expression': expression, **options}]}
    return E._value_targets_match(json.dumps({'answer': answer}), expected, state)


@pytest.mark.parametrize('path,result', [('$', None), ('$.value', {'value': None}),
    ('$.records[0].value', {'records': [{'value': None}]}),
    ('$.records[*].value', {'records': [{'value': None}, {'value': 2}]})])
def test_null_is_resolved_but_absent_tool_or_path_cannot_earn_credit(path, result):
    expression = {'tool_result': {'name': 'read', 'path': path}}
    wanted = [None, 2] if '[*]' in path else None
    assert E.value_resolution(expression, snapshot(result)) == {'resolved': True, 'value': wanted}
    assert match(expression, wanted, snapshot(result))
    assert not match(expression, wanted, ExpressionBusinessState({}, {}, {}))
    if path != '$':
        assert not match(expression, wanted, snapshot({}))
    assert not match(expression, 0, snapshot(result))


def test_null_default_requires_explicit_observed_missing_rule():
    read = {'tool_result': {'name': 'read', 'path': '$.value'}}
    observed = {'observed_value': {'value': read, 'missing': {'literal': 'UNOBSERVED'}}}
    assert match(observed, None, snapshot({'value': None}))
    assert not match(observed, 'UNOBSERVED', snapshot({'value': None}))
    assert match(observed, 'UNOBSERVED', snapshot({}))
    observed['observed_value']['missing'] = {'literal': None}
    assert match(observed, None, snapshot({}))
    assert not match(read, None, snapshot({}))


def test_null_predicates_arrays_checkpoints_and_tolerance_keep_missing_closed():
    read = {'tool_result': {'name': 'read', 'path': '$.value'}}
    predicate = {'op': 'eq', 'args': [read, {'literal': None}]}
    conditional = {'if': {'condition': predicate, 'then': {'literal': None}, 'else': read}}
    assert match(conditional, None, snapshot({'value': None}), tolerance=0)
    assert match(conditional, 2, snapshot({'value': 2}), tolerance=0)
    assert not match(conditional, None, snapshot({}))
    assert E.value_resolution(predicate, snapshot({}))['resolved'] is False
    assert match({'array': [read, {'literal': None}]}, [None, None], snapshot({'value': None}))
    assert not match({'array': [read, {'literal': None}]}, [None, None], snapshot({}))
    checkpoint = {'at_tool': {'name': 'read', 'expression': {'literal': None}}}
    assert not match(checkpoint, None, snapshot({'value': None}))
    assert match(checkpoint, None, ExpressionBusinessState({}, {}, {'read': {}}))
    assert match({'literal': [{'id': 1, 'value': None}]}, [{'id': 1, 'value': None}], {}, array_order=['id'])
    assert not match({'literal': [{'id': 1, 'value': None}]}, [{'id': True, 'value': None}], {}, array_order=['id'])


def test_nullable_state_goal_requires_a_present_field_and_resolved_expression():
    from env_factory.sandbox_runtime import BusinessGoalEvaluator
    predicate = {'table': 'items', 'where': {'id': 1}, 'count': 1,
                 'value_expressions': {'value': {'literal': None}}}
    assert BusinessGoalEvaluator.evaluate([predicate], {'items': [{'id': 1, 'value': None}]})
    assert not BusinessGoalEvaluator.evaluate([predicate], {'items': [{'id': 1}]})
    assert not BusinessGoalEvaluator.evaluate([predicate], {'items': [{'id': 1, 'value': 0}]})
    predicate['value_expressions']['value'] = {'lookup': {'table': 'missing', 'field': 'value', 'where': {}}}
    assert not BusinessGoalEvaluator.evaluate([predicate], {'items': [{'id': 1, 'value': None}]})


def test_missing_capture_cannot_become_a_successful_null_comparison():
    expression = {'$expr': {'op': 'eq', 'args': [{'$ref': 'value'}, {'literal': None}]}}
    assert E.resolve_capture_argument(expression, {'value': None}) is True
    with pytest.raises(SandboxError, match='unavailable evidence'):
        E.resolve_capture_argument(expression, {})


@pytest.mark.parametrize('kind,value,extra', [
    ('integer', 2, {}), ('number', 2.5, {}), ('boolean', False, {}), ('string', '', {}),
    ('array', [None], {'items': {'type': ['integer', 'null']}, 'maxItems': 2}),
    ('object', {'x': None}, {'properties': {'x': {'type': ['string', 'null']}},
                           'required': ['x'], 'additionalProperties': False})])
def test_nullable_schema_keeps_the_concrete_type_constraints(kind, value, extra):
    schema = {'type': [kind, 'null'], **extra}
    _schema(schema)
    validate_json_schema(schema, None)
    validate_json_schema(schema, value)
    with pytest.raises(SandboxError):
        validate_json_schema(schema, {'undeclared': 'wrong'} if kind == 'object' else 'wrong' if kind != 'string' else [])


@pytest.mark.parametrize('kind', [[], ['null', 'null'], ['integer', 'unsupported'], [['integer'], 'null'], None])
def test_unsupported_unions_fail_with_contract_diagnostic(kind):
    with pytest.raises(ValueError, match='ANSWER_CONTRACT_INVALID'):
        _schema({'type': kind})


def test_heterogeneous_array_union_matches_runtime_and_preserves_nested_constraints():
    schema = {'type': 'array', 'items': {'type': ['string', 'integer', 'boolean', 'null']}}
    _schema(schema)
    validate_json_schema(schema, ['id', 3, True, None])
    with pytest.raises(SandboxError):
        validate_json_schema(schema, ['id', 1.5])
    schema = {'type': ['array', 'object', 'null'], 'items': {'type': 'integer'}, 'maxItems': 2,
              'properties': {'value': {'type': 'boolean'}}, 'required': ['value'], 'additionalProperties': False}
    _schema(schema)
    for valid in [None, [1], {'value': False}]:
        validate_json_schema(schema, valid)
    for invalid in [[1, 2, 3], [True], {'value': 1}, {}]:
        with pytest.raises(SandboxError):
            validate_json_schema(schema, invalid)


def test_nullable_text_cannot_bypass_public_fallback_disclosure():
    value, _ = answer_source()
    value['answer_contract']['schema']['properties']['voice_types'] = {'type': ['string', 'null']}
    value['metric_implementations'][0]['expected']['targets'][0]['expression'] = {'if': {
        'condition': {'literal': False}, 'then': {'tool_result': {'name': 'read', 'path': '$.x'}},
        'else': {'literal': 'UNPUBLISHED'}}}
    with pytest.raises(ValueError, match='ANSWER_FALLBACK_UNDISCLOSED'):
        bind_answer_contract(value)


@pytest.mark.parametrize('path', ['$.', '$value', '$.records[bad]', 'records'])
def test_observation_uses_the_runtime_path_parser(path):
    value, _ = source()
    value['answer_fields']['current']['observation']['path'] = path
    with pytest.raises(ValueError, match='ANSWER_FIELDS_INVALID.*current'):
        expand_answer_fields(value)


def test_full_result_and_nullable_contract_compile_and_deliver_without_weakening_reward(tmp_path):
    from env_factory.generation.delivery_preflight import verify_delivery
    from env_factory.generation.lifecycle_preview import preview_lifecycle_task
    value, request = source()
    field = value['answer_fields']['count']
    field['schema'].update(type=['integer', 'null'], description='实际返回记录数；未查询为 null，空结果为 0。')
    field['missing'] = None
    projection = value['business_tools'][0]['implementation']['projection']
    types = {'text': 'string', 'integer': 'integer', 'real': 'number', 'boolean': 'boolean'}
    properties = {c['name']: {'type': types[c['type']]} for c in value['tables'][0]['columns'] if c['name'] in projection}
    record_schema = {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}
    value['answer_fields']['result'] = {'kind': 'observed', 'missing': None,
        'observation': {'tool': 'read_temperature', 'path': '$'},
        'schema': {'type': ['object', 'null'], 'description': '完整查询结果 records 和 count；未查询为 null。',
            'properties': {'records': {'type': 'array', 'items': record_schema}, 'count': {'type': 'integer'}},
            'required': ['records', 'count'], 'additionalProperties': False}}
    value['outcomes'][0]['fields'].append('result')
    records = [{key: row[key] for key in projection} for row in value['tables'][0]['rows']]
    answer = {**json.loads(value['reference']['answer']), 'result': {'records': records, 'count': len(records)}}
    value['reference']['answer'] = json.dumps(answer)
    original = copy.deepcopy(value)
    artifact = compile_source(value, root=tmp_path, request=request)
    assert value == original
    assert artifact['generation_pipeline']['verification']['passed']
    assert verify_delivery(artifact, tmp_path)['passed']
    context = preview_lifecycle_task({**artifact, 'artifacts': artifact}, tmp_path, return_context=True)
    assert E().evaluate_all(artifact['metric_implementations'], context)['facts'] == 1
    for wrong in [None, {'records': [], 'count': 2}, {'records': records, 'count': 0}]:
        bad = {**context, 'final_agent_response': json.dumps({**answer, 'result': wrong})}
        assert E().evaluate_all(artifact['metric_implementations'], bad)['facts'] == 0
