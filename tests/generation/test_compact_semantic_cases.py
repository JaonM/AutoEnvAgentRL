import copy
import json

import pytest

from env_factory.generation.agent_authoring import expand_source
from env_factory.generation.semantic_reward import compile_semantic_outcome
from tests.generation.test_semantic_reward import fixture


def paired_contracts():
    item, schema = fixture()
    schema['properties']['evidence'] = {'type': 'array', 'items': {'type': 'number'}}
    schema['required'].append('evidence')
    for case in item['semantic']['cases']:
        case['reference_id'] = 'reference_success'
        case['answer']['evidence'] = [17, 29]
    compact = copy.deepcopy(item)
    for case in compact['semantic']['cases']:
        case['answer_overrides'] = {'why': case.pop('answer')['why']}
    references = {'reference_success': json.dumps(item['semantic']['cases'][0]['answer'])}
    return item, compact, schema, references


def test_compact_cases_expand_to_identical_metrics_answers_and_labels():
    item, compact, schema, references = paired_contracts()
    original = copy.deepcopy(compact)
    full = compile_semantic_outcome(item, schema=schema, weight=0.5)
    expanded = compile_semantic_outcome(compact, schema=schema, weight=0.5, reference_answers=references)
    assert expanded == full
    assert compact == original
    expanded[1][0]['answer']['evidence'].append(42)
    assert expanded[1][1]['answer']['evidence'] == [17, 29]
    assert json.loads(references['reference_success'])['evidence'] == [17, 29]


def test_empty_override_uses_canonical_answer_without_skipping_validation():
    item, compact, schema, references = paired_contracts()
    compact['semantic']['cases'][0]['answer_overrides'] = {}
    assert compile_semantic_outcome(compact, schema=schema, weight=0.5, reference_answers=references) == (
        compile_semantic_outcome(item, schema=schema, weight=0.5))


def test_script_scoped_reference_is_preserved_for_execution_context_selection():
    _, compact, schema, references = paired_contracts()
    for case in compact['semantic']['cases']:
        case['reference_id'] += '-script-2'
    _, cases = compile_semantic_outcome(compact, schema=schema, weight=0.5, reference_answers=references)
    assert all(case['reference_id'] == 'reference_success-script-2' for case in cases)


@pytest.mark.parametrize('defect', ['unknown_reference', 'missing_reference', 'both_formats',
    'unowned_field', 'wrong_field_type', 'missing_evidence', 'duplicate', 'no_negative',
    'nonobject_reference', 'invalid_json', 'invalid_script_suffix'])
def test_compact_cases_preserve_existing_contract_requirements(defect):
    _, compact, schema, references = paired_contracts()
    cases = compact['semantic']['cases']
    if defect == 'unknown_reference': cases[0]['reference_id'] = 'unknown'
    elif defect == 'missing_reference': cases[0].pop('reference_id')
    elif defect == 'both_formats': cases[0]['answer'] = {'why': 'text'}
    elif defect == 'unowned_field': cases[0]['answer_overrides']['evidence'] = []
    elif defect == 'wrong_field_type': cases[0]['answer_overrides']['why'] = 42
    elif defect == 'missing_evidence': references['reference_success'] = '{"why":"text"}'
    elif defect == 'duplicate': cases[1] = copy.deepcopy(cases[0])
    elif defect == 'no_negative':
        for case in cases: case['expected'] = 'pass'
    elif defect == 'nonobject_reference': references['reference_success'] = '[]'
    elif defect == 'invalid_json': references['reference_success'] = '{'
    elif defect == 'invalid_script_suffix': cases[0]['reference_id'] += '-script-0'
    with pytest.raises(ValueError, match='SEMANTIC_REWARD_CASES'):
        compile_semantic_outcome(compact, schema=schema, weight=0.5, reference_answers=references)


def test_compact_authoring_keeps_identical_expanded_source():
    item, compact, schema, references = paired_contracts()
    source = {'business_tools': [], 'outcomes': [item],
        'description': {'public_input': {'initial_user_message': 'Explain the heading.', 'materials': []}},
        'answer_contract': {'format': 'json_object', 'schema': schema},
        'reference': {'calls': [], 'answer': references['reference_success']}}
    compact_source = copy.deepcopy(source)
    compact_source['outcomes'] = [compact]
    assert expand_source(source, {'seed': 1}) == expand_source(compact_source, {'seed': 1})
