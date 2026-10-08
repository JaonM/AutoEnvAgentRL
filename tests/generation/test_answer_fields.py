import copy
import json
from unittest.mock import patch

import pytest

from env_factory.generation.agent_authoring import compile_source, expand_source
from env_factory.generation.answer_fields import expand_answer_fields
from env_factory.sandbox_runtime import DeclarativeMetricEvaluator, ExpressionBusinessState, SandboxError
from .test_agent_authoring import fixture


def source():
    full, request = fixture('simple_agentic')
    value = {key: copy.deepcopy(full[key]) for key in ('version', 'description', 'environment_plan', 'tables')}
    table = value['tables'][0]
    table['columns'].append({'name': 'current', 'type': 'boolean', 'nullable': False})
    for row in table['rows']:
        row.update(name='青松库', current=True)
    implementation = copy.deepcopy(full['tool_implementations'][0])
    implementation['projection'].append('current')
    tool = full['tools'][0]['function']
    value['business_tools'] = [{'name': tool['name'], 'description': tool['description'],
        'parameters': tool['parameters'], 'implementation': implementation}]
    value['answer_fields'] = {
        'current': {'kind': 'observed', 'schema': {'type': 'boolean', 'description': '首条返回记录的现行标记'},
                    'observation': {'tool': 'read_temperature', 'path': '$.records[0].current'}, 'missing': False},
        'count': {'kind': 'observed', 'schema': {'type': 'integer', 'description': '查询返回记录数'},
                  'observation': {'tool': 'read_temperature', 'path': '$.count'}, 'missing': 0},
        'unique': {'kind': 'derived', 'schema': {'type': 'boolean', 'description': '返回记录是否恰好一条'},
                   'expression': {'op': 'eq', 'args': [{'answer_field': 'count'}, 1]}}}
    value['outcomes'] = [{'id': 'facts', 'rubric': '分别报告观察标记、记录数和唯一性',
                          'fields': ['current', 'count', 'unique']}]
    value['reference'] = {'calls': full['scenarios'][0]['steps'][:-1],
                          'answer': json.dumps({'current': True, 'count': 2, 'unique': False})}
    request['require_answer_fields'] = True
    return value, request


def test_actual_two_record_observation_is_not_overwritten_by_uniqueness(tmp_path):
    from env_factory.generation.lifecycle_preview import preview_lifecycle_task
    value, request = source()
    original = copy.deepcopy(value)
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as model:
        artifact = compile_source(value, root=tmp_path, request=request)
    model.assert_not_called()
    assert value == original
    assert artifact['generation_pipeline']['verification']['passed']
    context = preview_lifecycle_task({**artifact, 'artifacts': artifact}, tmp_path, return_context=True)
    evaluator = DeclarativeMetricEvaluator()
    correct = evaluator.evaluate_all(artifact['metric_implementations'], context)
    wrong = {**context, 'final_agent_response': json.dumps({'current': False, 'count': 2, 'unique': False})}
    assert correct['facts'] == 1 and evaluator.evaluate_all(artifact['metric_implementations'], wrong)['facts'] == 0
    public = artifact['public_input']
    assert public['answer_field_semantics']['current']['kind'] == 'observed'
    assert public['answer_contract']['schema']['properties']['current'] == value['answer_fields']['current']['schema']
    assert 'only when this path has no value' in public['initial_user_message']


@pytest.mark.parametrize('observed', [False, 0, '', [], {}, True])
def test_only_missing_evidence_uses_missing_value(observed):
    expression = {'observed_value': {'value': {'tool_result': {'name': 'read', 'path': '$.value'}},
                                      'missing': {'literal': 'missing'}}}
    def evaluate(snapshot):
        return DeclarativeMetricEvaluator._value_expression(expression, ExpressionBusinessState({}, {}, snapshot))
    assert evaluate({'$tool_results': {'read': {'value': observed}}}) == observed
    assert evaluate({}) == 'missing'
    assert evaluate({'$tool_results': {'read': {}}}) == 'missing'


@pytest.mark.parametrize('mutation', ['duplicate_schema', 'duplicate_rule', 'conditional_observation', 'cycle', 'unknown', 'semantic_owner'])
def test_conflicting_or_unowned_field_definitions_fail_before_models(mutation):
    value, request = source()
    if mutation == 'duplicate_schema': value['answer_contract'] = {}
    if mutation == 'duplicate_rule': value['outcomes'][0] = {'rule': {'source': 'final_agent_response'}}
    if mutation == 'conditional_observation': value['answer_fields']['current']['expression'] = {'literal': False}
    if mutation == 'cycle': value['answer_fields']['unique']['expression'] = {'answer_field': 'unique'}
    if mutation == 'unknown': value['answer_fields']['unique']['expression'] = {'answer_field': 'absent'}
    if mutation == 'semantic_owner': value['outcomes'].append({'semantic': {'fields': ['current']}})
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat') as model:
        with pytest.raises(ValueError, match='ANSWER_FIELDS_'):
            expand_source(value, request)
    model.assert_not_called()


def test_requested_shared_contract_cannot_be_omitted():
    value, request = source()
    value.pop('answer_fields')
    with pytest.raises(ValueError, match='ANSWER_FIELDS_REQUIRED'):
        expand_source(value, request)


def test_blueprint_freezes_meaning_and_role_before_implementation():
    from .test_business_blueprint import blueprint, implementation, request
    from env_factory.generation.business_blueprint import validate_blueprint, validate_implementation
    plan, materialized, req = blueprint(), implementation(), request()
    req['require_answer_fields'] = True
    plan['answer_fields'] = {'current': {'kind': 'observed', 'meaning': 'Observed current flag'}}
    materialized['answer_fields'] = {'current': {'kind': 'observed', 'schema': {
        'type': 'boolean', 'description': 'Observed current flag'}}}
    validate_blueprint(plan, req)
    validate_implementation(plan, materialized, req)
    materialized['answer_fields']['current']['kind'] = 'derived'
    with pytest.raises(ValueError, match='BLUEPRINT_DRIFT: answer field'):
        validate_implementation(plan, materialized, req)
    del plan['answer_fields']
    with pytest.raises(ValueError, match='BLUEPRINT: answer_fields'):
        validate_blueprint(plan, req)


def test_observed_primitive_cannot_hide_a_conditional_value():
    expression = {'observed_value': {'value': {'literal': False}, 'missing': {'literal': False}}}
    with pytest.raises(SandboxError, match='observed_value requires'):
        DeclarativeMetricEvaluator._value_expression(expression, {})


@pytest.mark.parametrize('explanation_mode', [None, 'runtime', 'fields'])
def test_generated_sandbox_rewards_observed_truth_under_ambiguity(tmp_path, explanation_mode, monkeypatch):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    from env_factory.generation.artifacts import write_task_artifact
    from env_factory.tasks.task import Task
    from scripts.sandbox.generate_sandbox_scaffold import generate
    monkeypatch.setenv('SANDBOX_EVALUATOR_MOCK', '0')
    monkeypatch.setenv('SANDBOX_LLM_API_KEY', 'offline-test')
    monkeypatch.setenv('SANDBOX_LLM_MODEL', 'offline-test')
    with_explanation = explanation_mode is not None
    value, request = explanation_source() if with_explanation else source()
    if with_explanation:
        value['answer_fields']['why']['evidence_mode'] = explanation_mode
    def model(messages, **kwargs):
        evidence = json.loads(messages[1]['content'])['runtime_evidence']
        assert evidence['verified_answer_context']['count']['value'] == 2
        assert evidence['verified_answer_context']['unique']['value'] is False
        if explanation_mode == 'fields':
            assert evidence['executable_field_facts']['count']['value'] == 2
            assert 'business_data' not in evidence and 'tool_results' not in evidence
        why = json.loads(evidence['final_agent_response'])['why']
        return {'label': 'pass' if why in {'候选不唯一，无法唯一选定。', '仍有多个候选，因此不能选定唯一对象。'} else 'fail'}
    with patch('env_factory.sandbox_runtime.RuntimeLLMClient.json_chat', side_effect=model):
        artifact = compile_source(value, root=tmp_path, request=request)
    path = write_task_artifact(tmp_path, Task(artifact['task'], artifact['environment'], artifact['metrics'],
        task_intent=artifact['task_intent'], artifacts=artifact), 'simple_agentic')
    contract = json.loads(path.read_text())
    contract.pop('actions')
    (tmp_path / 'BUILD_CONTRACT.json').write_text(json.dumps(contract))
    generate(tmp_path)
    project = Path(__file__).resolve().parents[2]
    for name in ('sandbox_runtime.py', 'runtime_llm.py'):
        shutil.copy2(project / 'src/env_factory' / name, tmp_path / name)
    script = tmp_path / 'verify_answer_fields.py'
    script.write_text('with_explanation = ' + repr(with_explanation) + '\nexplanation_mode = ' + repr(explanation_mode) + '\n' + '''import json
from app import create_app
from sandbox_runtime import RuntimeLLMClient
def model(self, messages, **kwargs):
    evidence = json.loads(messages[1]['content'])['runtime_evidence']
    assert evidence['verified_answer_context']['count']['value'] == 2
    assert evidence['verified_answer_context']['unique']['value'] is False
    assert set(json.loads(evidence['final_agent_response'])) == {'why'}
    if explanation_mode == 'fields':
        assert evidence['executable_field_facts']['count']['value'] == 2
        assert 'business_data' not in evidence and 'tool_results' not in evidence
    return {'label': 'pass'}
RuntimeLLMClient.json_chat = model
app = create_app()
headers = {'Authorization': 'Bearer test-fields'}
rewards = []
for current in [True, False]:
    assert app.handle('POST', '/v1/reset', {'seed': 17}, headers)[0] == 200
    status, result, _ = app.handle('POST', '/v1/tools/read_temperature', {'name': '青松库'})
    assert status == 200 and result['count'] == 2
    assert all(row['current'] is True for row in result['records'])
    answer = {'current': current, 'count': 2, 'unique': False}
    if with_explanation: answer['why'] = '候选不唯一，无法唯一选定。'
    assert app.handle('POST', '/v1/agent_response', {'content': json.dumps(answer)}, headers)[0] == 200
    status, result, _ = app.handle('GET', '/v1/reward', {}, headers)
    assert status == 200
    rewards.append(result['reward'])
assert rewards[0] == 1 and rewards[1] < 1, rewards
print(json.dumps(rewards))
''')
    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, 'SANDBOX_TRAINER_API_KEY': 'test-fields'}, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr


def explanation_source():
    value, request = source()
    value['answer_fields']['why'] = {'kind': 'explanation', 'context_fields': ['count', 'unique'],
        'schema': {'type': 'string', 'description': '解释为何不能唯一选定'}}
    answer = json.loads(value['reference']['answer'])
    cases = [({'why': text, **answer}, expected) for text, expected in [
        ('候选不唯一，无法唯一选定。', 'pass'), ('仍有多个候选，因此不能选定唯一对象。', 'pass'),
        ('可以直接唯一选定。', 'fail'), ('已查询。', 'fail')]]
    value['reference']['answer'] = json.dumps(cases[0][0])
    value['outcomes'].append({'id': 'reasoning', 'rubric': '解释歧义对选定的影响', 'semantic': {
        'fields': ['why'], 'criteria': ['说明候选不唯一导致不能唯一选定，无需重复结构化数值。'],
        'cases': [{'answer': answer, 'expected': label} for answer, label in cases]}})
    return value, request


@pytest.mark.parametrize('mode', ['runtime', 'fields'])
def test_shared_context_is_derived_once_and_published_without_cases(mode):
    from env_factory.generation.answer_contract import bind_answer_contract
    value, request = explanation_source()
    value['answer_fields']['why']['evidence_mode'] = mode
    expanded = expand_source(value, request)
    bind_answer_contract(expanded)
    metric = next(m for m in expanded['metrics'] if m['id'] == 'reasoning')
    assert metric['semantic_context_fields'] == ['count', 'unique']
    public = expanded['description']['public_input']
    assert public['answer_field_semantics']['why']['context_fields'] == ['count', 'unique']
    assert public['semantic_answer_criteria'][0]['context_fields'] == ['count', 'unique']
    assert metric.get('semantic_evidence_mode', 'runtime') == mode
    assert public['answer_field_semantics']['why'].get('evidence_mode', 'runtime') == mode
    assert public['semantic_answer_criteria'][0].get('evidence_mode', 'runtime') == mode
    assert '可以直接唯一选定' not in str(public)


@pytest.mark.parametrize('change', ['unknown', 'explanation', 'duplicate', 'observed_context', 'duplicate_definition'])
def test_invalid_shared_context_fails_early(change):
    value, request = explanation_source()
    if change == 'unknown': value['answer_fields']['why']['context_fields'] = ['missing']
    if change == 'explanation': value['answer_fields']['why']['context_fields'] = ['why']
    if change == 'duplicate': value['answer_fields']['why']['context_fields'] = ['count', 'count']
    if change == 'observed_context': value['answer_fields']['count']['context_fields'] = []
    if change == 'duplicate_definition': value['outcomes'][-1]['semantic']['context_fields'] = ['unique']
    with pytest.raises(ValueError, match='ANSWER_FIELDS_'):
        expand_source(value, request)


def test_blueprint_freezes_explanation_supporting_fields():
    from .test_business_blueprint import blueprint, implementation, request
    from env_factory.generation.business_blueprint import validate_blueprint, validate_implementation
    plan, materialized, req = blueprint(), implementation(), request()
    plan['answer_fields']['why'] = {'kind': 'explanation', 'meaning': 'Explain chosen option', 'context_fields': ['option']}
    materialized['answer_fields']['why'] = {'kind': 'explanation', 'schema': {
        'type': 'string', 'description': 'Explain chosen option'}, 'context_fields': ['option']}
    validate_implementation(plan, materialized, req)
    materialized['answer_fields']['why']['context_fields'] = []
    with pytest.raises(ValueError, match='BLUEPRINT_DRIFT: answer field'):
        validate_implementation(plan, materialized, req)
    plan['answer_fields']['why']['context_fields'] = ['why']
    with pytest.raises(ValueError, match='BLUEPRINT: explanation context'):
        validate_blueprint(plan, req)


def test_context_requires_deterministic_owner_even_for_legacy_contract():
    from env_factory.generation.answer_contract import bind_answer_contract
    value, request = explanation_source()
    expanded = expand_source(value, request)
    metric = next(m for m in expanded['metrics'] if m['id'] == 'reasoning')
    metric['semantic_context_fields'] = ['why']
    with pytest.raises(ValueError, match='context fields require deterministic'):
        bind_answer_contract(expanded)


@pytest.mark.parametrize('change', ['unknown_mode', 'missing_context', 'duplicate_mode', 'mixed_mode'])
def test_invalid_evidence_scope_is_rejected_during_authoring(change):
    value, request = explanation_source()
    value['answer_fields']['why']['evidence_mode'] = 'fields'
    if change == 'unknown_mode': value['answer_fields']['why']['evidence_mode'] = 'unknown'
    if change == 'missing_context': value['answer_fields']['why']['context_fields'] = []
    if change == 'duplicate_mode': value['outcomes'][-1]['semantic']['evidence_mode'] = 'fields'
    if change == 'mixed_mode':
        value['answer_fields']['other'] = {'kind': 'explanation', 'schema': {'type': 'string', 'description': 'Other'}}
        value['outcomes'][-1]['semantic']['fields'].append('other')
    with pytest.raises(ValueError, match='ANSWER_FIELDS_'):
        expand_source(value, request)


def test_blueprint_cannot_silently_shrink_evidence_scope():
    from .test_business_blueprint import blueprint, implementation, request
    from env_factory.generation.business_blueprint import validate_blueprint, validate_implementation
    plan, materialized, req = blueprint(), implementation(), request()
    plan['answer_fields']['why'] = {'kind': 'explanation', 'meaning': 'Explain option',
        'context_fields': ['option'], 'evidence_mode': 'fields'}
    materialized['answer_fields']['why'] = {'kind': 'explanation', 'schema': {
        'type': 'string', 'description': 'Explain option'}, 'context_fields': ['option'], 'evidence_mode': 'fields'}
    validate_blueprint(plan, req)
    validate_implementation(plan, materialized, req)
    materialized['answer_fields']['why']['evidence_mode'] = 'runtime'
    with pytest.raises(ValueError, match='BLUEPRINT_DRIFT: answer field'):
        validate_implementation(plan, materialized, req)
