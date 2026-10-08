from env_factory.tasks.task_quality import score_task, score_file, _numeric_private_dependencies
from tests.tasks.test_task_quality import valid_task
from tests.generation.test_agent_authoring import fixture
from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.artifacts import write_task_artifact
from env_factory.tasks.task import Task
from env_factory.graph.knowledge_graph import TaskType
from copy import deepcopy
import json
import pytest


def scoped_query_task():
    task = valid_task()
    task['tool_implementations'] = [{'tool_name': 'read_inventory', 'operation': 'select',
        'table': 'inventory', 'filters': [{'argument': 'category', 'column': 'category', 'operator': 'eq'}]}]
    task['metrics'] += [{'id': name, 'category': 'outcome', 'type': 'rule-based',
        'weight': 0.2, 'evaluator': {}} for name in ('scope', 'evidence')]
    for name, expression in [('scope', {'tool_argument': {'name': 'read_inventory', 'path': '$.category'}}),
                             ('evidence', {'tool_result': {'name': 'read_inventory', 'path': '$.records'}})]:
        task['metric_implementations'].append({'metric_id': name, 'source': 'final_agent_response',
            'path': '$', 'operator': 'value_targets', 'expected': {'answer_format': 'json_object',
                'targets': [{'key': name, 'expression': expression}]}, 'score_mapping': {'pass': 1, 'fail': 0}})
    return task


def test_query_scope_reward_is_supported_by_separate_evidence_reward():
    report = score_task(scoped_query_task())
    assert report.eligible, report.eligibility_failures


@pytest.mark.parametrize('defect', ['literal', 'unused_argument', 'unknown_tool', 'external_tool',
    'no_filter', 'no_table', 'no_evidence', 'zero_evidence_weight', 'literal_evidence', 'penalty_evidence'])
def test_scope_copy_cannot_replace_private_evidence(defect):
    task = scoped_query_task()
    scope, evidence = task['metric_implementations'][-2:]
    expression = scope['expected']['targets'][0]['expression']
    tool = task['tool_implementations'][0]
    if defect == 'literal':
        scope['expected']['targets'][0]['expression'] = {'literal': '办公设备'}
    elif defect == 'unused_argument':
        expression['tool_argument']['path'] = '$.unrelated'
    elif defect == 'unknown_tool':
        expression['tool_argument']['name'] = 'unknown'
    elif defect == 'external_tool':
        tool['operation'] = 'external'
    elif defect == 'no_filter':
        tool['filters'] = []
    elif defect == 'no_table':
        tool.pop('table')
    elif defect == 'no_evidence':
        task['metric_implementations'].remove(evidence)
    elif defect == 'zero_evidence_weight':
        task['metrics'][-1]['weight'] = 0
    elif defect == 'literal_evidence':
        evidence['expected']['targets'][0]['expression'] = {'literal': []}
    elif defect == 'penalty_evidence':
        task['metrics'][-1]['category'] = 'penalty'
    assert any('no private data dependency' in issue for issue in score_task(task).eligibility_failures)


def test_separate_scope_and_data_rewards_execute_and_reject_wrong_scope(tmp_path):
    source, request = fixture('simple_agentic')
    source['answer_contract'] = {'format': 'json_object', 'schema': {'type': 'object',
        'properties': {'temperature': {'type': 'number'}, 'scope': {'type': 'string'}},
        'required': ['temperature', 'scope'], 'additionalProperties': False}}
    source['description']['task'] = '查询内部库区青松库的温度，以 JSON 报告 temperature 和所查库区 scope。'
    source['description']['public_input']['initial_user_message'] = source['description']['task']
    source['metric_implementations'][0]['operator'] = 'value_targets'
    source['metric_implementations'][0]['expected'] = {'answer_format': 'json_object', 'targets': [
        {'key': 'temperature', 'expression': {
            'tool_result': {'name': 'read_temperature', 'path': '$.records[0].temperature'}}}]}
    source['metrics'][0]['weight'] = 0.1
    metric = deepcopy(source['metrics'][0])
    metric.update(id='scope', weight=0.9, rubric='报告实际查询的库区')
    source['metrics'].append(metric)
    source['metric_implementations'].append(deepcopy(scoped_query_task()['metric_implementations'][-2]))
    source['metric_implementations'][-1]['expected']['targets'][0]['expression'] = {
        'tool_argument': {'name': 'read_temperature', 'path': '$.name'}}
    success = source['scenarios'][0]
    success['steps'][-1]['content'] = json.dumps({'temperature': 18, 'scope': '青松库'})
    failure = deepcopy(success)
    failure.update(scenario_id='wrong_scope', kind='goal_failure')
    failure['steps'][-1]['content'] = json.dumps({'temperature': 18, 'scope': '白杨库'})
    source['scenarios'].append(failure)
    artifacts = compile_source(source, root=tmp_path, request=request)
    assert artifacts['generation_pipeline']['verification']['passed']
    path = write_task_artifact(tmp_path, Task(artifacts['task'], artifacts['environment'], artifacts['metrics'],
        task_type=TaskType(artifacts['task_type']), task_intent=artifacts['task_intent'],
        complexity=artifacts['complexity'], artifacts=artifacts), artifacts['training_category'])
    report = score_file(path)
    assert report.eligible, report.eligibility_failures


def test_field_scoped_semantics_preserve_factuality_checks():
    task = valid_task()
    metric = next(m for m in task['metrics'] if m['category'] == 'outcome')
    metric.update(semantic_fields=['explanation'], criteria=['核对回答与当前资料一致。'])
    task['public_input'] = {'answer_contract': {'schema': {'type': 'object', 'properties': {
        'explanation': {'type': 'string'}}}}}
    issue = '只读数据结果奖励缺少答案与业务证据核对：outcome'
    assert issue not in score_task(task).eligibility_failures
    metric['semantic_fields'] = ['missing']
    assert issue in score_task(task).eligibility_failures
    metric['semantic_fields'] = ['explanation']
    metric['criteria'] = ['无需核对工具结果与业务记录。']
    assert issue in score_task(task).eligibility_failures


def test_tool_result_dependency_must_be_real_private_read():
    value = {'tool_result': {'name': 'read', 'path': '$.count'}}
    tools = [{'tool_name': 'read', 'operation': 'aggregate_count', 'table': 'items'}]
    assert _numeric_private_dependencies(value, tools)
    assert not _numeric_private_dependencies(value, [])
    assert not _numeric_private_dependencies({'literal': value}, tools)
    assert not _numeric_private_dependencies(value, [dict(tools[0], operation='external')])


def test_observed_numeric_reward_uses_real_read_snapshot(tmp_path):
    source, request = fixture('simple_agentic')
    source['metric_implementations'][0]['expected']['targets'][0]['expression'] = {
        'tool_result': {'name': 'read_temperature', 'path': '$.records[0].temperature'}}
    artifacts = compile_source(source, root=tmp_path, request=request)
    path = write_task_artifact(tmp_path, Task(artifacts['task'], artifacts['environment'], artifacts['metrics'],
        task_type=TaskType(artifacts['task_type']), task_intent=artifacts['task_intent'],
        complexity=artifacts['complexity'], artifacts=artifacts), artifacts['training_category'])
    report = score_file(path)
    assert report.eligible, report.eligibility_failures


def test_observed_zero_count_is_sensitive_to_query_membership(tmp_path):
    source, request = fixture('simple_agentic')
    source['description']['task'] = '查询库区中名为未知库的匹配记录数量，回答数量：<数字>条。'
    source['description']['public_input']['initial_user_message'] = source['description']['task']
    source['description']['goal'] = '报告匹配记录数量'
    source['description']['expected_result'] = '准确的匹配数量'
    source['tools'][0]['function']['description'] = '统计指定库名的记录数量'
    source['tool_implementations'][0]['operation'] = 'aggregate_count'
    source['tool_implementations'][0].pop('projection')
    source['tool_implementations'][0].pop('result_field')
    success = source['scenarios'][0]
    success['steps'][0]['arguments']['name'] = '未知库'
    success['steps'][-1]['content'] = '数量：0条。'
    source['metrics'][0]['rubric'] = '匹配数量正确'
    target = source['metric_implementations'][0]['expected']['targets'][0]
    target.update(label='数量', unit='条', expression={
        'tool_result': {'name': 'read_temperature', 'path': '$.count'}})
    source = json.loads(json.dumps(source).replace('"read_temperature"', '"count_zones"'))
    artifacts = compile_source(source, root=tmp_path, request=request)
    path = write_task_artifact(tmp_path, Task(artifacts['task'], artifacts['environment'], artifacts['metrics'],
        task_type=TaskType(artifacts['task_type']), task_intent=artifacts['task_intent'],
        complexity=artifacts['complexity'], artifacts=artifacts), artifacts['training_category'])
    report = score_file(path)
    assert report.eligible, report.eligibility_failures
