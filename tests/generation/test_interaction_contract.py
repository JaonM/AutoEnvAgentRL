import copy
import json
from pathlib import Path
import pytest
from tests.generation.test_agent_authoring import fixture
from env_factory.generation.agent_authoring import compile_source


def interactive_source():
    source, request = fixture('multi_step_agentic')
    # The user alone knows which warehouse; it is not in the initial request.
    source['description']['public_input']['initial_user_message'] = '请先问我目标库区，再查询温度。我会选择库区并补充确认查询范围。区域暂按西区考虑。'
    source['tables'][0]['columns'].append({'name':'region', 'type':'text', 'nullable':False})
    for row in source['tables'][0]['rows']: row['region'] = '东区'
    source['tools'][0]['function']['parameters']['properties']['region'] = {'type':'string', 'description':'用户指定区域'}
    source['tools'][0]['function']['parameters']['required'].append('region')
    source['tool_implementations'][0]['filters'].append({'argument':'region', 'column':'region', 'operator':'eq'})
    source['scenarios'][0]['steps'][0]['arguments']['region'] = '东区'
    variants = []
    for index, kind in enumerate(['option_selection', 'constraint_update', 'correction']):
        variants.append({'stages':[
            {'id':'warehouse', 'kind':'information_required', 'purpose':'获得用户私有目标库区',
             'assistant_contains_all':['库区'], 'reference_response':'请问要查询哪个库区？',
             'user_reply':'请查询青松库。', 'retry_reply':'请先问我要查询的库区。',
             'private_fact':'青松库', 'before_tool':'lookup_zone', 'bind_to':{'tool':'lookup_zone','argument':'name'}},
            {'id':'choice', 'kind':kind, 'purpose':'确认用户最终区域，避免使用原始西区约束',
             'assistant_contains_all':['东区', '西区'] if kind == 'option_selection' else ['区域','西区'],
             'reference_response':'请确认按东区还是之前说的西区查询？' if kind == 'option_selection' else '请确认区域是否仍按西区？',
             'user_reply':'区域改为东区，请按新区域查询。', 'retry_reply':'请确认区域限制后再查询。',
             'private_fact':'东区', 'previous_fact':'西区', 'options':['东区', '西区'], 'before_tool':'lookup_zone',
             'bind_to':{'tool':'lookup_zone','argument':'region'}}]})
    source['interaction_contract'] = {'version':'1.0','variants':variants}
    request['require_interaction_contract'] = True
    return source, request


def test_authored_interactions_execute_all_branches_and_cannot_be_skipped(tmp_path):
    source, request = interactive_source()
    result = compile_source(source, root=tmp_path, request=request)
    assert result['user_simulation_policy']['mode'] == 'interactive_goal'
    checks = result['generation_pipeline']['verification']
    assert checks
    scripts = json.loads((tmp_path/'data/user_simulation/user_scripts.json').read_text())
    assert len(scripts) == 3
    assert len({json.dumps(s['interaction_protocol'],sort_keys=True) for s in scripts}) == 3
    assert any(s['operation'] == 'dialogue_turn' for case in result['acceptance_contract']['executable_scenarios'] for s in case['steps'])


def test_private_fact_diagnostic_identifies_published_metadata(tmp_path):
    source, request = interactive_source()
    source['description']['public_input']['semantic_answer_criteria'] = [
        {'criteria': ['应报告青松库的实际查询结果。']}]
    with pytest.raises(ValueError, match='private interaction fact already appears') as failure:
        compile_source(source, root=tmp_path, request=request)
    message = str(failure.value)
    assert 'warehouse' in message
    assert 'description.public_input.semantic_answer_criteria[0].criteria[0]' in message
    assert '应报告青松库的实际查询结果。' in message
    assert 'published answer schema and semantic criteria' in message


@pytest.mark.parametrize('change, message', [
    ('missing','requires task-specific'), ('duplicate','differ beyond'), ('leak','already appears'),
    ('binding','does not match'), ('confirmation','actual tool')])
def test_interaction_authoring_rejects_unexecutable_or_cosmetic_complexity(tmp_path, change, message):
    source, request = interactive_source()
    if change == 'missing': source.pop('interaction_contract')
    elif change == 'duplicate': source['interaction_contract']['variants'][1] = copy.deepcopy(source['interaction_contract']['variants'][0])
    elif change == 'leak': source['description']['public_input']['initial_user_message'] += '青松库'
    elif change == 'binding':
        for v in source['interaction_contract']['variants']:
            v['stages'][1]['bind_to']['argument'] = 'not_an_argument'
    elif change == 'confirmation': source['interaction_contract']['variants'][0]['stages'][0]['kind'] = 'execution_confirmation'
    with pytest.raises(ValueError, match=message):
        compile_source(source, root=tmp_path, request=request)


def test_generated_sandbox_executes_interactions_without_external_model(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from scripts.sandbox.generate_sandbox_scaffold import generate
    from env_factory.generation.artifacts import write_task_artifact
    from env_factory.tasks.task import Task
    source, request = interactive_source()
    for scenario in source['scenarios']:
        scenario['steps'].insert(0, {'operation':'reset', 'body':{'seed':17}})
        scenario['steps'].append({'operation':'reward'})
    artifacts = compile_source(source, root=tmp_path, request=request)
    path = write_task_artifact(tmp_path, Task(artifacts['task'], artifacts['environment'], artifacts['metrics'],
        task_intent=artifacts['task_intent'], artifacts=artifacts), 'multi_step_agentic')
    contract = json.loads(path.read_text())
    contract.pop('actions')
    (tmp_path/'BUILD_CONTRACT.json').write_text(json.dumps(contract))
    generate(tmp_path)
    project = Path(__file__).resolve().parents[2]
    for name in ('sandbox_runtime.py','runtime_llm.py'):
        shutil.copy2(project/'src/env_factory'/name, tmp_path/name)
    script = tmp_path/'exercise.py'
    script.write_text('''import json
from app import create_app
from sandbox_runtime import AcceptanceScenarioRunner, validate_json_schema
app = create_app()
headers = {'Authorization':'Bearer test-interaction'}
runner = AcceptanceScenarioRunner(app.handle, trainer_headers=headers,
    business_snapshot=app.business_snapshot, mutate_business_state=app.mutate_business_state)
contract = json.load(open('BUILD_CONTRACT.json'))
schema = next(endpoint['response_schema'] for endpoint in contract['requirements']['runtime_interface']['endpoints'] if endpoint['name'] == 'user_simulator')
for scenario in contract['acceptance_contract']['executable_scenarios']:
    result = runner.run(scenario)
    for step in result['history']:
        if step['operation'] == 'dialogue_turn':
            validate_json_schema(schema, step['body'])
print('all interaction scenarios passed')
''')
    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
        env={**os.environ, 'SANDBOX_TRAINER_API_KEY':'test-interaction'}, capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    report = tmp_path/'gate.json'
    checked = subprocess.run([sys.executable, str(project/'scripts/sandbox/validate_agentic_training_value.py'),
        '--root', str(tmp_path), '--output', str(report)], capture_output=True, text=True, timeout=30,
        env={**os.environ, 'SANDBOX_TRAINER_API_KEY':'test-interaction', 'PYTHONPATH':str(project/'src')})
    assert checked.returncode == 0, checked.stderr + report.read_text()
    assert json.loads(report.read_text())['hard_gates_passed']


def test_stateful_confirmation_is_required_before_write(tmp_path):
    source, request = interactive_source()
    source['description']['task_intent'] = 'modify'
    source['description']['task'] = '查询目标库区编号，将目标温度改为19，其他库区保持不变。'
    source['description']['public_input']['initial_user_message'] = '请先询问目标库区，查询后让我确认，原目标温度为18，我会另行给出新的目标。区域为东区。'
    # Region is public in this task, only the warehouse is hidden.
    source['environment_plan'].update(mode='stateful', requires_persistence=True)
    tool = source['tools'][1]['function']
    tool['parameters']['properties']['temperature'] = {'type':'integer', 'description':'新目标温度'}
    tool['parameters']['required'].append('temperature')
    source['tool_implementations'][1] = {'tool_name':'read_temperature', 'operation':'update', 'table':'zones',
        'result_field':'updated_count', 'selector':{'id':'id'}, 'changes':{'temperature':'temperature'}}
    source['scenarios'][0]['steps'][1]['arguments']['temperature'] = 19
    source['metric_implementations'][0].update(source='business_state', operator='eq',
        path='$.zones[?(@.name=="青松库")][0].temperature', expected=19)
    source['semantic_goal'] = {'row_predicates':[{'table':'zones','where':{'name':'青松库'},'values':{'temperature':19},'count':1}],
        'expected_delta':[{'table':'zones','where':{'name':'青松库'},'field':'temperature','before':18,'after':19}]}
    first = source['interaction_contract']['variants'][0]['stages'][0]
    source['interaction_contract']['variants'] = [{'stages':[first,
        {'id':'new_temperature', 'kind':'constraint_update', 'purpose':'用户将原18度目标改为新温度',
         'before_tool':'read_temperature', 'requires_tools':['lookup_zone'],
         'private_fact':19, 'previous_fact':18, 'bind_to':{'tool':'read_temperature','argument':'temperature'},
         'assistant_contains_all':['目标温度'], 'reference_response':'请问新的目标温度是多少？',
         'user_reply':'新的目标温度改为19。', 'retry_reply':'请向我询问新的目标温度。'},
        {'id':'approval', 'kind':'execution_confirmation', 'purpose':'用户确认写入19度',
         'before_tool':'read_temperature', 'requires_tools':['lookup_zone'],
         'assistant_contains_all':['19','确认'], 'reference_response':'已查到库区，是否确认将温度改为19？',
         'user_reply':'确认改为19。', 'retry_reply':'请先查询库区并明确要修改的温度，再让我确认。'}]}]
    artifacts = compile_source(source, root=tmp_path, request=request, script_count=1)
    checks = artifacts['generation_pipeline']['verification']
    assert checks['passed']


def test_skipped_tool_counterfactual_can_remain_before_confirmation(tmp_path):
    from env_factory.generation.delivery_preflight import verify_delivery
    from env_factory.tasks.task_routing import training_contract
    source = json.loads((Path(__file__).parents[1] / 'fixtures/generation/audit_confirmation_source.json').read_text())
    request = {'task_type':'Event', 'training_category':'multi_step_agentic', 'training_contract':training_contract('multi_step_agentic'),
               'available_environment_modes':['stateful'], 'require_interaction_contract':True,
               'graph_context': {'source':'neo4j_scene_path', 'hops':2, 'nodes':['义和团','围攻','使馆'],
                                 'keywords':['义和团','围攻','使馆'], 'relation':'SAME_EVENT_ELEMENT'}}
    artifacts = compile_source(source, root=tmp_path, request=request)
    assert verify_delivery(artifacts, tmp_path)['passed']
    report = json.loads((tmp_path/'prebuild_agentic_value.json').read_text())
    skipped = report['evidence']['counterfactuals']['skipped_tool_2']
    assert (skipped['status'] == 'completed' and skipped['reward'] <= .2) or 400 <= skipped.get('http_status', 0) < 500


def test_positive_dialogue_stage_assertions_are_still_enforced():
    from env_factory.sandbox_runtime import AcceptanceScenarioRunner, SandboxError
    def call(method, path, body, headers):
        return 200, {'user_query':'请先查询文书', 'interaction_stage':None}, {}
    runner = AcceptanceScenarioRunner(call)
    with pytest.raises(SandboxError, match='interaction branch'):
        runner.run({'steps':[{'operation':'dialogue_turn', 'content':'确认吗', 'expected_stage':'approval'}]})
    # Counterfactual dialogue deliberately omits the positive-stage assertion.
    assert runner.run({'steps':[{'operation':'dialogue_turn', 'content':'确认吗'}]})['history']
