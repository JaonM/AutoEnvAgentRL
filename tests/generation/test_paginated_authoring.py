import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.artifacts import write_task_artifact
from env_factory.tasks.task import Task
from scripts.sandbox.generate_sandbox_scaffold import generate
from .test_data_recipes import source


def test_compiled_standalone_sandbox_executes_reproducible_bulk_data_and_cursor_chain(tmp_path):
    value, request = source()
    table = value['tables'][0]
    table['rows'] = [{'id': index, 'name': '青松库', 'temperature': index % 31, 'current': True}
                     for index in range(1, 206)]
    tool = value['business_tools'][0]
    tool['parameters']['properties']['cursor'] = {'type': 'string', 'description': '上一页返回的游标'}
    tool['implementation'].update(pagination={'cursor_argument': 'cursor', 'page_size': 100}, order_by=['id'])
    value['complete_retrieval_tools'] = ['read_temperature']
    value['answer_fields'] = {'sum': {'kind': 'derived',
        'schema': {'type': 'integer', 'description': '全部青松库记录的温度之和'},
        'expression': {'aggregate': {'table': 'zones', 'field': 'temperature', 'where': {'name': '青松库'}, 'op': 'sum'}}}}
    value['outcomes'] = [{'id': 'sum_answer', 'rubric': '完整加总全部匹配记录', 'fields': ['sum']}]
    goal = '检索全部青松库温度资料并求温度总和，必须翻完所有页，不得遗漏或混入其他库区。'
    value['description'].update(task=goal, goal=goal, expected_result='全部匹配记录的温度和')
    value['description']['public_input']['initial_user_message'] = goal
    value['reference'] = {'calls': [
        {'tool_name': 'read_temperature', 'arguments': {'name': '青松库'}, 'capture': {'page2': '$.next_cursor'}},
        {'tool_name': 'read_temperature', 'arguments': {'name': '青松库', 'cursor': {'$ref': 'page2'}},
         'capture': {'page3': '$.next_cursor'}},
        {'tool_name': 'read_temperature', 'arguments': {'name': '青松库', 'cursor': {'$ref': 'page3'}}}],
        'answer': json.dumps({'sum': sum(r['temperature'] for r in table['rows'])})}
    artifact = compile_source(value, root=tmp_path, request=request)
    assert artifact['generation_pipeline']['verification']['passed']
    assert 'restart without a cursor' in artifact['tools'][0]['function']['description']
    path = write_task_artifact(tmp_path, Task(artifact['task'], artifact['environment'], artifact['metrics'],
        task_intent=artifact['task_intent'], artifacts=artifact), 'simple_agentic')
    contract = json.loads(path.read_text()); contract.pop('actions')
    (tmp_path / 'BUILD_CONTRACT.json').write_text(json.dumps(contract))
    generate(tmp_path)
    project = Path(__file__).resolve().parents[2]
    for name in ('sandbox_runtime.py', 'runtime_llm.py'):
        shutil.copy2(project / 'src/env_factory' / name, tmp_path / name)
    script = tmp_path / 'verify_pages.py'
    script.write_text('''import json
from app import create_app
app=create_app()
headers={'Authorization':'Bearer test-pages'}
assert app.handle('POST','/v1/reset',{'seed':73},headers)[0]==200
args={'name':'青松库'}
counts=[]; total=0
while True:
    status,page,_=app.handle('POST','/v1/tools/read_temperature',args)
    assert status==200 and page['total_count']==205
    counts.append(page['count'])
    total+=sum(r['temperature'] for r in page['records'])
    if not page['has_more']: break
    args['cursor']=page['next_cursor']
assert counts==[100,100,5],counts
rewards=[]
for answer in [total,total-1]:
    assert app.handle('POST','/v1/agent_response',{'content':json.dumps({'sum':answer})},headers)[0]==200
    status,result,_=app.handle('GET','/v1/reward',{},headers)
    assert status==200
    rewards.append(result['reward'])
assert rewards[0]==1 and rewards[1]<1,rewards
print(json.dumps({'pages':counts,'sum':total,'rewards':rewards}))
''')
    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, 'SANDBOX_TRAINER_API_KEY': 'test-pages'}, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
