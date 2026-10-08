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
from .test_agent_authoring import fixture


def test_generated_sandbox_uses_compiled_clock_origin_and_shared_schema(tmp_path):
    source, request = fixture('simple_agentic')
    source['business_lifecycle'] = {'version': '1.0',
        'clock': {'origin_datetime': '2026-10-08T09:00:00+08:00'}, 'dynamics': {'tick_seconds': 7}}
    artifacts = compile_source(source, root=tmp_path, request=request)
    path = write_task_artifact(tmp_path, Task(artifacts['task'], artifacts['environment'], artifacts['metrics'],
        task_intent=artifacts['task_intent'], artifacts=artifacts), 'simple_agentic')
    contract = json.loads(path.read_text())
    contract.pop('actions')
    (tmp_path / 'BUILD_CONTRACT.json').write_text(json.dumps(contract))
    generate(tmp_path)
    project = Path(__file__).resolve().parents[2]
    for name in ['sandbox_runtime.py', 'runtime_llm.py']:
        shutil.copy2(project / 'src/env_factory' / name, tmp_path / name)
    script = tmp_path / 'check_clock.py'
    script.write_text('''import json
from app import create_app
from sandbox_runtime import validate_json_schema
app = create_app()
headers = {'Authorization': 'Bearer clock-delivery'}
assert app.handle('POST', '/v1/reset', {'seed': 17}, headers)[0] == 200
contract = json.load(open('BUILD_CONTRACT.json'))
schema = next(e['response_schema'] for e in contract['requirements']['runtime_interface']['endpoints'] if e['name'] == 'time')
initial = app.handle('GET', '/v1/time')[1]
validate_json_schema(schema, initial)
assert initial['virtual_datetime'] == '2026-10-08T01:00:00+00:00'
assert app.handle('POST', '/v1/tools/read_temperature', {'name': '青松库'})[0] == 200
current = app.handle('GET', '/v1/time')[1]
validate_json_schema(schema, current)
assert current['tick'] == 1 and current['elapsed_seconds'] == 7
assert current['virtual_datetime'] == '2026-10-08T01:00:07+00:00'
assert app.handle('POST', '/v1/reset', {'seed': 18}, headers)[0] == 200
assert app.handle('GET', '/v1/time')[1] == initial
print('generated clock passed')
''')
    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path,
        env={**os.environ, 'SANDBOX_TRAINER_API_KEY': 'clock-delivery'},
        capture_output=True, text=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert 'generated clock passed' in run.stdout
