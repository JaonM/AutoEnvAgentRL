"""Generated service wiring must agree with author-side branch verification."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.artifacts import write_task_artifact
from env_factory.tasks.task import Task
from scripts.sandbox.generate_sandbox_scaffold import generate
from tests.generation.test_execution_routes import source_with_routes


def materialize(source, request, tmp_path):
    artifacts = compile_source(source, request=request, root=tmp_path)
    task_path = write_task_artifact(tmp_path, Task(artifacts['task'], artifacts['environment'], artifacts['metrics'],
        task_intent=artifacts['task_intent'], artifacts=artifacts), 'multi_step_agentic')
    shutil.copy2(task_path, tmp_path / 'BUILD_CONTRACT.json')
    generate(tmp_path)
    project = Path(__file__).resolve().parents[2]
    for name in ('sandbox_runtime.py', 'runtime_llm.py'):
        shutil.copy2(project / 'src' / 'env_factory' / name, tmp_path / name)
    return project


def test_generated_service_enforces_observed_refusal_and_stop_guard(tmp_path):
    from tests.generation.test_dialogue_terminal_routes import refusal_source
    source, request = refusal_source()
    materialize(source, request, tmp_path)
    exercise = r'''
import copy,json
from pathlib import Path
import app
from sandbox_runtime import AcceptanceScenarioRunner
root=Path.cwd(); contract=json.loads((root/'BUILD_CONTRACT.json').read_text())
probe=next(p for p in contract['execution_route_probes'] if p['route_id']=='declined')
users=contract['artifacts']['user_simulation_manifest']
path=root/users['root']/users['scripts_file']; scripts=json.loads(path.read_text())
for script in scripts: script['variables'].update(probe['user_variables'])
path.write_text(json.dumps(scripts))
service=app.create_app(db_path=root/'refusal.sqlite3')
runner=AcceptanceScenarioRunner(service.handle, trainer_headers={'Authorization':'Bearer local-route-test'})
for scenario in probe['scenarios']:
    if scenario.get('kind')!='goal_success': continue
    result=runner.run(scenario)
    reward=result['history'][-1]['body']
    assert reward['reward']==1 and reward['execution_route']=='declined',reward
    for defect in ['missing_dialogue','continued_call']:
        steps=copy.deepcopy(scenario['steps'])
        if defect=='missing_dialogue':
            steps=[s for s in steps if s['operation']!='dialogue_turn']
        else:
            index=next(i for i,s in enumerate(steps) if s['operation']=='agent_response')
            steps.insert(index,{'operation':'tool_call','tool_name':'lookup_zone',
                'arguments':{'name':'A库','region':'东区'},'expected_status':409})
        outcome=runner.run({'steps':steps,'assertions':[]})
        if defect=='continued_call':
            rejected=next(h for h in outcome['history'] if h['body'].get('error',{}).get('code')=='USER_REQUEST_STOPPED')
            assert rejected['status']==409,rejected
        assert outcome['history'][-1]['body']['reward']==0,outcome
'''
    result = subprocess.run([sys.executable, '-c', exercise], cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(tmp_path), 'SANDBOX_TRAINER_API_KEY': 'local-route-test'},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('design', ['stateful_routes', 'decision_table', 'readonly_routes'])
def test_generated_application_executes_write_and_no_write_routes(tmp_path, design):
    from tests.generation.test_decision_table import decision_source
    from tests.generation.test_readonly_execution_routes import readonly_routes
    factory = {'stateful_routes': source_with_routes, 'decision_table': decision_source,
               'readonly_routes': readonly_routes}[design]
    source, request = factory()
    materialize(source, request, tmp_path)
    # Import in a fresh interpreter: do not accidentally reuse an earlier test app.
    exercise = r'''
import copy,json
from pathlib import Path
import app
from sandbox_runtime import AcceptanceScenarioRunner
root=Path.cwd(); contract=json.loads((root/'BUILD_CONTRACT.json').read_text())
manifest=contract['artifacts']['data_manifest']
primary=next(s['execution_route'] for s in contract['acceptance_contract']['executable_scenarios'] if s.get('kind')=='goal_success')
cases=[{'route_id':primary,'initial_changes':[], 'scenarios':contract['acceptance_contract']['executable_scenarios']},
       *contract['execution_route_probes']]
reports=[]
for case in cases:
    for change in case['initial_changes']:
        table=next(t for t in manifest['tables'] if t['table_name']==change['table'])
        path=root/manifest['root']/table['rows_file']
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if all(row[k]==v for k,v in change['where'].items()): row.update(change['values'])
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    service=app.create_app(db_path=root/(case['route_id']+'.sqlite3'))
    runner=AcceptanceScenarioRunner(service.handle, trainer_headers={'Authorization':'Bearer local-route-test'})
    for scenario in case['scenarios']:
        if scenario.get('kind')!='goal_success': continue
        steps=[{'operation':'reset','body':{'seed':17}},
               *[copy.deepcopy(s) for s in scenario['steps'] if s['operation'] not in {'reset','reward'}],
               {'operation':'reward'}]
        result=runner.run({'steps':steps,'assertions':[]})
        reward=result['history'][-1]['body']
        assert reward['reward']>=1-1e-9 and reward['execution_route']==case['route_id'],reward
        for step in steps:
            if step['operation']=='agent_response':step['content']='unknown'
        wrong=runner.run({'steps':steps,'assertions':[]})['history'][-1]['body']
        assert wrong['reward']<=.2+1e-9,wrong
        reports.append(reward)
(root/'route-service-report.json').write_text(json.dumps(reports))
'''
    result = subprocess.run([sys.executable, '-c', exercise], cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(tmp_path), 'SANDBOX_TRAINER_API_KEY': 'local-route-test'},
        capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    reports = json.loads((tmp_path / 'route-service-report.json').read_text())
    expected = ['found', 'unresolved'] if design == 'readonly_routes' else ['adjust', 'already_set']
    assert [r['execution_route'] for r in reports] == expected
    assert reports[1]['inactive_components'] == [
        'process_read_temperature' if design == 'readonly_routes' else 'process_update_temperature']


@pytest.mark.parametrize('captured_key', [False, True])
def test_final_agentic_gate_accepts_idempotent_retry_without_inflating_probes(tmp_path, captured_key):
    from copy import deepcopy
    from tests.generation.test_dynamic_state_goal import dynamic_source
    source, request = dynamic_source()
    steps = source['scenarios'][0]['steps']
    steps[2]['idempotency_key'] = 'same-request'
    if captured_key:
        source['tool_implementations'][0]['projection'].append('name')
        steps[0]['capture']['request_key'] = '$.records[0].name'
        steps[2]['idempotency_key'] = {'$ref': 'request_key'}
    steps.insert(3, deepcopy(steps[2]))
    for scenario in source['scenarios']:
        scenario['steps'].insert(0, {'operation': 'reset', 'body': {'seed': 17}})
        scenario['steps'].append({'operation': 'reward'})
    project = materialize(source, request, tmp_path)
    report_path = tmp_path / 'gate.json'
    result = subprocess.run([sys.executable, str(project / 'scripts/sandbox/validate_agentic_training_value.py'),
        '--root', str(tmp_path), '--output', str(report_path)],
        env={**os.environ, 'PYTHONPATH': str(project / 'src'), 'SANDBOX_TRAINER_API_KEY': 'local-retry-test',
             'SANDBOX_EVALUATOR_MOCK': '1'}, capture_output=True, text=True, timeout=30)
    assert report_path.exists(), result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report['hard_gates_passed'], report['failures']
    assert report['evidence']['idempotent_call_groups'] == [[1], [2], [3, 4]]
    probes = report['evidence']['counterfactuals']
    assert 'skipped_tool_3' in probes and 'skipped_tool_4' not in probes
    assert 'corrupted_arguments_3' in probes and 'corrupted_arguments_4' not in probes
    from scripts.certify_training_materials import valid_reward_calibration
    task = json.loads((tmp_path / 'BUILD_CONTRACT.json').read_text())
    assert not valid_reward_calibration(report, task)  # offline remains uncertified
    # Unit-check the certifier's matrix shape independently of its live-mode gate.
    # This synthetic copy is never written back as actual validation evidence.
    assert valid_reward_calibration({**report, 'validation_mode': 'live_evaluator'}, task)
    # The certifier recomputes groups from the task, ignoring claimed grouping.
    changed = deepcopy(task)
    success = next(s for s in changed['acceptance_contract']['executable_scenarios'] if s.get('kind') == 'goal_success')
    [s for s in success['steps'] if s.get('operation') == 'tool_call'][-1]['idempotency_key'] = 'different-request'
    assert not valid_reward_calibration({**report, 'validation_mode': 'live_evaluator'}, changed)


def test_optional_repeat_read_has_positive_probe_and_certifier_coverage(tmp_path):
    from copy import deepcopy
    from tests.generation.test_dynamic_state_goal import dynamic_source
    from scripts.certify_training_materials import valid_reward_calibration
    source, request = dynamic_source()
    steps = source['scenarios'][0]['steps']
    repeat = {**deepcopy(steps[1]), 'optional_observation': True}
    repeat.pop('step_id', None)
    steps.insert(-1, repeat)
    for scenario in source['scenarios']:
        scenario['steps'].insert(0, {'operation': 'reset', 'body': {'seed': 17}})
        scenario['steps'].append({'operation': 'reward'})
    project = materialize(source, request, tmp_path)
    report_path = tmp_path / 'gate.json'
    result = subprocess.run([sys.executable, str(project / 'scripts/sandbox/validate_agentic_training_value.py'),
        '--root', str(tmp_path), '--output', str(report_path)],
        env={**os.environ, 'PYTHONPATH': str(project / 'src'), 'SANDBOX_TRAINER_API_KEY': 'local-repeat-test',
             'SANDBOX_EVALUATOR_MOCK': '1'}, capture_output=True, text=True, timeout=30)
    assert report_path.exists(), result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert report['hard_gates_passed'], report['failures']
    assert report['evidence']['counterfactuals']['optional_observation_4']['reward'] == 1
    task = json.loads((tmp_path / 'BUILD_CONTRACT.json').read_text())
    synthetic = {**report, 'validation_mode': 'live_evaluator'}
    assert valid_reward_calibration(synthetic, task)
    changed = deepcopy(synthetic)
    changed['evidence']['counterfactuals']['optional_observation_4']['reward'] = 0
    assert not valid_reward_calibration(changed, task)
