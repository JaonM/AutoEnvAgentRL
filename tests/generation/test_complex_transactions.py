import copy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from tests.generation.test_authorization_contract import approved_source


def complex_source():
    """Small mechanism control, not a claim about generated business difficulty."""
    source, request = approved_source()
    request['difficulty_profile'] = 'complex_write_transactions'
    source['realism_requirements'] = {'version': '1.0', 'mechanisms': ['compensation'],
                                       'compensation_tools': ['restore_temperature']}
    source['description']['public_input']['initial_user_message'] += (
        '请确认温度调整和第11批记录1、2的登记（状态1）。登记服务失败时，先恢复原温度再重试登记，成功后重新调整温度。')
    source['tables'].append({'table_name': 'orders', 'description': '调整登记', 'primary_key': ['id'],
        'columns': [{'name': name, 'type': 'integer', 'nullable': False} for name in ['id', 'batch', 'status']],
        'foreign_keys': [], 'indexes': [], 'constraints': [],
        'rows': [{'id': i, 'batch': 11, 'status': 0} for i in [1, 2]]})
    read = source['tool_implementations'][1]
    read['projection'].append('temperature')
    steps = source['scenarios'][0]['steps']
    steps[1]['capture']['original_temperature'] = '$.records[0].temperature'
    for name, parameters, impl in [
        ('register_orders', {'batch': {'type': 'integer', 'description': '公开批号'}},
         {'operation': 'update', 'table': 'orders', 'selector': {'batch': 'batch'},
          'changes': {}, 'derived_changes': {'status': 1}}),
        ('restore_temperature', {'id': {'type': 'integer', 'description': '已查询库区编号'},
                                 'value': {'type': 'integer', 'description': '先前读取的温度'}},
         {'operation': 'update', 'table': 'zones', 'selector': {'id': 'id'}, 'changes': {'value': 'temperature'}})]:
        source['tools'].append({'type': 'function', 'function': {'name': name, 'description': name,
            'parameters': {'type': 'object', 'properties': parameters, 'required': list(parameters), 'additionalProperties': False}}})
        source['tool_implementations'].append({'tool_name': name, 'result_field': 'updated_count', **impl})
        source['actions'].append({'name': name, 'description': name, 'atomicity_rationale': '事务内单表写入',
                                  'inputs': [], 'outputs': [], 'preconditions': [], 'effects': []})
        source['tool_bindings'].append({'tool_name': name, 'action_name': name})
        source['capability_plan'].append({'action_name': name, 'kind': 'environment_operation', 'requires_tool': True,
            'dependencies': ['read_temperature'], 'reason': '完成失败恢复和登记'})
        source['reward_key_steps'].append({'step_id': name, 'action_name': name, 'required_for_goal': True,
            'dependencies': ['read_temperature'], 'rationale': '恢复失败的业务流程'})
    policies = source['business_lifecycle']['tools']
    policies['update_temperature']['require_idempotency'] = True
    for name, temperature in [('register_orders', 18), ('restore_temperature', 25)]:
        policies[name] = {'require_idempotency': True, 'requires_grants': [name], 'guards': [
            {'id': 'recovery_state', 'table': 'zones', 'selector': {'name': '青松库'},
             'conditions': {'temperature': {'eq': temperature}}}]}
    source['business_lifecycle']['dynamics'] = {'faults': [
        {'id': 'registration_unavailable', 'tool': 'register_orders', 'call_index': 1, 'phase': 'before', 'status': 503}]}
    for variant in source['interaction_contract']['variants']:
        variant['stages'].extend([
            {'id': 'registration', 'kind': 'execution_confirmation', 'purpose': '确认登记范围',
             'assistant_contains_all': ['登记'], 'reference_response': '是否登记第11批记录1、2到状态1？',
             'user_reply': '同意登记第11批记录1、2到状态1。', 'retry_reply': '请确认登记范围。',
             'before_tool': 'register_orders', 'requires_tools': ['update_temperature'],
             'grant_permissions': ['register_orders'], 'approval_terms': {'register_orders': {
                 'scope': 'terms', 'arguments': {'batch': 11}, 'postconditions': [
                     {'table': 'orders', 'where': {'id': i}, 'values': {'status': 1}, 'count': 1} for i in [1, 2]]}}},
            {'id': 'restore', 'kind': 'execution_confirmation', 'purpose': '授权故障恢复',
             'assistant_contains_all': ['恢复'], 'reference_response': '恢复原温度18度后重试登记吗？',
             'user_reply': '同意恢复原温度18度后重试登记。', 'retry_reply': '请确认恢复温度。',
             'before_tool': 'restore_temperature', 'requires_tools': ['update_temperature'],
             'grant_permissions': ['restore_temperature'], 'approval_terms': {'restore_temperature': {
                 'scope': 'terms', 'postconditions': [
                     {'table': 'zones', 'where': {'name': '青松库'}, 'values': {'temperature': 18}, 'count': 1}]}}}])
    steps[2]['idempotency_key'] = 'initial-adjustment'
    registration = {'operation': 'tool_call', 'tool_name': 'register_orders', 'arguments': {'batch': 11},
                    'expected_status': 200, 'idempotency_key': 'registration'}
    failed = {**copy.deepcopy(registration), 'expected_status': 503}
    restore = {'operation': 'tool_call', 'tool_name': 'restore_temperature',
        'arguments': {'id': {'$ref': 'zone_id'}, 'value': {'$ref': 'original_temperature'}},
        'expected_status': 200, 'idempotency_key': 'restore'}
    finish = {**copy.deepcopy(steps[2]), 'idempotency_key': 'final-adjustment'}
    steps[-1:-1] = [failed, restore, registration, finish, copy.deepcopy(registration)]
    source['semantic_goal']['row_predicates'].extend([
        {'table': 'orders', 'where': {'id': i}, 'values': {'status': 1}, 'count': 1} for i in [1, 2]])
    for scenario in source['scenarios']:
        scenario['steps'].insert(0, {'operation': 'reset', 'body': {'seed': 17}})
        scenario['steps'].append({'operation': 'reward'})
    return source, request


def test_complex_profile_requires_actual_execution_witness(tmp_path):
    source, request = complex_source()
    task = compile_source(source, root=tmp_path, request=request)
    report = task['generation_pipeline']['verification']
    assert report['difficulty_coverage']['passed']
    assert all(row['transaction_witness']['audit_valid'] for row in report['difficulty_observations'])
    proof = report['difficulty_observations'][0]['transaction_witness']
    assert proof['changed_objects'] == 3
    assert proof['compensated_fault_ids'] == ['registration_unavailable']
    assert proof['idempotent_replay_tools'] == ['register_orders']


def test_delivered_standalone_gate_recomputes_transaction_witness(tmp_path):
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys
    from env_factory.generation.artifacts import write_task_artifact
    from env_factory.tasks.task import Task
    from scripts.sandbox.generate_sandbox_scaffold import generate
    source, request = complex_source()
    artifact = compile_source(source, root=tmp_path, request=request)
    # Same correct business behavior without a redundant transport replay. It
    # may be selected for policy counterfactuals, but another reference must be
    # allowed to establish the profile's replay coverage independently.
    references = artifact['acceptance_contract']['executable_scenarios']
    no_replay = copy.deepcopy(next(s for s in references if s.get('kind') == 'goal_success'))
    no_replay['scenario_id'] = '00-success-without-replay'
    position = max(i for i, step in enumerate(no_replay['steps']) if step.get('tool_name') == 'register_orders')
    no_replay['steps'].pop(position)
    references.insert(0, no_replay)
    path = write_task_artifact(tmp_path, Task(artifact['task'], artifact['environment'], artifact['metrics'],
        task_intent=artifact['task_intent'], artifacts=artifact), 'multi_step_agentic')
    contract = json.loads(path.read_text()); contract.pop('actions')
    (tmp_path / 'BUILD_CONTRACT.json').write_text(json.dumps(contract))
    generate(tmp_path)
    project = Path(__file__).resolve().parents[2]
    for name in ('sandbox_runtime.py', 'runtime_llm.py'):
        shutil.copy2(project / 'src/env_factory' / name, tmp_path / name)
    command = [sys.executable, str(project / 'scripts/sandbox/validate_agentic_training_value.py'),
        '--root', str(tmp_path), '--output', str(tmp_path / 'independent_gate.json')]
    env = {**os.environ, 'PYTHONPATH': str(project / 'src'), 'SANDBOX_TRAINER_API_KEY': 'test-transaction',
           'SANDBOX_EVALUATOR_MOCK': '1'}
    completed = subprocess.run(command, cwd=project, capture_output=True, text=True, env=env, timeout=45)
    assert (tmp_path / 'independent_gate.json').exists(), completed.stderr
    report = json.loads((tmp_path / 'independent_gate.json').read_text())
    assert completed.returncode == 0, report
    assert report['evidence']['transaction_witness']['changed_objects'] == 3
    assert report['evidence']['transaction_witness']['compensated_fault_ids'] == ['registration_unavailable']
    runs = report['evidence']['transaction_coverage_runs']
    assert len(runs) == 2
    assert runs[0]['scenario_id'] == '00-success-without-replay'
    assert runs[0]['reward'] == 1 and not runs[0]['witness']['idempotent_replay_tools']
    # A copied authoring report cannot cover a delivered runtime without audit.
    contract['task_spec'].pop('transaction_contract')
    (tmp_path / 'BUILD_CONTRACT.json').write_text(json.dumps(contract))
    completed = subprocess.run(command, cwd=project, capture_output=True, text=True, env=env, timeout=45)
    report = json.loads((tmp_path / 'independent_gate.json').read_text())
    assert completed.returncode == 1, report
    assert 'complex_transaction_coverage' in report['failed_gates']


@pytest.mark.parametrize('defect', ['no_compensation', 'no_idempotency', 'unbound_grant'])
def test_complex_contract_rejects_missing_mechanisms_before_execution(tmp_path, defect):
    source, request = complex_source()
    if defect == 'no_compensation':
        source.pop('realism_requirements')
    elif defect == 'no_idempotency':
        source['business_lifecycle']['tools']['register_orders']['require_idempotency'] = False
    else:
        for variant in source['interaction_contract']['variants']:
            variant['stages'][-1]['approval_terms']['restore_temperature'] = {'scope': 'operation'}
    with pytest.raises(ValueError, match='DIFFICULTY_PROFILE_MISSING'):
        compile_source(source, root=tmp_path, request=request, structural_preview=True)
