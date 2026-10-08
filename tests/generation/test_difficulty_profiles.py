import copy
import json
import random
from collections import Counter

import pytest

from env_factory.generation.difficulty_profiles import allocate, augment_realism, parse_mix, requirements, require_coverage


def test_frozen_mix_is_reproducible_and_keeps_all_ten_planned_slots():
    mix = 'standard=4,long_horizon_planning=2,large_scale_retrieval=2,complex_write_transactions=2'
    def run(seed):
        return allocate(['multi_step_agentic'] * 10, mix, rng=random.Random(seed), available_modes=['reference_data', 'stateful'])
    assert run(19) == run(19)
    assert Counter(run(19)) == {'standard': 4, 'long_horizon_planning': 2, 'large_scale_retrieval': 2, 'complex_write_transactions': 2}
    assert run(19) != run(20)


@pytest.mark.parametrize('text', ['', 'unknown=1', 'standard=nan', 'standard=-1', 'standard=0', 'standard=1,standard=1'])
def test_invalid_mix_rejected(text):
    with pytest.raises(ValueError, match='DIFFICULTY_MIX'):
        parse_mix(text)


@pytest.mark.parametrize('categories,modes,intent', [
    (['simple_agentic'], ['stateful'], None),
    (['multi_step_agentic'], ['reference_data'], None),
    (['multi_step_agentic'], ['stateful'], 'query')])
def test_incompatible_request_never_downgrades_profile(categories, modes, intent):
    with pytest.raises(ValueError, match='INCOMPATIBLE'):
        allocate(categories, 'complex_write_transactions=1', rng=random.Random(0), available_modes=modes, task_intent=intent)


def test_required_mechanisms_survive_merging_and_requirements_are_not_mutable_aliases():
    req = {'difficulty_profile': 'complex_write_transactions', 'training_category': 'multi_step_agentic',
           'realism_requirements': {'version': '1.0', 'mechanisms': ['external_change']}}
    original = copy.deepcopy(req)
    assert augment_realism(req)['mechanisms'] == ['authorization', 'external_change', 'fault_recovery', 'idempotency']
    assert req == original
    value = requirements('large_scale_retrieval'); value['min_dataset_rows'] = 1
    assert requirements('large_scale_retrieval')['min_dataset_rows'] == 10000
    from env_factory.generation.difficulty_profiles import validate_request
    with pytest.raises(ValueError, match='frozen requirements differ'):
        validate_request({'difficulty_profile': 'large_scale_retrieval', 'difficulty_requirements': value})


def retrieval_source():
    from .test_agent_authoring import fixture
    full, request = fixture('multi_step_agentic')
    source = {key: copy.deepcopy(full[key]) for key in ('version', 'description', 'environment_plan', 'tables')}
    source['tables'].append({'table_name': 'readings', 'description': '逐次温度记录', 'primary_key': ['id'],
        'columns': [{'name': 'id', 'type': 'integer', 'nullable': False},
                    {'name': 'zone_id', 'type': 'integer', 'nullable': False},
                    {'name': 'temperature', 'type': 'integer', 'nullable': False}],
        'foreign_keys': [{'column': 'zone_id', 'ref_table': 'zones', 'ref_column': 'id'}],
        'indexes': [], 'constraints': [],
        'rows': [{'id': i, 'zone_id': 821, 'temperature': i % 31} for i in range(1, 206)],
        'row_recipe': {'count': 10000, 'columns': {'id': {'sequence': {'start': 1000, 'step': 1}},
            'zone_id': {'literal': 547}, 'temperature': {'integer': {'min': -10, 'max': 40}}}}})
    source['business_tools'] = [{**copy.deepcopy(tool['function']), 'implementation': copy.deepcopy(impl)}
                               for tool, impl in zip(full['tools'], full['tool_implementations'])]
    tool = source['business_tools'][1]
    tool['parameters']['properties']['cursor'] = {'type': 'string', 'description': '上一页游标'}
    tool['implementation'].update(table='readings', filters=[{'argument': 'id', 'column': 'zone_id', 'operator': 'eq'}],
        projection=['id', 'temperature'], order_by=['id'], pagination={'cursor_argument': 'cursor', 'page_size': 100})
    source['answer_fields'] = {'sum': {'kind': 'derived', 'schema': {'type': 'integer', 'description': '青松库全部温度记录之和'},
        'expression': {'aggregate': {'table': 'readings', 'field': 'temperature', 'op': 'sum',
            'where': {'zone_id': {'lookup': {'table': 'zones', 'field': 'id', 'where': {'name': '青松库'}}}}}}}}
    source['outcomes'] = [{'id': 'sum_answer', 'rubric': '完整加总青松库温度记录，排除其他库区', 'fields': ['sum']}]
    source['reference'] = {'calls': [
        {'tool_name': 'lookup_zone', 'arguments': {'name': '青松库'}, 'capture': {'zone_id': '$.records[0].id'}},
        {'tool_name': 'read_temperature', 'arguments': {'id': {'$ref': 'zone_id'}}, 'capture': {'page2': '$.next_cursor'}},
        {'tool_name': 'read_temperature', 'arguments': {'id': {'$ref': 'zone_id'}, 'cursor': {'$ref': 'page2'}},
         'capture': {'page3': '$.next_cursor'}},
        {'tool_name': 'read_temperature', 'arguments': {'id': {'$ref': 'zone_id'}, 'cursor': {'$ref': 'page3'}}}],
        'answer': json.dumps({'sum': sum(i % 31 for i in range(1, 206))})}
    goal = '查询青松库的全部温度记录，沿分页游标读取所有页，汇总温度，排除其他库区。'
    source['description'].update(task=goal, goal=goal, expected_result='全部匹配记录的温度之和')
    source['description']['public_input']['initial_user_message'] = goal
    request.update(difficulty_profile='large_scale_retrieval', seed=73, require_answer_fields=True)
    return source, request


def test_compiled_large_profile_proves_actual_complete_three_page_cross_table_read(tmp_path):
    from env_factory.generation.agent_authoring import compile_source
    value, request = retrieval_source()
    result = compile_source(value, root=tmp_path, request=request)
    report = result['generation_pipeline']['verification']
    assert report['difficulty_coverage']['passed']
    observations = report['difficulty_observations']
    assert any(row['retrieval_complete'] and row['retrieval_chains'][0]['pages'] == 3
               and set(row['read_tables']) == {'zones', 'readings'} for row in observations)
    assert result['task_spec']['difficulty_requirements']['min_dataset_rows'] == 10000
    assert '即使已知最终答案' in result['public_input']['initial_user_message']
    # Inspect the omitted terminal page specifically: preceding pages must run
    # with this new episode's cursors, not fail because baseline cursors expired.
    assert report['checks']['skip_0_4']


def test_standalone_reward_rejects_known_correct_answer_with_incomplete_pages(tmp_path):
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    from env_factory.generation.agent_authoring import compile_source
    from env_factory.generation.artifacts import write_task_artifact
    from env_factory.tasks.task import Task
    from scripts.sandbox.generate_sandbox_scaffold import generate
    source, request = retrieval_source()
    artifact = compile_source(source, root=tmp_path, request=request)
    path = write_task_artifact(tmp_path, Task(artifact['task'], artifact['environment'], artifact['metrics'],
        task_intent=artifact['task_intent'], artifacts=artifact), 'multi_step_agentic')
    contract = json.loads(path.read_text()); contract.pop('actions')
    (tmp_path / 'BUILD_CONTRACT.json').write_text(json.dumps(contract))
    generate(tmp_path)
    project = Path(__file__).resolve().parents[2]
    for name in ('sandbox_runtime.py', 'runtime_llm.py'):
        shutil.copy2(project / 'src/env_factory' / name, tmp_path / name)
    script = tmp_path / 'check_complete_pages.py'
    script.write_text('''import json
from app import create_app
app = create_app()
headers = {'Authorization': 'Bearer test-retrieval'}
scores = {}
for variant in ['omit_last', 'repeat_first', 'complete']:
    assert app.handle('POST', '/v1/reset', {'seed': 73}, headers)[0] == 200
    status, data, _ = app.handle('POST', '/v1/tools/lookup_zone', {'name': '青松库'})
    assert status == 200
    identity = data['records'][0]['id']
    args = {'id': identity}
    for index in range(2 if variant == 'omit_last' else 3):
        status, page, _ = app.handle('POST', '/v1/tools/read_temperature', args)
        assert status == 200
        if variant != 'repeat_first': args = {'id': identity, 'cursor': page['next_cursor']}
    answer = {'sum': sum(i % 31 for i in range(1, 206))}
    assert app.handle('POST', '/v1/agent_response', {'content': json.dumps(answer)}, headers)[0] == 200
    status, reward, _ = app.handle('GET', '/v1/reward', {}, headers)
    assert status == 200
    scores[variant] = reward['reward']
assert scores == {'omit_last': 0, 'repeat_first': 0, 'complete': 1}, scores
print(json.dumps(scores))
''')
    run = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True,
        env={**os.environ, 'SANDBOX_TRAINER_API_KEY': 'test-retrieval'}, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    gate = subprocess.run([sys.executable, str(project / 'scripts/sandbox/validate_agentic_training_value.py'),
        '--root', str(tmp_path), '--output', str(tmp_path / 'independent_gate.json')],
        cwd=project, capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': str(project / 'src'), 'SANDBOX_TRAINER_API_KEY': 'test-retrieval',
             'SANDBOX_EVALUATOR_MOCK': '1'}, timeout=45)
    report = json.loads((tmp_path / 'independent_gate.json').read_text())
    assert gate.returncode == 0, report


def test_retrieval_coverage_rejects_short_or_unrelated_reads():
    artifact = {'task_spec': {'difficulty_requirements': requirements('large_scale_retrieval')}}
    good = {'retrieval_complete': True, 'retrieval_chains': [{'pages': 3}], 'read_tables': ['a', 'b']}
    assert require_coverage(artifact, [good])['passed']
    for bad in [{**good, 'retrieval_complete': False}, {**good, 'retrieval_chains': [{'pages': 2}]},
                {**good, 'read_tables': ['a']}]:
        with pytest.raises(ValueError, match='DIFFICULTY_EXECUTION_UNCOVERED'):
            require_coverage(artifact, [bad])


def test_declared_unused_edges_and_cursor_hops_are_not_planning_depth():
    from env_factory.generation.difficulty_profiles import observed_dependency_depth
    edges = [{'edge': {'from_tool': f't{i}', 'to_tool': f't{i+1}', 'result_path': '$.id', 'argument_path': '$.id'},
              'established': True} for i in range(7)]
    assert observed_dependency_depth({'dependency_edges': edges}) == 8
    for change in ('unused', 'cursor', 'coarse'):
        altered = copy.deepcopy(edges)
        if change == 'unused': altered[3]['established'] = False
        if change == 'cursor': altered[3]['edge']['result_path'] = '$.next_cursor'
        if change == 'coarse': altered[3]['edge'].pop('result_path')
        assert observed_dependency_depth({'dependency_edges': altered}) < 8


def test_actual_eight_stage_capture_chain_and_recovery_establish_long_profile(tmp_path):
    from .test_agent_authoring import fixture
    from env_factory.generation.agent_authoring import compile_source
    full, request = fixture('multi_step_agentic')
    source = {key: copy.deepcopy(full[key]) for key in ('version', 'description', 'environment_plan', 'tables')}
    table = source['tables'][0]
    table['columns'].append({'name': 'next_id', 'type': 'integer', 'nullable': False})
    table['rows'] = [{'id': 100+i, 'next_id': 101+i, 'name': '青松库' if i == 0 else f'后续检查{i}', 'temperature': 18+i}
                     for i in range(8)]
    source['business_tools'], calls = [], []
    for i in range(8):
        name, argument = f'read_stage_{i}', 'name' if i == 0 else 'id'
        source['business_tools'].append({'name': name, 'description': f'读取第{i+1}阶段库区温度检查及后续记录标识',
            'parameters': {'type': 'object', 'properties': {argument: {'type': 'string' if i == 0 else 'integer', 'description': '查询目标标识'}},
                           'required': [argument], 'additionalProperties': False},
            'implementation': {'operation': 'select', 'table': 'zones',
                'filters': [{'argument': argument, 'column': argument, 'operator': 'eq'}],
                'projection': ['next_id', 'temperature']}})
        call = {'tool_name': name, 'arguments': {argument: '青松库' if i == 0 else {'$ref': f'next{i-1}'}}}
        if i == 3:
            call['idempotency_key'] = 'stage-four'
            calls.append({**copy.deepcopy(call), 'expected_status': 503})
        if i < 7: call['capture'] = {f'next{i}': '$.records[0].next_id'}
        calls.append(call)
    source['answer_fields'] = {'temperature': {'kind': 'observed', 'schema': {'type': 'integer', 'description': '最后阶段实际返回温度'},
        'observation': {'tool': 'read_stage_7', 'path': '$.records[0].temperature'}, 'missing': -999}}
    source['outcomes'] = [{'id': 'temperature_answer', 'rubric': '沿实际链条查到最后阶段温度', 'fields': ['temperature']}]
    source['reference'] = {'calls': calls, 'answer': json.dumps({'temperature': 25})}
    source['business_lifecycle'] = {'version': '1.0', 'dynamics': {'tick_seconds': 1, 'faults': [
        {'id': 'stage_four_unavailable', 'tool': 'read_stage_3', 'call_index': 1, 'phase': 'before', 'status': 503}]}}
    request.update(difficulty_profile='long_horizon_planning', seed=17)
    result = compile_source(source, root=tmp_path, request=request)
    report = result['generation_pipeline']['verification']
    assert report['difficulty_coverage']['passed']
    assert any(row['dependency_depth'] == 8 and row['recovered_fault_ids'] == ['stage_four_unavailable']
               for row in report['difficulty_observations'])
