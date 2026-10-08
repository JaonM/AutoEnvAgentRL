import json
from pathlib import Path

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.artifacts import write_task_artifact
from env_factory.graph.knowledge_graph import TaskType
from env_factory.tasks.task import Task
from env_factory.tasks.task_quality import score_file, score_task
from tests.generation.test_interaction_contract import interactive_source


FAILURE = '关键业务真值依赖未定义的后续用户回复'


def delivered(tmp_path):
    source, request = interactive_source()
    request['require_structured_interactions'] = True
    artifacts = compile_source(source, root=tmp_path, request=request)
    path = write_task_artifact(tmp_path, Task(artifacts['task'], artifacts['environment'],
        artifacts['metrics'], task_type=TaskType(artifacts['task_type']),
        task_intent='execute', complexity=artifacts['complexity'], artifacts=artifacts),
        artifacts['training_category'])
    return path, json.loads(path.read_text())


def test_defined_interactive_reply_is_not_unspecified_business_truth(tmp_path):
    path, task = delivered(tmp_path)
    assert FAILURE in score_task(task).eligibility_failures  # No file evidence supplied.
    assert FAILURE not in score_file(path).eligibility_failures


@pytest.mark.parametrize('defect', ['flag_only', 'missing_file', 'empty_scripts', 'missing_fact',
    'wrong_fact', 'missing_binding', 'unknown_argument', 'missing_dialogue',
    'wrong_call', 'uncovered_variant', 'malformed_reply', 'outside_manifest'])
def test_interaction_declaration_cannot_bypass_unresolved_input_gate(tmp_path, defect):
    path, task = delivered(tmp_path)
    users = task['artifacts']['user_simulation_manifest']
    script_path = tmp_path / users['root'] / users['scripts_file']
    scripts = json.loads(script_path.read_text())
    stage = scripts[-1]['interaction_protocol']['stages'][0]
    successes = [case for case in task['acceptance_contract']['executable_scenarios']
                 if case.get('kind') == 'goal_success']
    if defect == 'flag_only':
        task['artifacts'].pop('user_simulation_manifest')
    elif defect == 'missing_file':
        script_path.unlink()
    elif defect == 'empty_scripts':
        scripts = []
    elif defect == 'missing_fact':
        stage.pop('private_fact')
    elif defect == 'wrong_fact':
        stage['private_fact'] = '不存在的库区'
    elif defect == 'missing_binding':
        stage.pop('bind_to')
    elif defect == 'unknown_argument':
        stage['bind_to']['argument'] = 'not_a_parameter'
    elif defect == 'malformed_reply':
        stage['reference_response'] = '{}'
    elif defect == 'outside_manifest':
        users['root'] = '../elsewhere'
    elif defect == 'uncovered_variant':
        task['acceptance_contract']['executable_scenarios'] = [successes[0]]
    else:
        for case in successes:
            if defect == 'missing_dialogue':
                case['steps'] = [s for s in case['steps'] if s.get('operation') != 'dialogue_turn']
            else:
                for step in case['steps']:
                    if step.get('tool_name') == 'lookup_zone':
                        step['arguments']['name'] = '不存在的库区'
    if defect != 'missing_file':
        script_path.write_text(json.dumps(scripts, ensure_ascii=False))
    path.write_text(json.dumps(task, ensure_ascii=False))
    # The qualification helper fails closed before later file/runtime checks.
    report = score_task(task, task_root=tmp_path)
    assert FAILURE in report.eligibility_failures
