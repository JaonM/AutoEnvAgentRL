from copy import deepcopy
import json

import pytest

from env_factory.generation.agent_authoring import compile_source
from env_factory.generation.artifacts import write_task_artifact
from env_factory.generation.lifecycle_preview import preview_lifecycle_task
from env_factory.graph.knowledge_graph import TaskType
from env_factory.tasks.task import Task
from env_factory.tasks.task_quality import score_file
from tests.generation.test_realism_requirements import recovery_source
from tests.construction.test_gate_reform import script


def delivered_recovery(tmp_path):
    source, request = recovery_source()
    a = compile_source(source, root=tmp_path, request=request)
    path = write_task_artifact(tmp_path, Task(desc=a['task'], env=a['environment'],
        metrics=a['metrics'], task_type=TaskType(a['task_type']), task_intent=a['task_intent'],
        complexity=a['complexity'], artifacts=a), a['training_category'])
    return path, json.loads(path.read_text())


def test_fault_recovery_passes_both_prebuild_callers(tmp_path):
    path, task = delivered_recovery(tmp_path)
    assert preview_lifecycle_task(task, tmp_path)
    quality = score_file(path).to_dict()
    assert not any('预演失败' in finding for finding in quality['findings']), quality
    report = script('sandbox/assess_task_buildability.py').assess(tmp_path)
    assert not any(i['code'] == 'SUCCESS_TOOL_PREVIEW_INVALID' for i in report['issues']), report


def test_wrong_fault_expectation_and_wrong_business_goal_still_fail(tmp_path):
    _, task = delivered_recovery(tmp_path)
    wrong = deepcopy(task)
    steps = next(s['steps'] for s in wrong['acceptance_contract']['executable_scenarios']
                 if s.get('kind') == 'goal_success')
    failed = next(s for s in steps if s.get('expected_status') == 503)
    failed['expected_status'] = 200
    with pytest.raises(Exception, match='expected HTTP 200, got 503'):
        preview_lifecycle_task(wrong, tmp_path)
    wrong = deepcopy(task)
    predicate = wrong['task_spec']['goal_contract']['row_predicates'][0]
    predicate['values']['temperature'] = -999
    predicate.get('value_expressions', {}).pop('temperature', None)
    with pytest.raises(ValueError, match='postconditions'):
        preview_lifecycle_task(wrong, tmp_path)
