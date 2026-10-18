"""Task design constraints are verified from behavior, independently of author claims."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_core import FakeRunner
from test_generate import SCENE, Writer
from test_world import GOALS, world_request

from env_factory.authoring import pipeline
from env_factory.production import task_requirements as quality, world


def test_measure_excludes_rejected_writes_and_checks_actual_state():
    request = {'tables': {'t': []}, 'user_facts': {'goals': [{'process': 'p'}]}}
    result = {'final_state': {'t': []}, 'trace': [
        {'kind': 'tool', 'tool': 'write', 'status': 'error', 'write': True},
        {'kind': 'tool', 'tool': 'handoff', 'status': 'ok', 'write': False}]}
    measured = quality.measure(result, request)
    assert measured['writes'] == 0 and not measured['state_changed']
    assert measured['tools'] == ['handoff'] and measured['tool_calls'] == 1
    with patch.object(quality, 'run_reference', return_value=result):
        request['time'] = {'start': 0}
        assert not quality.evaluate(None, request, {'min_writes': 1})['passed']
        assert quality.evaluate(None, request, {'required_tools': ['handoff'], 'state_effect': 'unchanged'})['passed']
        assert not quality.evaluate(None, request, {'required_tools': ['write']})['passed']


def test_profile_fails_before_model_call_on_unknown_or_inconsistent_requirements():
    for value in ({'tau2_specific': True}, {'min_writes': True}, {'min_writes': 2, 'max_writes': 1},
                  {'state_effect': 'refused'}, {'required_tools': 'handoff'}):
        with pytest.raises(ValueError):
            quality.validate(value)
    with pytest.raises(ValueError):
        quality.profile({'by_shape': {'unused_domain': {}}})
    profile = quality.profile({'default': {'min_goals': 1}, 'by_shape': {'complex': {'min_writes': 2}}})
    assert quality.select(profile, 'complex') == {'min_goals': 1, 'min_writes': 2}
    assert quality.select(profile, None) == {'min_goals': 1}


@pytest.fixture
def certified(tmp_path):
    status = pipeline.build({**world_request(GOALS), 'user_request': '续借一下'}, tmp_path / 'world',
                            runner=FakeRunner(), perturbed=5)
    assert status['training_ready'], status
    return Path(status['bundle'])


def author_task(processes=('p1',)):
    facts = copy.deepcopy(world_request(GOALS)['user_facts'])
    facts['goals'] = [{'process': p, 'params': {'loan_id': 'L1'}} for p in processes]
    return {'title': '续借任务', 'user_facts': facts, 'user_scene': SCENE, 'user_script': {'traits': ['话少']}}


def test_all_assigned_processes_required_and_automatically_repaired(certified, tmp_path):
    roles = []
    def runner(folder, prompt, **options):
        roles.append(options['role'])
        processes = ('p1',) if options['role'] == 'task_author' else ('p1', 'p2')
        (folder / 'task.json').write_text(json.dumps(author_task(processes), ensure_ascii=False))
    row = world.make_task(certified, tmp_path / 'task', index=0, processes=['p1', 'p2'], script=[],
                          runner=runner, client=Writer(), python=sys.executable)
    assert row['passed'], row
    assert roles == ['task_author', 'task_repair'] and row['task_repairs_used'] == 1
    initial = json.loads((tmp_path / 'task/attempt-00/task_result.json').read_text())
    assert initial['stage'] == 'task' and not initial['passed']
    assert (Path(row['bundle']) / 'env.py').read_bytes() == (certified / 'env.py').read_bytes()


def test_task_cannot_downgrade_requested_difficulty(certified, tmp_path):
    roles = []
    def runner(folder, prompt, **options):
        roles.append(options['role'])
        (folder / 'task.json').write_text(json.dumps(author_task(), ensure_ascii=False))
    row = world.make_task(certified, tmp_path / 'task', index=0, processes=['p1'], script=[], runner=runner,
                          client=Writer(), python=sys.executable, requirements={'min_writes': 99}, repairs=1)
    assert not row['passed'] and row['stage'] == 'task_requirements'
    assert roles == ['task_author', 'task_repair'] and row['requirements'] == {'min_writes': 99}
    assert row['task_quality']['actual']['writes'] < 99


def test_task_author_cannot_modify_the_assignment(certified, tmp_path):
    def runner(folder, prompt, **options):
        (folder / 'assignment.json').write_text('{}')
        (folder / 'task.json').write_text(json.dumps(author_task()))
    row = world.make_task(certified, tmp_path / 'task', index=0, processes=['p1'], script=[], runner=runner,
                          client=Writer(), python=sys.executable)
    assert not row['passed'] and row['stage'] == 'task_author_error' and row['task_attempts'] == 1


def test_required_sufficiency_provider_failure_does_not_admit_task(certified, tmp_path):
    from env_factory.core.llm import ProviderError
    with patch('env_factory.certify.sufficiency.run', side_effect=ProviderError('unavailable')):
        row = world.guard({'passed': True}, certified, certified, tmp_path, sufficiency_client=object())
    assert not row['passed'] and row['stage'] == 'sufficiency_infrastructure'


def test_missing_natural_acceptance_result_is_not_exported(tmp_path):
    rows = [{'id': 'world-00-t000', 'passed': True, 'bundle': str(tmp_path/'a')},
            {'id': 'world-00-t001', 'passed': True, 'bundle': str(tmp_path/'b')}]
    (tmp_path/'tasks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    world.write_tasks(tmp_path, {'world-00': {'id': 'world-00', 'training_ready': True}}, {},
                      {'world-00-t000': 'unsolved'})
    data = json.loads((tmp_path/'tasks.json').read_text())
    assert not data['tasks'] and len(data['excluded']) == 2
    assert 'missing' in data['excluded'][1]['reason']


def test_coverage_never_substitutes_labels_for_measured_complexity():
    rows = [{'id': 'a', 'requested_shape': 'complex', 'passed': False, 'assigned_processes': ['p'],
             'task_quality': {'actual': {'writes': 0, 'state_changed': False}}}]
    report = quality.coverage(rows)
    group = report['by_requested_shape']['complex']
    assert group['attempted'] == group['measured'] == 1
    assert group['certified'] == group['natural_passed'] == group['state_changes'] == 0
    assert group['successful_writes'] == [0] and not report['natural_acceptance_run']
