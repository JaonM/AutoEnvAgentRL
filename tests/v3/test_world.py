"""Domain worlds: world and task checks, overlays, composition and task certification. No model calls."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_env_factory_v3 import FakeRunner, library_request  # noqa: E402
from test_generate import SCENE, Writer  # noqa: E402

from env_factory_v3.authoring import pipeline  # noqa: E402
from env_factory_v3.production import world  # noqa: E402

GOALS = {f'p{i}': {'description': f'process {i}', 'params': {'loan_id': 'str'}} for i in range(6)}


def world_request():
    request = copy.deepcopy(library_request())
    request.pop('user_request')
    request.update(goal_schema=GOALS, user_scene=copy.deepcopy(SCENE))
    request['user_facts']['goals'] = [{'process': 'p0', 'params': {'loan_id': 'L1'}}]
    rows = request['tables']['books']
    request['tables']['books'] = rows + [{**rows[0], 'id': f'BX{i}'} for i in range(200)]
    return request


class WorldCheckTests(unittest.TestCase):
    def test_world_needs_goal_schema_size_and_valid_sample_goals(self):
        folder = Path(tempfile.mkdtemp())
        (folder / 'request.json').write_text(json.dumps(world_request(), ensure_ascii=False))
        self.assertEqual(world.check_world(folder)[0], [])
        small = world_request()
        small['tables']['books'] = small['tables']['books'][:3]
        small['user_facts']['goals'] = [{'process': 'nope', 'params': {}}]
        small['goal_schema'] = {'p0': GOALS['p0']}
        (folder / 'request.json').write_text(json.dumps(small, ensure_ascii=False))
        problems = world.check_world(folder)[0]
        self.assertTrue(any('goal_schema' in p for p in problems))
        self.assertTrue(any('rows' in p for p in problems))

    def test_overlays_only_touch_existing_rows_and_columns(self):
        tables = world_request()['tables']
        changed, problems = world.apply_overlay(tables, [{'table': 'loans', 'id': 'L1', 'set': {'due_day': 99}}])
        self.assertEqual(problems, [])
        self.assertEqual(next(r for r in changed['loans'] if r['id'] == 'L1')['due_day'], 99)
        self.assertNotEqual(next(r for r in tables['loans'] if r['id'] == 'L1')['due_day'], 99)  # world untouched
        _, problems = world.apply_overlay(tables, [{'table': 'loans', 'id': 'L1', 'set': {'colour': 'red'}},
                                                   {'table': 'loans', 'id': 'nope', 'set': {}}])
        self.assertEqual(len(problems), 2)

    def test_task_check_enforces_assigned_process_and_script(self):
        folder = Path(tempfile.mkdtemp())
        base = world_request()
        (folder / 'world.json').write_text(json.dumps(base, ensure_ascii=False))
        (folder / 'assignment.json').write_text(json.dumps({'id': 'w-t000', 'processes': ['p1'], 'script': ['multi_goal']}))
        task = {'title': 't', 'user_facts': base['user_facts'], 'user_scene': SCENE,
                'user_script': {'traits': ['话少']}}
        (folder / 'task.json').write_text(json.dumps(task, ensure_ascii=False))
        problems = world.check_task(folder)[0]
        self.assertTrue(any('assigned process' in p for p in problems))
        self.assertTrue(any('multi_goal' in p for p in problems))
        task['user_facts'] = {**base['user_facts'], 'goals': [{'process': 'p1', 'params': {'loan_id': 'L1'}}]}
        task['user_script']['goals'] = ['续借', '问候补']
        (folder / 'task.json').write_text(json.dumps(task, ensure_ascii=False))
        self.assertEqual(world.check_task(folder)[0], [])


class TaskCertificationTests(unittest.TestCase):
    def test_a_task_is_certified_on_a_copy_of_the_world_bundle(self):
        root = Path(tempfile.mkdtemp())
        status = pipeline.build({**world_request(), 'user_request': '续借一下'}, root / 'world', runner=FakeRunner(),
                                perturbed=20)
        self.assertTrue(status['training_ready'], status)

        def task_author(folder, prompt, *, timeout, role):
            base = json.loads((Path(folder) / 'world.json').read_text())
            facts = {**world_request()['user_facts'], 'goals': [{'process': 'p1', 'params': {'loan_id': 'L1'}}]}
            (Path(folder) / 'task.json').write_text(json.dumps({
                'title': '续借任务', 'user_facts': facts, 'user_scene': SCENE, 'user_script': {'traits': ['话少']},
                'overlay': [{'table': 'books', 'id': base['tables']['books'][0]['id'], 'set': {}}]},
                ensure_ascii=False))
            return {'role': role, 'status': 'exited'}

        row = world.make_task(status['bundle'], root / 'tasks' / 't000', index=0, processes=['p1'], script=[],
                              runner=task_author, client=Writer(), python=sys.executable)
        self.assertTrue(row['passed'], row)
        task = json.loads((Path(row['bundle']) / 'request.json').read_text())
        self.assertEqual(task['user_facts']['goals'][0]['process'], 'p1')
        self.assertTrue(task['user_request'])
        world_env = (Path(status['bundle']) / 'env.py').read_text()
        self.assertEqual((Path(row['bundle']) / 'env.py').read_text(), world_env)  # the world's certified env

    def test_task_plan_covers_every_process(self):
        plan = world.plan_tasks({'id': 'w', 'goal_schema': GOALS}, 12)
        self.assertEqual({p for processes, _ in plan for p in processes}, set(GOALS))
        self.assertTrue(all(2 <= len([e for e in script if e != 'identity']) <= 4 for _, script in plan))


if __name__ == '__main__':
    unittest.main()
