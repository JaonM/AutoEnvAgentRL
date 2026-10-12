"""End-to-end run command. No model calls: generation, build and policy models are scripted."""
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

from env_factory_v3 import run  # noqa: E402
from rl.data.tasks import load_tasks  # noqa: E402

DOMAINS = [{'name': 'clinic appointments', 'detail': '门诊预约'}, {'name': 'car rental', 'detail': '租车'}]


class LineRunner(FakeRunner):
    """Request authors write the library task (with a scene, different titles); builders use fixtures."""

    def __call__(self, folder, prompt, *, timeout, role):
        if role == 'request_author':
            self.roles.append(role)
            assignment = json.loads((Path(folder) / 'assignment.json').read_text())
            request = copy.deepcopy(library_request())
            request.pop('user_request')
            index = int(assignment['id'].split('-')[1])
            request.update(category='multi_step_agentic', title=f'续借任务{index}', user_scene=copy.deepcopy(SCENE),
                           business_rules=request['business_rules'] + f'（编号{index}的门店：' + '甲乙丙丁戊' * index + '）')
            (Path(folder) / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
            return {'role': role, 'status': 'exited'}
        return super().__call__(folder, prompt, timeout=timeout, role=role)


class VariedWriter(Writer):
    """Distinct openings per task, as real scenes give."""

    def chat(self, messages, tools=None, **parameters):
        self.prompts.append(messages[-1]['content'])
        lines = ['两本书还有3天到期，能续的都续上吧。', '我在租车门店附近，想问下押金什么时候能退回来，订单早就结束了。']
        return {'content': lines[(len(self.prompts) - 1) % len(lines)]}


def options(**extra):
    base = dict(count=2, domains_file=None, split=None, rubric_share=0.0, parallel=1, repairs=2, docker=False,
                policy_k=0, policy_user_mode='structured', eval_share=0.0, runner=LineRunner(),
                opening_client=Writer(), min_reference_calls={})  # the 4-call fixture is below the real floor
    base.update(extra)
    return base


class RunTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.domains = self.root / 'domains.json'
        self.domains.write_text(json.dumps(DOMAINS))

    def plan_complex_only(self):
        # Two slots of a 2-task plan are direct/simple; make every slot complex so the fixture fits.
        from env_factory_v3 import generate
        original = generate.plan
        generate.plan = lambda domains, count, rubric_share=0.0: [
            (domains[i % len(domains)], 'multi_step_agentic', 'complex') for i in range(count)]
        self.addCleanup(setattr, generate, 'plan', original)
        similar, generate.SIMILAR = generate.SIMILAR, 1.01  # the fixture tasks are near-identical on purpose
        self.addCleanup(setattr, generate, 'SIMILAR', similar)

    def test_end_to_end_manifest_is_rl_loadable_and_resumes(self):
        self.plan_complex_only()
        opts = options(domains_file=str(self.domains), eval_share=0.5, opening_client=VariedWriter())
        code, value = run.run(self.root / 'run', **opts)
        self.assertEqual(code, 0)
        self.assertEqual(value['summary']['trainable'], 2)
        self.assertEqual({t['split'] for t in value['tasks']}, {'train', 'eval'})
        self.assertTrue(all(set(t) == {'id', 'sandbox', 'split', 'weight'} for t in value['tasks']))
        tasks = load_tasks(manifest=str(self.root / 'run' / 'tasks.json'))
        self.assertEqual(len(tasks), 2)
        self.assertTrue((self.root / 'run' / 'report.md').exists())
        detail = value['details'][value['tasks'][0]['id']]
        self.assertEqual(detail['user_modes'], ['structured', 'natural'])
        built = len(opts['runner'].roles)
        code, again = run.run(self.root / 'run', **opts)
        self.assertEqual(len(opts['runner'].roles), built)  # nothing regenerated or rebuilt
        self.assertEqual(again['summary'], value['summary'])

    def test_eval_split_that_rl_rejects_is_folded_into_train(self):
        self.plan_complex_only()
        code, value = run.run(self.root / 'run', **options(domains_file=str(self.domains), eval_share=0.5))
        self.assertEqual(code, 0)  # identical openings: rl's near-duplicate check rejects any eval split
        self.assertEqual({t['split'] for t in value['tasks']}, {'train'})
        self.assertIn('eval_split_dropped', value)
        load_tasks(manifest=str(self.root / 'run' / 'tasks.json'))

    def test_unsolved_policy_gate_moves_tasks_to_excluded(self):
        self.plan_complex_only()

        class Silent:
            model = 'silent'

            def chat(self, messages, tools=None, **parameters):
                return {'content': '{}'}

        code, value = run.run(self.root / 'run', **options(domains_file=str(self.domains), policy_k=2,
                                                           policy_client=Silent()))
        self.assertEqual(code, 1)
        self.assertEqual(value['tasks'], [])
        self.assertEqual(len(value['excluded']), 2)
        self.assertTrue(all('policy gate' in row['reason'] for row in value['excluded']))

    def test_tasks_below_the_difficulty_floor_get_no_image_or_policy_gate(self):
        self.plan_complex_only()
        played = []

        class Recorder:
            model = 'recorder'

            def chat(self, messages, tools=None, **parameters):
                played.append(1)
                return {'content': '{}'}

        opts = options(domains_file=str(self.domains), min_reference_calls={'complex': 6}, policy_k=2,
                       policy_client=Recorder())
        code, value = run.run(self.root / 'run', **opts)
        self.assertEqual((code, value['tasks'], played), (1, [], []))
        self.assertTrue(all(row['failure_class'] == 'too_easy' and 'below the floor of 6' in row['reason']
                            for row in value['excluded']))
        self.assertEqual({d['reference_calls'] for d in value['details'].values()}, {4})

    def test_generation_rejections_are_listed_as_excluded(self):
        code, value = run.run(self.root / 'run', **options(domains_file=str(self.domains)))
        rejected = [row for row in value['excluded'] if row['failure_class'] == 'request']
        self.assertTrue(rejected)  # the 2-task plan asks for direct/simple tasks the fixture cannot satisfy
        self.assertTrue(all(row['reason'].startswith('generation rejected') for row in rejected))


if __name__ == '__main__':
    unittest.main()
