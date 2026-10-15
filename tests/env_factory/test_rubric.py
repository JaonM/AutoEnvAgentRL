"""Rubric reward for subjective tasks. No model calls: the judge's client is scripted."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_core import LIBRARY, library_modules, library_request  # noqa: E402

from env_factory.authoring import pipeline
from env_factory.certify import bundle, gates
from env_factory.core import rubric  # noqa: E402
from env_factory.production import generate
from env_factory.runtime import run_policy, run_reference  # noqa: E402

REPLY = ("    return {'renewed_loan_ids': renewed, 'blocked': blocked}",
         "    return {'reply': f\"续借成功：{'、'.join(renewed) or '无'}；不能续借：\"\n"
         "            f\"{'、'.join(b['loan_id'] + '（' + str(b['active_hold_count']) + '人排队）' for b in blocked) or '无'}\"}")
RUBRIC = {'criteria': [
    {'id': 'reports_renewals', 'weight': 3, 'basis': '成功续借更新到期日与续借次数',
     'description': 'Tells the user which loans were renewed.',
     'levels': {'0': 'does not say', '1': 'partly', '2': 'names every renewed loan'}},
    {'id': 'explains_blocked', 'weight': 2, 'basis': '对续不了的书报告同书目有效预约人数',
     'description': 'For loans that could not be renewed, gives the number of active holds.',
     'levels': {'0': 'omitted', '1': 'vague', '2': 'states the count per loan'}},
    {'id': 'own_loans_only', 'weight': 1, 'basis': '仅本人有效借阅可续借',
     'description': "Only acts on the user's own active loans.",
     'levels': {'0': 'acts on others', '1': 'unclear', '2': 'own loans only'}}]}


def rubric_request():
    return {**library_request(), 'reward_type': 'rubric'}


class ScriptedJudge:
    """Full marks when the reply talks about renewals and quotes no shifted digit (see _fabricate)."""

    model = 'scripted-judge'

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, **parameters):
        self.calls += 1
        prompt = messages[-1]['content']
        reply = prompt.split('助理最后对用户说的话：\n', 1)[1].split('\n\n注意', 1)[0]
        ids = [line.split('：', 1)[0].lstrip('- ').split('（')[0] for line in prompt.splitlines()
               if line.startswith('- ') and '（权重' in line]
        level = 2 if '续借' in reply and not any(c in reply for c in '456789') else 0
        return {'content': json.dumps({'criteria': [{'id': i, 'level': level, 'reason': 'scripted'} for i in ids]})}


def judge():
    return rubric.Judge(ScriptedJudge())


class RubricUnitTests(unittest.TestCase):
    def test_rubric_check(self):
        request = rubric_request()
        self.assertEqual(rubric.check_rubric(RUBRIC, request), [])
        bad = json.loads(json.dumps(RUBRIC))
        bad['criteria'][0]['basis'] = 'not in the rules'
        bad['criteria'][1]['id'] = 'plain_language'
        problems = rubric.check_rubric(bad, request)
        self.assertTrue(any('verbatim' in p for p in problems))
        self.assertTrue(any('plain_language' in p for p in problems))

    def test_grade_is_zero_on_a_safety_violation_and_pays_efficiency_only_on_success(self):
        env, _, _ = library_modules((LIBRARY / 'env.py').read_text().replace(*REPLY))
        request = rubric_request()
        result = run_reference(env, request['tables'], request['user_facts'], start=10)
        good = rubric.grade(result, RUBRIC, request, request['user_facts'], judge(), reference_calls=result['tool_calls'])
        self.assertTrue(good['success'])
        self.assertAlmostEqual(good['reward'], 1.0)
        generic = rubric.grade({**result, 'answer': {'reply': gates.GENERIC_REPLY}}, RUBRIC, request,
                               request['user_facts'], judge(), reference_calls=result['tool_calls'])
        self.assertEqual((generic['success'], generic['reward']), (False, 0.0))
        unsafe = rubric.grade({**result, 'safety_violations': 1}, RUBRIC, request, request['user_facts'], judge(),
                              reference_calls=result['tool_calls'])
        self.assertEqual(unsafe['reward'], 0.0)

    def test_doing_nothing_never_scores(self):
        env, _, _ = library_modules((LIBRARY / 'env.py').read_text().replace(*REPLY))
        request = rubric_request()
        result = run_reference(env, request['tables'], request['user_facts'], start=10)

        class ProcessOnly(ScriptedJudge):
            """Full marks for the conversation-only criterion and grounded, nothing else."""

            def chat(self, messages, tools=None, **parameters):
                self.calls += 1
                ids = [line.split('：', 1)[0].lstrip('- ').split('（')[0] for line in messages[-1]['content'].splitlines()
                       if line.startswith('- ') and '（权重' in line]
                return {'content': json.dumps({'criteria': [
                    {'id': i, 'level': 2 if i in ('own_loans_only', 'grounded') else 0, 'reason': 'x'} for i in ids]})}

        client = ProcessOnly()
        vacuous = rubric.grade({**result, 'answer': {'reply': '好的'}}, RUBRIC, request, request['user_facts'],
                               rubric.Judge(client), reference_calls=result['tool_calls'])
        self.assertEqual(vacuous['reward'], 0.0)
        silent = rubric.grade({**result, 'answer': {'reply': '  '}}, RUBRIC, request, request['user_facts'],
                              rubric.Judge(client), reference_calls=result['tool_calls'])
        self.assertEqual((silent['reward'], client.calls), (0.0, 1))  # an empty reply is not even judged

    def test_qualification_probes_discriminate(self):
        env, _, _ = library_modules((LIBRARY / 'env.py').read_text().replace(*REPLY))
        report = gates.qualify_rubric(env, RUBRIC, rubric.SCHEMA, rubric_request(), judge=judge())
        self.assertTrue(report['passed'], report)
        self.assertEqual(report['gate_names'], list(gates.RUBRIC_GATES))
        self.assertLessEqual(report['rubric']['probe_scores']['fabricated'], 0.0)
        self.assertEqual(report['rubric']['probe_scores']['null_policy_reward'], 0.0)
        self.assertIn('flip:approve_first', report['valid_worlds'])

    def test_a_reply_without_the_facts_fails_the_rubric_gate(self):
        lazy = (LIBRARY / 'env.py').read_text().replace(REPLY[0], "    return {'reply': '好的'}")
        env, _, _ = library_modules(lazy)
        report = gates.qualify_rubric(env, RUBRIC, rubric.SCHEMA, rubric_request(), judge=judge())
        self.assertFalse(report['rubric']['passed'])

    def test_generator_plans_subjective_tasks(self):
        shapes = [s for _, _, s in generate.plan([{'name': 'x'}], 30, rubric_share=0.2)]
        self.assertEqual(shapes.count('subjective'), 6)
        self.assertEqual(shapes.count('direct'), 3)


class RubricRunner:
    def __init__(self):
        self.roles = []

    def __call__(self, folder, prompt, *, timeout, role):
        self.roles.append(role)
        folder = Path(folder)
        if role == 'author':
            (folder / 'env.py').write_text((LIBRARY / 'env.py').read_text().replace(*REPLY))
        elif role == 'rubric_author':
            (folder / 'rubric.json').write_text(json.dumps(RUBRIC, ensure_ascii=False))
        return {'role': role, 'status': 'exited', 'seconds': 0.0, 'usage': {}, 'attempts': 1}


class RubricPipelineTests(unittest.TestCase):
    def test_build_serve_and_check_offline(self):
        runner, shared = RubricRunner(), judge()
        root = Path(tempfile.mkdtemp())
        status = pipeline.build(rubric_request(), root / 'build', runner=runner, judge=shared)
        self.assertTrue(status['training_ready'], status)
        self.assertNotIn('schema', runner.roles)
        self.assertNotIn('verifier', runner.roles)
        folder = Path(status['bundle'])
        self.assertTrue((folder / 'rubric.json').exists())
        self.assertFalse((folder / 'verifier.py').exists())
        from rl.data.tasks import load_tasks  # rl trains rubric bundles too
        manifest = root / 'tasks.json'
        manifest.write_text(json.dumps({'tasks': [{'id': 't', 'sandbox': str(folder), 'split': 'train'}]}))
        self.assertEqual([t.id for t in load_tasks(manifest=str(manifest))], ['t'])
        loaded = bundle.load_bundle(folder)
        self.assertIsNone(loaded['verifier'])
        offline = bundle.qualify_loaded(loaded, judge=None)
        self.assertTrue(offline['passed'])
        service = bundle.service(loaded, judge=shared)
        episode = service.reset(world='original')['episode']
        record = service._get(episode)
        record['episode'].submit({'reply': '续借成功：L1'})
        graded = service.reward(episode)
        self.assertEqual(graded['reward_type'], 'rubric')
        self.assertIn('rubric_score', graded)


if __name__ == '__main__':
    unittest.main()
