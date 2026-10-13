"""Reward trust: acceptable alternatives, simulator audit, untrainable episodes, disagreement mining,
policy sufficiency and the third oracle. No model calls."""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_env_factory_v3 import LIBRARY, FakeRunner, library_modules, library_request  # noqa: E402
from test_natural_user import Reader, ScriptedUser  # noqa: E402

from env_factory_v3.authoring import pipeline  # noqa: E402
from env_factory_v3.certify import bundle, sufficiency  # noqa: E402
from env_factory_v3.core import disclosure, reward  # noqa: E402
from env_factory_v3.production import disagreement, world  # noqa: E402
from env_factory_v3.serving import server  # noqa: E402
from rl.data.trajectory import make_samples  # noqa: E402

SCHEMA = {'fields': {'a': {'type': 'integer', 'description': 'a'}}}
INITIAL = {'t': [{'id': 1, 'v': 0}]}


def outcome(a, v):
    return {'answer': {'a': a}, 'final_state': {'t': [{'id': 1, 'v': v}]}}


def result(a, v):
    return {**outcome(a, v), 'safety_violations': 0, 'tool_calls': 1}


class Canned:
    """A chat client that returns fixed contents in turn."""

    model = 'canned'

    def __init__(self, *contents):
        self.contents, self.prompts = list(contents), []

    def chat(self, messages, tools=None, **parameters):
        self.prompts.append(messages[-1]['content'])
        return {'content': self.contents.pop(0) if len(self.contents) > 1 else self.contents[0]}


class AlternativeOutcomeTests(unittest.TestCase):
    def test_the_best_acceptable_outcome_counts(self):
        expected = {**outcome(1, 1), 'alternatives': [outcome(2, 2)]}
        grade = reward.grade(result(2, 2), expected, SCHEMA, initial=INITIAL, reference_calls=1)
        self.assertEqual((grade['success'], grade['reward'], grade['outcome']), (True, 1.0, 1))
        self.assertEqual(reward.grade(result(1, 1), expected, SCHEMA, initial=INITIAL, reference_calls=1)['outcome'], 0)
        wrong = reward.grade(result(3, 3), expected, SCHEMA, initial=INITIAL, reference_calls=1)
        self.assertFalse(wrong['success'])

    def test_alternatives_cannot_repeat_or_do_nothing(self):
        self.assertEqual(reward.alternative_problems({**outcome(1, 1), 'alternatives': [outcome(2, 2)]}, SCHEMA, INITIAL), [])
        problems = reward.alternative_problems({**outcome(1, 1), 'alternatives': [outcome(1, 1), outcome(5, 0)]},
                                               SCHEMA, INITIAL)
        self.assertTrue(any('repeats' in p for p in problems))
        self.assertTrue(any('unchanged' in p for p in problems))
        many = {**outcome(1, 1), 'alternatives': [outcome(i, i) for i in range(2, 7)]}
        self.assertTrue(reward.alternative_problems(many, SCHEMA, INITIAL))


class SimulatorAuditTests(unittest.TestCase):
    def test_faults_must_quote_the_user_and_name_a_fact(self):
        facts = {'reader_id': 'R1'}
        said = '我不记得读者号了'
        self.assertTrue(disclosure.verified({'kind': 'withheld', 'fact': 'reader_id', 'quote': '不记得读者号'}, facts, said))
        self.assertFalse(disclosure.verified({'kind': 'withheld', 'fact': 'reader_id', 'quote': '编造的话'}, facts, said))
        self.assertFalse(disclosure.verified({'kind': 'withheld', 'fact': 'nope', 'quote': '不记得读者号'}, facts, said))
        self.assertTrue(disclosure.verified({'kind': 'invented', 'quote': '不记得读者号'}, facts, said))

    def test_a_failed_episode_with_a_simulator_slip_is_marked_invalid(self):
        from env_factory_v3.core.extract import Extractor
        env, oracle, schema = library_modules()
        fault = {'faults': [{'kind': 'withheld', 'fact': 'reader_id', 'quote': '好的（1）', 'reason': 'x'}]}
        auditor = disclosure.Auditor(Canned(json.dumps(fault, ensure_ascii=False)))
        service = server.Service(env, oracle, schema, library_request(), user_mode='natural', user_client=ScriptedUser(),
                                 extractor=Extractor(Reader()), auditor=auditor, audit=True)
        episode = service.reset(world='original')['episode']
        service.step(episode, {'type': 'chat', 'message': '请问您的读者号？'})
        graded = service.reward(episode)
        self.assertFalse(graded['success'])
        self.assertIs(graded['valid'], False)
        self.assertEqual(graded['simulator_audit']['faults'][0]['fact'], 'reader_id')
        clean = server.Service(env, oracle, schema, library_request(), user_mode='natural', user_client=ScriptedUser(),
                               extractor=Extractor(Reader()), auditor=disclosure.Auditor(Canned('{"faults": []}')),
                               audit=True)
        episode = clean.reset(world='original')['episode']
        clean.step(episode, {'type': 'chat', 'message': '请问您的读者号？'})
        self.assertIs(clean.reward(episode)['valid'], True)

    def test_a_write_without_an_approval_fact_is_declined_not_an_error(self):
        env, oracle, schema = library_modules()
        service = server.Service(env, oracle, schema, library_request(), user_mode='natural', user_client=ScriptedUser(),
                                 audit=False)
        env.approval_key = lambda tool, args, facts: 'approve_another_process'  # e.g. a world's other process
        episode = service.reset(world='original')['episode']
        reply = service.step(episode, {'type': 'chat', 'message': '要不要续借？',
                                       'proposal': {'tool': 'renew_loan', 'args': {'loan_id': 'L1', 'reader_id': 'R1'}}})
        self.assertTrue(reply['ok'], reply)
        self.assertIsNone(reply['result']['consent'])

    def test_untrainable_episodes_get_no_gradient_and_set_no_baseline(self):
        def episode(reward_, trainable=True):
            return {'seed': 1, 'final_reward': reward_, 'trainable': trainable, 'bootstrap': 0.0,
                    'actions': [{'tokens': [1, 2], 'old_logp': [0.0, 0.0], 'reward': reward_, 'terminated': True}]}
        samples = make_samples({'task_id': 't', 'episodes': [episode(1.0), episode(0.0), episode(0.0, False)]}, 'grpo')
        advantages = [s['advantage'] for s in samples]
        self.assertAlmostEqual(advantages[0], -advantages[1])
        self.assertEqual((advantages[2], samples[2]['weight']), (0.0, 0))


class LibraryBundle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        status = pipeline.build(library_request(), cls.root / 'build', runner=FakeRunner(), perturbed=20)
        assert status['training_ready'], status
        cls.bundle = Path(status['bundle'])


class DisagreementTests(LibraryBundle):
    def test_failures_ending_the_same_way_are_a_convergent_cluster(self):
        call = {'id': 'c1', 'type': 'function', 'function': {'name': 'renew_loan', 'arguments': json.dumps(
            {'loan_id': 'L1', 'consent': 'consent-1'})}}
        messages = [{'role': 'system', 'content': 's'}, {'role': 'assistant', 'content': '', 'tool_calls': [call]},
                    {'role': 'tool', 'tool_call_id': 'c1', 'content': json.dumps({'ok': True})}]
        failed = {'world': 'original', 'success': False, 'answer': {'renewed_loan_ids': ['L1']}, 'messages': messages}
        report = {'played': 3, 'successes': 0, 'episodes': [failed, copy.deepcopy(failed),
                                                             {**failed, 'valid': False}]}
        row = disagreement.mine(self.bundle, report)
        self.assertTrue(row['flagged'])
        self.assertEqual((row['failures'], row['skipped_invalid'], row['clusters'][0]['episodes']), (2, 1, 2))
        self.assertEqual(row['clusters'][0]['writes'], [['renew_loan', {'loan_id': 'L1'}]])
        judged = disagreement.mine(self.bundle, {**report, 'episodes': [failed]},
                                   Canned('{"verdict": "policy_gap", "reason": "r", "quote": "q"}'))
        self.assertEqual((judged['flagged'], judged['reasons']), (True, ['policy_gap']))


class SufficiencyTests(LibraryBundle):
    def test_a_prediction_equal_to_the_reference_passes(self):
        loaded = bundle.load_bundle(self.bundle)
        _, writes, answer = sufficiency.reference_view(loaded)
        right = json.dumps({'writes': [{'tool': t, 'args': a} for t, a in writes], 'answer': answer}, ensure_ascii=False)
        self.assertTrue(sufficiency.check(loaded, Canned(right))['passed'])
        wrong = json.dumps({'writes': [], 'answer': {**answer, 'renewed_loan_ids': []}}, ensure_ascii=False)
        report = sufficiency.check(loaded, Canned(wrong))
        self.assertFalse(report['passed'])
        self.assertIn('renewed_loan_ids', report['gaps']['fields'])


class ThirdOracleTests(LibraryBundle):
    def test_a_disagreeing_third_oracle_rejects_the_task(self):
        folder = Path(tempfile.mkdtemp())
        shutil.copy(LIBRARY / 'oracle.py', folder / 'oracle.py')
        self.assertTrue(world.third_opinion(self.bundle, folder / 'oracle.py')['passed'])
        source = (LIBRARY / 'oracle.py').read_text().replace("loan['due_day'] += book['renewal_days']",
                                                              "loan['due_day'] += book['renewal_days'] + 1")
        (folder / 'oracle.py').write_text(source)
        opinion = world.third_opinion(self.bundle, folder / 'oracle.py')
        self.assertFalse(opinion['passed'])
        self.assertTrue(opinion['disagreements'])
        # guard: a world folder layout <world>/build/bundle with <world>/oracle3
        world_dir = Path(tempfile.mkdtemp())
        (world_dir / 'build').mkdir()
        shutil.copytree(self.bundle, world_dir / 'build' / 'bundle')
        (world_dir / 'oracle3').mkdir()
        (world_dir / 'oracle3' / 'oracle.py').write_text(source)
        (world_dir / 'oracle3' / 'third_oracle.json').write_text('{"passed": true}')
        row = world.guard({'id': 't', 'passed': True}, world_dir / 'build' / 'bundle', self.bundle, Path(tempfile.mkdtemp()))
        self.assertEqual((row['passed'], row['stage']), (False, 'third_oracle'))


if __name__ == '__main__':
    unittest.main()
