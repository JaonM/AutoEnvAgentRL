"""Reward trust: acceptable alternatives, simulator audit, untrainable episodes, disagreement mining,
policy sufficiency. No model calls."""
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


class AnswerComparisonTests(unittest.TestCase):
    OBJ = {'fields': {'refund': {'type': 'object', 'description': 'r', 'properties': {
        'return_id': {'type': 'string'}, 'amount_cents': {'type': 'integer'},
        'destination': {'type': 'string'}, 'status': {'type': 'string', 'enum': ['done', 'refused']}}}}}

    def test_typography_does_not_matter(self):
        from env_factory_v3.core import answers
        schema = {'fields': {'card': {'type': 'string', 'description': 'c'}}}
        self.assertTrue(answers.compare(schema, {'card': '尾号6628的信用卡'}, {'card': '尾号 6628 的信用卡'})['card'])
        self.assertTrue(answers.compare(schema, {'card': 'Visa ending 4242'}, {'card': 'visa  ending 4242'})['card'])

    def test_one_wrong_sub_field_costs_a_share_not_the_field(self):
        from env_factory_v3.core import answers
        want = {'refund': {'return_id': 'R-77', 'amount_cents': 4300, 'destination': 'Visa 4821', 'status': 'done'}}
        got = {'refund': {**want['refund'], 'amount_cents': 4000}}
        self.assertAlmostEqual(answers.scores(self.OBJ, want, got)['refund'], 2 / 3)  # the status earns no share
        self.assertFalse(answers.compare(self.OBJ, want, got)['refund'])

    def test_a_status_or_a_null_alone_earns_nothing(self):
        from env_factory_v3.core import answers
        want = {'refund': {'return_id': 'R-77', 'amount_cents': 4300, 'destination': None, 'status': 'done'}}
        only_status = {'refund': {'return_id': None, 'amount_cents': None, 'destination': None, 'status': 'done'}}
        self.assertEqual(answers.scores(self.OBJ, want, only_status)['refund'], 0.0)
        wrong_status = {'refund': {**want['refund'], 'status': 'refused'}}
        self.assertEqual(answers.scores(self.OBJ, want, wrong_status)['refund'], 0.5)

    def test_amount_strings_compare_as_numbers(self):
        from env_factory_v3.core import answers
        schema = {'fields': {'total': {'type': 'string', 'description': 't'}}}
        for said in ('12.5', '$12.50', '12.50'):
            self.assertTrue(answers.compare(schema, {'total': '12.50'}, {'total': said})['total'], said)
        self.assertFalse(answers.compare(schema, {'total': '12.50'}, {'total': '12.51'})['total'])

    def test_naming_another_record_is_wrong_even_for_an_internal_id(self):
        from env_factory_v3.core import answers
        schema = {'fields': {'plan': {'type': 'string', 'description': 'p'}}}
        known = ['I want the cheapest plan']
        self.assertTrue(answers.compare(schema, {'plan': 'plan-basic'}, {'plan': None}, known)['plan'])
        self.assertFalse(answers.compare(schema, {'plan': 'plan-basic'}, {'plan': 'plan-pro'}, known)['plan'])

    def test_another_name_of_the_expected_record_is_right(self):
        from env_factory_v3.core import answers
        tables = {'gift_cards': [{'id': 'gc-new', 'code': 'GIFT-88', 'status': 'active'},
                                 {'id': 'gc-old', 'code': 'GIFT-12', 'status': 'active'}]}
        aliases = answers.aliases_of(tables)
        spec = {'type': 'string'}
        self.assertEqual(answers.score(spec, 'gc-new', 'GIFT-88', ['x'], aliases), 1.0)  # the card's own number
        self.assertEqual(answers.score(spec, 'gc-new', 'GIFT-12', ['x'], aliases), 0.0)  # another card
        self.assertNotIn('active', aliases)  # a value shared by several rows names none of them

    def test_a_named_record_counts_as_said(self):
        from env_factory_v3.core import extract
        trace = [{'kind': 'tool', 'status': 'ok', 'result': [{'id': 'plan-basic', 'name': 'Basic Saver'},
                                                             {'id': 'plan-pro', 'name': 'Pro Unlimited'}]}]
        records = extract.records_of(trace)
        spec = {'type': 'string'}
        self.assertEqual(extract.verify('plan-basic', spec, 'I recommend the Basic Saver plan.', records), 'plan-basic')
        self.assertIsNone(extract.verify('plan-pro', spec, 'I recommend the Basic Saver plan.', records))

    def test_internal_ids_the_user_never_saw_are_not_graded(self):
        from env_factory_v3.core import answers
        want = {'refund': {'return_id': 'return-ord-102', 'amount_cents': 4300, 'destination': 'Visa 4821',
                           'status': 'done'}}
        got = {'refund': {**want['refund'], 'return_id': None}}
        self.assertFalse(answers.compare(self.OBJ, want, got)['refund'])  # strict without `known`
        self.assertTrue(answers.compare(self.OBJ, want, got, known=['I want to return my jacket'])['refund'])
        # an id the user gave in the opening is graded
        self.assertFalse(answers.compare(self.OBJ, want, got, known=['return return-ord-102 please'])['refund'])
        grade = reward.grade({'answer': got, 'final_state': INITIAL, 'safety_violations': 0, 'tool_calls': 1},
                             {'answer': want, 'final_state': INITIAL}, self.OBJ, initial=INITIAL, reference_calls=1,
                             known=['I want to return my jacket'])
        self.assertTrue(grade['success'], grade)


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


class TaskScaleTests(unittest.TestCase):
    def test_task_scale_uses_recent_rewards_of_the_task(self):
        from rl.data.trajectory import TaskScale
        tracker = TaskScale(window=64, minimum=4)
        group = lambda rewards: {'task_id': 't', 'episodes': [{'final_reward': r} for r in rewards]}
        self.assertIsNone(tracker(group([1.0, 0.0])))  # too little history: the group's own spread
        scale = tracker(group([1.0, 0.0]))
        self.assertAlmostEqual(scale, 0.5)
        samples = make_samples({'task_id': 't', 'episodes': [
            {'seed': 1, 'final_reward': r, 'bootstrap': 0.0,
             'actions': [{'tokens': [1], 'old_logp': [0.0], 'reward': r, 'terminated': True}]} for r in (1.0, 0.5)]},
            'grpo', scale=0.5)
        self.assertAlmostEqual(samples[0]['advantage'], 0.5)


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


class FuzzTests(unittest.TestCase):
    def test_correct_tools_pass_and_crashes_and_hidden_state_are_found(self):
        from env_factory_v3.certify import fuzz
        env, _, _ = library_modules()
        request = library_request()
        self.assertTrue(fuzz.fuzz(env, request['tables'], 10, calls=15, walks=5)['passed'])

        class Broken:
            def make_tools(self):
                seen = []

                def lookup(db, now, loan_id: str):
                    """Look up a loan.

                    Args:
                        loan_id: the loan
                    """
                    seen.append(loan_id)  # call history outside the tables
                    if len(seen) > 1:
                        raise ValueError('boom')
                    return {'loan_id': loan_id}
                return {'lookup': lookup}
        report = fuzz.fuzz(Broken(), request['tables'], 10, calls=5, walks=2)
        self.assertFalse(report['passed'])
        self.assertTrue(report['tools']['lookup']['hidden_state'] or report['tools']['lookup']['crash'])


if __name__ == '__main__':
    unittest.main()


class PlainConsentTests(unittest.TestCase):
    def service(self):
        env, oracle, schema = library_modules()
        request = library_request()
        for hold in request['tables']['holds']:
            hold['active'] = False  # both loans renewable
        return server.Service(env, oracle, schema, request, user_mode='natural', user_client=ScriptedUser(),
                              audit=False, valid_worlds=['original', 'flip:approve_second']), request

    def test_plain_consent_has_no_token_and_no_proposal_channel(self):
        from env_factory_v3.serving import agent_view
        service, request = self.service()
        reset = service.reset(world='original', consent='plain')
        write = next(t for t in reset['instructions']['tools'] if t['kind'] == 'write')
        self.assertNotIn('consent', write['parameters']['properties'])
        names = [t['function']['name'] for t in agent_view.tools(reset['instructions'], request['category'], answer_tool=False)]
        self.assertNotIn('v3_chat', names)
        system = agent_view.messages(reset['instructions'], answer_tool=False)[0]['content']
        self.assertNotIn('consent', system)
        self.assertIn('同意', service._get(reset['episode'])['user'].messages[0]['content'])

    def test_a_write_runs_only_when_the_user_agreed(self):
        service, request = self.service()
        facts = request['user_facts']
        episode = service.reset(world='flip:approve_second', consent='plain')['episode']
        record = service._get(episode)
        first, second = record['facts']['loan_ids'][:2]
        reader = record['facts']['reader_id']
        ok = service.step(episode, {'type': 'tool', 'name': 'renew_loan', 'args': {'loan_id': first, 'reader_id': reader}})
        self.assertTrue(ok['ok'], ok)
        refused = service.step(episode, {'type': 'tool', 'name': 'renew_loan', 'args': {'loan_id': second, 'reader_id': reader}})
        self.assertEqual(refused['kind'], 'business')
        self.assertEqual(record['episode'].safety_violations, 1)
        self.assertNotEqual(facts.get('approve_second'), record['facts'].get('approve_second'))  # the flipped world
