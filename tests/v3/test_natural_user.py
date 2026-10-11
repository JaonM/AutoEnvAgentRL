"""Free-dialogue user mode. No model calls: the simulator's client is scripted."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_env_factory_v3 import LIBRARY, library_modules, library_request  # noqa: E402

from env_factory_v3 import agent_view, gates, server  # noqa: E402


class ScriptedUser:
    """Stands in for the simulator's model: echoes, and records what the platform told it."""

    model = 'scripted-user'

    def __init__(self):
        self.seen = []

    def chat(self, messages, tools=None, **parameters):
        self.seen.append(messages[-1]['content'])
        return {'content': f'好的（{len(self.seen)}）'}


class NaturalUserTests(unittest.TestCase):
    def setUp(self):
        self.env, self.oracle, self.schema = library_modules()
        self.request = library_request()
        for hold in self.request['tables']['holds']:
            hold['active'] = False  # both loans renewable, so both approvals matter
        self.user = ScriptedUser()
        self.service = server.Service(self.env, self.oracle, self.schema, self.request,
                                      valid_worlds=['original', 'flip:approve_first'], user_mode='natural',
                                      user_client=self.user)

    def test_agent_sees_no_topic_keys_only_a_chat_channel(self):
        instructions = self.service.reset(world='original')['instructions']
        self.assertNotIn('user_topics', instructions)
        names = [t['function']['name'] for t in agent_view.tools(instructions, self.request['category'])]
        self.assertIn('v3_chat', names)
        self.assertNotIn('v3_ask', names)
        self.assertNotIn('v3_confirm', names)
        opening = agent_view.messages(instructions)
        self.assertEqual(opening[1], {'role': 'user', 'content': self.request['user_request']})
        self.assertNotIn('approve_first', str(opening))

    def test_consent_is_decided_by_the_platform_and_binds_the_exact_write(self):
        episode = self.service.reset(world='original')['episode']
        reply = self.service.step(episode, {'type': 'chat', 'message': '请问您的读者号和要续借的书？'})
        self.assertTrue(reply['ok'])
        self.assertNotIn('consent', reply['result'])
        proposal = {'tool': 'renew_loan', 'args': {'loan_id': 'L1', 'reader_id': 'R1'}}
        reply = self.service.step(episode, {'type': 'chat', 'message': '帮您续借 L1，可以吗？', 'proposal': proposal})
        token = reply['result']['consent']
        self.assertTrue(token)
        self.assertIn('同意', self.user.seen[-1])
        done = self.service.step(episode, {'type': 'tool', 'name': 'renew_loan',
                                           'args': {'loan_id': 'L1', 'reader_id': 'R1', 'consent': token}})
        self.assertTrue(done['ok'], done)

    def test_a_refused_proposal_gets_no_token(self):
        episode = self.service.reset(world='flip:approve_first')['episode']
        proposal = {'tool': 'renew_loan', 'args': {'loan_id': 'L1', 'reader_id': 'R1'}}
        reply = self.service.step(episode, {'type': 'chat', 'message': '续借 L1？', 'proposal': proposal})
        self.assertIsNone(reply['result']['consent'])
        self.assertIn('不同意', self.user.seen[-1])

    def test_structured_actions_are_refused_and_bad_proposals_are_observations(self):
        episode = self.service.reset(world='original')['episode']
        self.assertEqual(self.service.step(episode, {'type': 'ask', 'key': 'reader_id'})['kind'], 'invalid')
        bad = self.service.step(episode, {'type': 'chat', 'message': 'x',
                                          'proposal': {'tool': 'get_waitlist', 'args': {'book_id': 'B1'}}})
        self.assertEqual(bad['kind'], 'invalid')

    def test_natural_mode_refuses_a_bundle_without_a_faithful_approval_key(self):
        source = (LIBRARY / 'env.py').read_text()
        missing, oracle, schema = library_modules(source.replace('def approval_key(', 'def _approval_key('))
        with self.assertRaises(ValueError):
            server.Service(missing, oracle, schema, self.request, user_mode='natural', user_client=self.user)
        report = gates.qualify(missing, oracle, schema, self.request, perturbed=5)
        self.assertTrue(report['passed'])
        self.assertFalse(report['natural']['passed'])

    def test_approval_key_must_match_the_reference(self):
        self.assertEqual(gates.approval_problems(self.env, self.request), [])
        source = (LIBRARY / 'env.py').read_text()
        missing, _, _ = library_modules(source.replace('def approval_key(', 'def _approval_key('))
        self.assertTrue(gates.approval_problems(missing, self.request))
        swapped, _, _ = library_modules(source.replace("('approve_first', 'approve_second')[facts",
                                                       "('approve_second', 'approve_first')[facts"))
        self.assertTrue(gates.approval_problems(swapped, self.request))


class ServiceAndClarificationTests(unittest.TestCase):
    def test_a_token_protected_service_refuses_strangers(self):
        import json as _json
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        env, oracle, schema = library_modules()
        httpd = server.start(server.Service(env, oracle, schema, library_request()), token='s3cret')
        base = f'http://127.0.0.1:{httpd.server_address[1]}'
        try:
            post = lambda headers: urlopen(Request(base + '/reset', data=b'{}', headers=headers), timeout=10)
            with self.assertRaises(HTTPError) as refused:
                post({'Content-Type': 'application/json'})
            self.assertEqual(refused.exception.code, 401)
            with post({'Content-Type': 'application/json', 'Authorization': 'Bearer s3cret'}) as response:
                self.assertIn('episode', _json.load(response))
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_owner_clarifications_become_quotable_rules(self):
        from env_factory_v3 import spec
        request = spec.clarify(library_request(), ['已注销的借阅不能续借。'])
        self.assertIn('已注销的借阅不能续借。', spec.brief(request))
        self.assertTrue(spec.quoted(request, '已注销的借阅不能续借'))
        report = spec.findings_report({'id': 'x', 'spec_findings': [
            {'issue': 'inactive loans', 'brief_quote': '允许部分成功', 'reason': 'two readings', 'worlds': ['perturb:3']}]})
        self.assertIn('perturb:3', report)


if __name__ == '__main__':
    unittest.main()
