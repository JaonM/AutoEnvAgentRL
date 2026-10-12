"""Free-dialogue user mode. No model calls: the simulator's client is scripted."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_env_factory_v3 import LIBRARY, library_modules, library_request  # noqa: E402

from env_factory_v3.certify import gates  # noqa: E402
from env_factory_v3.serving import agent_view, server


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


class AgentViewFormatTests(unittest.TestCase):
    def instructions(self, writes=True):
        env, oracle, schema = library_modules()
        service = server.Service(env, oracle, schema, library_request(), user_mode='natural', user_client=ScriptedUser())
        instructions = service.reset(world='original')['instructions']
        if not writes:
            instructions = {**instructions, 'tools': [t for t in instructions['tools'] if t['kind'] == 'read']}
        return instructions

    def test_tool_descriptions_are_summaries_without_internal_arguments(self):
        for tool in self.instructions()['tools']:
            self.assertNotIn('Args:', tool['description'])
            self.assertNotIn('db:', tool['description'])

    def test_system_prompt_is_chinese_without_a_second_copy_of_the_answer_format(self):
        system = agent_view.messages(self.instructions())[0]['content']
        self.assertTrue(system.startswith('你是一名业务助理'))
        self.assertIn('业务规则：', system)
        self.assertNotIn('answer_schema', system)
        self.assertIn('v3_chat', system)

    def test_read_only_tasks_get_no_approval_tool(self):
        instructions = self.instructions(writes=False)
        names = [t['function']['name'] for t in agent_view.tools(instructions, 'multi_step_agentic')]
        self.assertNotIn('v3_chat', names)
        self.assertNotIn('v3_chat', agent_view.messages(instructions)[0]['content'])

    def test_renaming_works_inside_chinese_text(self):
        names = agent_view.naming(0, 'heldout')
        system = agent_view.messages(self.instructions(), names)[0]['content']
        self.assertIn('permit_code', system)
        self.assertNotIn('consent', system)
        self.assertNotIn('v3_answer', system)


class PlainSpeechTests(unittest.TestCase):
    def test_plain_messages_are_speech_and_only_v3_answer_ends_the_episode(self):
        import json as _json
        from env_factory_v3.certify import policy_gate as policy
        env, oracle, schema = library_modules()
        service = server.Service(env, oracle, schema, library_request(), user_mode='natural', user_client=ScriptedUser())
        httpd = server.start(service)

        class Agent:
            model = 'scripted-agent'

            def __init__(self):
                self.turn = 0

            def chat(self, messages, tools=None, **parameters):
                self.turn += 1
                if self.turn == 1:
                    return {'content': '请问您的读者号和要续借的书号是多少？'}
                self.last_user = messages[-1]
                return {'content': '', 'tool_calls': [{'id': 'a1', 'type': 'function', 'function': {
                    'name': 'v3_answer', 'arguments': _json.dumps({'renewed_loan_ids': [], 'blocked': []})}}]}

        agent = Agent()
        try:
            result = policy.play(f'http://127.0.0.1:{httpd.server_address[1]}', agent, 'original',
                                 'multi_step_agentic')
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertTrue(result['answered'])
        self.assertEqual(result['turns'], 2)
        self.assertEqual(agent.last_user['role'], 'user')
        self.assertTrue(agent.last_user['content'].startswith('好的'))


class Leaver(ScriptedUser):
    """Says goodbye (with the platform's end marker) once it has been told something."""

    def chat(self, messages, tools=None, **parameters):
        self.seen.append(messages[-1]['content'])
        return {'content': '好的谢谢，我知道了，再见\n[结束]' if '续借' in messages[-1]['content'] else '嗯'}


class ConversationEndTests(unittest.TestCase):
    def service(self, user, request=None, **options):
        env, oracle, schema = library_modules()
        return server.Service(env, oracle, schema, request or library_request(), user_mode='natural',
                              user_client=user, **options)

    def test_idle_talk_is_warned_once_then_closed_without_an_answer(self):
        service = self.service(ScriptedUser())
        episode = service.reset(world='original')['episode']
        replies = [service.step(episode, {'type': 'chat', 'message': f'第{i}句'}) for i in range(1, 5)]
        self.assertEqual([r['result'].get('ended') for r in replies], [None, None, 'warning', 'closed'])
        self.assertFalse(replies[-1]['result']['submitted'])
        refused = service.step(episode, {'type': 'tool', 'name': 'list_loans', 'args': {'reader_id': 'R1'}})
        self.assertEqual(refused['kind'], 'finished')

    def test_a_tool_call_resets_the_idle_count(self):
        service = self.service(ScriptedUser())
        episode = service.reset(world='original')['episode']
        for i in range(2):
            service.step(episode, {'type': 'chat', 'message': f'第{i}句'})
        service.step(episode, {'type': 'tool', 'name': 'list_loans', 'args': {'reader_id': 'R1'}})
        later = [service.step(episode, {'type': 'chat', 'message': 'x'})['result'].get('ended') for _ in range(2)]
        self.assertEqual(later, [None, None])

    def test_the_user_saying_goodbye_ends_and_the_marker_is_never_shown(self):
        service = self.service(Leaver())
        episode = service.reset(world='original')['episode']
        reply = service.step(episode, {'type': 'chat', 'message': '您的两本书都续借好了'})
        self.assertEqual(reply['result']['ended'], 'warning')
        self.assertNotIn('[结束]', reply['result']['reply'])

    def test_a_subjective_task_is_graded_on_the_last_thing_said(self):
        from test_rubric import RUBRIC, REPLY, judge
        env, oracle, schema = library_modules((LIBRARY / 'env.py').read_text().replace(*REPLY))
        from env_factory_v3.core import rubric
        request = {**library_request(), 'reward_type': 'rubric'}
        service = server.Service(env, None, rubric.SCHEMA, request, user_mode='natural', user_client=Leaver(),
                                 rubric=RUBRIC, judge=judge())
        episode = service.reset(world='original')['episode']
        service.step(episode, {'type': 'chat', 'message': '续借成功：L1、L2'})
        closed = service.step(episode, {'type': 'chat', 'message': '续借成功：L1、L2，祝您阅读愉快'})
        self.assertEqual((closed['result']['ended'], closed['result']['submitted']), ('closed', True))
        self.assertEqual(service._get(episode)['episode'].answer, {'reply': '续借成功：L1、L2，祝您阅读愉快'})
        self.assertIn('rubric_score', service.reward(episode))


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
        from env_factory_v3.core import spec
        request = spec.clarify(library_request(), ['已注销的借阅不能续借。'])
        self.assertIn('已注销的借阅不能续借。', spec.brief(request))
        self.assertTrue(spec.quoted(request, '已注销的借阅不能续借'))
        report = spec.findings_report({'id': 'x', 'spec_findings': [
            {'issue': 'inactive loans', 'brief_quote': '允许部分成功', 'reason': 'two readings', 'worlds': ['perturb:3']}]})
        self.assertIn('perturb:3', report)


if __name__ == '__main__':
    unittest.main()
