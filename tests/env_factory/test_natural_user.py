"""Free-dialogue user mode. No model calls: the simulator's client is scripted."""
from __future__ import annotations

import json
import os
import sys
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_core import LIBRARY, library_modules, library_request  # noqa: E402

from env_factory.certify import gates  # noqa: E402
from env_factory.serving import agent_view, server

# The simulator audit calls a judge model; these tests cover the dialogue itself (see test_reward_trust).
_AUDIT_OFF = unittest.mock.patch.dict(os.environ, {'SIMULATOR_AUDIT': 'off'})


def setUpModule():
    _AUDIT_OFF.start()


def tearDownModule():
    _AUDIT_OFF.stop()


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
        self.env, self.verifier, self.schema = library_modules()
        self.request = library_request()
        for hold in self.request['tables']['holds']:
            hold['active'] = False  # both loans renewable, so both approvals matter
        self.user = ScriptedUser()
        self.service = server.Service(self.env, self.verifier, self.schema, self.request,
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
        missing, verifier, schema = library_modules(source.replace('def approval_key(', 'def _approval_key('))
        with self.assertRaises(ValueError):
            server.Service(missing, verifier, schema, self.request, user_mode='natural', user_client=self.user)
        report = gates.qualify(missing, verifier, schema, self.request, perturbed=5)
        self.assertFalse(report['natural']['passed'])
        self.assertFalse(report['passed'])  # an unservable bundle is not certified

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
        env, verifier, schema = library_modules()
        service = server.Service(env, verifier, schema, library_request(), user_mode='natural', user_client=ScriptedUser())
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


class SystemPromptTests(unittest.TestCase):
    def test_the_agent_records_the_result_and_tells_the_user(self):
        env, verifier, schema = library_modules()
        service = server.Service(env, verifier, schema, library_request(), user_mode='natural', user_client=ScriptedUser())
        system = agent_view.messages(service.reset(world='original')['instructions'])[0]['content']
        self.assertIn('记录处理结果', system)
        self.assertIn('把结果告诉用户', system)


class EnglishTests(unittest.TestCase):
    def service(self, user=None):
        env, verifier, schema = library_modules()
        request = {**library_request(), 'language': 'en'}
        return server.Service(env, verifier, schema, request, user_mode='natural', user_client=user or ScriptedUser())

    def test_an_english_task_shows_the_agent_english_platform_texts(self):
        instructions = self.service().reset(world='original')['instructions']
        self.assertEqual(instructions['language'], 'en')
        system = agent_view.messages(instructions, answer_tool=False)[0]['content']
        self.assertTrue(system.startswith('You are a customer service agent'))
        self.assertIn('Policy:', system)
        chat = next(t for t in agent_view.tools(instructions, 'multi_step_agentic') if t['function']['name'] == 'v3_chat')
        self.assertIn('Ask the user to approve', chat['function']['description'])
        self.assertIn('System notice', agent_view.left_notice('en'))

    def test_the_english_user_speaks_english_and_stops_with_its_own_marker(self):
        class Bye(ScriptedUser):
            def chat(self, messages, tools=None, **parameters):
                self.seen.append(messages[0]['content'])
                return {'content': 'Great, thanks. Bye!\n[END]'}
        user = Bye()
        service = self.service(user)
        episode = service.reset(world='original')['episode']
        reply = service.step(episode, {'type': 'chat', 'message': 'Both books are renewed.'})
        self.assertEqual(reply['result']['ended'], 'closed')
        self.assertNotIn('[END]', reply['result']['reply'])
        self.assertTrue(user.seen[0].startswith('You are an ordinary customer'))

    def test_dollar_amounts_count_as_said(self):
        from env_factory.core.numerals import said
        self.assertTrue(said('Your refund is $3.50.', 350))
        self.assertTrue(said('That comes to $1,234.50 in total.', 123450))


class ScriptTests(unittest.TestCase):
    def test_the_user_acts_out_its_script_and_the_platform_ends_the_pushback(self):
        env, verifier, schema = library_modules()
        request = {**library_request(), 'user_script': {
            'traits': ['着急'], 'goals': ['续借', '问候补'], 'forgotten': ['邮箱'],
            'pushback': {'demand': '免掉逾期费', 'rounds': 1, 'then': 'accept'}}}
        user = ScriptedUser()
        service = server.Service(env, verifier, schema, request, user_mode='natural', user_client=user)
        episode = service.reset(world='original')['episode']
        record = service._get(episode)
        system = record['user'].messages[0]['content']
        self.assertIn('免掉逾期费', system)
        self.assertIn('不记得：邮箱', system)
        for message in ('逾期费不能免', '规定如此', '还有别的吗'):
            service.step(episode, {'type': 'chat', 'message': message})
        notices = [m['content'] for m in record['user'].messages if m['role'] == 'user']
        self.assertNotIn('坚持够了', notices[0])
        self.assertIn('坚持够了', notices[1])                    # after the demand plus one insistence
        self.assertEqual(sum('坚持够了' in n for n in notices), 1)  # said once

    def test_the_author_brief_tells_how_the_user_behaves(self):
        from env_factory.core import spec
        request = {**library_request(), 'user_script': {'traits': ['话少'], 'forgotten': ['email'],
                                                        'goals': ['续借', '问候补']}}
        text = spec.brief(request)
        self.assertIn('How the user behaves', text)
        self.assertIn('api.ask raises KeyError', text)


class PlainSpeechTests(unittest.TestCase):
    def test_plain_messages_are_speech_and_the_user_ends_the_episode(self):
        import json as _json
        from env_factory.certify import policy_gate as policy
        env, verifier, schema = library_modules()
        service = server.Service(env, verifier, schema, library_request(), user_mode='natural', user_client=Leaver())
        httpd = server.start(service)

        class Agent:
            model = 'scripted-agent'

            def __init__(self):
                self.turn = 0

            def chat(self, messages, tools=None, **parameters):
                self.turn += 1
                if self.turn == 1:
                    return {'content': '请问您的读者号和书号是多少？'}  # the Leaver stays until results are told
                if self.turn == 2:
                    self.after_question = messages[-1]
                    return {'content': '', 'tool_calls': [{'id': 'a1', 'type': 'function', 'function': {
                        'name': 'v3_answer', 'arguments': _json.dumps({'renewed_loan_ids': [], 'blocked': []})}}]}
                return {'content': '已记录：这次没有可以续借的书。'}

        agent = Agent()
        try:
            result = policy.play(f'http://127.0.0.1:{httpd.server_address[1]}', agent, 'original',
                                 'multi_step_agentic')
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertTrue(result['answered'])
        self.assertEqual(result['turns'], 3)  # speech, record, speech; then the user said goodbye
        self.assertEqual(agent.after_question['role'], 'user')
        self.assertIsNotNone(result['reward'])


class Leaver(ScriptedUser):
    """Says goodbye (with the platform's end marker) once it has been told something."""

    def chat(self, messages, tools=None, **parameters):
        self.seen.append(messages[-1]['content'])
        return {'content': '好的谢谢，我知道了，再见\n[结束]' if '续借' in messages[-1]['content'] else '嗯'}


class Reader:
    """Stands in for the extractor's model: reads nothing it was not told."""

    model = 'scripted-reader'

    def __init__(self, answer=None):
        self.answer, self.prompts = answer, []

    def chat(self, messages, tools=None, **parameters):
        self.prompts.append(messages[-1]['content'])
        return {'content': json.dumps(self.answer or {'renewed_loan_ids': None, 'blocked': None})}


class ConversationEndTests(unittest.TestCase):
    def service(self, user, request=None, **options):
        from env_factory.core.extract import Extractor
        env, verifier, schema = library_modules()
        options.setdefault('extractor', Extractor(Reader()))
        return server.Service(env, verifier, schema, request or library_request(), user_mode='natural',
                              user_client=user, **options)

    def test_idle_talk_is_warned_once_then_closed_without_an_answer(self):
        service = self.service(ScriptedUser())
        episode = service.reset(world='original')['episode']
        replies = [service.step(episode, {'type': 'chat', 'message': f'第{i}句'}) for i in range(1, 5)]
        self.assertEqual([r['result'].get('ended') for r in replies], [None, None, 'warning', 'closed'])
        self.assertFalse(replies[-1]['result']['recorded'])
        graded = service.reward(episode)  # nothing recorded and nothing said: the extracted answer is empty
        self.assertEqual((graded['answer_source'], graded['success']), ('extracted', False))
        refused = service.step(episode, {'type': 'tool', 'name': 'list_loans', 'args': {'reader_id': 'R1'}})
        self.assertEqual(refused['kind'], 'finished')

    def test_an_answer_told_but_not_recorded_is_extracted_from_speech_and_verified(self):
        from env_factory.core.extract import Extractor
        told = {'renewed_loan_ids': ['L1'], 'blocked': [{'loan_id': 'L2', 'active_hold_count': 1}]}
        reader = Reader(told)
        service = self.service(Leaver(), extractor=Extractor(reader))
        episode = service.reset(world='original')['episode']
        service.step(episode, {'type': 'tool', 'name': 'renew_loan', 'args': {'loan_id': 'L1', 'reader_id': 'R1'}})
        closed = service.step(episode, {'type': 'chat', 'message': '续借好了 L1；L2 有 1 人排队，续不了。'})
        self.assertEqual((closed['result']['ended'], closed['result']['recorded']), ('closed', False))
        graded = service.reward(episode)
        self.assertEqual(graded['answer_source'], 'extracted')
        self.assertEqual(graded['answer'], told)                      # every value was actually said
        self.assertNotIn('renew_loan', reader.prompts[0])             # the reader never sees tool calls

    def test_the_reader_is_told_the_last_statement_counts_and_sees_the_whole_conversation(self):
        from env_factory.core.extract import Extractor
        reader = Reader()
        Extractor(reader).extract({'fields': {'renewed_loan_ids': {'type': 'array', 'items': {'type': 'string'},
                                                                   'description': 'x'}}},
                                  ['先说一下：L1 可以续借。', '更正：L1 续不了，L2 续好了。'])
        prompt = reader.prompts[0]
        self.assertIn('以最后一次为准', prompt)
        self.assertLess(prompt.index('先说一下'), prompt.index('更正'))  # every message, in order

    def test_an_extracted_value_that_was_never_said_is_dropped(self):
        from env_factory.core.extract import verify
        spec = {'type': 'object', 'properties': {'loan_id': {'type': 'string'}, 'active_hold_count': {'type': 'integer'}}}
        self.assertEqual(verify({'loan_id': 'L2', 'active_hold_count': 3}, spec, 'L2 有 1 人排队'),
                         {'loan_id': 'L2', 'active_hold_count': None})

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
        answer = {'renewed_loan_ids': ['L1'], 'blocked': []}
        recorded = service.step(episode, {'type': 'answer', 'answer': answer})
        self.assertEqual(recorded['result'], {'recorded': True})  # recording does not end the conversation
        reply = service.step(episode, {'type': 'chat', 'message': '您的两本书都续借好了'})
        self.assertEqual((reply['result']['ended'], reply['result']['recorded']), ('closed', True))
        self.assertNotIn('[结束]', reply['result']['reply'])
        self.assertEqual(service._get(episode)['episode'].answer, answer)

    def test_a_subjective_task_is_graded_on_the_last_thing_said(self):
        from test_rubric import RUBRIC, REPLY, judge
        env, verifier, schema = library_modules((LIBRARY / 'env.py').read_text().replace(*REPLY))
        from env_factory.core import rubric
        request = {**library_request(), 'reward_type': 'rubric'}
        service = server.Service(env, None, rubric.SCHEMA, request, user_mode='natural', user_client=Leaver(),
                                 rubric=RUBRIC, judge=judge())
        episode = service.reset(world='original')['episode']
        closed = service.step(episode, {'type': 'chat', 'message': '续借成功：L1、L2，祝您阅读愉快'})
        self.assertEqual((closed['result']['ended'], closed['result']['recorded']), ('closed', True))
        self.assertEqual(service._get(episode)['episode'].answer, {'reply': '续借成功：L1、L2，祝您阅读愉快'})
        self.assertIn('rubric_score', service.reward(episode))


class ServiceAndClarificationTests(unittest.TestCase):
    def test_a_token_protected_service_refuses_strangers(self):
        import json as _json
        from urllib.error import HTTPError
        from urllib.request import Request, urlopen
        env, verifier, schema = library_modules()
        httpd = server.start(server.Service(env, verifier, schema, library_request()), token='s3cret')
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
        from env_factory.core import spec
        request = spec.clarify(library_request(), ['已注销的借阅不能续借。'])
        self.assertIn('已注销的借阅不能续借。', spec.brief(request))
        self.assertTrue(spec.quoted(request, '已注销的借阅不能续借'))
        report = spec.findings_report({'id': 'x', 'spec_findings': [
            {'issue': 'inactive loans', 'brief_quote': '允许部分成功', 'reason': 'two readings', 'worlds': ['perturb:3']}]})
        self.assertIn('perturb:3', report)


if __name__ == '__main__':
    unittest.main()
