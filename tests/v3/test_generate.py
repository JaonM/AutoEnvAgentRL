"""Task generation front end. No model calls: request author sessions are scripted."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_env_factory_v3 import library_request  # noqa: E402

from env_factory_v3.production import generate  # noqa: E402

DOMAINS = [{'name': 'clinic appointments'}, {'name': 'car rental'}]


SCENE = {'situation': '借的两本书快到期了', 'goal': '能续的都续上', 'known': {},
         'wants_to_know': ['续上了哪些', '续不了的为什么']}


def natural(request, title):
    """Stage-one output: rules, tables, facts and a scene, no opening."""
    request = copy.deepcopy(request)
    request.pop('user_request')
    request.update(title=title, user_scene=copy.deepcopy(SCENE))
    return request


def script_for(elements):
    """A user_script that satisfies the assigned elements on the library task (approve_first/second)."""
    script = {'traits': ['话少']}
    if 'fallback' in elements:
        script['fallbacks'] = ['先续第一本', '不行就续第二本']
    if 'pushback' in elements:
        script['pushback'] = {'demand': '免掉逾期费', 'rounds': 2, 'then': 'accept'}
    if 'multi_goal' in elements:
        script['goals'] = ['续借', '问候补人数']
    if 'forgotten' in elements:
        script['forgotten'] = ['email']
    if 'out_of_scope' in elements:
        script['out_of_scope'] = '顺便帮我订个餐厅'
        script['goals'] = script.get('goals', ['续借']) + ['顺便帮我订个餐厅']
    if 'wrong_first' in elements:
        script['wrong_first'] = {'reader_id': 'R9X'}  # matches nothing in the library tables
    if 'ambiguous' in elements:
        script['ambiguous'] = {'phrase': '我那本书', 'fact': 'loan_ids'}
    if 'inquisitive' in elements:
        script['inquisitive'] = ['续借要收费吗？']
    if 'change_of_mind' in elements:
        script['change_of_mind'] = {'from': '续借第一本', 'to': '续借第二本', 'after': '听说第一本不能续'}
    if 'clarify' in elements:
        script['clarify'] = {'options': ['续借第一本', '续借第二本'], 'decides_by': '能不能续'}
    return script if elements else None


POLICY = '只能为借阅人本人续借；有其他读者排队的书不能续借，要告诉用户排队人数。'


def policy_reply(prompt):
    """The policy writer and checker share the opening writer's client; answer them plainly."""
    if prompt.startswith('你要把一份写给开发者的业务实现规格'):
        return {'content': POLICY}
    if prompt.startswith('下面是一份业务实现规格'):
        return {'content': '{"problems": []}'}
    return None


class Writer:
    """Stands in for the opening writer; records what it was shown."""

    model = 'scripted-writer'

    def __init__(self, replies=None):
        self.prompts, self.replies = [], list(replies or [])

    def chat(self, messages, tools=None, **parameters):
        policy = policy_reply(messages[-1]['content'])
        if policy:
            return policy
        self.prompts.append(messages[-1]['content'])
        return {'content': self.replies.pop(0) if self.replies else '两本书还有3天到期，能续的都续上吧，续不了的跟我说下原因。'}


class OpeningTests(unittest.TestCase):
    def test_red_flags_catch_protocol_and_virtual_time_but_not_plain_speech(self):
        from env_factory_v3.production.openings import persona, red_flags
        self.assertIn('交代确认流程', red_flags('每件寄之前先告诉我走哪家、多少钱，等我确认再寄'))
        self.assertIn('交代确认流程', red_flags('先把分配发我，我确认后你再留'))
        self.assertIn('虚拟时间', red_flags('当前时间是10，预约甲开诊时间30'))
        self.assertIn('虚拟时间', red_flags('今天是第10天，课包还剩几课时'))
        for plain in ('我这次停车181分钟，收费牌上写第一个小时3元，要交多少钱？', '先帮我确认下这个号还能不能挂，下午两点那个。',
                      '我想查一下上次租车的押金现在能不能解冻；订单号我有。'):
            self.assertEqual(red_flags(plain), [], plain)
        self.assertEqual(persona('gen-007'), persona('gen-007'))


class GenerateTests(unittest.TestCase):
    def test_stock_phrases_are_detected_and_rewritten(self):
        texts = ['退款能到账吗，弄完跟我说下结果哈', '车位还有吗，弄完跟我说下结果哈', '押金呢，弄完跟我说下结果哈', '我想问下运费']
        self.assertIn('跟我说下结果', ''.join(generate.repeated_phrases(texts)))
        request = natural(library_request(), 'x')
        request.update(id='gen-001', user_request=texts[0])
        output = Path(tempfile.mkdtemp())
        (output / 'requests').mkdir()
        writer = Writer(['书快到期了，能续就续上吧。'])
        self.assertEqual(generate.respread([request, *[{**request, 'id': f'gen-00{i}', 'user_request': t}
                                                       for i, t in ((2, texts[1]), (3, texts[2]))]],
                                           writer, output)[:1], ['gen-001'])
        self.assertIn('不要使用这些已被其他用户用滥的说法', writer.prompts[0])

    def test_plan_follows_the_default_mix(self):
        shapes = [shape for _, _, shape in generate.plan(DOMAINS, 30)]
        self.assertEqual([shapes.count(s) for s in ('direct', 'simple', 'ordinary', 'single_write', 'complex')],
                         [3, 6, 4, 8, 9])

    def test_request_check_rejects_bad_scenes_and_bad_approvals(self):
        folder = Path(tempfile.mkdtemp())
        request = natural(library_request(), 'x')
        (folder / 'assignment.json').write_text(json.dumps({'category': request['category'], 'shape': 'complex'}))
        (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
        self.assertEqual(generate.check_request(folder)[0], [])
        request['user_scene']['goal'] = '续借，办不成的别影响已经续好的，当前时间是10'
        request['user_facts']['approve_first'] = 'yes'
        (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
        problems = generate.check_request(folder)[0]
        self.assertTrue(any('别影响' in p for p in problems))
        self.assertTrue(any('虚拟时间' in p for p in problems))
        self.assertTrue(any('approve_first' in p for p in problems))
        request['user_scene']['goal'] = '能续的都续上'
        request['user_facts']['approve_first'] = True
        request['user_scene']['wants_to_know'] = ['a', 'b', 'c', 'd']
        (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
        self.assertTrue(any('wants_to_know' in p for p in generate.check_request(folder)[0]))
        request['user_scene']['wants_to_know'] = ['a']
        del request['user_facts']['approve_second']
        (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
        self.assertTrue(any('at least two writes' in p for p in generate.check_request(folder)[0]))
        del request['user_scene']
        (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
        self.assertTrue(any('user_scene' in p for p in generate.check_request(folder)[0]))

    def test_agent_policy_hides_internals_and_must_cover_the_spec(self):
        from env_factory_v3.production import agent_policy as policies
        from env_factory_v3.serving import server
        request = library_request()
        self.assertTrue(policies.internals(request, '仅当 active_hold_count 为 0 时可续借，见 holds 表'))
        self.assertTrue(policies.internals(request, '首先查询借阅，然后调用续借'))
        self.assertEqual(policies.internals(request, POLICY), [])

        class Checker:
            """Finds a gap in the first draft only."""

            def __init__(self):
                self.drafts = 0

            def chat(self, messages, tools=None, **parameters):
                prompt = messages[-1]['content']
                if prompt.startswith('你要把一份写给开发者的业务实现规格'):
                    self.drafts += 1
                    return {'content': POLICY if self.drafts > 1 else '可以续借。'}
                gap = '可以续借。' in prompt.split('客服政策手册：', 1)[1]
                return {'content': json.dumps({'problems': ['漏了排队的书不能续借'] if gap else []}, ensure_ascii=False)}

        checker = Checker()
        policy, problems = policies.write(checker, request)
        self.assertEqual((policy, problems, checker.drafts), (POLICY, [], 2))
        env, oracle, schema = __import__('test_env_factory_v3').library_modules()
        service = server.Service(env, oracle, schema, {**request, 'agent_policy': POLICY})
        self.assertEqual(service.reset(world='original')['instructions']['business_rules'], POLICY)

    def test_english_share_spreads_english_tasks_and_english_checks_apply(self):
        from env_factory_v3.production import agent_policy, openings
        langs = generate.languages(10, 0.7)
        self.assertEqual(langs.count('en'), 7)
        self.assertEqual(generate.languages(4, 0.0), ['zh'] * 4)
        self.assertTrue(openings.red_flags('Please confirm with me first before each step', 'en'))
        self.assertEqual(openings.red_flags('Hi, I want to cancel my order and get $12 back.', 'en'), [])
        request = {**library_request(), 'language': 'en'}
        clean = 'Only the borrower may renew. Loans with other readers waiting cannot be renewed; tell the user how many.'
        self.assertEqual(agent_policy.internals(request, clean, ('renew', 'list_loans')), [])  # ordinary words
        self.assertTrue(agent_policy.internals(request, 'Call list_loans, then renew.', ('list_loans',)))

    def test_generic_table_readers_are_rejected(self):
        from env_factory_v3 import checks

        def read_table(db, now, table: str):
            """Read one table."""

        def dump_holds(db, now):
            """Every hold."""

        def get_loan(db, now, loan_id: str):
            """One loan."""

        request = library_request()
        problems = checks.api_problems({'read_table': read_table, 'dump_holds': dump_holds, 'get_loan': get_loan},
                                       request)
        self.assertEqual(sorted(p.split(':')[0] for p in problems), ['tool dump_holds', 'tool read_table'])

    def test_script_elements_are_assigned_and_checked(self):
        shapes = ['direct', 'simple', 'complex', 'complex'] * 25
        assigned = generate.scripts(len(shapes), shapes)
        self.assertTrue(all(not a for a, s in zip(assigned, shapes) if s == 'direct'))
        elements = lambda a: [e for e in a if e != 'identity']  # identification is a norm, not counted
        self.assertTrue(all(2 <= len(elements(a)) <= 4 for a, s in zip(assigned, shapes) if s != 'direct'))
        self.assertGreater(sum('fallback' in a for a in assigned), sum('forgotten' in a for a in assigned))
        request = library_request()
        self.assertTrue(generate.script_problems(request, ['pushback']))  # missing user_script
        everything = ['fallback', 'pushback', 'multi_goal', 'forgotten', 'out_of_scope']
        request['user_script'] = script_for(everything)
        self.assertEqual(generate.script_problems(request, everything), [])
        request['user_script']['forgotten'] = ['reader_id']  # still in user_facts
        self.assertTrue(generate.script_problems(request, ['forgotten']))
        self.assertIn('Pushback', generate.script_prompt(['pushback']))

    def test_numbers_said_in_chinese_count_as_said(self):
        scene = {'known': {'身份': '芭蕾舞者', '所需次数': 2, '课时': 12, '单价': 18000, '押金': 350, '人数': 10}}
        self.assertEqual(generate.missing_numbers('要上两次，一共十二课时，每节一百八，押金三块五，十个人', scene), [])
        self.assertEqual(generate.missing_numbers('要上2次，12课时，180元，3块5，10人', scene), [])
        self.assertEqual(generate.missing_numbers('每节一百八，押金三块五，十个人', scene), [2, 12])
        self.assertTrue({'一百零五', '一千零五', '两万零三百'} <= generate.chinese(105) | generate.chinese(1005)
                        | generate.chinese(20300))
        self.assertNotIn('两十', generate.chinese(20))

    def test_opening_writer_sees_only_the_scene_and_must_say_every_number(self):
        request = natural(library_request(), 'x')
        request['user_scene']['known'] = {'停车分钟': 181, '首小时': 300, '每半小时': 120}
        writer = Writer(['停了181分钟，首小时3元，每半小时1.2元，多少钱？等我确认再交。',
                         '停了181分钟，首小时3块，后面每半小时1块2，算下多少钱？'])
        opening, _, problems = generate.write_opening(writer, request, 'gen-001')
        self.assertEqual(problems, [])
        self.assertIn('1块2', opening)
        self.assertNotIn(request['business_rules'][:12], writer.prompts[0])
        self.assertIn('交代确认流程', writer.prompts[1])  # the first try's red flag is fed back
        stuck = Writer(['停了很久，多少钱？'] * 3)
        self.assertIsNone(generate.write_opening(stuck, request, 'gen-001')[0])

    def test_generate_keeps_valid_requests_and_drops_near_duplicates(self):
        base = library_request()

        def runner(folder, prompt, *, timeout, role):
            assignment = json.loads((Path(folder) / 'assignment.json').read_text())
            index = int(assignment['id'].split('-')[1])
            request = natural(base, '续借与候补')  # sessions 0 and 1 write the same task
            request['category'] = assignment['category']
            if assignment['shape'] == 'direct':
                request['tables'] = {name: [] for name in request['tables']}
                request['user_scene']['known'] = {'剩余天数': 3}
            if assignment.get('script'):
                request['user_script'] = script_for(assignment['script'])
            if index >= 2:
                request['title'] = f'不同的任务 {index}'
                request['business_rules'] = f'完全不同的规则编号{index}：' + 'abcdefghij' * index
            (Path(folder) / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
            return {'role': role, 'status': 'exited'}

        output = Path(tempfile.mkdtemp()) / 'gen'
        summary = generate.generate(DOMAINS, 5, output, runner=runner, parallel=1, opening_client=Writer())
        kept = sorted(p.name for p in (output / 'requests').glob('*.json'))
        self.assertIn('gen-000.json', kept)
        self.assertNotIn('gen-001.json', kept)
        self.assertEqual(json.loads((output / 'requests' / 'gen-000.json').read_text())['agent_policy'], POLICY)
        self.assertTrue(any('near-duplicate' in ' '.join(r['problems']) for r in summary['rows']))


if __name__ == '__main__':
    unittest.main()


class RecoveryScriptTests(unittest.TestCase):
    def test_wrong_first_and_identity_checks(self):
        from env_factory_v3.production import generate
        request = {'tables': {'orders': [{'id': 'W123'}]}, 'user_facts': {'order_id': 'W123', 'approve_x': True},
                   'user_scene': {'known': {}}, 'user_script': {'traits': ['brief'], 'wrong_first': {'order_id': 'W124'}}}
        self.assertEqual(generate.script_problems(request, ['wrong_first', 'identity']), [])
        request['user_script']['wrong_first'] = {'order_id': 'W123'}
        self.assertTrue(any('wrong value' in p for p in generate.script_problems(request, ['wrong_first'])))
        request['user_scene']['known'] = {'order': 'W123'}
        self.assertTrue(any('identity' in p for p in generate.script_problems(request, ['identity'])))

    def test_identity_is_assigned_to_most_interactive_tasks(self):
        from env_factory_v3.production import generate
        plan = generate.scripts(200, ['complex'] * 200)
        share = sum('identity' in s for s in plan) / 200
        self.assertGreater(share, 0.6)
        self.assertTrue(all('identity' in s for s in plan if 'wrong_first' in s))

    def test_simulated_user_is_told_both_values(self):
        from env_factory_v3.serving import user_sim
        text = user_sim.script_text({'wrong_first': {'order_id': 'W124'}}, 'en', {'order_id': 'W123'})
        self.assertIn('W124', text)
        self.assertIn('W123', text)


class ClarificationScriptTests(unittest.TestCase):
    def test_ambiguous_and_clarify_are_checked_and_acted(self):
        from env_factory_v3.production import generate
        from env_factory_v3.serving import user_sim
        request = copy.deepcopy(library_request())
        request['user_script'] = script_for(['ambiguous', 'clarify'])
        self.assertEqual(generate.script_problems(request, ['ambiguous', 'clarify']), [])
        request['user_script']['ambiguous'] = {'phrase': 'x', 'fact': 'nope'}
        self.assertTrue(generate.script_problems(request, ['ambiguous']))
        text = user_sim.script_text(script_for(['ambiguous', 'clarify']), 'zh')
        self.assertIn('我那本书', text)
        self.assertIn('能不能续', text)


class CodexChatTests(unittest.TestCase):
    def test_each_call_is_its_own_session_and_returns_answer_md(self):
        import tempfile
        from env_factory_v3.authoring.codex import CodexChat
        seen = []

        def runner(folder, prompt, *, timeout, role):
            seen.append(Path(folder))
            (Path(folder) / 'answer.md').write_text(f'answer {len(seen)}')
            return {'status': 'exited'}
        chat = CodexChat(Path(tempfile.mkdtemp()), runner=runner, role='agent_policy')
        self.assertEqual(chat.chat([{'role': 'user', 'content': 'write'}])['content'], 'answer 1')
        self.assertEqual(chat.chat([{'role': 'user', 'content': 'check'}])['content'], 'answer 2')
        self.assertNotEqual(seen[0], seen[1])
        self.assertTrue(chat.model.startswith('codex:'))


class CodexIsolationTests(unittest.TestCase):
    def test_reads_outside_the_session_are_violations(self):
        import tempfile
        from env_factory_v3.authoring.codex import PROJECT, isolation_violations
        log = Path(tempfile.mkdtemp()) / 'events.jsonl'
        log.write_text('\n'.join([
            '{"cmd": "python -m env_factory_v3.checks env"}',
            '{"cmd": "cat ../oracle/verifier.py"}',
            f'{{"cmd": "cat {PROJECT}/output/run/build/gen-001/oracle/verifier.py"}}',
            '{"cmd": "ls /tmp/envfactory-codex/session-other1"}',
            '{"cmd": "ls /tmp/envfactory-codex/session-mine99"}']))
        found = isolation_violations(log, own='session-mine99')
        self.assertEqual(len(found), 3)
        self.assertFalse(any('checks env' in line or 'mine99' in line for line in found))
