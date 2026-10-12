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

from env_factory_v3 import generate  # noqa: E402

DOMAINS = [{'name': 'clinic appointments'}, {'name': 'car rental'}]


SCENE = {'situation': '借的两本书快到期了', 'goal': '能续的都续上', 'known': {},
         'wants_to_know': ['续上了哪些', '续不了的为什么']}


def natural(request, title):
    """Stage-one output: rules, tables, facts and a scene, no opening."""
    request = copy.deepcopy(request)
    request.pop('user_request')
    request.update(title=title, user_scene=copy.deepcopy(SCENE))
    return request


class Writer:
    """Stands in for the opening writer; records what it was shown."""

    model = 'scripted-writer'

    def __init__(self, replies=None):
        self.prompts, self.replies = [], list(replies or [])

    def chat(self, messages, tools=None, **parameters):
        self.prompts.append(messages[-1]['content'])
        return {'content': self.replies.pop(0) if self.replies else '两本书还有3天到期，能续的都续上吧，续不了的跟我说下原因。'}


class OpeningTests(unittest.TestCase):
    def test_red_flags_catch_protocol_and_virtual_time_but_not_plain_speech(self):
        from env_factory_v3.openings import persona, red_flags
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

    def test_plan_follows_the_v2_mix(self):
        shapes = [shape for _, _, shape in generate.plan(DOMAINS, 30)]
        self.assertEqual([shapes.count(s) for s in ('direct', 'simple', 'ordinary', 'complex')], [6, 6, 6, 12])

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
        self.assertTrue(any('near-duplicate' in ' '.join(r['problems']) for r in summary['rows']))


if __name__ == '__main__':
    unittest.main()
