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


def natural(request, title):
    request = copy.deepcopy(request)
    request.update(title=title, user_request='我想续借两本书，能续的帮我续一下，最后告诉我续上了哪些。')
    return request


class GenerateTests(unittest.TestCase):
    def test_plan_follows_the_v2_mix(self):
        shapes = [shape for _, _, shape in generate.plan(DOMAINS, 30)]
        self.assertEqual([shapes.count(s) for s in ('direct', 'simple', 'ordinary', 'complex')], [6, 6, 6, 12])

    def test_request_check_rejects_unnatural_openings_and_bad_approvals(self):
        folder = Path(tempfile.mkdtemp())
        request = natural(library_request(), 'x')
        (folder / 'assignment.json').write_text(json.dumps({'category': request['category'], 'shape': 'complex'}))
        (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
        self.assertEqual(generate.check_request(folder)[0], [])
        request['user_request'] = '续借费是300分，办不成的别影响已经续好的。'
        request['user_facts']['approve_first'] = 'yes'
        (folder / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
        problems = generate.check_request(folder)[0]
        self.assertTrue(any('别影响' in p for p in problems))
        self.assertTrue(any('元' in p for p in problems))
        self.assertTrue(any('approve_first' in p for p in problems))

    def test_generate_keeps_valid_requests_and_drops_near_duplicates(self):
        base = library_request()

        def runner(folder, prompt, *, timeout, role):
            assignment = json.loads((Path(folder) / 'assignment.json').read_text())
            index = int(assignment['id'].split('-')[1])
            request = natural(base, '续借与候补')  # sessions 0 and 1 write the same task
            request['category'] = assignment['category']
            if assignment['shape'] == 'direct':
                request['tables'] = {name: [] for name in request['tables']}
            if index >= 2:
                request['title'] = f'不同的任务 {index}'
                request['business_rules'] = f'完全不同的规则编号{index}：' + 'abcdefghij' * index
            (Path(folder) / 'request.json').write_text(json.dumps(request, ensure_ascii=False))
            return {'role': role, 'status': 'exited'}

        output = Path(tempfile.mkdtemp()) / 'gen'
        summary = generate.generate(DOMAINS, 5, output, runner=runner, parallel=1)
        kept = sorted(p.name for p in (output / 'requests').glob('*.json'))
        self.assertIn('gen-000.json', kept)
        self.assertNotIn('gen-001.json', kept)
        self.assertTrue(any('near-duplicate' in ' '.join(r['problems']) for r in summary['rows']))


if __name__ == '__main__':
    unittest.main()
