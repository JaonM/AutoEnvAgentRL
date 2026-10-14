"""Exact fork and restore of a served episode (tree rollouts, reverse curriculum). No model calls."""
from __future__ import annotations

import json
import os
import sys
import unittest
import unittest.mock
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_env_factory_v3 import library_modules, library_request  # noqa: E402
from test_natural_user import ScriptedUser  # noqa: E402

from env_factory_v3.serving import server  # noqa: E402

PROPOSAL = {'tool': 'renew_loan', 'args': {'loan_id': 'L1', 'reader_id': 'R1'}}


class ForkTests(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch.dict(os.environ, {'V3_SIMULATOR_AUDIT': 'off'})
        patcher.start()
        self.addCleanup(patcher.stop)
        env, oracle, schema = library_modules()
        request = library_request()
        for hold in request['tables']['holds']:
            hold['active'] = False
        self.user = ScriptedUser()
        self.service = server.Service(env, oracle, schema, request, user_mode='natural', user_client=self.user)

    def advance(self, episode):
        self.service.step(episode, {'type': 'chat', 'message': '您的读者号？'})
        self.service.step(episode, {'type': 'tool', 'name': 'check_renewal', 'args': {'loan_id': 'L1', 'reader_id': 'R1'}})
        return self.service.step(episode, {'type': 'chat', 'message': '续借 L1 可以吗？', 'proposal': PROPOSAL})

    def test_a_fork_continues_exactly_and_independently(self):
        episode = self.service.reset(world='original')['episode']
        token = self.advance(episode)['result']['consent']
        fork = self.service.fork(episode)['episode']
        original, copy_ = self.service.episodes[episode], self.service.episodes[fork]
        self.assertEqual(copy_['user'].messages, original['user'].messages)
        self.assertIsNot(copy_['user'].messages, original['user'].messages)
        self.assertIs(copy_['user'].client, self.user)  # the model client is shared, not copied
        write = {'type': 'tool', 'name': 'renew_loan', 'args': {**PROPOSAL['args'], 'consent': token}}
        done = self.service.step(fork, write)
        self.assertTrue(done['ok'], done)  # the consent token survived the fork
        self.assertNotEqual(copy_['episode'].db, original['episode'].db)  # the original is untouched
        again = self.service.step(episode, write)
        self.assertEqual(again, done)  # same action from the same state, same observation
        self.assertEqual(copy_['episode'].db, original['episode'].db)
        self.assertEqual(copy_['episode'].trace, original['episode'].trace)

    def test_restore_replays_a_recording_without_the_user_model(self):
        episode = self.service.reset(world='original')['episode']
        actions = [{'type': 'chat', 'message': '您的读者号？'},
                   {'type': 'tool', 'name': 'check_renewal', 'args': {'loan_id': 'L1', 'reader_id': 'R1'}},
                   {'type': 'chat', 'message': '续借 L1 可以吗？', 'proposal': PROPOSAL}]
        recording = []
        for action in actions:
            result = self.service.step(episode, action)
            recording.append({'action': action, **({'reply': result['result']['reply']} if action['type'] == 'chat' else {})})
        recorded = self.service.episodes[episode]
        calls = len(self.user.seen)
        restored = self.service.restore('original', recording)
        self.assertEqual(len(self.user.seen), calls)  # no user-model call
        copy_ = self.service.episodes[restored['episode']]
        self.assertEqual(copy_['user'].messages, recorded['user'].messages)
        self.assertEqual(copy_['episode'].db, recorded['episode'].db)
        self.assertIs(copy_['user'].client, self.user)

    def test_fork_over_http(self):
        httpd = server.start(self.service)
        self.addCleanup(httpd.shutdown)
        base = f'http://127.0.0.1:{httpd.server_address[1]}'

        def post(path, body):
            data = json.dumps(body).encode()
            with urllib.request.urlopen(urllib.request.Request(base + path, data=data, method='POST',
                                                               headers={'Content-Type': 'application/json'})) as r:
                return json.loads(r.read())
        episode = post('/reset', {'world': 'original'})['episode']
        fork = post('/fork', {'episode': episode})
        self.assertNotEqual(fork['episode'], episode)
        self.assertTrue(post('/step', {'episode': fork['episode'],
                                       'action': {'type': 'tool', 'name': 'check_renewal',
                                                  'args': {'loan_id': 'L1', 'reader_id': 'R1'}}})['ok'])


if __name__ == '__main__':
    unittest.main()
