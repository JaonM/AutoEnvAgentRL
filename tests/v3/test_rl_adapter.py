"""V3 bundle -> rl runtime adapter. No model calls: the bundle comes from the fake-runner pipeline."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_env_factory_v3 import FakeRunner, library_request  # noqa: E402

from env_factory_v3 import bundle, pipeline  # noqa: E402
from env_factory_v3.runtime import BusinessError  # noqa: E402
from rl.data.tasks import load_tasks  # noqa: E402
from rl.environments.v3 import V3DockerEpisode, V3Episode, is_v3_bundle  # noqa: E402


def call(name, args, identifier):
    return {'role': 'assistant', 'content': '', 'tool_calls': [{'id': identifier, 'type': 'function',
            'function': {'name': name, 'arguments': json.dumps(args)}}]}


class ScriptedPolicy:
    """Drives the bundle's reference agent through the policy-facing tool-call protocol."""

    def __init__(self, episode, skip_consent=False):
        self.episode, self.skip_consent, self.count = episode, skip_consent, 0

    def _send(self, name, args):
        self.count += 1
        self.episode.step(call(name, args, f'c{self.count}'))
        result = json.loads(self.episode.messages[-1]['content'])
        if not result.get('ok'):
            if result.get('kind') == 'business':
                raise BusinessError(result['error'])
            raise KeyError(result['error'])
        return result['result']

    def ask(self, key):
        return self._send('v3_ask', {'key': key})

    def confirm(self, key, proposal):
        return 'forged' if self.skip_consent else self._send('v3_confirm', {'key': key, 'proposal': proposal})

    def call(self, name, **args):
        return self._send(name, args)


class V3AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp())
        status = pipeline.build(library_request(), cls.root / 'build', runner=FakeRunner(), perturbed=20)
        assert status['training_ready'], status
        cls.bundle = Path(status['bundle'])

    def setUp(self):
        self.env = V3Episode(self.bundle)
        self.addCleanup(self.env.close)

    def play(self, policy):
        from env_factory_v3.bundle import load_bundle
        answer = load_bundle(self.bundle)['env'].reference(policy)
        self.env.step(call('v3_answer', answer, 'final'))
        return self.env.finish()

    def test_reference_through_the_policy_protocol_earns_full_reward(self):
        messages = self.env.reset(0)
        self.assertNotIn('user_facts', json.dumps(messages, ensure_ascii=False))
        names = {t['function']['name'] for t in self.env.tools}
        self.assertTrue({'v3_ask', 'v3_confirm', 'v3_answer'} <= names)
        result = self.play(ScriptedPolicy(self.env))
        self.assertTrue(self.env.terminated)
        self.assertEqual(result['final_reward'], 1.0, result['score'])

    def test_write_without_consent_earns_zero(self):
        self.env.reset(0)
        try:
            self.play(ScriptedPolicy(self.env, skip_consent=True))
        except (BusinessError, KeyError):
            self.env.step(call('v3_answer', {'renewed_loan_ids': [], 'blocked': []}, 'final'))
        self.assertEqual(self.env.finish()['final_reward'], 0.0)

    def test_protocol_errors_are_observations(self):
        self.env.reset(0)
        reward, done = self.env.step({'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'nope'}]})
        self.assertEqual((reward, done), (0.0, False))
        self.assertIn('protocol_error', self.env.messages[-1]['content'])
        self.env.step('not json')
        self.assertEqual(self.env.finish()['score']['reason'], 'invalid_final_answer')

    def test_snapshot_restore_continues_the_same_episode(self):
        self.env.reset(3)
        policy = ScriptedPolicy(self.env)
        policy.ask('reader_id')
        saved = self.env.snapshot()
        restored = V3Episode(self.bundle)
        self.addCleanup(restored.close)
        restored.restore(saved)
        self.assertEqual(restored.world, self.env.world)
        self.assertEqual(restored.messages, self.env.messages)
        policy = ScriptedPolicy(restored)
        policy.count = 10
        answer = __import__('env_factory_v3.bundle', fromlist=['load_bundle']).load_bundle(self.bundle)['env'].reference(policy)
        restored.step(call('v3_answer', answer, 'final'))
        self.assertEqual(restored.finish()['final_reward'], 1.0)

    def test_finished_episodes_are_released(self):
        self.env.reset(0)
        first = self.env.episode
        self.env.reset(1)
        self.assertNotIn(first, self.env.service.episodes)

    def test_task_loader_accepts_v3_bundles(self):
        self.assertTrue(is_v3_bundle(self.bundle))
        manifest = self.root / 'tasks.json'
        manifest.write_text(json.dumps({'tasks': [{'id': 'v3-library', 'sandbox': str(self.bundle)}]}))
        tasks = load_tasks(manifest=str(manifest))
        self.assertEqual(tasks[0].sandbox, str(self.bundle.resolve()))
        self.assertTrue(tasks[0].identity)


class RequalifyTests(unittest.TestCase):
    def test_bundle_from_another_platform_is_refused_until_requalified(self):
        root = Path(tempfile.mkdtemp())
        status = pipeline.build(library_request(), root / 'build', runner=FakeRunner(), perturbed=20)
        folder = Path(status['bundle'])
        manifest = json.loads((folder / 'manifest.json').read_text())
        manifest['platform_identity'] = 'an-older-platform'
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            bundle.load_bundle(folder)
        self.assertTrue(bundle.requalify(folder)['passed'])
        loaded = bundle.load_bundle(folder)
        self.assertEqual(loaded['manifest']['requalified'], [bundle.platform_identity()])

    def test_requalify_never_signs_a_failing_bundle(self):
        root = Path(tempfile.mkdtemp())
        status = pipeline.build(library_request(), root / 'build', runner=FakeRunner(), perturbed=20)
        folder = Path(status['bundle'])
        manifest = json.loads((folder / 'manifest.json').read_text())
        manifest['platform_identity'] = 'an-older-platform'
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        source = (folder / 'oracle.py').read_text().replace("renewed.append(loan_id)", "pass", 1)
        (folder / 'oracle.py').write_text(source)
        manifest['files']['oracle.py'] = bundle.sha256(folder / 'oracle.py')
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        self.assertFalse(bundle.requalify(folder)['passed'])
        with self.assertRaises(ValueError):
            bundle.load_bundle(folder)


class PolicyGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp())
        cls.bundle = Path(pipeline.build(library_request(), root / 'build', runner=FakeRunner(), perturbed=20)['bundle'])

    def test_a_policy_that_answers_nothing_leaves_the_task_unsolved(self):
        from env_factory_v3 import policy

        class Silent:
            model = 'silent'

            def chat(self, messages, tools):
                self.tools = [t['function']['name'] for t in tools]
                return {'content': '{}'}

        client = Silent()
        report = policy.gate(self.bundle, client, k=2, parallel=1)
        self.assertEqual((report['status'], report['played'], report['successes']), ('unsolved', 2, 0))
        self.assertIn('v3_ask', client.tools)
        self.assertTrue((self.bundle / policy.RESULT).exists())

    def test_provider_failures_are_infrastructure_not_task_results(self):
        from env_factory_v3 import policy

        class Broken:
            model = 'broken'

            def chat(self, messages, tools):
                raise policy.ProviderError('HTTP 503')

        report = policy.gate(self.bundle, Broken(), k=2, parallel=1)
        self.assertEqual((report['status'], report['played']), ('infrastructure', 0))


@unittest.skipUnless(os.environ.get('V3_DOCKER_TESTS') == '1' and shutil.which('docker'),
                     'set V3_DOCKER_TESTS=1 to build and run real containers')
class DockerTests(unittest.TestCase):
    def test_verified_container_serves_a_full_reward_episode(self):
        from env_factory_v3 import docker
        root = Path(tempfile.mkdtemp())
        status = pipeline.build(library_request(), root / 'build', runner=FakeRunner(), perturbed=20)
        folder = Path(status['bundle'])
        metadata = docker.build(folder)
        self.assertTrue(metadata['container_check']['passed'])
        env = V3DockerEpisode(folder)
        try:
            env.reset(0)
            answer = bundle.load_bundle(folder)['env'].reference(ScriptedPolicy(env))
            env.step(call('v3_answer', answer, 'final'))
            self.assertEqual(env.finish()['final_reward'], 1.0)
            inspect = json.loads(docker.docker('inspect', env.container))[0]['HostConfig']
            self.assertTrue(inspect['ReadonlyRootfs'])
            self.assertEqual(inspect['CapDrop'], ['ALL'])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
