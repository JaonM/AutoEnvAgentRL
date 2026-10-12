"""V3 bundle -> rl runtime adapter. No model calls: the bundle comes from the fake-runner pipeline."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_env_factory_v3 import LIBRARY, FakeRunner, library_request  # noqa: E402

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


_MODE = unittest.mock.patch.dict(os.environ, {'ENVFACTORY_V3_USER_MODE': 'structured',
                                              'ENVFACTORY_V3_TOOL_NAMES': 'fixed'})


def setUpModule():
    _MODE.start()  # these tests drive the structured protocol by canonical names; natural/random are the defaults


def tearDownModule():
    _MODE.stop()


class DefaultModeTests(unittest.TestCase):
    def test_training_defaults_to_free_dialogue(self):
        root = Path(tempfile.mkdtemp())
        status = pipeline.build(library_request(), root / 'build', runner=FakeRunner(), perturbed=20)
        environ = {k: v for k, v in os.environ.items() if k != 'ENVFACTORY_V3_USER_MODE'}
        with unittest.mock.patch.dict(os.environ, environ, clear=True), \
                unittest.mock.patch('env_factory_v3.llm.ChatClient'):  # no simulator calls in this test
            env = V3Episode(Path(status['bundle']))
            self.addCleanup(env.close)
            env.reset(0)
        self.assertEqual(env.user_mode, 'natural')
        names = {tool['function']['name'] for tool in env.tools}
        self.assertNotIn('v3_ask', names)
        self.assertNotIn('v3_confirm', names)


class RenamedToolTests(unittest.TestCase):
    """Platform tool names change per episode; the policy must use the names it is given."""

    @classmethod
    def setUpClass(cls):
        root = Path(tempfile.mkdtemp())
        status = pipeline.build(library_request(), root / 'build', runner=FakeRunner(), perturbed=20)
        cls.bundle = Path(status['bundle'])

    def episode(self, naming):
        with unittest.mock.patch.dict(os.environ, {'ENVFACTORY_V3_TOOL_NAMES': naming}):
            env = V3Episode(self.bundle)
        self.addCleanup(env.close)
        return env

    def test_reference_under_random_names_earns_full_reward(self):
        from env_factory_v3 import agent_view
        env = self.episode('random')
        seed = next(s for s in range(50) if agent_view.naming(s)['v3_ask'] != 'v3_ask'
                    and agent_view.naming(s)['consent'] != 'consent')
        env.reset(seed)
        names = env.naming
        offered = {t['function']['name'] for t in env.tools}
        self.assertNotIn('v3_answer', offered)
        self.assertIn(names['v3_answer'], offered)
        self.assertIn(names['v3_ask'], env.messages[0]['content'])
        write = next(t for t in env.tools if names['consent'] in t['function']['parameters']['properties'])
        self.assertNotIn('consent', write['function']['parameters']['properties'])

        class Renamed(ScriptedPolicy):
            def _send(self, name, args):
                if name in ('v3_ask', 'v3_confirm'):
                    name = names[name]
                elif 'consent' in args:
                    args = dict(args)
                    args[names['consent']] = args.pop('consent')
                return super()._send(name, args)

        loaded = bundle.load_bundle(self.bundle)
        answer = loaded['env'].reference(Renamed(env))
        env.step(call(names['v3_answer'], answer, 'final'))
        self.assertEqual(env.finish()['final_reward'], 1.0)

    def test_heldout_names_count_calls_to_tools_that_are_not_offered(self):
        env = self.episode('heldout')
        env.reset(0)
        env.step(call('v3_answer', {}, 'x1'))     # a memorised platform name
        env.step(call('made_up_tool', {}, 'x2'))  # an unknown name
        result = env.finish()
        self.assertEqual((result['unknown_tool_calls'], result['platform_tool_hallucinations']), (2, 1))
        self.assertEqual(result['tool_naming'], 'heldout')


class NaturalEndTests(unittest.TestCase):
    def test_idle_speech_gets_a_notice_naming_the_answer_tool_then_closes(self):
        root = Path(tempfile.mkdtemp())
        status = pipeline.build(library_request(), root / 'build', runner=FakeRunner(), perturbed=20)

        class Murmur:
            model = 'scripted-user'

            def __init__(self, *args, **kwargs):
                pass

            def chat(self, messages, tools=None, **parameters):
                return {'content': '嗯'}

        with unittest.mock.patch.dict(os.environ, {'ENVFACTORY_V3_USER_MODE': 'natural',
                                                   'ENVFACTORY_V3_TOOL_NAMES': 'random'}), \
                unittest.mock.patch('env_factory_v3.llm.ChatClient', Murmur):
            env = V3Episode(Path(status['bundle']))
            self.addCleanup(env.close)
            env.reset(3)
            for turn in range(3):
                env.step({'role': 'assistant', 'content': f'第{turn}句'})
            self.assertIn(env.naming['v3_answer'], env.messages[-1]['content'])
            self.assertFalse(env.terminated)
            _, done = env.step({'role': 'assistant', 'content': '还在吗'})
        self.assertTrue(done)
        self.assertEqual(env.finish()['score']['reason'], 'user_left_without_answer')


class ContainerLifetimeTests(unittest.TestCase):
    def test_serving_containers_remove_themselves_when_idle(self):
        from env_factory_v3 import docker
        calls = []

        def fake(*args, **kwargs):
            calls.append(args)
            return {'run': 'c0ffee', 'port': '127.0.0.1:1'}.get(args[0], '')

        with unittest.mock.patch.object(docker, 'docker', fake), \
                unittest.mock.patch.object(docker, 'image_for', lambda folder: 'sha256:x'), \
                unittest.mock.patch.object(docker, 'remove', lambda container: None):
            with self.assertRaises(RuntimeError):
                docker.run('/nowhere', timeout=0.1)
        run = next(c for c in calls if c[0] == 'run')
        self.assertIn('--rm', run)
        self.assertEqual(run[run.index('--idle-exit') + 1], str(docker.IDLE_EXIT))


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


class ToolCrashTests(unittest.TestCase):
    """A crash of an authored tool is an observation; repeated crashes quarantine the sandbox."""

    @classmethod
    def setUpClass(cls):
        source = (LIBRARY / 'env.py').read_text().replace(
            "    _find(db['books'], book_id, 'book')",
            "    if book_id == 'CRASH':\n        raise IndexError('authored bug')\n    _find(db['books'], book_id, 'book')")

        class CrashRunner(FakeRunner):
            pass
        root = Path(tempfile.mkdtemp())
        status = pipeline.build(library_request(), root / 'build', runner=CrashRunner(env_source=source), perturbed=20)
        assert status['training_ready'], status
        cls.bundle = Path(status['bundle'])

    def test_service_reports_a_tool_crash_and_the_episode_goes_on(self):
        env = V3Episode(self.bundle)
        self.addCleanup(env.close)
        env.reset(0)
        env.step(call('get_waitlist', {'book_id': 'CRASH'}, 'c1'))
        reply = json.loads(env.messages[-1]['content'])
        self.assertEqual((reply['ok'], reply['kind']), (False, 'tool_error'))
        self.assertIn('IndexError', reply['error'])
        self.assertFalse(env.terminated)
        env.step(call('v3_ask', {'key': 'reader_id'}, 'c2'))
        self.assertTrue(json.loads(env.messages[-1]['content'])['ok'])
        self.assertEqual(env.finish()['tool_errors'], 1)

    def test_repeated_crashes_quarantine_the_sandbox(self):
        from rl.errors import EnvironmentProtocolError
        os.environ['ENVFACTORY_V3_TOOL_ERROR_LIMIT'] = '2'
        self.addCleanup(os.environ.pop, 'ENVFACTORY_V3_TOOL_ERROR_LIMIT')
        env = V3Episode(self.bundle)
        self.addCleanup(env.close)
        env.reset(0)
        env.step(call('get_waitlist', {'book_id': 'CRASH'}, 'c1'))
        with self.assertRaises(EnvironmentProtocolError):
            env.step(call('get_waitlist', {'book_id': 'CRASH'}, 'c2'))


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

    def test_platform_identity_covers_exactly_the_runtime_import_closure(self):
        import ast
        root = bundle.PACKAGE_ROOT
        imported = set()
        for name in bundle.RUNTIME_MODULES:
            for node in ast.walk(ast.parse((root / name).read_text())):
                if isinstance(node, ast.ImportFrom) and node.module == 'env_factory_v3':
                    imported |= {f'{alias.name}.py' for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and (node.module or '').startswith('env_factory_v3.'):
                    imported.add(node.module.split('.', 1)[1] + '.py')
        self.assertLessEqual(imported, set(bundle.RUNTIME_MODULES))  # runtime never reaches orchestration
        for orchestration in ('pipeline.py', 'run.py', 'generate.py', 'prompts.py', 'policy.py', 'codex.py'):
            self.assertNotIn(orchestration, bundle.RUNTIME_MODULES)

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

    def test_an_unanswered_episode_earns_zero_like_in_training(self):
        from env_factory_v3 import policy

        class Busy:
            model = 'busy'

            def chat(self, messages, tools=None, **parameters):
                return {'content': '', 'tool_calls': [{'id': f'c{len(messages)}', 'type': 'function', 'function': {
                    'name': 'v3_ask', 'arguments': json.dumps({'key': 'reader_id'})}}]}

        report = policy.gate(self.bundle, Busy(), k=1, parallel=1, max_turns=3)
        episode = report['episodes'][0]
        self.assertEqual((episode['reward'], episode['answered'], episode['reason']), (0.0, False, 'no_final_answer'))

    def test_an_unreachable_service_is_infrastructure_not_a_crash(self):
        from env_factory_v3 import policy
        with self.assertRaises(policy.ProviderError):
            policy.play('http://127.0.0.1:9', object(), 'original', 'multi_step_agentic')

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
