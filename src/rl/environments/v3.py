"""V3 policy runtime: the qualified V3 bundle served over its own loopback HTTP API, in process
(V3Episode) or from its verified hardened container (V3DockerEpisode).

The policy sees only the public instructions (request, rules, tool specs, askable user topics,
answer schema). Private facts, the oracle and source tables stay inside the service. V3 bundles
are deterministic, so a continuation snapshot is the world plus the actions taken so far and
restore replays them.
"""
from __future__ import annotations

import copy
import json
import os
import random
import re
import secrets
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from env_factory_v3.serving import agent_view
from rl.errors import EnvironmentProtocolError, InfrastructureError

def is_v3_bundle(root):
    root = Path(root)
    # Business bundles carry the oracle; subjective (rubric-graded) bundles carry rubric.json instead.
    return all((root / name).is_file() for name in ('manifest.json', 'gates.json', 'env.py')) and \
        any((root / name).is_file() for name in ('oracle.py', 'rubric.json'))


def bundle_identity(root):
    """Content identity of a V3 bundle: the hashes its manifest certifies (verified on load)."""
    from env_factory_v3.certify.bundle import load_bundle
    from env_factory_v3.runtime import canonical
    import hashlib
    loaded = load_bundle(root)
    return hashlib.sha256(canonical(loaded['manifest']['files']).encode()).hexdigest()


RESTORED = ('messages', 'tools', 'terminated', 'score', 'answer', 'unknown_tool_calls',
            'platform_tool_hallucinations')


class V3Episode:
    transport = 'in_process'

    def __init__(self, root):
        from env_factory_v3.certify import bundle
        from env_factory_v3.core.llm import load_env_file
        from rl.paths import PROJECT_ROOT
        load_env_file(PROJECT_ROOT / '.env')
        # natural (default): free dialogue with a simulated user, so the agent must find out what to ask
        # and what to confirm; structured: ask/confirm by topic keys listed in the tool schema.
        self.user_mode = os.environ.get('ENVFACTORY_V3_USER_MODE', 'natural')
        self.token = secrets.token_hex(16)  # only this runtime may drive its service
        self.tool_error_limit = int(os.environ.get('ENVFACTORY_V3_TOOL_ERROR_LIMIT', '3'))
        # random (default): platform tools and the consent argument are renamed per episode so the
        # policy reads its tool list; fixed: canonical names; heldout: names never used in training.
        self.tool_naming = os.environ.get('ENVFACTORY_V3_TOOL_NAMES', 'random')
        # never (default): in free dialogue the agent delivers by talking and the service extracts the
        # answer from its speech, as on τ²-bench; random: half the episodes (by seed) offer the answer
        # tool; always: every episode does.
        self.answer_tool_mode = os.environ.get('ENVFACTORY_V3_ANSWER_TOOL', 'never')
        self.tool_errors = 0
        self.root = Path(root).resolve()
        loaded = bundle.load_bundle(self.root)
        self.base_url = self._start(loaded)
        self.opener = build_opener(ProxyHandler({}))
        self.category = loaded['request']['category']
        self.task = {'task': loaded['request']['user_request']}
        self.episode, self.world, self.seed = None, None, None
        self.messages, self.tools, self.trace, self.actions = [], [], [], []
        self.terminated, self.score, self.answer, self.call_ids = False, None, None, set()
        self._closed = False

    # -- transport --------------------------------------------------------------
    def _start(self, loaded):
        from env_factory_v3.serving import server
        user = None
        if self.user_mode == 'natural':
            from env_factory_v3.core.llm import ChatClient
            user = ChatClient('V3_USER')
        from env_factory_v3.certify import bundle
        self.service = bundle.service(loaded, user_mode=self.user_mode, user_client=user)
        self.server = server.start(self.service, token=self.token)
        return f'http://127.0.0.1:{self.server.server_address[1]}'

    def _stop(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, route, body):
        started = time.monotonic()
        request = Request(self.base_url + route, data=json.dumps(body, ensure_ascii=False).encode(),
                          method='POST', headers={'Content-Type': 'application/json',
                                                  'Authorization': f'Bearer {self.token}'})
        try:
            with self.opener.open(request, timeout=120) as response:
                status, value = response.status, json.load(response)
        except HTTPError as error:
            status, value = error.code, json.load(error)
        except (URLError, TimeoutError, OSError) as error:
            raise InfrastructureError('V3 HTTP transport failed') from error
        if status >= 500:
            raise InfrastructureError(f'V3 service failed on {route}; no fallback')
        self.trace.append({'path': route, 'status': status, 'request': copy.deepcopy(body),
                           'response': copy.deepcopy(value), 'seconds': time.monotonic() - started})
        return status, value

    def _step(self, action):
        self.actions.append(copy.deepcopy(action))
        status, reply = self.request('/step', {'episode': self.episode, 'action': action})
        if status != 200:
            raise InfrastructureError(f'V3 step rejected by transport: {status}')
        if reply.get('kind') == 'tool_error':
            # A crash of the sandbox's own tool is an observation for the agent; repeated crashes
            # mean a defective sandbox, which rl quarantines instead of retrying or training on it.
            self.tool_errors += 1
            if self.tool_errors >= self.tool_error_limit:
                raise EnvironmentProtocolError(f'V3 sandbox tools crashed {self.tool_errors} times in one episode: '
                                               f'{reply.get("error")}')
        return reply

    # -- episode ----------------------------------------------------------------
    def reset(self, seed, world=None):
        self._forget()
        status, response = self.request('/reset', {'world': world} if world else {'seed': int(seed)})
        if status != 200:
            raise InfrastructureError('V3 reset failed')
        self.episode, self.world, self.seed = response['episode'], response['world'], seed
        self.trace, self.actions, self.tool_errors = [], [], 0
        self.unknown_tool_calls, self.platform_tool_hallucinations = 0, 0
        self.terminated, self.score, self.answer, self.call_ids = False, None, None, set()
        public = response['instructions']
        self.names = {tool['name'] for tool in public['tools']}
        self.language = agent_view.language(public)
        arguments = {arg for tool in public['tools'] for arg in tool['parameters']['properties']}
        self.naming = agent_view.naming(int(seed), self.tool_naming, avoid=self.names | arguments)
        self.answer_tool = (self.user_mode != 'natural' or self.answer_tool_mode == 'always' or
                            (self.answer_tool_mode == 'random' and random.Random(int(seed) * 7919 + 1).random() < 0.5))
        self.tools = agent_view.tools(public, self.category, self.naming, self.answer_tool)
        self.messages = agent_view.messages(public, self.naming, self.answer_tool)
        return self.messages

    def _dispatch(self, name, params):
        try:
            step = agent_view.action(name, params, self.names, self.naming)
        except KeyError as error:
            self.unknown_tool_calls += 1
            self.platform_tool_hallucinations += name in agent_view.platform_names()
            return {'ok': False, 'kind': 'invalid', 'error': str(error)}
        if step is None and self.user_mode != 'natural':
            return self._submit(params)
        if step is None:  # free dialogue: the answer is recorded; the user ends the conversation
            self.answer = copy.deepcopy(params)
            return self._step({'type': 'answer', 'answer': params})
        reply = self._step(step)
        if step['type'] == 'chat':
            notice = self._after_chat(reply)
            if notice:
                reply = {**reply, 'notice': notice}
        return reply

    def _after_chat(self, reply):
        """The user may leave: warned once (returns the notice for the agent), then the episode closes."""
        result = reply.get('result') or {}
        if reply.get('kind') == 'finished' or result.get('ended') == 'closed':
            # The user left: graded on the final state, the latest recorded answer and what was said.
            self.terminated = True
            if self.score is None:
                self._grade()
        elif result.get('ended') == 'warning':
            return agent_view.left_notice(self.language, self.naming['v3_answer'] if self.answer_tool else None)
        return None

    def _grade(self):
        status, grade = self.request('/reward', {'episode': self.episode})
        self.score = grade if status == 200 and 'reward' in grade else \
            {'reward': 0.0, 'success': False, 'reason': 'grading_failed'}

    def _submit(self, answer):
        self.answer = copy.deepcopy(answer)
        reply = self._step({'type': 'answer', 'answer': answer})
        self.terminated = True
        if reply.get('ok'):
            self._grade()
        else:
            self.score = {'reward': 0.0, 'success': False, 'reason': 'invalid_final_answer'}
        return {'ok': True, 'result': {'submitted': True}} if reply.get('ok') else reply

    def step(self, action):
        if self.terminated:
            raise ValueError('episode already terminated')
        message = {'role': 'assistant', 'content': action} if isinstance(action, str) else copy.deepcopy(action)
        try:
            if not isinstance(message, dict) or message.get('role') != 'assistant' or message.get('protocol_error'):
                raise ValueError('expected valid assistant message')
            calls = message.get('tool_calls') or []
            if not isinstance(calls, list) or not isinstance(message.get('content') or '', str):
                raise ValueError('invalid assistant message')
            parsed, new_ids = [], set()
            for index, call in enumerate(calls):
                if not isinstance(call, dict) or call.get('type') != 'function':
                    raise ValueError('function tool call required')
                identifier = call.setdefault('id', f'call_{len(self.messages)}_{index}')
                if (not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', identifier)
                        or identifier in self.call_ids | new_ids):
                    raise ValueError('unique safe tool call id required')
                fn = call['function']
                params = json.loads(fn['arguments']) if isinstance(fn['arguments'], str) else fn['arguments']
                if not isinstance(params, dict):
                    raise ValueError('tool arguments must be object')
                fn['arguments'] = json.dumps(params, ensure_ascii=False)
                parsed.append((fn['name'], params, identifier))
                new_ids.add(identifier)
        except (ValueError, TypeError, KeyError) as error:
            self.messages.extend([{'role': 'assistant', 'content': str(action)},
                                  {'role': 'user', 'content': json.dumps({'protocol_error': str(error)})}])
            return 0.0, False
        self.call_ids.update(new_ids)
        self.messages.append(message)
        if calls:
            for name, params, identifier in parsed:
                result = ({'ok': False, 'kind': 'finished', 'error': 'episode terminated'} if self.terminated
                          else self._dispatch(name, params))
                self.messages.append({'role': 'tool', 'tool_call_id': identifier, 'name': name,
                                      'content': json.dumps(result, ensure_ascii=False)})
        elif self.user_mode == 'natural':
            # Free dialogue: a plain message is speech to the user; their reply is the next user turn.
            said = self._step({'type': 'chat', 'message': message.get('content') or ''})
            notice = self._after_chat(said)
            text = (said.get('result') or {}).get('reply') or said.get('error', '')
            self.messages.append({'role': 'user', 'content': f'{text}\n\n{notice}' if notice else text})
        else:
            answer = agent_view.final_answer(message.get('content'))
            if answer is None:
                self.terminated = True
                self.score = {'reward': 0.0, 'success': False, 'reason': 'invalid_final_answer'}
            else:
                self._submit(answer)
        return 0.0, self.terminated

    def finish(self):
        if self.score is None and self.user_mode == 'natural' and self.episode is not None:
            self._grade()  # step limit reached: the service closes the conversation and grades it
        if self.score is None:
            self.score = {'reward': 0.0, 'success': False, 'reason': 'no_final_answer'}
        return {'final_reward': float(self.score['reward']), 'terminated': self.terminated,
                'trace': copy.deepcopy(self.trace), 'score': copy.deepcopy(self.score),
                'world': self.world, 'user_mode': self.user_mode, 'tool_errors': self.tool_errors,
                'tool_naming': self.tool_naming, 'answer_tool': self.answer_tool,
                'unknown_tool_calls': self.unknown_tool_calls,
                'platform_tool_hallucinations': self.platform_tool_hallucinations,
                # False when the simulator audit found the user, not the agent, at fault (server.audit_simulator)
                'trainable': self.score.get('valid', True) is not False}

    # -- continuation (trainer-side only) -----------------------------------------
    def snapshot(self):
        return {'bundle_identity': bundle_identity(self.root), 'world': self.world, 'seed': self.seed,
                'actions': copy.deepcopy(self.actions),
                **{k: copy.deepcopy(getattr(self, k)) for k in RESTORED},
                'call_ids': sorted(self.call_ids)}

    def restore(self, snapshot):
        if snapshot['bundle_identity'] != bundle_identity(self.root):
            raise ValueError('V3 continuation bundle mismatch')
        self.reset(snapshot['seed'], world=snapshot['world'])
        for action in snapshot['actions']:
            self._step(action)
        for key in RESTORED:
            setattr(self, key, copy.deepcopy(snapshot[key]))
        self.call_ids = set(snapshot['call_ids'])

    def _forget(self):
        # The service keeps every episode it served; release ours so long runs stay bounded.
        if self.episode is not None:
            self.request('/release', {'episode': self.episode})

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._forget()
        finally:
            self._stop()


class V3DockerEpisode(V3Episode):
    """Same runtime against the bundle's verified, hardened container (see env_factory_v3.certify.docker)."""

    transport = 'docker'

    def _start(self, loaded):
        from env_factory_v3.certify import docker
        started = docker.run(self.root, user_mode=self.user_mode, token=self.token)
        self.container = started['container']
        return started['url']

    def reset(self, seed, world=None):
        # An idle serving container stops itself (docker.IDLE_EXIT); start a fresh one if it has.
        try:
            with self.opener.open(self.base_url + '/health', timeout=5) as response:
                alive = json.load(response).get('ok') is True
        except (URLError, TimeoutError, OSError, ValueError):
            alive = False
        if not alive:
            from env_factory_v3.certify import docker
            docker.remove(self.container)
            self.episode = None  # the old service and its episodes are gone
            self.base_url = self._start(None)
        return super().reset(seed, world=world)

    def _stop(self):
        from env_factory_v3.certify import docker
        docker.remove(self.container)
