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
import re
import secrets
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from env_factory_v3 import agent_view
from rl.errors import EnvironmentProtocolError, InfrastructureError

def is_v3_bundle(root):
    root = Path(root)
    return all((root / name).is_file() for name in ('manifest.json', 'gates.json', 'env.py', 'oracle.py'))


def bundle_identity(root):
    """Content identity of a V3 bundle: the hashes its manifest certifies (verified on load)."""
    from env_factory_v3.bundle import load_bundle
    from env_factory_v3.runtime import canonical
    import hashlib
    loaded = load_bundle(root)
    return hashlib.sha256(canonical(loaded['manifest']['files']).encode()).hexdigest()


class V3Episode:
    transport = 'in_process'

    def __init__(self, root):
        from env_factory_v3 import bundle
        from env_factory_v3.llm import load_env_file
        from rl.paths import PROJECT_ROOT
        load_env_file(PROJECT_ROOT / '.env')
        # structured: ask/confirm by topic key; natural: free dialogue with a simulated user.
        self.user_mode = os.environ.get('ENVFACTORY_V3_USER_MODE', 'structured')
        self.token = secrets.token_hex(16)  # only this runtime may drive its service
        self.tool_error_limit = int(os.environ.get('ENVFACTORY_V3_TOOL_ERROR_LIMIT', '3'))
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
        from env_factory_v3 import server
        user = None
        if self.user_mode == 'natural':
            from env_factory_v3.llm import ChatClient
            user = ChatClient('V3_USER')
        from env_factory_v3 import bundle
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
        self.terminated, self.score, self.answer, self.call_ids = False, None, None, set()
        public = response['instructions']
        self.names = {tool['name'] for tool in public['tools']}
        self.tools = agent_view.tools(public, self.category)
        self.messages = agent_view.messages(public)
        return self.messages

    def _dispatch(self, name, params):
        try:
            step = agent_view.action(name, params, self.names)
        except KeyError as error:
            return {'ok': False, 'kind': 'invalid', 'error': str(error)}
        return self._submit(params) if step is None else self._step(step)

    def _submit(self, answer):
        self.answer = copy.deepcopy(answer)
        reply = self._step({'type': 'answer', 'answer': answer})
        self.terminated = True
        if reply.get('ok'):
            status, grade = self.request('/reward', {'episode': self.episode})
            self.score = grade if status == 200 and 'reward' in grade else \
                {'reward': 0.0, 'success': False, 'reason': 'grading_failed'}
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
            self.messages.append({'role': 'user', 'content': (said.get('result') or {}).get('reply')
                                  or said.get('error', '')})
        else:
            answer = agent_view.final_answer(message.get('content'))
            if answer is None:
                self.terminated = True
                self.score = {'reward': 0.0, 'success': False, 'reason': 'invalid_final_answer'}
            else:
                self._submit(answer)
        return 0.0, self.terminated

    def finish(self):
        if self.score is None:
            self.score = {'reward': 0.0, 'success': False, 'reason': 'no_final_answer'}
        return {'final_reward': float(self.score['reward']), 'terminated': self.terminated,
                'trace': copy.deepcopy(self.trace), 'score': copy.deepcopy(self.score),
                'world': self.world, 'user_mode': self.user_mode, 'tool_errors': self.tool_errors, 'trainable': True}

    # -- continuation (trainer-side only) -----------------------------------------
    def snapshot(self):
        return {'bundle_identity': bundle_identity(self.root), 'world': self.world, 'seed': self.seed,
                'actions': copy.deepcopy(self.actions),
                **{k: copy.deepcopy(getattr(self, k)) for k in ('messages', 'tools', 'terminated', 'score', 'answer')},
                'call_ids': sorted(self.call_ids)}

    def restore(self, snapshot):
        if snapshot['bundle_identity'] != bundle_identity(self.root):
            raise ValueError('V3 continuation bundle mismatch')
        self.reset(snapshot['seed'], world=snapshot['world'])
        for action in snapshot['actions']:
            self._step(action)
        for key in ('messages', 'tools', 'terminated', 'score', 'answer'):
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
    """Same runtime against the bundle's verified, hardened container (see env_factory_v3.docker)."""

    transport = 'docker'

    def _start(self, loaded):
        from env_factory_v3 import docker
        started = docker.run(self.root, user_mode=self.user_mode, token=self.token)
        self.container = started['container']
        return started['url']

    def _stop(self):
        from env_factory_v3 import docker
        docker.remove(self.container)
