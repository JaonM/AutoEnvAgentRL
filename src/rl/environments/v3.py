"""V3 policy runtime: the qualified V3 bundle served over its own loopback HTTP API.

The policy sees only the public instructions (request, rules, tool specs, askable user topics,
answer schema). Private facts, the oracle and source tables stay inside the service. V3 bundles
are deterministic, so a continuation snapshot is the world plus the actions taken so far and
restore replays them.
"""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from rl.errors import InfrastructureError

RESERVED = ('v3_ask', 'v3_confirm', 'v3_answer')


def function(name, description, properties, required=()):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties, 'required': list(required)}}}


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
    def __init__(self, root):
        from env_factory_v3 import bundle, server
        self.root = Path(root).resolve()
        loaded = bundle.load_bundle(self.root)
        self.service = server.Service(loaded['env'], loaded['oracle'], loaded['schema'], loaded['request'],
                                      valid_worlds=loaded['valid_worlds'])
        self.server = server.start(self.service)
        self.base_url = f'http://127.0.0.1:{self.server.server_address[1]}'
        self.opener = build_opener(ProxyHandler({}))
        self.category = loaded['request']['category']
        self.task = {'task': loaded['request']['user_request']}
        self.episode, self.world, self.seed = None, None, None
        self.messages, self.tools, self.trace, self.actions = [], [], [], []
        self.terminated, self.score, self.answer, self.call_ids = False, None, None, set()
        self._closed = False

    # -- transport --------------------------------------------------------------
    def request(self, route, body):
        started = time.monotonic()
        request = Request(self.base_url + route, data=json.dumps(body, ensure_ascii=False).encode(),
                          method='POST', headers={'Content-Type': 'application/json'})
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
        return reply

    # -- episode ----------------------------------------------------------------
    def reset(self, seed, world=None):
        self._forget()
        status, response = self.request('/reset', {'world': world} if world else {'seed': int(seed)})
        if status != 200:
            raise InfrastructureError('V3 reset failed')
        self.episode, self.world, self.seed = response['episode'], response['world'], seed
        self.trace, self.actions = [], []
        self.terminated, self.score, self.answer, self.call_ids = False, None, None, set()
        public = response['instructions']
        self.names = {tool['name'] for tool in public['tools']}
        if self.names & set(RESERVED):
            raise ValueError('business tool collides with a V3 runtime tool')
        self.tools = [function(t['name'], t['description'], copy.deepcopy(t['parameters']['properties']),
                               t['parameters']['required']) for t in public['tools']]
        approvals = [t['key'] for t in public['user_topics'] if t['kind'] == 'approval']
        facts = [t['key'] for t in public['user_topics'] if t['kind'] != 'approval']
        if facts:
            self.tools.append(function('v3_ask', 'Ask the user for one private fact by its topic key.',
                                       {'key': {'type': 'string', 'enum': facts}}, ['key']))
        if approvals:
            self.tools.append(function(
                'v3_confirm', 'Ask the user to approve one exact write. Returns a consent token, or null when '
                'the user declines; pass the token as `consent` to exactly that write.',
                {'key': {'type': 'string', 'enum': approvals},
                 'proposal': {'type': 'object', 'properties': {'tool': {'type': 'string'}, 'args': {'type': 'object'}},
                              'required': ['tool', 'args']}}, ['key', 'proposal']))
        fields = public['answer_schema']['fields']
        self.tools.append(function('v3_answer', 'Submit the final answer and end the episode.',
                                   copy.deepcopy(fields), list(fields)))
        if self.category == 'direct_response':
            self.tools = [self.tools[-1]]
        instructions = ('Handle the user request using only the public tools. Never guess private user facts: '
                        'use v3_ask. Before every write call v3_confirm with the exact tool and args, then pass '
                        'the returned token as `consent`. Finish with v3_answer, or a final JSON object with '
                        'exactly the answer schema fields.')
        public_view = {k: public[k] for k in ('title', 'user_request', 'business_rules', 'user_topics',
                                              'answer_schema', 'time')}
        self.messages = [{'role': 'system', 'content': instructions},
                         {'role': 'user', 'content': json.dumps(public_view, ensure_ascii=False)}]
        return self.messages

    def _dispatch(self, name, params):
        if name in self.names:
            return self._step({'type': 'tool', 'name': name, 'args': params})
        if name == 'v3_ask':
            return self._step({'type': 'ask', 'key': params.get('key')})
        if name == 'v3_confirm':
            return self._step({'type': 'confirm', 'key': params.get('key'), 'proposal': params.get('proposal')})
        if name == 'v3_answer':
            return self._submit(params)
        return {'ok': False, 'kind': 'invalid', 'error': f'unknown tool {name}'}

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
        else:
            text = (message.get('content') or '').strip()
            if text.startswith('```') and text.endswith('```'):
                text = text.split('\n', 1)[-1].rsplit('```', 1)[0]
            try:
                answer = json.loads(text)
                if not isinstance(answer, dict):
                    raise ValueError
            except ValueError:
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
                'world': self.world, 'trainable': True}

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
        # The service keeps every episode it served; drop ours so long runs stay bounded.
        if self.episode is not None:
            with self.service.lock:
                self.service.episodes.pop(self.episode, None)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._forget()
        self.server.shutdown()
        self.server.server_close()
