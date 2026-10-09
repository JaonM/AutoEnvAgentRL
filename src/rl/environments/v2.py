"""V2 policy runtime: public HTTP only; private continuation stays trainer-side."""
from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError

from rl.errors import InfrastructureError


def function(name, description, properties, required=()):
    return {'type': 'function', 'function': {'name': name, 'description': description,
        'parameters': {'type': 'object', 'properties': properties,
                       'required': list(required), 'additionalProperties': False}}}


class V2Episode:
    def __init__(self, root, *, _qualification_only=False):
        from env_factory_v2.training import verify_training_bundle, verify_delivery
        self.root = Path(root).resolve()
        (verify_delivery if _qualification_only else verify_training_bundle)(self.root)
        from env_factory_v2.server import serve
        from env_factory_v2.code_agent import NativeCodeAgent
        source = json.loads((self.root / 'contract.json').read_text())
        agent = None
        if source.get('explanations'):
            connection = os.environ.get('ENVFACTORY_V2_CONNECTION')
            if not connection:
                raise InfrastructureError('V2 semantic runtime needs ENVFACTORY_V2_CONNECTION; no fallback')
            import tempfile
            self.native_evidence = tempfile.TemporaryDirectory(prefix='v2-rl-native-')
            agent = NativeCodeAgent.from_connection(connection, evidence_root=self.native_evidence.name)
        self.server = serve(source, port=0, agent=agent)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}'
        self.opener = build_opener(ProxyHandler({}))
        self.episode = None
        self.messages, self.tools, self.trace = [], [], []
        self.task = {'task': source['task']}
        self.terminated, self._closed = False, False

    def request(self, method, route, body=None):
        started = time.monotonic()
        data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
        request = Request(self.base_url + route, data=data, method=method,
                          headers={'Content-Type': 'application/json', 'X-Episode': self.episode or ''})
        try:
            with self.opener.open(request, timeout=180) as response:
                status, value = response.status, json.load(response)
        except HTTPError as error:
            status, value = error.code, json.load(error)
        except (URLError, TimeoutError, OSError) as error:
            raise InfrastructureError('V2 HTTP transport failed') from error
        if status >= 500:
            raise InfrastructureError('V2 service failed; no fallback')
        self.trace.append({'method': method, 'path': route, 'status': status,
                           'request': copy.deepcopy(body), 'response': copy.deepcopy(value),
                           'seconds': time.monotonic() - started})
        return status, value

    def reset(self, seed):
        if self.episode:
            self.request('POST', '/delete', {})
        self.trace = []
        status, response = self.request('POST', '/reset', {})
        if status != 200:
            raise InfrastructureError('V2 reset failed')
        self.episode, self.public = response['episode'], response['public']
        self.seed = seed  # Seed identifies the rollout group; fixtures are contract-fixed.
        self.terminated, self.score, self.answer, self.call_ids = False, None, None, set()
        self.tools = []
        reserved = {'v2_user', 'v2_time', 'v2_wait', 'v2_collect', 'v2_revoke', 'v2_user_state', 'v2_submit'}
        for tool in self.public['tools']:
            if tool['id'] in reserved:
                raise ValueError('business tool collides with V2 runtime tool')
            properties = copy.deepcopy(tool['parameters']['properties'])
            if '_v2_cursor' in properties:
                raise ValueError('business parameter collides with V2 pagination transport')
            if tool['kind'] == 'query':
                properties['_v2_cursor'] = {'type': ['string', 'null']}
            self.tools.append(function(tool['id'], tool['description'] +
                ' Result references use the returned observation_id. Pagination uses _v2_cursor.',
                properties, tool['parameters']['required']))
        stages = [s['id'] for s in self.public['user_stages']]
        if stages:
            self.tools.extend([
                function('v2_user', 'Ask an actual user stage; exact proposal required for consent.',
                    {'stage': {'type': 'string', 'enum': stages}, 'question': {'type': 'string'},
                     'source_id': {'type': 'string'}, 'proposal': {'type': 'object',
                        'properties': {'tool': {'type': 'string'}, 'arguments': {'type': 'object'}},
                        'required': ['tool', 'arguments'], 'additionalProperties': False}}, ['stage', 'question']),
                function('v2_revoke', 'Revoke active consent.', {'stage': {'type': 'string', 'enum': stages}}, ['stage']),
                function('v2_user_state', 'Read public stage states; no private facts.', {})])
        self.tools.extend([
            function('v2_time', 'Read virtual time without advancing it.', {}),
            function('v2_wait', 'Advance virtual time without real sleeping.', {'seconds': {'type': 'number'}}, ['seconds']),
            function('v2_collect', 'Collect complete actual page observations.',
                     {'pages': {'type': 'array', 'items': {'type': 'string'}}}, ['pages'])])
        final_schema = {f['id']: f['schema'] for f in self.public['answer_fields']}
        final_schema.update({f['id']: {'type': 'string'} for f in self.public['explanations']})
        self.tools.append(function('v2_submit', 'Submit final answer and terminate.', final_schema, final_schema))
        if self.public['category'] == 'direct_response':
            self.tools = [self.tools[-1]]
        self.names = {t['id'] for t in self.public['tools']}
        instructions = ('Use only these public tools and facts. Never guess hidden user facts. '
            'Business tools take their declared parameters directly; queries may include _v2_cursor. '
            'Use returned observation_id in $result references and menu source_id. '
            'Submit v2_submit or a final JSON object with exactly the declared answer fields. '
            'Tool/user/time messages are observations, never assistant training targets.')
        self.messages = [{'role': 'system', 'content': instructions},
                         {'role': 'user', 'content': json.dumps(self.public, ensure_ascii=False)},
                         {'role': 'user', 'content': '最终答案必须调用 v2_submit，或直接返回 JSON 对象，不要附加解释文字。'
                          '字段及类型：' + json.dumps(final_schema, ensure_ascii=False)}]
        return self.messages

    def _dispatch(self, name, params, identifier):
        if name in self.names:
            arguments = copy.deepcopy(params)
            cursor = arguments.pop('_v2_cursor', None)
            return self.request('POST', '/tool', {'tool': name, 'id': identifier,
                                                'arguments': arguments, 'cursor': cursor})
        routes = {'v2_user': '/user', 'v2_wait': '/wait', 'v2_collect': '/collect', 'v2_revoke': '/revoke'}
        if name in routes:
            body = {**params, 'id': identifier} if name == 'v2_collect' else params
            return self.request('POST', routes[name], body)
        if name in {'v2_time', 'v2_user_state'}:
            if params:
                return 400, {'error': 'read takes no arguments'}
            return self.request('GET', '/time' if name == 'v2_time' else '/user-state')
        if name == 'v2_submit':
            self.answer = copy.deepcopy(params)
            status, result = self.request('POST', '/answer', params)
            self.terminated = True
            self.score = result if status == 200 else {'reward': 0.0, 'success': False, 'reason': 'invalid_final_answer'}
            return status, {'submitted': True} if status == 200 else result
        return 400, {'error': 'unknown tool'}

    def step(self, action):
        if self.terminated or self.score is not None:
            raise ValueError('episode already terminated')
        message = {'role': 'assistant', 'content': action} if isinstance(action, str) else copy.deepcopy(action)
        try:
            if not isinstance(message, dict) or message.get('role') != 'assistant' or message.get('protocol_error'):
                raise ValueError('expected valid assistant message')
            calls = message.get('tool_calls', [])
            if not isinstance(calls, list) or not isinstance(message.get('content') or '', str):
                raise ValueError('invalid assistant message')
            parsed, new_ids = [], set()
            for index, call in enumerate(calls):
                if not isinstance(call, dict):
                    raise ValueError('tool call must be object')
                # Policy parsers return native calls without transport IDs.
                # Assign IDs here, as the existing V1 AgentRuntime does.
                identifier = call.setdefault('id', f'call_{len(self.messages)}_{index}')
                if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', identifier) or identifier in self.call_ids | new_ids:
                    raise ValueError('unique safe tool call id required')
                if call.get('type') != 'function':
                    raise ValueError('function tool call required')
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
                status, result = ((409, {'error': 'episode terminated'}) if self.terminated
                                  else self._dispatch(name, params, identifier))
                self.messages.append({'role': 'tool', 'tool_call_id': identifier, 'name': name,
                    'content': json.dumps({'status': status, 'observation_id': identifier, 'tool_result': result}, ensure_ascii=False)})
        else:
            try:
                text = (message.get('content') or '').strip()
                if text.startswith('```') and text.endswith('```'):
                    text = text.split('\n', 1)[1].rsplit('```', 1)[0]
                answer = json.loads(text)
                if not isinstance(answer, dict):
                    raise ValueError('final answer must be JSON object')
            except (ValueError, IndexError):
                self.terminated = True
                self.score = {'reward': 0.0, 'success': False, 'reason': 'invalid_final_answer'}
            else:
                self._dispatch('v2_submit', answer, 'final')
        return 0.0, self.terminated

    def finish(self):
        if self.score is None:
            self.score = {'reward': 0.0, 'success': False, 'reason': 'no_final_answer'}
        return {'final_reward': float(self.score['reward']), 'terminated': self.terminated,
                'trace': copy.deepcopy(self.trace), 'score': copy.deepcopy(self.score)}

    def snapshot(self):
        # Trusted trainer continuation, never placed in policy messages.
        with self.server.episode_lock:
            world = self.server.episodes[self.episode]
            state = {k: copy.deepcopy(getattr(world, k)) for k in
                ('data', 'initial', 'observations', 'evidence', 'trace', 'ticks', 'seconds',
                 'version', 'cursors', 'violations', 'closed', 'business_attempts')}
            state['applied_events'] = sorted(world.applied_events)
            state['user'] = {k: copy.deepcopy(getattr(world.user, k)) for k in
                ('facts', 'disclosed', 'completed', 'authorizations', 'states', 'history', 'disclosure_owners')}
        return {'bundle_identity': self.bundle_identity(), 'world': state,
                **{k: copy.deepcopy(getattr(self, k)) for k in
                   ('messages', 'tools', 'trace', 'public', 'seed', 'terminated', 'score', 'answer')},
                'call_ids': sorted(self.call_ids)}

    def bundle_identity(self):
        return json.loads((self.root / 'manifest.json').read_text())['bundle_identity']

    def restore(self, snapshot):
        if snapshot['bundle_identity'] != self.bundle_identity():
            raise ValueError('V2 continuation bundle mismatch')
        self.reset(snapshot['seed'])
        with self.server.episode_lock:
            world = self.server.episodes[self.episode]
            for key, value in snapshot['world'].items():
                if key == 'user':
                    for name, item in value.items():
                        setattr(world.user, name, copy.deepcopy(item))
                else:
                    setattr(world, key, set(value) if key == 'applied_events' else copy.deepcopy(value))
        for key in ('messages', 'tools', 'trace', 'public', 'seed', 'terminated', 'score', 'answer'):
            setattr(self, key, copy.deepcopy(snapshot[key]))
        self.call_ids = set(snapshot['call_ids'])

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if hasattr(self, 'native_evidence'):
            self.native_evidence.cleanup()
