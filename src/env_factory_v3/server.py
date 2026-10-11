"""Episode service over HTTP (stdlib only).

POST /reset   {"world": optional name, "seed": optional int}  -> {"episode", "world", "instructions"}
POST /step    {"episode", "action": {...}}                     -> {"ok", "result" | "error", "now"}
        actions: {"type": "tool", "name", "args"} | {"type": "ask", "key"}
                 {"type": "confirm", "key", "proposal": {"tool", "args"}} | {"type": "wait", "seconds"}
                 {"type": "answer", "answer"}
POST /reward  {"episode"}                                      -> grade (after the answer)
GET  /health
Private facts, the oracle and source tables are never exposed; a business rejection is an
observation (ok=false, kind=business), not a transport error.
"""
from __future__ import annotations

import copy
import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from env_factory_v3 import reward, spec
from env_factory_v3.runtime import BusinessError, Episode, EpisodeFinished, tool_specs
from env_factory_v3.worlds import variants


class Service:
    def __init__(self, env, oracle, schema, request, *, valid_worlds=None, perturbed=20):
        self.env, self.oracle, self.schema, self.request = env, oracle, schema, request
        worlds = {name: (data, facts) for name, data, facts in
                  variants(request['tables'], request['user_facts'], perturbed=perturbed)}
        names = valid_worlds or ['original']
        self.worlds = {name: worlds[name] for name in names if name in worlds}
        self.order = [name for name in names if name in self.worlds]
        self.episodes, self.lock = {}, threading.Lock()
        tools = env.make_tools()
        self.instructions = {
            'title': request['title'], 'user_request': request['user_request'],
            'business_rules': request['business_rules'], 'tools': tool_specs(tools),
            'user_topics': spec.fact_topics(request), 'answer_schema': schema,
            'time': {'start': request['time']['start']},
            'protocol': 'Use ask to learn user facts and confirm before every write; '
                        'pass the returned consent token to that exact write; finish with answer.'}

    def reset(self, world=None, seed=None):
        name = world if world is not None else self.order[(seed or 0) % len(self.order)]
        if name not in self.worlds:
            raise KeyError(f'unknown world {name}')
        data, facts = self.worlds[name]
        start = self.request['time']['start']
        expected = self.oracle.expected(copy.deepcopy(data), copy.deepcopy(facts), start)
        episode = Episode(self.env.make_tools(), data, facts, start=start)
        identifier = uuid.uuid4().hex
        with self.lock:
            self.episodes[identifier] = {'episode': episode, 'expected': expected, 'world': name, 'data': data}
        return {'episode': identifier, 'world': name, 'instructions': self.instructions}

    def _get(self, identifier):
        with self.lock:
            if identifier not in self.episodes:
                raise KeyError('unknown episode')
            return self.episodes[identifier]

    def step(self, identifier, action):
        episode = self._get(identifier)['episode']
        kind = action.get('type')
        try:
            if kind == 'tool':
                result = episode.call(action['name'], **(action.get('args') or {}))
            elif kind == 'ask':
                result = episode.ask(action['key'])
            elif kind == 'confirm':
                result = episode.confirm(action['key'], action['proposal'])
            elif kind == 'wait':
                result = episode.wait(action['seconds'])
            elif kind == 'answer':
                result = episode.submit(action.get('answer'))
            else:
                return {'ok': False, 'kind': 'invalid', 'error': f'unknown action type {kind}', 'now': episode.now}
        except BusinessError as exc:
            return {'ok': False, 'kind': 'business', 'error': str(exc), 'now': episode.now}
        except EpisodeFinished as exc:
            return {'ok': False, 'kind': 'finished', 'error': str(exc), 'now': episode.now}
        except (KeyError, ValueError, TypeError) as exc:
            return {'ok': False, 'kind': 'invalid', 'error': str(exc), 'now': episode.now}
        return {'ok': True, 'result': result, 'now': episode.now}

    def reward(self, identifier):
        record = self._get(identifier)
        episode = record['episode']
        if not episode.finished:
            return {'error': 'episode has no answer yet'}
        reference_calls = record.get('reference_calls')
        if reference_calls is None:
            from env_factory_v3.runtime import run_reference
            data, facts = self.worlds[record['world']]
            reference_calls = run_reference(self.env, data, facts, start=self.request['time']['start'])['tool_calls']
            record['reference_calls'] = reference_calls
        grade = reward.grade(episode.result(), record['expected'], self.schema,
                             initial=record['data'], reference_calls=reference_calls)
        grade.pop('state_diff', None)
        return {**grade, 'world': record['world']}


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, body):
            payload = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            return

        def do_GET(self):
            if self.path == '/health':
                return self._send(200, {'ok': True, 'worlds': len(service.order)})
            return self._send(404, {'error': 'not found'})

        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/reset':
                    return self._send(200, service.reset(body.get('world'), body.get('seed')))
                if self.path == '/step':
                    return self._send(200, service.step(body['episode'], body['action']))
                if self.path == '/reward':
                    return self._send(200, service.reward(body['episode']))
                return self._send(404, {'error': 'not found'})
            except KeyError as exc:
                return self._send(404, {'error': str(exc)})
            except Exception as exc:
                return self._send(500, {'error': f'{type(exc).__name__}: {exc}'})
    return Handler


def start(service, host='127.0.0.1', port=0):
    server = ThreadingHTTPServer((host, port), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class HttpAPI:
    """Client-side api with the same call/ask/confirm surface as an Episode."""

    BusinessError = BusinessError

    def __init__(self, base_url, episode):
        self.base_url, self.episode = base_url, episode

    def _post(self, path, body):
        request = Request(self.base_url + path, data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    def _step(self, action):
        reply = self._post('/step', {'episode': self.episode, 'action': action})
        if not reply['ok']:
            if reply['kind'] == 'business':
                raise BusinessError(reply['error'])
            raise KeyError(reply['error'])
        return reply['result']

    def call(self, name, **args):
        return self._step({'type': 'tool', 'name': name, 'args': args})

    def ask(self, key):
        return self._step({'type': 'ask', 'key': key})

    def confirm(self, key, proposal):
        return self._step({'type': 'confirm', 'key': key, 'proposal': proposal})

    def wait(self, seconds):
        return self._step({'type': 'wait', 'seconds': seconds})


def replay_check(env, oracle, schema, request):
    """G5: the reference through the real HTTP API earns the same full reward."""
    service = Service(env, oracle, schema, request)
    server = start(service)
    try:
        base = f'http://127.0.0.1:{server.server_address[1]}'
        session = HttpAPI(base, None)
        reset = session._post('/reset', {'world': 'original'})
        session.episode = reset['episode']
        answer = env.reference(session)
        session._step({'type': 'answer', 'answer': answer})
        grade = session._post('/reward', {'episode': session.episode})
        leaked = '"user_facts"' in json.dumps(reset['instructions'], ensure_ascii=False)
        return {'passed': grade.get('success') is True and grade.get('reward') == 1.0 and not leaked,
                'reward': grade.get('reward'), 'instructions_leak_private': leaked}
    finally:
        server.shutdown()
        server.server_close()
