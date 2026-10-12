"""Episode service over HTTP (stdlib only).

POST /reset   {"world": optional name, "seed": optional int}  -> {"episode", "world", "instructions"}
POST /step    {"episode", "action": {...}}                     -> {"ok", "result" | "error", "now"}
        actions: {"type": "tool", "name", "args"} | {"type": "ask", "key"}
                 {"type": "confirm", "key", "proposal": {"tool", "args"}} | {"type": "wait", "seconds"}
                 {"type": "answer", "answer"}
                 natural user mode: {"type": "chat", "message", "proposal"?} replaces ask/confirm
POST /reward  {"episode"}                                      -> grade (after the answer)
POST /release {"episode"}                                      -> {"released": bool}
GET  /health
With V3_SERVICE_TOKEN set (serve) every POST needs `Authorization: Bearer <token>`.
Private facts, the oracle and source tables are never exposed; a business rejection is an
observation (ok=false, kind=business), not a transport error, and so is a crash of an authored
tool (ok=false, kind=tool_error). Only platform and model-provider failures return 500.
"""
from __future__ import annotations

import copy
import hmac
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen

from env_factory_v3.core import reward, spec
from env_factory_v3.core.worlds import variants
from env_factory_v3.runtime import BusinessError, Episode, EpisodeFinished, ToolError, tool_specs


# Free dialogue ends when the user says goodbye or after this many agent messages in a row that
# neither call a tool nor propose a write. The agent is then told once to submit; if it talks on,
# the episode closes: a business task without its structured answer earns 0, a subjective task is
# graded on what the agent last told the user.
IDLE_LIMIT = 3


class Service:
    def __init__(self, env, oracle, schema, request, *, valid_worlds=None, perturbed=20, user_mode='structured',
                 user_client=None, rubric=None, judge=None):
        if user_mode not in ('structured', 'natural'):
            raise ValueError(f'unknown user mode {user_mode}')
        if user_mode == 'natural' and user_client is None:
            raise ValueError('natural user mode needs a user simulator client')
        self.env, self.oracle, self.schema, self.request = env, oracle, schema, request
        self.user_mode, self.user_client = user_mode, user_client
        # business: graded exactly against the oracle; rubric: graded by criteria and a judge model.
        self.reward_type = request.get('reward_type', 'business')
        if self.reward_type == 'rubric' and rubric is None:
            raise ValueError('a rubric task needs its rubric')
        self.rubric, self._judge = rubric, judge
        if user_mode == 'natural':
            from env_factory_v3.certify.gates import approval_problems
            problems = approval_problems(env, request)
            if problems:
                raise ValueError(f'bundle is not ready for the natural user: {problems[:3]}')
        self.approvals = {t['key'] for t in spec.fact_topics(request) if t['kind'] == 'approval'}
        worlds = {name: (data, facts) for name, data, facts in
                  variants(request['tables'], request['user_facts'], perturbed=perturbed)}
        names = valid_worlds or ['original']
        self.worlds = {name: worlds[name] for name in names if name in worlds}
        self.order = [name for name in names if name in self.worlds]
        self.episodes, self.lock = {}, threading.Lock()
        self.last_request = time.monotonic()
        tools = env.make_tools()
        self.instructions = {
            'title': request['title'], 'user_request': request['user_request'],
            # The agent sees the business-language policy when the task has one, never the spec's internals.
            'business_rules': request.get('agent_policy') or request['business_rules'], 'tools': tool_specs(tools),
            'user_topics': spec.fact_topics(request), 'answer_schema': schema,
            'time': {'start': request['time']['start'],
                     'advancing': 'scheduled_events' in request['tables']},
            'protocol': 'Use ask to learn user facts and confirm before every write; '
                        'pass the returned consent token to that exact write; finish with answer.',
            'user_mode': user_mode}
        if user_mode == 'natural':
            # The agent must find out what to ask; no topic keys, no approval keys.
            self.instructions.pop('user_topics')
            self.instructions['protocol'] = ('Talk with the user in natural language (chat). Before every write, '
                                             'propose that exact write in a chat message; the reply carries a '
                                             'consent token when the user agrees. Finish with answer.')

    def reset(self, world=None, seed=None):
        name = world if world is not None else self.order[(seed or 0) % len(self.order)]
        if name not in self.worlds:
            raise KeyError(f'unknown world {name}')
        data, facts = self.worlds[name]
        start = self.request['time']['start']
        expected = (self.oracle.expected(copy.deepcopy(data), copy.deepcopy(facts), start)
                    if self.reward_type == 'business' else None)
        episode = Episode(self.env.make_tools(), data, facts, start=start)
        identifier = uuid.uuid4().hex
        record = {'episode': episode, 'expected': expected, 'world': name, 'data': data, 'facts': facts}
        if self.user_mode == 'natural':
            from env_factory_v3.serving.user_sim import UserSimulator
            record['user'] = UserSimulator(self.user_client, self.request, facts, self.approvals)
        with self.lock:
            self.episodes[identifier] = record
        return {'episode': identifier, 'world': name, 'instructions': self.instructions}

    def _get(self, identifier):
        with self.lock:
            if identifier not in self.episodes:
                raise KeyError('unknown episode')
            return self.episodes[identifier]

    def chat(self, record, action):
        """One natural-language turn. Consent is decided here, never by the simulator."""
        episode, proposal = record['episode'], action.get('proposal')
        token, decision = None, None
        if proposal is not None:
            try:
                key = self.env.approval_key(proposal['tool'], copy.deepcopy(proposal['args']),
                                            copy.deepcopy(record['facts']))
            except Exception as exc:
                raise ValueError(f'proposal does not name an approvable write: {exc}') from exc
            if key not in self.approvals:
                raise ValueError(f'approval_key returned {key!r}, not an approval topic')
            token = episode.confirm(key, proposal)
            decision = token is not None
        user = record['user']
        reply = user.reply(str(action.get('message') or ''), decision)
        episode.trace.append({'kind': 'chat', 'message': action.get('message'), 'reply': reply,
                              'proposal': copy.deepcopy(proposal), 'time': episode.now})
        record['idle'] = 0 if proposal is not None else record.get('idle', 0) + 1
        result = {'reply': reply, **({'consent': token} if proposal is not None else {})}
        if getattr(user, 'ended', False) or record['idle'] >= IDLE_LIMIT:
            result.update(self._end(record, str(action.get('message') or '')))
        return result

    def _end(self, record, last_message):
        """The user has left: warn once, then close the episode."""
        if not record.get('warned'):
            record['warned'] = True
            return {'ended': 'warning'}
        record['closed'] = True
        episode = record['episode']
        if self.reward_type == 'rubric' and last_message.strip():
            episode.submit({'reply': last_message})
            return {'ended': 'closed', 'submitted': True}
        return {'ended': 'closed', 'submitted': False}

    def step(self, identifier, action):
        record = self._get(identifier)
        episode = record['episode']
        kind = action.get('type')
        if self.user_mode == 'natural' and kind in ('ask', 'confirm'):
            return {'ok': False, 'kind': 'invalid', 'error': 'use chat to talk with the user', 'now': episode.now}
        if self.user_mode == 'structured' and kind == 'chat':
            return {'ok': False, 'kind': 'invalid', 'error': 'chat needs the natural user mode', 'now': episode.now}
        if record.get('closed'):
            return {'ok': False, 'kind': 'finished', 'error': 'the user has left the conversation', 'now': episode.now}
        if kind in ('tool', 'wait'):
            record['idle'] = 0
        try:
            if kind == 'chat':
                result = self.chat(record, action)
            elif kind == 'tool':
                result = episode.call(action['name'], **(action.get('args') or {}))
            elif kind == 'ask':
                result = episode.ask(action['key'])
            elif kind == 'confirm':
                result = episode.confirm(action['key'], action['proposal'])
            elif kind == 'wait':
                if not self.instructions['time']['advancing']:
                    return {'ok': False, 'kind': 'invalid', 'error': 'the clock is fixed for this task',
                            'now': episode.now}
                result = episode.wait(action['seconds'])
            elif kind == 'answer':
                result = episode.submit(action.get('answer'))
            else:
                return {'ok': False, 'kind': 'invalid', 'error': f'unknown action type {kind}', 'now': episode.now}
        except BusinessError as exc:
            return {'ok': False, 'kind': 'business', 'error': str(exc), 'now': episode.now}
        except ToolError as exc:  # the sandbox's own code failed: an observation, never a 500
            return {'ok': False, 'kind': 'tool_error', 'error': str(exc), 'now': episode.now}
        except EpisodeFinished as exc:
            return {'ok': False, 'kind': 'finished', 'error': str(exc), 'now': episode.now}
        except (KeyError, ValueError, TypeError) as exc:
            return {'ok': False, 'kind': 'invalid', 'error': str(exc), 'now': episode.now}
        return {'ok': True, 'result': result, 'now': episode.now}

    def release(self, identifier):
        with self.lock:
            return {'released': self.episodes.pop(identifier, None) is not None}

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
        if self.reward_type == 'rubric':
            from env_factory_v3.core import rubric
            grade = rubric.grade(episode.result(), self.rubric, self.request, record['facts'], self.judge(),
                                 reference_calls=reference_calls)
        else:
            result = episode.result()
            # In free dialogue the agent must also have told the user its results (reward.told).
            speech = ([e.get('message') for e in result['trace'] if e.get('kind') == 'chat']
                      if self.user_mode == 'natural' else None)
            grade = reward.grade(result, record['expected'], self.schema, initial=record['data'],
                                 reference_calls=reference_calls, speech=speech)
            grade.pop('state_diff', None)
        return {**grade, 'world': record['world'], 'reward_type': self.reward_type}

    def judge(self):
        """The rubric judge, created on first use from V3_JUDGE_* (falling back to LLM_*)."""
        if self._judge is None:
            from env_factory_v3.core.llm import ChatClient
            from env_factory_v3.core.rubric import Judge
            self._judge = Judge(ChatClient('V3_JUDGE'))
        return self._judge


def make_handler(service, token=None):
    class Handler(BaseHTTPRequestHandler):
        def _authorized(self):
            return token is None or hmac.compare_digest(self.headers.get('Authorization', ''), f'Bearer {token}')

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
            service.last_request = time.monotonic()  # the serving container exits when nobody calls it
            if not self._authorized():
                return self._send(401, {'error': 'missing or wrong service token'})
            try:
                length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(length) or b'{}')
                if self.path == '/reset':
                    return self._send(200, service.reset(body.get('world'), body.get('seed')))
                if self.path == '/step':
                    return self._send(200, service.step(body['episode'], body['action']))
                if self.path == '/reward':
                    return self._send(200, service.reward(body['episode']))
                if self.path == '/release':
                    return self._send(200, service.release(body['episode']))
                return self._send(404, {'error': 'not found'})
            except KeyError as exc:
                return self._send(404, {'error': str(exc)})
            except Exception as exc:
                return self._send(500, {'error': f'{type(exc).__name__}: {exc}'})
    return Handler


def start(service, host='127.0.0.1', port=0, token=None):
    """Serve in a daemon thread; with a token every POST needs `Authorization: Bearer <token>`."""
    server = ThreadingHTTPServer((host, port), make_handler(service, token))
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


def replay_check(env, oracle, schema, request, *, rubric=None, judge=None):
    """G5: the reference through the real HTTP API earns the same full reward (a passing rubric
    score for rubric tasks)."""
    service = Service(env, oracle, schema, request, rubric=rubric, judge=judge)
    server = start(service)
    try:
        base = f'http://127.0.0.1:{server.server_address[1]}'
        session = HttpAPI(base, None)
        reset = session._post('/reset', {'world': 'original'})
        session.episode = reset['episode']
        try:
            answer = env.reference(session)
        except Exception as exc:  # the reference depends on something the service does not offer
            return {'passed': False, 'reward': None, 'error': f'{type(exc).__name__}: {exc}'[:300]}
        session._step({'type': 'answer', 'answer': answer})
        grade = session._post('/reward', {'episode': session.episode})
        leaked = '"user_facts"' in json.dumps(reset['instructions'], ensure_ascii=False)
        full = grade.get('success') is True and (grade.get('reward') == 1.0 or rubric is not None)
        return {'passed': full and not leaked, 'reward': grade.get('reward'), 'instructions_leak_private': leaked}
    finally:
        server.shutdown()
        server.server_close()
