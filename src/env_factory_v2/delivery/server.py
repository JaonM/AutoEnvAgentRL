"""Public tool-only API: private contract and user state stay on the server."""
from __future__ import annotations

import json
import copy
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from env_factory_v2.contracts.contract import ContractError, require
from env_factory_v2.execution.reward import grade
from env_factory_v2.execution.runtime import Sandbox
from env_factory_v2.execution.user_simulator import validate_language_event


def serve(source, *, host='127.0.0.1', port=8080, agent=None, user_mode='structured', user_agent_factory=None):
    require(user_mode in {'structured', 'free_dialogue'}, 'invalid user mode')
    episodes = {}
    user_agents = {}
    lock = threading.RLock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value):
            body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            with lock:
                if self.path == '/health':
                    return self.reply(200, {'version': '2.0', 'status': 'ready'})
                sandbox = episodes.get(self.headers.get('X-Episode'))
                if self.path == '/time' and sandbox is not None:
                    return self.reply(200, sandbox.observe_time())
                if self.path == '/user-state' and sandbox is not None and user_mode == 'structured':
                    return self.reply(200, sandbox.user.snapshot())
                return self.reply(404, {'error': 'unknown route/episode'})

        def do_POST(self):
            self.connection.settimeout(10)
            try:
                size = int(self.headers.get('Content-Length', '0'))
                require(0 < size <= 1024 * 1024, 'request size limit')
                body = json.loads(self.rfile.read(size))
                require(isinstance(body, dict), 'object request required')
                with lock:
                    if self.path == '/reset':
                        require(not body, 'public reset cannot mutate private fixtures')
                        require(len(episodes) < 128, 'episode capacity reached; delete completed episodes')
                        identity = secrets.token_hex(24)
                        sandbox = Sandbox(source)
                        public = copy.deepcopy(sandbox.compiled['public'])
                        if user_mode == 'free_dialogue':
                            from env_factory_v2.execution.free_dialogue import DialogueUser
                            sandbox.user = DialogueUser(sandbox.source)
                            if sandbox.source['category'] != 'direct_response':
                                if user_agent_factory:
                                    user_agents[identity] = user_agent_factory(identity)
                                else:
                                    require(agent is not None, 'free dialogue needs native Code Agent; no fallback')
                                    from env_factory_v2.native.code_agent import NativeCodeAgent
                                    user_agents[identity] = NativeCodeAgent(base_url=agent.base_url, token=agent.token,
                                        timeout=agent.timeout, evidence_root=agent.evidence_root / ('user-' + identity))
                            public.pop('user_stages', None)
                            public.pop('user_statechart', None)
                            public.pop('success', None)
                            public['runtime_api'].pop('user-state', None)
                            public['user_interaction'] = {'mode': 'free_dialogue', 'api': '/chat',
                                'description': 'Send natural language; user facts are obtained through conversation.'}
                        episodes[identity] = sandbox
                        return self.reply(200, {'episode': identity, 'public': public})
                    sandbox = episodes.get(self.headers.get('X-Episode'))
                    require(sandbox is not None, 'unknown episode')
                    if self.path == '/chat':
                        require(user_mode == 'free_dialogue' and self.headers.get('X-Episode') in user_agents,
                                'free dialogue is not available')
                        require(set(body) <= {'message', 'proposal', 'source_id'} and 'message' in body, 'invalid chat request')
                        from env_factory_v2.execution.free_dialogue import converse
                        result = converse(sandbox, user_agents[self.headers['X-Episode']], **body)
                    elif self.path == '/tool':
                        require(set(body) <= {'tool', 'arguments', 'id', 'cursor', 'authorization'} and {'tool', 'arguments', 'id'} <= set(body), 'invalid tool request')
                        if user_mode == 'free_dialogue' and sandbox.tools.get(body['tool'], {}).get('kind') == 'transaction':
                            try:
                                sandbox.user.check_token(body.get('authorization'), body['tool'], sandbox.resolve(body['arguments']))
                            except ContractError:
                                sandbox.violations.append('unauthorized_write')
                                raise
                        result = sandbox.call(body['tool'], body['arguments'], call_id=body['id'], cursor=body.get('cursor'))
                    elif self.path == '/user':
                        require(user_mode == 'structured', 'structured user events are unavailable in free dialogue')
                        require(set(body) <= {'stage', 'question', 'proposal', 'source_id'} and 'question' in body, 'invalid user request')
                        natural = 'stage' not in body
                        if natural:
                            require(agent is not None, 'natural-language interpretation requires native Code Agent')
                            event = validate_language_event(agent.interpret_user(body['question'], sandbox.compiled['public']['user_stages']), sandbox.user)
                            body['stage'] = event['stage_id']
                        result = sandbox.interact(body['stage'], question=body['question'], proposal=body.get('proposal'), source_id=body.get('source_id'))
                        if agent is not None and natural:
                            result = agent.express_user(result)
                    elif self.path == '/answer':
                        result = grade(sandbox, body, judge=agent.judge if agent is not None else None)
                    elif self.path == '/wait':
                        require(set(body) == {'seconds'}, 'invalid wait request')
                        result = sandbox.wait(body['seconds'])
                    elif self.path == '/collect':
                        require(set(body) == {'pages', 'id'}, 'invalid page collection request')
                        result = sandbox.collect(body['pages'], call_id=body['id'])
                    elif self.path == '/revoke':
                        require(user_mode == 'structured', 'withdrawal requires a user dialogue event')
                        require(set(body) == {'stage'}, 'invalid revoke request')
                        result = sandbox.revoke(body['stage'])
                    elif self.path == '/delete':
                        require(not body, 'invalid delete request')
                        del episodes[self.headers['X-Episode']]
                        user_agents.pop(self.headers['X-Episode'], None)
                        result = {'deleted': True}
                    else:
                        return self.reply(404, {'error': 'unknown route'})
                    return self.reply(200, result)
            except (ContractError, ValueError, TypeError, KeyError) as exc:
                self.reply(400, {'error': str(exc)})
            except Exception:
                self.reply(503, {'error': 'native service unavailable; no fallback',
                    'error_code': 'user_simulation_failure' if self.path == '/chat' else 'native_service_failure',
                    'trainable': False})

    server = ThreadingHTTPServer((host, port), Handler)
    # Trusted in-process trainer continuation; not an HTTP endpoint.
    server.episodes, server.episode_lock = episodes, lock
    return server
