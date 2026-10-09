"""Public tool-only API: private contract and user state stay on the server."""
from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .contract import ContractError, require
from .reward import grade
from .runtime import Sandbox
from .user_simulator import validate_language_event


def serve(source, *, host='127.0.0.1', port=8080, agent=None):
    episodes = {}
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
                    return self.reply(200, sandbox.time())
                if self.path == '/user-state' and sandbox is not None:
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
                        episodes[identity] = sandbox
                        return self.reply(200, {'episode': identity, 'public': sandbox.compiled['public']})
                    sandbox = episodes.get(self.headers.get('X-Episode'))
                    require(sandbox is not None, 'unknown episode')
                    if self.path == '/tool':
                        require(set(body) <= {'tool', 'arguments', 'id', 'cursor'} and {'tool', 'arguments', 'id'} <= set(body), 'invalid tool request')
                        result = sandbox.call(body['tool'], body['arguments'], call_id=body['id'], cursor=body.get('cursor'))
                    elif self.path == '/user':
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
                        require(set(body) == {'stage'}, 'invalid revoke request')
                        result = sandbox.revoke(body['stage'])
                    elif self.path == '/delete':
                        require(not body, 'invalid delete request')
                        del episodes[self.headers['X-Episode']]
                        result = {'deleted': True}
                    else:
                        return self.reply(404, {'error': 'unknown route'})
                    return self.reply(200, result)
            except (ContractError, ValueError, TypeError, KeyError) as exc:
                self.reply(400, {'error': str(exc)})
            except Exception:
                self.reply(503, {'error': 'native service unavailable; no fallback'})

    server = ThreadingHTTPServer((host, port), Handler)
    # Trusted in-process trainer continuation; not an HTTP endpoint.
    server.episodes, server.episode_lock = episodes, lock
    return server
