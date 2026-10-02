"""Docker sandbox HTTP service with isolated sessions and trainer-only snapshots.

Mounted into qualified sandbox images by rl.kubernetes; uses their existing app.
"""
import base64
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time


def artifact_identity(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def snapshot_database(path):
    with sqlite3.connect(path) as source, sqlite3.connect(':memory:') as target:
        source.backup(target)
        return base64.b64encode(target.serialize()).decode('ascii')


def restore_database(path, value):
    # WAL-mode serialized databases cannot be reopened as SQLite :memory:.
    # A temporary file preserves SQLite's journal semantics on Linux and macOS.
    with tempfile.TemporaryDirectory(prefix='rl-restore-') as directory:
        restored = Path(directory) / 'snapshot.sqlite3'
        restored.write_bytes(base64.b64decode(value, validate=True))
        with sqlite3.connect(restored) as source, sqlite3.connect(path) as target:
            source.backup(target)


class SessionService:
    def __init__(self, factory, root, identity, key, *, max_sessions=128, ttl=3600):
        self.factory, self.root, self.identity, self.key = factory, Path(root), identity, key
        self.root.mkdir(parents=True, exist_ok=True)
        self.sessions, self.lock = {}, threading.RLock()
        self.max_sessions, self.ttl = max_sessions, ttl

    def handle(self, method, path, body, headers):
        if method == 'GET' and path == '/health':
            return 200, {'status': 'ok', 'artifact_identity': self.identity}, {}
        if not hmac.compare_digest(headers.get('Authorization', ''), 'Bearer ' + self.key):
            return 401, {'error': 'unauthorized'}, {}
        session = headers.get('X-RL-Session', '')
        if not re.fullmatch(r'[a-f0-9]{32}', session):
            return 400, {'error': 'invalid session'}, {}
        with self.lock:
            for name, item in list(self.sessions.items()):
                if time.monotonic() - item['used'] > self.ttl and item['lock'].acquire(blocking=False):
                    try:
                        self._remove(name)
                    finally:
                        item['lock'].release()
            item = self.sessions.get(session)
            if item is None:
                if path not in ('/v1/reset', '/_rl/restore') or method != 'POST':
                    return 409, {'error': 'session lost; restore committed snapshot'}, {}
                if len(self.sessions) >= self.max_sessions:
                    return 503, {'error': 'session capacity exhausted'}, {}
                directory = self.root / session
                directory.mkdir(exist_ok=True)
                item = {'app': self.factory(db_path=directory / 'episode.sqlite3'),
                        'lock': threading.RLock(), 'used': time.monotonic(), 'directory': directory}
                self.sessions[session] = item
            item['lock'].acquire()
            item['used'] = time.monotonic()
        try:
            database = item['app'].episode_store.db_path
            if path == '/_rl/snapshot' and method == 'GET':
                return 200, {'database': snapshot_database(database)}, {}
            if path == '/_rl/restore' and method == 'POST':
                restore_database(database, body['database'])
                return 200, {'restored': True}, {}
            if path == '/_rl/close' and method == 'POST':
                # Remove after releasing the session lock to preserve lock ordering.
                return 200, {'closed': True}, {}
            return item['app'].handle(method, path, body, headers)
        finally:
            item['used'] = time.monotonic()
            item['lock'].release()
            if path == '/_rl/close' and method == 'POST':
                with self.lock, item["lock"]:
                    if self.sessions.get(session) is item:
                        self._remove(session)

    def _remove(self, session):
        item = self.sessions.pop(session, None)
        if item:
            close = getattr(item['app'], 'close', None)
            if callable(close):
                close()
            shutil.rmtree(item['directory'], ignore_errors=True)


def make_server(service, host='0.0.0.0', port=8000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Never log authorization headers or trainer-only snapshot bodies.

        def dispatch(self):
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if size < 0 or size > 64 * 1024 * 1024:
                    self.send_error(413)
                    return
                body = json.loads(self.rfile.read(size)) if size else None
                status, value, _ = service.handle(self.command, self.path, body, self.headers)
            except (ValueError, KeyError, TypeError):
                status, value = 400, {'error': 'invalid request'}
            except Exception as error:
                print(json.dumps({"event": "request_failed", "path": self.path,
                                  "error_type": type(error).__name__}), file=sys.stderr, flush=True)
                status, value = 500, {'error': 'sandbox service error'}
            payload = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        do_GET = dispatch
        do_POST = dispatch
    return ThreadingHTTPServer((host, port), Handler)


def main():
    root = Path(os.environ.get('RL_SANDBOX_ROOT', '/app')).resolve()
    hashes = json.loads(Path(os.environ['RL_ARTIFACT_HASHES_FILE']).read_text())
    if not hashes:
        raise ValueError('missing qualified artifact hashes')
    for name, digest in hashes.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'container artifact identity mismatch: {name}')
    os.environ.pop('SANDBOX_EVALUATOR_MOCK', None)
    os.environ['SANDBOX_MUTATION_MODE'] = 'disabled'
    sys.path.insert(0, str(root))
    from app import create_app
    service = SessionService(create_app, '/tmp/rl-sessions', artifact_identity(hashes),
                             os.environ['SANDBOX_TRAINER_API_KEY'])
    with make_server(service) as server:
        server.serve_forever()


if __name__ == '__main__':
    main()
