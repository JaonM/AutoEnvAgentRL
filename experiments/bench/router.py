"""Sticky router over several serve_openai.py processes on the same GPU.

A single process decodes one request at a time; with thinking, decoding dominates and the GPU is far
from busy, so several processes serve in parallel. A conversation always goes to the same process
(hash of its first three messages: on τ²-bench the system prompt and the agent's greeting are shared
by every task of a domain, the user's opening is not), so that process's prefix cache keeps hitting.

    python experiments/bench/router.py --port 8766 --backends 8771 8772 8773
"""
from __future__ import annotations

import argparse
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--backends', type=int, nargs='+', required=True)
    args = parser.parse_args()
    opener = build_opener(ProxyHandler({}))

    def backend(body):
        messages = body.get('messages') or []
        key = json.dumps([m.get('content') for m in messages[:3]], ensure_ascii=False)
        return args.backends[int(hashlib.sha256(key.encode()).hexdigest(), 16) % len(args.backends)]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            return

        def relay(self, status, payload):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            try:
                for port in args.backends:
                    opener.open(f'http://127.0.0.1:{port}/health', timeout=5).read()
                return self.relay(200, b'{"ok": true}')
            except OSError:
                return self.relay(503, b'{"ok": false}')

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get('Content-Length') or 0))
            port = backend(json.loads(raw or b'{}'))
            request = Request(f'http://127.0.0.1:{port}{self.path}', data=raw, method='POST',
                              headers={'Content-Type': 'application/json'})
            try:
                with opener.open(request, timeout=1800) as response:
                    return self.relay(response.status, response.read())
            except HTTPError as error:
                return self.relay(error.code, error.read())

    ThreadingHTTPServer.request_queue_size = 128
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
