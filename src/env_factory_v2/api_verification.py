"""Execute a positive trajectory through the real public HTTP API."""
from __future__ import annotations

import json
import threading
from urllib.request import Request, urlopen

from .contract import require
from .server import serve


def verify_api(source, report, *, agent=None):
    scenario = next(s for s in source['scenarios'] if s['expect_success'] and not s.get('mutations') and not s.get('user_overrides'))
    reference = next(r['first'] for r in report['records'] if r['kind'] == 'scenario' and r['id'] == scenario['id'])
    # This verifier is trusted and can use the reference answer. The agent-facing
    # service never publishes that answer recipe or the private source tables.
    server = serve(source, port=0, agent=agent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    episode = None
    def post(route, body):
        request = Request(base + route, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', **({'X-Episode': episode} if episode else {})})
        with urlopen(request, timeout=240) as response:
            return json.load(response)
    try:
        with urlopen(base + '/health', timeout=5) as response:
            require(json.load(response)['status'] == 'ready', 'container HTTP health failed')
        reset = post('/reset', {})
        episode = reset['episode']
        for step in scenario['steps']:
            kind = step['kind']
            if kind == 'user':
                post('/user', {key: value for key, value in step.items() if key != 'kind'})
            elif kind == 'call':
                post('/tool', {key: value for key, value in step.items() if key != 'kind'})
            elif kind == 'traverse':
                pages, cursor = [], None
                while True:
                    call_id = f'{step["id"]}_page{len(pages)}'
                    page = post('/tool', {'tool': step['tool'], 'arguments': step['arguments'], 'id': call_id, 'cursor': cursor})
                    pages.append(call_id)
                    cursor = page['next_cursor']
                    if cursor is None:
                        break
                post('/collect', {'pages': pages, 'id': step['id']})
            elif kind == 'wait':
                post('/wait', {'seconds': step['seconds']})
            elif kind == 'time':
                with urlopen(Request(base + '/time', headers={'X-Episode': episode}), timeout=5) as response:
                    json.load(response)
            elif kind == 'revoke':
                post('/revoke', {'stage': step['stage']})
        result = post('/answer', reference['answer'])
        require(result['success'] is True, 'public HTTP reference failed')
        with urlopen(Request(base + '/time', headers={'X-Episode': episode}), timeout=5) as response:
            clock = json.load(response)
        require(clock == reference['time'], 'HTTP and in-process virtual times disagree')
        post('/delete', {})
        return {'passed': True, 'scenario': scenario['id'], 'time': clock, 'reward': result['reward']}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
