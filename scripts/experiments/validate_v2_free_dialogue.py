"""Diagnostic native policy + native free-user HTTP rollout; not a yield sample."""
import argparse
import json
import threading
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from env_factory_v2.code_agent import NativeCodeAgent, object_schema
from env_factory_v2.server import serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--connection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = json.loads(args.source.read_text())
    user_transport = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output / 'user-native')
    policy = NativeCodeAgent.from_connection(args.connection, evidence_root=args.output / 'policy-native')
    server = serve(source, port=0, agent=user_transport, user_mode='free_dialogue')
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    episode = ''
    trace = []
    def request(route, body):
        req = Request(f'http://127.0.0.1:{server.server_port}' + route,
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'X-Episode': episode})
        try:
            with urlopen(req, timeout=1200) as response:
                value = json.load(response)
        except HTTPError as error:
            value = json.load(error)
            trace.append({'route': route, 'request': body, 'status': error.code, 'response': value})
            (args.output / 'failure.json').write_text(json.dumps({'passed': False, 'trace': trace}, ensure_ascii=False, indent=2))
            if error.code >= 500:
                raise
            return value
        trace.append({'route': route, 'request': body, 'response': value})
        return value
    try:
        initial = request('/reset', {})
        episode = initial['episode']
        history = []
        schema = object_schema({'tool': {'type': 'string'}, 'arguments_json': {'type': 'string'}})
        score = None
        for index in range(16):
            intent, _ = policy.ask('Complete this user task using actual public tools. Never guess private facts. '
                'Select one business tool or v2_chat or v2_submit. Return arguments as a JSON object string. '
                'v2_chat accepts message, optional exact proposal and actual menu source_id. '
                'Business calls use exact canonical disclosed_facts values returned by chat, $result references to observation IDs, '
                'and optional _v2_cursor. Transactions require _v2_authorization from user confirmation. '
                'v2_submit accepts final answer fields. No host tools.',
                {'public': initial['public'], 'history': history}, schema)
            params = json.loads(intent['arguments_json'])
            identifier = f'call_{index}'
            if intent['tool'] == 'v2_chat':
                result = request('/chat', params)
            elif intent['tool'] == 'v2_submit':
                score = request('/answer', params)
                history.append({'intent': intent, 'result': score})
                break
            else:
                cursor = params.pop('_v2_cursor', None)
                token = params.pop('_v2_authorization', None)
                result = request('/tool', {'tool': intent['tool'], 'arguments': params, 'id': identifier,
                    'cursor': cursor, **({'authorization': token} if token else {})})
            history.append({'intent': intent, 'observation_id': identifier, 'result': result})
        passed = bool(score and score['success'])
        report = {'scope': 'diagnostic_native_free_dialogue_http_rollout_not_yield', 'passed': passed,
            'score': score, 'history': history, 'trace': trace,
            'user_dialogue': server.episodes[episode].user.dialogue}
        (args.output / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'passed': passed, 'steps': len(history), 'output': str(args.output)}))
        return 0 if passed else 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == '__main__':
    raise SystemExit(main())
