"""Policy gate: can a strong model, seeing only what a trained policy sees, solve the task?

An independent OpenAI-compatible model (not the one that wrote env.py and oracle.py) plays k
episodes through the bundle's HTTP service and the shared agent view, one world per episode in
the bundle's served order. A task no strong policy solves gives a learner no successful
trajectories to start from. Provider failures are infrastructure errors, never business results.

    python -m env_factory_v3 policy BUNDLE [--k 4] [--prefix LLM]   # uses <PREFIX>_API_KEY/BASE_URL/MODEL
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

from env_factory_v3 import agent_view, bundle, server

RESULT = 'policy.json'


class ProviderError(Exception):
    """The model provider failed; the episode says nothing about the task."""


def load_env_file(path='.env'):
    """KEY=VALUE lines into os.environ, never overriding what is already set."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip().strip('\'"'))


class ChatClient:
    def __init__(self, prefix='LLM', *, timeout=180, retries=2):
        self.base_url = os.environ[f'{prefix}_BASE_URL'].rstrip('/')
        self.api_key = os.environ[f'{prefix}_API_KEY']
        self.model = os.environ[f'{prefix}_MODEL']
        self.timeout, self.retries = timeout, retries
        self.opener = build_opener(ProxyHandler())

    def chat(self, messages, tools):
        body = json.dumps({'model': self.model, 'messages': messages, 'tools': tools, 'stream': False}).encode()
        for attempt in range(self.retries + 1):
            request = Request(self.base_url + '/chat/completions', data=body, method='POST',
                              headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.api_key})
            try:
                with self.opener.open(request, timeout=self.timeout) as response:
                    return json.load(response)['choices'][0]['message']
            except HTTPError as error:
                if error.code not in (408, 429) and error.code < 500:
                    raise ProviderError(f'provider rejected the request: HTTP {error.code}') from error
                failure = f'HTTP {error.code}'
            except (URLError, TimeoutError, OSError, KeyError, ValueError) as error:
                failure = type(error).__name__
            if attempt < self.retries:
                time.sleep(5 * (attempt + 1))
        raise ProviderError(f'provider failed after retries: {failure}')


def _post(opener, url, body):
    request = Request(url, data=json.dumps(body, ensure_ascii=False).encode(), method='POST',
                      headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=60) as response:
        return json.load(response)


def play(base_url, client, world, category, *, max_turns=30):
    """One episode on `world`; returns its grade and transcript."""
    opener = build_opener(ProxyHandler({}))
    reset = _post(opener, base_url + '/reset', {'world': world})
    episode, instructions = reset['episode'], reset['instructions']
    names = {tool['name'] for tool in instructions['tools']}
    tools = agent_view.tools(instructions, category)
    messages = agent_view.messages(instructions)
    answered, calls, turns = False, 0, 0
    step = lambda action: _post(opener, base_url + '/step', {'episode': episode, 'action': action})
    try:
        while not answered and turns < max_turns:
            turns += 1
            reply = client.chat(messages, tools)
            message = {'role': 'assistant', 'content': reply.get('content') or ''}
            if reply.get('tool_calls'):
                message['tool_calls'] = reply['tool_calls']
            messages.append(message)
            if not reply.get('tool_calls'):
                answer = agent_view.final_answer(reply.get('content'))
                step({'type': 'answer', 'answer': answer if answer is not None else {}})
                answered = True
                break
            for call in reply['tool_calls']:
                name = call['function']['name']
                try:
                    params = json.loads(call['function'].get('arguments') or '{}')
                    action = agent_view.action(name, params, names)
                except (ValueError, KeyError) as error:
                    result = {'ok': False, 'kind': 'invalid', 'error': str(error)}
                else:
                    if answered:
                        result = {'ok': False, 'kind': 'finished', 'error': 'episode already answered'}
                    elif action is None:
                        result, answered = step({'type': 'answer', 'answer': params}), True
                    else:
                        calls += action['type'] == 'tool'
                        result = step(action)
                messages.append({'role': 'tool', 'tool_call_id': call.get('id', name), 'name': name,
                                 'content': json.dumps(result, ensure_ascii=False)})
        if not answered:
            step({'type': 'answer', 'answer': {}})
        grade = _post(opener, base_url + '/reward', {'episode': episode})
        return {'world': world, 'reward': grade.get('reward'), 'success': grade.get('success') is True,
                'fields': grade.get('fields'), 'state_ok': grade.get('state_ok'), 'safety': grade.get('safety'),
                'turns': turns, 'tool_calls': calls, 'answered': answered, 'messages': messages}
    finally:
        _post(opener, base_url + '/release', {'episode': episode})


def gate(folder, client, *, k=4, parallel=4, max_turns=30):
    """k episodes on the bundle's served worlds; passes when at least one succeeds."""
    folder = Path(folder)
    loaded = bundle.load_bundle(folder)
    service = server.Service(loaded['env'], loaded['oracle'], loaded['schema'], loaded['request'],
                             valid_worlds=loaded['valid_worlds'])
    httpd = server.start(service)
    base_url = f'http://127.0.0.1:{httpd.server_address[1]}'
    worlds = [service.order[i % len(service.order)] for i in range(k)]

    def one(world):
        try:
            return play(base_url, client, world, loaded['request']['category'], max_turns=max_turns)
        except ProviderError as error:
            return {'world': world, 'infrastructure': str(error)}

    try:
        with ThreadPoolExecutor(parallel) as pool:
            episodes = list(pool.map(one, worlds))
    finally:
        httpd.shutdown()
        httpd.server_close()
    played = [e for e in episodes if 'infrastructure' not in e]
    successes = sum(e['success'] for e in played)
    report = {'model': client.model, 'k': k, 'played': len(played), 'successes': successes,
              'success_rate': round(successes / len(played), 3) if played else None,
              'mean_reward': round(sum(e['reward'] or 0 for e in played) / len(played), 3) if played else None,
              'status': ('infrastructure' if not played else 'passed' if successes else 'unsolved'),
              'platform_identity': loaded['manifest']['platform_identity'], 'episodes': episodes}
    (folder / RESULT).write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report
