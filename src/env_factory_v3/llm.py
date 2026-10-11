"""Minimal OpenAI-compatible chat client (stdlib only) for host-side model roles: the policy
gate's player and the natural-language user simulator. Provider failures are infrastructure."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


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
    def __init__(self, prefix='LLM', *, fallback='LLM', timeout=180, retries=2):
        get = lambda name: os.environ.get(f'{prefix}_{name}') or os.environ[f'{fallback}_{name}']
        self.base_url = get('BASE_URL').rstrip('/')
        self.api_key = get('API_KEY')
        self.model = get('MODEL')
        self.timeout, self.retries = timeout, retries
        self.opener = build_opener(ProxyHandler())

    def chat(self, messages, tools=None, **parameters):
        payload = {'model': self.model, 'messages': messages, 'stream': False, **parameters}
        if tools:
            payload['tools'] = tools
        body = json.dumps(payload).encode()
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
