"""OpenAI-compatible chat server for an MLX model, rendered and parsed exactly as rl trains it.

Benchmarks (τ²-bench via litellm) call POST /v1/chat/completions with messages and tools. The
prompt is the model's own chat template with the tools (thinking disabled, as in the phase-1
training runs), generation runs with mlx-lm, and the output is parsed into an assistant message
with tool calls by rl.environments.chat.completion: the same rendering and parsing the policy saw
during SFT and GRPO. Requests are served one at a time (one model on the GPU); each
conversation's prefix is reused from a KV cache (PrefixCache), since every turn resends the whole
conversation (~6k tokens). --no-cache recomputes every prompt. Greedy outputs with and without the
cache can differ in rare numerical ties; compare arms served the same way.

    python experiments/v3/bench/serve_openai.py --model models/Qwen3-0.6B --port 8765
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class PrefixCache:
    """KV caches of recent conversations, one entry per conversation (Qwen3-0.6B: ~0.11 MB per token).

    Every turn resends the whole conversation; the entry sharing the longest prefix with the new
    prompt is taken out, trimmed to that prefix and extended by generation, then stored again under
    the new tokens. Keep more entries than concurrent conversations, or entries are evicted before
    their next turn. (Batched decoding was tried: on Apple GPUs it made 5k-token prefills slower
    and its extracted caches kept whole batch buffers alive.)"""

    def __init__(self, size, max_bytes):
        self.size, self.max_bytes, self.entries = size, max_bytes, []  # [(tokens, cache, nbytes)], oldest first

    def take(self, prompt):
        """(cache or None, tokens already in it) for the best entry; that entry leaves the store."""
        from mlx_lm.models.cache import trim_prompt_cache
        best, best_len = None, 0
        for index, (tokens, _, _) in enumerate(self.entries):
            n = 0
            limit = min(len(tokens), len(prompt) - 1)  # at least one prompt token must be processed
            while n < limit and tokens[n] == prompt[n]:
                n += 1
            if n > best_len:
                best, best_len = index, n
        if best is None or best_len < 64:
            return None, 0
        tokens, cache, _ = self.entries.pop(best)
        if len(tokens) > best_len:
            trim_prompt_cache(cache, len(tokens) - best_len)
        return cache, best_len

    def put(self, tokens, cache):
        nbytes = sum(c.nbytes for c in cache)
        self.entries.append((list(tokens), cache, nbytes))
        while len(self.entries) > self.size or sum(e[2] for e in self.entries) > self.max_bytes:
            self.entries.pop(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--max-tokens', type=int, default=1024, help='default completion budget per call')
    parser.add_argument('--max-context', type=int, default=32768)
    parser.add_argument('--thinking', action='store_true', help='enable the thinking template (phase 2)')
    parser.add_argument('--log', default=None, help='JSONL of every request and completion')
    parser.add_argument('--no-cache', action='store_true', help='recompute every prompt (no prefix cache)')
    parser.add_argument('--cache-size', type=int, default=32,
                        help='conversations kept; must exceed the client concurrency')
    parser.add_argument('--cache-gb', type=float, default=20.0,
                        help='KV memory for kept conversations (Qwen3-0.6B: a 6k-token conversation ~0.7 GB)')
    args = parser.parse_args()

    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    from rl.environments.chat import completion

    model, tokenizer = load(args.model)
    lock = threading.Lock()
    prefixes = None if args.no_cache else PrefixCache(args.cache_size, int(args.cache_gb * 2 ** 30))
    log = open(args.log, 'a') if args.log else None

    def normalize(messages):
        """OpenAI messages -> chat-template messages (tool-call arguments as objects, no None content)."""
        out = []
        for message in messages:
            message = dict(message)
            if message.get('content') is None:
                message['content'] = ''
            if isinstance(message.get('content'), list):  # content parts
                message['content'] = ''.join(p.get('text', '') for p in message['content'] if isinstance(p, dict))
            for call in message.get('tool_calls') or []:
                arguments = call.get('function', {}).get('arguments')
                if isinstance(arguments, str):
                    try:
                        call['function']['arguments'] = json.loads(arguments or '{}')
                    except ValueError:
                        pass
            out.append(message)
        return out

    def chat(body):
        messages, tools = normalize(body['messages']), body.get('tools') or None
        options = {'tools': tools} if tools else {}
        prompt = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               enable_thinking=args.thinking, **options)
        budget = int(body.get('max_tokens') or body.get('max_completion_tokens') or args.max_tokens)
        if len(prompt) + budget > args.max_context:
            raise ValueError(f'context budget exceeded: {len(prompt)} prompt tokens')
        temperature = float(body.get('temperature') if body.get('temperature') is not None else 0.0)
        sampler = make_sampler(temp=temperature)
        tokens = []
        with lock:
            cache, cached = prefixes.take(prompt) if prefixes else (None, 0)
            if prefixes and cache is None:
                cache = make_prompt_cache(model)
            for token, _ in generate_step(mx.array(prompt[cached:]), model, max_tokens=budget, sampler=sampler,
                                          prompt_cache=cache):
                token = int(token)
                tokens.append(token)
                if token in tokenizer.eos_token_ids:
                    break
            if prefixes:
                # generate_step may already have fed the last sampled token: trust the cache's own offset.
                prefixes.put((prompt + tokens)[:cache[0].offset], cache)
        parsed = completion(tokenizer, tokens, tools if tools else None)
        message = parsed.get('action') or {'role': 'assistant', 'content': parsed['text']}
        message = {k: v for k, v in message.items() if k in ('role', 'content', 'tool_calls')}
        for index, call in enumerate(message.get('tool_calls') or []):
            call.setdefault('id', f'call_{uuid.uuid4().hex[:12]}_{index}')
        finish = 'tool_calls' if message.get('tool_calls') else (
            'stop' if tokens and tokens[-1] in tokenizer.eos_token_ids else 'length')
        if log:
            log.write(json.dumps({'time': time.time(), 'prompt_tokens': len(prompt), 'completion_tokens': len(tokens),
                                  'message': message}, ensure_ascii=False) + '\n')
            log.flush()
        return {'id': f'chatcmpl-{uuid.uuid4().hex}', 'object': 'chat.completion', 'created': int(time.time()),
                'model': body.get('model', args.model),
                'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                'usage': {'prompt_tokens': len(prompt), 'completion_tokens': len(tokens),
                          'total_tokens': len(prompt) + len(tokens)}}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            return

        def send(self, status, body):
            payload = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.rstrip('/') in ('/v1/models', '/models'):
                return self.send(200, {'object': 'list', 'data': [{'id': args.model, 'object': 'model'}]})
            return self.send(200 if self.path == '/health' else 404, {'ok': self.path == '/health'})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
                if self.path.rstrip('/').endswith('/chat/completions'):
                    return self.send(200, chat(body))
                return self.send(404, {'error': {'message': f'unknown route {self.path}'}})
            except ValueError as error:
                return self.send(400, {'error': {'message': str(error), 'type': 'invalid_request_error'}})

    ThreadingHTTPServer.request_queue_size = 128  # the default backlog (5) resets concurrent clients
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    print(f'serving {args.model} on http://127.0.0.1:{args.port}/v1', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
