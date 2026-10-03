"""Worker-local continuous decode batches with independent request identities."""
import time

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from .errors import ContextBudgetExceeded
from .chat import completion


class BatchDecoder:
    def __init__(self, policy, *, greedy=False):
        self.policy = policy
        self.sampler = make_sampler(temp=0. if greedy else 1.)
        self.requests = {}
        self.order = []
        self.cache = None
        self.forward_calls = 0
        self.max_batch_size = 0

    def add(self, key, messages, max_tokens, max_context, *, tools=None):
        if key in self.requests:
            raise ValueError('duplicate decode request')
        prompt = self.policy.encode(messages, tools=tools)
        if not prompt or max_tokens < 1:
            raise ValueError('nonempty prompt and positive token budget required')
        if len(prompt) + max_tokens > max_context:
            raise ContextBudgetExceeded('policy context budget exceeded; refusing silent context truncation')
        started = time.perf_counter()
        def prefill():
            if hasattr(self.policy, '_iter_prefill'):
                return (yield from self.policy._iter_prefill(mx.array([prompt])))
            return self.policy._prefill(mx.array([prompt]))
        self.requests[key] = dict(prompt=prompt, tools=tools, tokens=[], old_logp=[], old_values=[],
            prefill=prefill(), current=prompt[-1], limit=max_tokens, started=started,
            compute=0., prefill_seconds=0., reused_tokens=0, batch_sizes=[])

    def tick(self):
        if not self.requests:
            return {}
        # Advance bounded prefill chunks, then decode all ready rows. Long new
        # prompts no longer require a complete prefill before existing rows run.
        for request in self.requests.values():
            if 'prefill' not in request:
                continue
            before = getattr(self.policy, '_prefill_stats', {}).get('reused_tokens', 0)
            started = time.perf_counter()
            try:
                next(request['prefill'])
            except StopIteration as done:
                request['cache'] = done.value
                del request['prefill']
            elapsed = time.perf_counter() - started
            request['prefill_seconds'] += elapsed
            request['compute'] += elapsed
            request['reused_tokens'] += getattr(self.policy, '_prefill_stats', {}).get('reused_tokens', 0) - before
        started = time.perf_counter()
        order = [key for key, request in self.requests.items() if 'prefill' not in request]
        if not order:
            return {}
        if order != self.order:
            # Preserve existing caches across membership changes, including unequal lengths.
            previous = {key: i for i, key in enumerate(self.order)}
            caches = [[c.extract(previous[key]) for c in self.cache] if key in previous
                      else self.requests[key]['cache'] for key in order]
            self.cache = [caches[0][i].merge([c[i] for c in caches]) for i in range(len(caches[0]))]
            for request in self.requests.values():
                request.pop('cache', None)
            self.order = order
        ids = mx.array([[self.requests[key]['current']] for key in order])
        logp, values = self.policy._batch_step(ids, self.cache)
        chosen = self.sampler(logp)
        selected = mx.take_along_axis(logp, chosen[:, None], axis=-1).squeeze(-1)
        mx.eval(chosen, selected, *([] if values is None else [values]))
        if not bool(mx.all(mx.isfinite(selected))) or (values is not None and not bool(mx.all(mx.isfinite(values)))):
            raise ValueError('non-finite batched behavior statistics')
        tokens, probabilities = chosen.tolist(), selected.tolist()
        predictions = values.tolist() if values is not None else None
        self.forward_calls += 1
        self.max_batch_size = max(self.max_batch_size, len(order))
        elapsed = time.perf_counter()-started
        finished = {}
        for i, key in enumerate(order):
            request = self.requests[key]
            token = tokens[i]
            request['tokens'].append(token)
            request['old_logp'].append(probabilities[i])
            if predictions is not None:
                request['old_values'].append(predictions[i])
            request['current'] = token
            request['compute'] += elapsed / len(order)
            request['batch_sizes'].append(len(order))
            eos = token in self.policy.tokenizer.eos_token_ids
            if eos or len(request['tokens']) >= request['limit']:
                finished[key] = dict(prompt=request['prompt'], tokens=request['tokens'],
                    old_logp=request['old_logp'],
                    **({'old_values': request['old_values']} if predictions is not None else {}),
                    **completion(self.policy.tokenizer, request['tokens'], request['tools']),
                    generation_finish='eos' if eos else 'token_limit',
                    behavior_statistics_source='batched_sampling_forward',
                    decode_batch_sizes=request['batch_sizes'],
                    generation_seconds=time.perf_counter()-request['started'],
                    generation_compute_seconds=request['compute'], verification_seconds=0.)
                finished[key].update(generation_prefill_seconds=request['prefill_seconds'],
                                     generation_decode_seconds=max(0., request['compute'] - request['prefill_seconds']),
                                     prefix_reused_tokens=request['reused_tokens'])
        for key in finished:
            del self.requests[key]
        # Immediately remove finished KV rows; slots can be reused before the next tick.
        if finished:
            keep = [i for i, key in enumerate(order) if key not in finished]
            if keep:
                for cache in self.cache:
                    cache.filter(mx.array(keep))
                self.order = [order[i] for i in keep]
            else:
                self.cache, self.order = None, []
        return finished
