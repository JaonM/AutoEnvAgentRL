"""Worker-local continuous decode batches with independent request identities."""
import time

import mlx.core as mx
from mlx_lm.sample_utils import make_sampler

from .errors import ContextBudgetExceeded


class BatchDecoder:
    def __init__(self, policy, *, greedy=False):
        self.policy = policy
        self.sampler = make_sampler(temp=0. if greedy else 1.)
        self.requests = {}
        self.order = []
        self.cache = None
        self.forward_calls = 0
        self.max_batch_size = 0

    def add(self, key, messages, max_tokens, max_context):
        if key in self.requests:
            raise ValueError('duplicate decode request')
        prompt = self.policy.encode(messages)
        if not prompt or max_tokens < 1:
            raise ValueError('nonempty prompt and positive token budget required')
        if len(prompt) + max_tokens > max_context:
            raise ContextBudgetExceeded('policy context budget exceeded; refusing silent context truncation')
        started = time.perf_counter()
        cache = self.policy._prefill(mx.array([prompt]))
        mx.eval([c.state for c in cache])
        self.requests[key] = dict(prompt=prompt, tokens=[], old_logp=[], old_values=[],
            cache=cache, current=prompt[-1], limit=max_tokens, started=started,
            compute=time.perf_counter()-started, batch_sizes=[])

    def tick(self):
        if not self.requests:
            return {}
        started = time.perf_counter()
        order = list(self.requests)
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
                    text=self.policy.tokenizer.decode(request['tokens'], skip_special_tokens=True),
                    generation_finish='eos' if eos else 'token_limit',
                    behavior_statistics_source='batched_sampling_forward',
                    decode_batch_sizes=request['batch_sizes'],
                    generation_seconds=time.perf_counter()-request['started'],
                    generation_compute_seconds=request['compute'], verification_seconds=0.)
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
