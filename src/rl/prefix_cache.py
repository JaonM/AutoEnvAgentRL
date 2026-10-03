"""Exact, version-local prefix reuse. KV arrays are detached copies of cache objects."""
import copy
import mlx.core as mx


def clone(caches):
    result = copy.deepcopy(caches)
    # MLX arrays use immutable graph values; cache containers must not be shared.
    return result


def iter_prefill(policy, prompt):
    from mlx_lm.models.cache import make_prompt_cache
    tokens = tuple(prompt[0].tolist())
    prefix = tokens[:-1]
    budget = policy._prefix_cache_tokens
    cache, offset = make_prompt_cache(policy.lm), 0
    if budget:
        candidates = [key for key in policy._prefix_cache if len(key) <= len(prefix) and prefix[:len(key)] == key]
        if candidates:
            key = max(candidates, key=len)
            cache, offset = clone(policy._prefix_cache[key]), len(key)
            policy._prefill_stats['cache_hits'] += 1
            policy._prefill_stats['reused_tokens'] += offset
            policy._prefix_cache.move_to_end(key)
    for start in range(offset, len(prefix), policy._prefill_chunk_size):
        stop = min(start + policy._prefill_chunk_size, len(prefix))
        policy.lm.model(prompt[:, start:stop], cache=cache)
        mx.eval([c.state for c in cache])
        if budget and stop <= budget:
            key = prefix[:stop]
            policy._prefix_cache[key] = clone(cache)
            policy._prefix_cache.move_to_end(key)
            while sum(map(len, policy._prefix_cache)) > budget:
                policy._prefix_cache.popitem(last=False)
        if stop < len(prefix):
            yield None
    return cache
