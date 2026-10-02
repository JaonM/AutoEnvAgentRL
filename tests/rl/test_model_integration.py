"""Opt-in real-model numerical check: RL_TEST_MODEL=/path/to/mlx/model pytest ..."""
import os
import pytest

MODEL = os.environ.get('RL_TEST_MODEL')
pytestmark = pytest.mark.skipif(not MODEL, reason='set RL_TEST_MODEL for local Metal model integration')


def test_sampling_and_scoring_agree_on_tool_context():
    mx = pytest.importorskip('mlx.core', exc_type=ImportError)
    from rl.model import Policy
    mx.random.seed(42)
    policy = Policy(MODEL, temperature=1.5, critic=True)
    messages = [
        {'role': 'system', 'content': 'Output only JSON: {"kind":"respond","content":"answer"}.'},
        {'role': 'user', 'content': '采购项目的记录如下，请求和 quantity，输出采购总数量：数字件。'},
        {'role': 'assistant', 'content': '{"kind":"tool","name":"query_entries","arguments":{"account_id":573827}}'},
        {'role': 'user', 'content': '{"tool_result":{"records":[{"quantity":27},{"quantity":80},{"quantity":162},{"quantity":169}]}}'},
    ]
    sample = policy.sample(messages, max_tokens=128)
    assert sample['behavior_scoring_max_logp_error'] < .001
    assert len(sample['old_values']) == len(sample['old_logp']) == len(sample['tokens'])
    assert all(value == 0 for value in sample['old_values'])
    assert sample['tokens']
    assert bool(mx.all(mx.isfinite(mx.array(sample['old_logp']))))


def test_interleaved_generation_keeps_separate_caches_and_behavior_probabilities():
    from rl.model import Policy
    policy=Policy(MODEL,temperature=1.5)
    iterators=[policy.iter_sample([{'role':'user','content':question}],max_tokens=16)
               for question in ['Compute 12 + 7.','Compute 31 - 4.']]
    results=[]
    while iterators:
        for iterator in list(iterators):
            try:next(iterator)
            except StopIteration as done:
                results.append(done.value)
                iterators.remove(iterator)
    assert len(results)==2
    assert results[0]['prompt']!=results[1]['prompt']
    assert all(r['behavior_scoring_max_logp_error']<=.001 for r in results)
    assert all('old_values' not in r for r in results)
    assert not hasattr(policy, 'critic')
    with pytest.raises(ValueError, match='requires a PPO critic'):
        policy.value(results[0]['prompt'])


def test_batched_decode_dynamic_membership_and_behavior_statistics():
    import mlx.core as mx
    from rl.model import Policy
    policy = Policy(MODEL, temperature=1.5, critic=True)
    decoder = policy.batch_decoder(greedy=True)
    first = [{'role': 'user', 'content': 'Compute 12 + 7.'}]
    second = [{'role': 'user', 'content': 'Explain why water freezes in a very cold room.'}]
    decoder.add('short', first, 1, 4096)
    decoder.add('long', second, 4, 4096)
    # Observe the exact distributions that actually generated the sampled tokens.
    observed = []
    original = policy._batch_step
    def capture(ids, cache):
        logp, values = original(ids, cache)
        mx.eval(logp, values)
        observed.append((logp.tolist(), values.tolist()))
        return logp, values
    policy._batch_step = capture
    done = decoder.tick()
    assert 'short' in done
    token = done['short']['tokens'][0]
    assert done['short']['old_logp'][0] == observed[0][0][0][token]
    assert done['short']['old_values'][0] == observed[0][1][0]
    # Reuse a departed slot while another sequence retains its KV history.
    decoder.add('short', first, 2, 4096)
    results = []
    while decoder.requests:
        results.extend(decoder.tick().values())
    assert decoder.max_batch_size == 2
    assert sum(len(r['tokens']) for r in results) >= 2
    assert all(len(r['old_values']) == len(r['tokens']) == len(r['old_logp']) for r in results)
    assert decoder.cache is None
    assert all(r['behavior_statistics_source'] == 'batched_sampling_forward' for r in results)


def test_batched_decode_ragged_cache_matches_independent_greedy_sequences():
    from rl.model import Policy
    policy = Policy(MODEL)
    messages = [[{'role':'user', 'content':text}] for text in
                ['Compute 12 + 7.', 'Count from one to ten, with one number on each line.']]
    expected = [policy.sample(message, max_tokens=8, greedy=True)['tokens'] for message in messages]
    decoder = policy.batch_decoder(greedy=True)
    decoder.add(0, messages[0], 8, 4096)
    results = decoder.tick()
    # New, longer prompt joins a request whose cache has already advanced.
    decoder.add(1, messages[1], 8, 4096)
    while decoder.requests:
        results.update(decoder.tick())
    assert [results[i]['tokens'] for i in range(2)] == expected
    assert all('old_values' not in result for result in results.values())


def test_batched_decode_eos_and_context_validation():
    import mlx.core as mx
    from rl.errors import ContextBudgetExceeded
    from rl.model import Policy
    policy = Policy(MODEL)
    decoder = policy.batch_decoder()
    message = [{'role':'user', 'content':'Hello'}]
    with pytest.raises(ContextBudgetExceeded):
        decoder.add('a', message, 8, 1)
    assert not decoder.requests
    decoder.add('a', message, 8, 4096)
    with pytest.raises(ValueError, match='duplicate'):
        decoder.add('a', message, 8, 4096)
    eos = next(iter(policy.tokenizer.eos_token_ids))
    decoder.sampler = lambda logp: mx.array([eos] * logp.shape[0])
    result = decoder.tick()['a']
    assert result['generation_finish'] == 'eos'
    assert result['tokens'] == [eos]
    assert decoder.cache is None and not decoder.requests
    assert decoder.tick() == {}
    assert decoder.forward_calls == 1
