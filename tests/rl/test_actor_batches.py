import pytest
mx = pytest.importorskip('mlx.core', exc_type=ImportError)
from rl.actor import collate, physical_batches, actor_loss


def sample(prompt, tokens, weight=.5):
    return {'prompt': prompt, 'tokens': tokens, 'old_logp': [-1.] * len(tokens),
            'advantage': 1., 'weight': weight, 'loss_mask': [1] * len(tokens)}


def test_physical_batch_padding_and_masks_preserve_logical_weights():
    rows = [sample([1, 2], [3, 4, 5]), sample([1], [2])]
    rows[0]['loss_mask'] = [1, 0, 1]
    batch = collate(rows)
    assert batch['ids'].shape == (2, 5)
    assert batch['weights'].tolist() == [[.25, 0., .25], [.5, 0., 0.]]
    assert float(mx.sum(batch['weights'])) == pytest.approx(1.)
    assert len(list(physical_batches(rows, 2, 10))) == 1
    assert len(list(physical_batches(rows, 2, 8))) == 2


def test_masked_and_padded_positions_cannot_change_loss_or_gradient():
    rows = [sample([1], [2, 3]), sample([1], [2])]
    rows[0]['loss_mask'] = [1, 0]
    batch = collate(rows)
    class Policy:
        def __init__(self, scores): self.scores = scores
        def batch_token_stats(self, *args, return_entropy=False):
            return self.scores, None, mx.ones_like(self.scores)
    def loss(scores):
        return actor_loss(Policy(scores), batch, algorithm='grpo', clip=.2, beta=.01, value_coefficient=.5)[0]
    x = mx.array([[-1., -2.], [-1., -3.]])
    y = mx.array([[-1., -1000.], [-1., -1000.]])
    assert float(loss(x)) == pytest.approx(float(loss(y)))
    grad = mx.grad(loss)(y)
    assert grad[:, 1].tolist() == [0., 0.]


def test_numeric_guard_splits_before_falling_back_to_cached():
    import mlx.nn as nn
    import mlx.optimizers as optim
    from rl.actor import ActorTrainer
    from rl.train import Config
    from rl.timing import StageTimes
    class Model(nn.Module):
        def batch_token_stats(self, ids, prompts, responses):
            error = .03 if ids.shape[0] > 1 else .002
            return mx.full((ids.shape[0], max(responses)), -1.-error), None
        def cached_token_stats(self, ids, prompt):
            return mx.full((ids.shape[1]-prompt,), -1.), None
    model = Model()
    trainer = ActorTrainer(model, model, optim.Adam(learning_rate=1e-6), Config(), StageTimes())
    rows = [sample([1], [2, 3]), sample([1], [4, 5])]
    values = trainer.score(rows, verify=True)
    assert [len(chunk) for chunk in trainer.chunks(rows)] == [1, 1]
    assert trainer.numeric_fallbacks == 1
    assert values[id(rows[0])] == pytest.approx([-1.002, -1.002])
    assert trainer.shape_key([rows[0]]) not in trainer.fallback_shapes
    restored = ActorTrainer(model, model, optim.Adam(learning_rate=1e-6), Config(), StageTimes())
    restored.restore_state(trainer.state())
    assert restored.fallback_shapes == trainer.fallback_shapes


@pytest.mark.parametrize('cached', [False, True])
def test_exact_policy_entropy_masks_weights_and_cached_path(cached):
    import math
    from types import SimpleNamespace
    from rl.model import Policy
    # Different histories have uniform or skewed next-token distributions.
    logits = mx.array([[0., 0.], [0., math.log(3.)]])
    model = SimpleNamespace(
        lm=SimpleNamespace(model=lambda ids: logits[ids], lm_head=__import__('mlx.nn', fromlist=['Identity']).Identity(),
                           args=SimpleNamespace(tie_word_embeddings=False)),
        _temperature=2., _logits_chunk_size=1, _has_critic=False, _entropy=Policy._entropy,
        _prefill=lambda ids: None)
    def step(ids, cache):
        scores = logits[ids[0, 0]] / model._temperature
        return scores - mx.logsumexp(scores), None
    model._step = step
    model.batch_token_stats = lambda *args, **kw: Policy.batch_token_stats(model, *args, **kw)
    model.cached_token_stats = lambda *args, **kw: Policy.cached_token_stats(model, *args, **kw)
    rows = [sample([0], [1, 0], .25), sample([1], [0], .75)]
    rows[0]['loss_mask'] = [1, 0]
    batch = collate(rows)
    batch['cached_fallback'] = cached
    _, details = actor_loss(model, batch, algorithm='grpo', clip=.2, beta=0., value_coefficient=0.)
    p = math.sqrt(3.) / (1 + math.sqrt(3.))
    expected = .25 * math.log(2) + .75 * (-p * math.log(p) - (1-p) * math.log(1-p))
    assert float(details[-1]) == pytest.approx(expected, abs=1e-6)
    scores = mx.array([0., -float('inf')])
    assert float(Policy._entropy(scores)) == 0.
    assert mx.grad(lambda x: Policy._entropy(x))(mx.array([-.5, -1.])).tolist() == [0., 0.]


def test_reference_cache_is_bounded_and_independent_of_actor_updates():
    import mlx.nn as nn
    from rl.actor import ActorTrainer
    from rl.train import Config
    from rl.timing import StageTimes
    calls = []
    class Model(nn.Module):
        def batch_token_stats(self, ids, prompts, responses):
            calls.append('batch')
            return mx.full((ids.shape[0], max(responses)), -1.), None
    model = Model()
    trainer = ActorTrainer(model, model, None, Config(reference_cache_tokens=4), StageTimes())
    a, b = sample([1], [2]), sample([3], [4])
    trainer.score([a, b], reference=True)
    assert len(calls) == 1
    trainer.cached_policy_scores.clear()  # What a policy update does.
    trainer.score([a, b], reference=True)
    assert len(calls) == 1 and trainer.reference_cache_hits == 2
    trainer.score([sample([5], [6])], reference=True)
    assert trainer.reference_cache_size <= 4
    trainer.score([a], reference=True)
    assert len(calls) == 3


def test_periodic_checks_recheck_and_switch_to_strict_on_drift():
    import mlx.nn as nn
    from rl.actor import ActorTrainer
    from rl.train import Config
    from rl.timing import StageTimes
    calls, error = [], [0.]
    class Model(nn.Module):
        def batch_token_stats(self, ids, prompts, responses):
            return mx.full((ids.shape[0], max(responses)), -1. - error[0]), None
        def cached_token_stats(self, ids, prompt):
            calls.append('cached')
            return mx.full((ids.shape[1] - prompt,), -1.), None
    model = Model()
    trainer = ActorTrainer(model, model, None, Config(numerical_check_interval=2), StageTimes())
    rows = [sample([1], [2])]
    for step in (1, 2):
        trainer.check_step = step
        trainer.cached_policy_scores.clear()
        trainer.score(rows, verify=True)
    assert len(calls) == 1
    trainer.check_step = 3
    trainer.cached_policy_scores.clear()
    error[0] = .1
    trainer.score(rows, verify=True)
    assert len(calls) == 2 and trainer.strict_checks
    state = trainer.state()
    restored = ActorTrainer(model, model, None, Config(), StageTimes())
    restored.restore_state(state)
    assert restored.strict_checks and restored.check_step == 3


@pytest.mark.parametrize('failed', [False, True])
def test_periodic_evaluation_preserves_mlx_rng_even_on_failure(failed):
    from rl.evaluation import preserve_mlx_rng
    mx.random.seed(123)
    expected = mx.random.uniform(shape=(4,)).tolist()
    mx.random.seed(123)
    try:
        with preserve_mlx_rng():
            mx.eval(mx.random.uniform(shape=(100,)))
            if failed:
                raise RuntimeError('evaluation failed')
    except RuntimeError:
        pass
    assert mx.random.uniform(shape=(4,)).tolist() == expected
