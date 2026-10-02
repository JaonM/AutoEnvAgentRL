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
        def batch_token_stats(self, *args): return self.scores, None
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
