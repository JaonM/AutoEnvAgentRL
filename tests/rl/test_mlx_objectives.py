import pytest
mx = pytest.importorskip('mlx.core', exc_type=ImportError)
nn = pytest.importorskip('mlx.nn', exc_type=ImportError)
from rl.objectives import clipped_surrogate, sampled_kl, clipped_value_loss
from rl.qat import QATLinear, fake_quantize


def test_ppo_clips_both_advantage_signs_and_stops_behavior_gradients():
    new = mx.log(mx.array([1.3, .7]))
    old, advantages = mx.zeros((2,)), mx.array([1., -1.])
    assert clipped_surrogate(new, old, advantages).tolist() == pytest.approx([-1.2, .8])
    assert mx.grad(lambda x: clipped_surrogate(new, x, advantages).sum())(old).tolist() == [0, 0]
    assert mx.grad(lambda x: clipped_surrogate(x, old, advantages).sum())(new).tolist() == [0, 0]


def test_kl_and_clipped_critic():
    assert sampled_kl(mx.array([-.3]), mx.array([-.3])).item() == 0
    assert sampled_kl(mx.array([-.1]), mx.array([-.3])).item() >= 0
    assert clipped_value_loss(mx.array(1.), mx.array(0.), mx.array(1.)).item() == pytest.approx(.32)


def test_qat_forward_really_quantizes_but_master_gets_gradient():
    import mlx.optimizers as optim
    mx.random.seed(7)
    qat = QATLinear(nn.Linear(64,16))
    x = mx.random.normal((4,64))
    assert float(mx.linalg.norm(fake_quantize(qat.weight)-qat.weight)) > 0
    gradient = mx.grad(lambda w: fake_quantize(w).sum())(qat.weight)
    assert bool(mx.all(gradient == 1))
    old = mx.array(qat.weight)
    loss, grads = nn.value_and_grad(qat, lambda module: mx.mean(module(x)**2))(qat)
    opt = optim.Adam(1e-3)
    opt.update(qat, grads)
    mx.eval(qat.parameters(), opt.state)
    assert float(mx.linalg.norm(qat.weight - old)) > 0
    packed = qat.deployed()
    restored = mx.dequantize(packed.weight, packed.scales, packed.biases, group_size=64,bits=4)
    assert float(mx.max(mx.abs(restored-fake_quantize(qat.weight)))) < 1e-6
    # Different Metal dense and packed matmul kernels have rounding error.
    assert float(mx.max(mx.abs(qat(x)-packed(x)))) < .005


def test_qat_export_reload_roundtrip(tmp_path):
    from rl.model import Policy
    # Exercise the real exporter/loader without downloading a language model.
    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy.lm = nn.Sequential(QATLinear(nn.Linear(64, 16)))
    x = mx.random.normal((2, 64))
    before = policy.lm(x)
    mx.eval(before)
    description = policy.export_qat(tmp_path)
    assert description
    policy.load_qat_export(tmp_path)
    assert not any(isinstance(module, QATLinear) for _, module in policy.lm.named_modules())
    assert float(mx.max(mx.abs(before - policy.lm(x)))) < .005


def test_policy_digest_excludes_critic_and_detects_packed_change():
    from rl.model import Policy
    policy = Policy.__new__(Policy)
    nn.Module.__init__(policy)
    policy.lm = nn.Sequential(QATLinear(nn.Linear(64,16)))
    policy.critic = nn.Linear(16,1)
    original = policy.digest(policy_only=True)
    packed = policy.digest(effective=True)
    policy.critic.weight = policy.critic.weight + 1
    assert policy.digest(policy_only=True) == original
    assert policy.digest(effective=True) == packed
    policy.lm.layers[0].weight = policy.lm.layers[0].weight + .5
    assert policy.digest(policy_only=True) != original
    assert policy.digest(effective=True) != packed


def test_sandbox_gradient_accumulation_matches_direct_weighted_objective():
    # Unequal action/token lengths must not give long episodes extra weight.
    from rl.batching import combine_samples, minibatches
    group = [{'episode_index': 0, 'weight': .125, 'x': [1.]},
             {'episode_index': 0, 'weight': .375, 'x': [2., 3., 4.]},
             {'episode_index': 1, 'weight': .5, 'x': [5., 6.]}]
    samples = combine_samples([group, group], 2)
    batch = [(s, None, None) for s in samples]
    w = mx.array(.3)
    def action_loss(w, sample):
        return mx.mean((w * mx.array(sample['x']) - 1.) ** 2)
    direct = mx.grad(lambda w: sum(s['weight'] * action_loss(w, s) for s in samples))(w)
    parts = list(minibatches(batch, 1, seed=3))
    accumulated = [sum(s['weight'] * 2 * mx.grad(lambda w: action_loss(w, s))(w)
                       for s, _, _ in part) for part in parts]
    assert float(sum(accumulated) / 2) == pytest.approx(float(direct), abs=1e-6)
