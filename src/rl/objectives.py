"""MLX PPO/GRPO objectives. Inputs contain assistant tokens only."""
import mlx.core as mx


def clipped_surrogate(new_logp, old_logp, advantage, epsilon=0.2):
    ratio = mx.exp(new_logp - mx.stop_gradient(old_logp))
    advantage = mx.stop_gradient(advantage)
    return -mx.minimum(ratio * advantage, mx.clip(ratio, 1 - epsilon, 1 + epsilon) * advantage)


def sampled_kl(new_logp, reference_logp):
    delta = mx.stop_gradient(reference_logp) - new_logp
    return mx.exp(delta) - delta - 1


def clipped_value_loss(value, old_value, target, epsilon=0.2):
    old_value, target = mx.stop_gradient(old_value), mx.stop_gradient(target)
    clipped = old_value + mx.clip(value - old_value, -epsilon, epsilon)
    return 0.5 * mx.maximum(mx.square(value - target), mx.square(clipped - target))
