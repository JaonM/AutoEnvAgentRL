"""Reference advantage estimators, independent of any accelerator backend."""
from __future__ import annotations

import math


def gae(rewards, values, terminated, *, bootstrap=0.0, gamma=0.99, lam=0.95,
        discounts=None, lambdas=None):
    """GAE over policy timesteps; truncation bootstraps, termination does not."""
    if not (len(rewards) == len(values) == len(terminated)) or not rewards:
        raise ValueError("GAE requires equally sized non-empty trajectories")
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("discounts must be in [0, 1]")
    if not all(math.isfinite(x) for x in [*rewards, *values, bootstrap]):
        raise ValueError("non-finite trajectory")
    discounts = [gamma] * len(rewards) if discounts is None else discounts
    lambdas = [lam] * len(rewards) if lambdas is None else lambdas
    if len(discounts) != len(rewards) or len(lambdas) != len(rewards):
        raise ValueError('discount / lambda lengths must match trajectory')
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in [*discounts, *lambdas]):
        raise ValueError('discounts and lambdas must be in [0, 1]')
    advantages = [0.0] * len(rewards)
    carry = 0.0
    next_value = bootstrap
    for i in reversed(range(len(rewards))):
        continuation = 0.0 if terminated[i] else 1.0
        discount = discounts[i] * continuation
        delta = rewards[i] + discount * next_value - values[i]
        carry = delta + discount * lambdas[i] * carry
        advantages[i] = carry
        next_value = values[i]
    return advantages, [a + v for a, v in zip(advantages, values)]


def group_advantages(rewards, epsilon=1e-8):
    """GRPO population normalization for completions of the SAME initial task."""
    if len(rewards) < 2 or not all(math.isfinite(x) for x in rewards):
        raise ValueError("GRPO needs at least two finite rewards per group")
    mean = sum(rewards) / len(rewards)
    deviation = math.sqrt(sum((x - mean) ** 2 for x in rewards) / len(rewards))
    return [(x - mean) / (deviation + epsilon) for x in rewards]
