"""Use the same trajectory/token weights for logging and gradient accumulation."""
import math


class WeightedMetrics:
    names = ('loss', 'policy_loss', 'kl', 'value_loss', 'clip_fraction', 'behavior_kl')

    def __init__(self):
        self.weight = 0.
        self.sums = dict.fromkeys(self.names, 0.)

    def add(self, weight, values):
        if not math.isfinite(weight) or weight <= 0 or len(values) != len(self.names):
            raise ValueError('invalid metric weight/shape')
        for name, value in zip(self.names, values):
            if not math.isfinite(value):
                raise FloatingPointError(f'non-finite metric: {name}')
            self.sums[name] += weight * value
        self.weight += weight

    def result(self):
        if not math.isclose(self.weight, 1., abs_tol=1e-6):
            raise ValueError(f'objective sample weights must sum to 1: {self.weight}')
        return dict(self.sums)
