import pytest
from rl.metrics import WeightedMetrics


def test_logged_loss_uses_same_weights_as_optimizer():
    metrics=WeightedMetrics()
    metrics.add(.25,[1.,1.,0.,0.,0.,0.,2.])
    metrics.add(.75,[3.,3.,0.,0.,0.,0.,4.])
    assert metrics.result()['loss'] == pytest.approx(2.5)
    assert metrics.result()['policy_loss'] == pytest.approx(2.5)
    assert metrics.result()['policy_entropy'] == pytest.approx(3.5)


def test_incomplete_and_nonfinite_metric_batches_rejected():
    metrics=WeightedMetrics()
    metrics.add(.5,[1.]*7)
    with pytest.raises(ValueError,match='sum to 1'):metrics.result()
    with pytest.raises(FloatingPointError):metrics.add(.5,[float('inf')]*7)
