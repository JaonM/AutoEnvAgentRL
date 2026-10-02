import pytest
from rl.advantages import gae, group_advantages


def test_terminal_cuts_bootstrap_and_future_credit():
    advantage, returns = gae([0, 1], [.2, .3], [False, True], bootstrap=99, gamma=1, lam=1)
    assert advantage == pytest.approx([.8, .7])
    assert returns == pytest.approx([1, 1])


def test_truncation_bootstraps_instead_of_inventing_failure():
    assert gae([0], [.2], [False], bootstrap=.8, gamma=1)[1] == pytest.approx([.8])
    assert gae([0], [.2], [True], bootstrap=.8, gamma=1)[1] == pytest.approx([0])


def test_grpo_centers_groups_and_zero_variance_is_zero():
    assert group_advantages([0, 1]) == pytest.approx([-1, 1])
    assert group_advantages([.5, .5]) == [0, 0]
    with pytest.raises(ValueError):
        group_advantages([1])
