import pytest
from rl.batching import combine_samples, minibatches
from rl.train import Config


def test_defaults_and_independent_sandbox_batch_sizes():
    config = Config(sandbox='x', output='y')
    config.validate()
    assert config.batch_size == config.mini_batch_size == 1
    config = Config(sandbox='x', output='y', rollout_group=4, batch_size=5, mini_batch_size=2)
    config.validate()  # Sandbox counts need not be multiples of the rollout count.
    assert config.rollout_group == 4 and config.epochs == 2
    assert config.optimization_passes == 1


@pytest.mark.parametrize('options', [dict(batch_size=0), dict(mini_batch_size=0),
    dict(batch_size=-1), dict(batch_size=2, mini_batch_size=3), dict(optimization_passes=0)])
def test_invalid_batch_configuration(options):
    with pytest.raises(ValueError):
        Config(sandbox='x', output='y', **options).validate()


def test_sandbox_groups_stay_whole_with_short_minibatch_tail():
    original = [{'episode_index': 0, 'weight': .125, 'advantage': -1., 'action': 0},
                {'episode_index': 0, 'weight': .375, 'advantage': -1., 'action': 1},
                {'episode_index': 1, 'weight': .5, 'advantage': 1., 'action': 0}]
    samples = combine_samples([original, original, original], 2)
    batch = [(s, None, None) for s in samples]
    parts = list(minibatches(batch, 2, seed=42))
    assert parts == list(minibatches(batch, 2, seed=42))
    assert [len({s['sandbox_index'] for s, _, _ in part}) for part in parts] == [2, 1]
    for sandbox in range(3):
        belonging = [part for part in parts if any(s['sandbox_index'] == sandbox for s, _, _ in part)]
        assert len(belonging) == 1  # Both trajectories, all actions, in exactly one step.
        found = [s for s, _, _ in belonging[0] if s['sandbox_index'] == sandbox]
        assert [s['advantage'] for s in found] == [-1., -1., 1.]
        assert [s['action'] for s in found] == [0, 1, 0]
    assert sum(s['weight'] for s in samples) == pytest.approx(1.)
    for part in parts:
        weight = sum(s['weight'] for s, _, _ in part)
        assert sum(s['weight'] / weight for s, _, _ in part) == pytest.approx(1.)
