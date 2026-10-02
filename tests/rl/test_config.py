import pytest
from rl.train import Config


@pytest.mark.parametrize('settings', [
    {'algorithm': 'invalid'}, {'temperature': 0}, {'temperature': float('inf')}, {'rollout_group': 1}, {'learning_rate': float('nan')},
    {'max_policy_lag': -1}, {'clip': 1.},
    {'gamma': 1.1}, {'rollout_timeout': 0}, {'max_context': 64},
])
def test_invalid_configuration_rejected_before_model_load(settings):
    with pytest.raises(ValueError):
        Config(sandbox='unused', output='unused', **settings).validate()


def test_ppo_accepts_single_trajectory():
    Config(sandbox='unused', output='unused', algorithm='ppo', rollout_group=1).validate()


def test_task_manifest_configuration():
    Config(tasks='tasks.json', output='run').validate()
    with pytest.raises(ValueError, match='exactly one'):
        Config(sandbox='sandbox', tasks='tasks.json', output='run').validate()
    with pytest.raises(ValueError, match='exactly one'):
        Config(output='run').validate()


@pytest.mark.parametrize('algorithm', ['ppo', 'grpo'])
def test_algorithms_support_replay_without_mode(algorithm):
    config = Config(sandbox='task', output='run', algorithm=algorithm,
                    replay_source='prior-run', replay_batches_per_batch=1, max_groups=1, updates=2)
    config.validate()
    assert config.algorithm == algorithm
    assert 'mode' not in Config.__dataclass_fields__


@pytest.mark.parametrize('algorithm', ['vtrace', 'auto'])
def test_removed_algorithms_rejected(algorithm):
    with pytest.raises(ValueError, match='unsupported algorithm'):
        Config(sandbox='task', output='run', algorithm=algorithm).validate()
