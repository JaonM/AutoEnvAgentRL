"""Reverse curriculum: late starts from recorded successes, moving earlier as groups succeed."""
import json
import time

from rl.parallel_rollout import EnvironmentPool, parallel_rollouts
from rl.rollout.curriculum import Curriculum
from test_tree_rollout import Policy, snapshot_environment


def library(tmp_path):
    path = tmp_path / 'library.json'
    path.write_text(json.dumps({'hard': [{'seed': 1, 'snapshots': [{'taken': ['a']}, {'taken': ['a', 'b']},
                                                                  {'taken': ['a', 'b', 'c']}]}]}))
    return path


def group(successes, total=4):
    return [{'score': {'success': i < successes}} for i in range(total)]


def test_a_hard_task_starts_late_and_moves_earlier_after_two_good_groups(tmp_path):
    curriculum = Curriculum(library(tmp_path), tmp_path / 'state.json')
    assert curriculum.start('easy') is None  # not in the library: plays from the beginning
    first = curriculum.start('hard')
    assert (first['level'], first['prefix_steps']) == (0, 3)  # a quarter of the 4 recorded turns left
    assert curriculum.record('hard', 0, group(3)) == 0  # one good group is not enough
    assert curriculum.record('hard', 0, group(3)) == 1
    assert curriculum.start('hard')['prefix_steps'] == 2
    assert curriculum.record('hard', 0, group(4)) == 1  # a stale group from the old level does not count
    curriculum.record('hard', 1, group(4)), curriculum.record('hard', 1, group(4))
    assert curriculum.start('hard')['prefix_steps'] == 1
    curriculum.record('hard', 2, group(4)), curriculum.record('hard', 2, group(4))
    assert curriculum.start('hard') is None  # the last level is the normal reset
    assert curriculum.record('hard', 3, group(0, total=0)) is None  # nothing trainable: no update


def test_a_curriculum_group_restores_one_start_for_every_episode(tmp_path):
    start = {'snapshot': {'taken': ['a', 'b']}, 'prefix_steps': 2, 'level': 1}
    pool = EnvironmentPool(str(tmp_path), 2, target=snapshot_environment)
    try:
        episodes = parallel_rollouts(Policy(), pool, {'rollout_group': 3, 'max_steps': 3, 'max_tokens': 8,
                                                      'max_context': 32, 'algorithm': 'grpo', 'rollout_timeout': 10,
                                                      '_start': start}, 5)
    finally:
        pool.close()
    assert len(episodes) == 3
    assert all(e['taken'][:2] == ['a', 'b'] and len(e['actions']) == 1 for e in episodes)  # only the last step played
    assert all(e['curriculum'] == {'prefix_steps': 2, 'level': 1} for e in episodes)
