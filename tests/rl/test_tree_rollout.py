"""Tree rollouts: trunks, exact-snapshot branches from the least certain states, and segment / Tree-GRPO credit."""
import itertools
import time

import pytest

from rl.parallel_rollout import EnvironmentPool, parallel_rollouts
from rl.data.trajectory import make_samples, tree_advantages
from rl.rollout.scheduler import choose_forks


def snapshot_environment(connection, root, parent_pid):
    """Three steps per episode; reward 1 when any action said 'good'. Snapshots are the exact action list."""
    taken = []
    while True:
        operation, value = connection.recv()
        started = time.time()
        if operation == 'close':
            break
        reply = None
        if operation == 'reset':
            taken = []
        elif operation == 'restore':
            taken = list(value['taken'])
        elif operation == 'snapshot':
            reply = {'taken': list(taken), 'messages': [{'role': 'user', 'content': str(taken)}]}
        elif operation == 'step':
            taken.append(value)
            reply = (0., len(taken) >= 3)
        else:
            reply = {'final_reward': float('good' in taken), 'terminated': True, 'trace': [], 'taken': list(taken)}
        connection.send({'value': reply, 'messages': [{'role': 'user', 'content': str(taken)}],
                         'terminated': operation in ('finish',) or len(taken) >= 3,
                         'started_at': started, 'finished_at': time.time()})
    connection.close()


class Policy:
    """Says 'bad' on trunks' first steps; the second action of every episode is the uncertain one (low log-prob)."""
    texts = itertools.cycle(['bad', 'bad', 'bad', 'bad', 'bad', 'bad', 'good', 'bad', 'good', 'bad'])

    def iter_sample(self, messages, max_tokens, max_context):
        yield None
        steps = messages[0]['content'].count("'")  // 2
        text = next(self.texts)
        return {'prompt': [1], 'tokens': [2], 'text': text, 'old_logp': [-2.0 if steps == 1 else -0.1],
                'old_values': [0.]}


def test_tree_rollouts_branch_from_exact_snapshots(tmp_path):
    pool = EnvironmentPool(str(tmp_path), 2, target=snapshot_environment)
    try:
        episodes = parallel_rollouts(Policy(), pool, {'rollout_group': 2, 'max_steps': 3, 'max_tokens': 8,
                                                      'max_context': 32, 'algorithm': 'grpo', 'rollout_timeout': 10,
                                                      'rollout_mode': 'tree', 'tree_forks': 2, 'tree_branches': 2}, 7)
    finally:
        pool.close()
    trunks = [e for e in episodes if e['tree']['role'] == 'trunk']
    branches = [e for e in episodes if e['tree']['role'] == 'branch']
    assert len(trunks) == 2 and len(branches) == 4
    assert {(b['tree']['trunk'], b['tree']['fork_step']) for b in branches} == {(0, 1), (1, 1)}  # the uncertain step
    for b in branches:
        trunk = episodes[b['tree']['trunk']]
        assert b['taken'][:1] == trunk['taken'][:1]  # continued from the trunk's exact state
        assert len(b['actions']) == 3 - b['tree']['fork_step']  # the step limit counts the prefix
    assert all(e['seed'] == 7 for e in episodes)


def episode(reward, actions, **tree):
    return {'seed': 1, 'final_reward': reward, 'bootstrap': 0.,
            'actions': [{'prompt': [1], 'tokens': [2], 'old_logp': [-.1], 'reward': 0., 'terminated': a == actions - 1}
                        for a in range(actions)], **({'tree': tree} if tree else {})}


def test_segment_credit_follows_the_values_around_each_segment():
    # trunk 0 fails (R 0) after 4 actions; from its state after 2 actions, both branches succeed (V = 2/3);
    # trunk 1 succeeds. Root value = mean of trunks = 0.5.
    episodes = [episode(0, 4, role='trunk'), episode(1, 3, role='trunk'),
                episode(1, 2, role='branch', trunk=0, fork_step=2), episode(1, 2, role='branch', trunk=0, fork_step=2)]
    adv = tree_advantages(episodes, kept=range(4), scale=1.0)
    v_fork, root = 2 / 3, 0.5
    assert adv[0][:2] == pytest.approx([(v_fork - root) + (v_fork - root)] * 2)  # the first segment helped
    assert adv[0][2:] == pytest.approx([(0 - v_fork) + (0 - root)] * 2)  # the second lost it
    assert adv[2] == pytest.approx([(1 - v_fork) + (1 - root)] * 2)
    assert adv[1] == pytest.approx([(1 - root) + (1 - root)] * 3)


def test_tree_grpo_credit_and_untrainable_branches():
    episodes = [episode(0, 4, role='trunk'), episode(1, 3, role='trunk'),
                episode(1, 2, role='branch', trunk=0, fork_step=2), episode(0, 2, role='branch', trunk=0, fork_step=2)]
    adv = tree_advantages(episodes, kept=[0, 1, 2], scale=1.0, mode='tree_grpo')  # episode 3: simulator at fault
    overall = 2 / 3
    leaf0 = (0 - 0.5) + (0 - overall)  # tree of trunk 0 = {0, 2}
    leaf2 = (1 - 0.5) + (1 - overall)
    assert adv[0][:2] == pytest.approx([(leaf0 + leaf2) / 2] * 2)  # shared by the trunk and branch 2
    assert adv[0][2:] == pytest.approx([leaf0] * 2)
    assert adv[3] == [0.0, 0.0]


def test_make_samples_uses_tree_credit_per_action():
    episodes = [episode(0, 4, role='trunk'), episode(1, 3, role='trunk'),
                episode(1, 2, role='branch', trunk=0, fork_step=2), episode(1, 2, role='branch', trunk=0, fork_step=2)]
    samples = make_samples({'episodes': episodes}, 'grpo', scale=1.0)
    first = [s['advantage'] for s in samples if s['episode_index'] == 0]
    assert first[0] > 0 > first[3]  # one trajectory, opposite credit on its two segments


def test_forks_spread_over_trunks_and_skip_untrainable():
    actions = lambda logps: [{'old_logp': [lp]} for lp in logps]
    trunks = [{'actions': actions([-.1, -3., -.1])}, {'actions': actions([-.1, -.2, -2.5])},
              {'actions': actions([-.1, -5., -5.]), 'trainable': False}]
    snaps = {0: [{'messages': []}] * 2, 1: [{'messages': []}] * 2, 2: [{'messages': []}] * 2}
    assert choose_forks(trunks, snaps, 2) == [(0, 1), (1, 2)]
