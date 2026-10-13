"""Validated assistant-token trajectories with explicit environment time boundaries."""
import math

from rl.algorithms.advantages import gae, group_advantages


def assign_episode_reward(actions, result):
    """A completed rollout is scored once; only its final action receives reward."""
    reward = float(result['final_reward'])
    if not actions or not math.isfinite(reward):
        raise ValueError('terminal scoring requires actions and a finite reward')
    for action in actions:
        action['reward'] = 0.0
    actions[-1]['reward'] = reward


class TaskScale:
    """Recent trainable rewards per task; the GRPO advantage scale of Config.advantage_std == "task"."""

    def __init__(self, window=64, minimum=8):
        self.window, self.minimum, self.history = window, minimum, {}

    def __call__(self, group):
        rewards = [e['final_reward'] for e in group['episodes'] if trainable(e)]
        seen = self.history.setdefault(group.get('task_id'), [])
        seen.extend(rewards)
        del seen[:-self.window]
        if len(seen) < self.minimum:
            return None  # too little history: use the group's own spread
        mean = sum(seen) / len(seen)
        return math.sqrt(sum((r - mean) ** 2 for r in seen) / len(seen))


def trainable(episode):
    """False when the environment marked the episode invalid for training (its reward is not evidence)."""
    return episode.get('trainable', True) is not False


def make_samples(group, algorithm, gamma=1.0, lam=.95, discount_unit='action', scale=None):
    if algorithm not in {'ppo', 'grpo'}:
        raise ValueError('unsupported trajectory objective')
    if discount_unit not in {'action', 'token'}:
        raise ValueError('discount_unit must be action or token')
    episodes = group['episodes']
    if not episodes or len({(e.get('task_id', group.get('task_id')), e['seed']) for e in episodes}) != 1:
        raise ValueError('a group must share the same initial task/seed')
    kept = [i for i, e in enumerate(episodes) if trainable(e)]
    grouped = None
    if algorithm == 'grpo':
        # An untrainable episode (e.g. the simulated user misled the agent) neither sets the group
        # baseline nor gets a gradient.
        grouped = [0.0] * len(episodes)
        rewards = [episodes[i]['final_reward'] for i in kept]
        if scale is not None and len(kept) >= 2:
            # Task-level scale: group mean, but the spread of this task's recent rewards (see Config.advantage_std).
            mean = sum(rewards) / len(rewards)
            values = [(r - mean) / (scale + 1e-8) for r in rewards]
        else:
            values = group_advantages(rewards) if len(kept) >= 2 else []
        for i, value in zip(kept, values):
            grouped[i] = value
    samples = []
    for index, episode in enumerate(episodes):
        actions = episode['actions']
        if not actions or not math.isfinite(episode['final_reward']):
            raise ValueError('empty or non-finite episode')
        rewards, values, terminals, discounts, lambdas = [], [], [], [], []
        for j, action in enumerate(actions):
            n = len(action['tokens'])
            old_logp = action['old_logp']
            action_values = action.get('old_values', [0.] * n)
            if not n or len(old_logp) != n or len(action_values) != n:
                raise ValueError('invalid assistant token / behavior statistics alignment')
            if not all(math.isfinite(x) for x in [*old_logp, *action_values, action['reward']]):
                raise ValueError('non-finite behavior statistics')
            if action['terminated'] and j != len(actions) - 1:
                raise ValueError('actions after terminal transition')
            rewards.extend([0.] * (n - 1) + [action['reward']])
            terminals.extend([False] * (n - 1) + [action['terminated']])
            discounts.extend(([1.] * (n - 1) + [gamma]) if discount_unit == 'action' else [gamma] * n)
            lambdas.extend(([1.] * (n - 1) + [lam]) if discount_unit == 'action' else [lam] * n)
            values.extend(action_values)
        if algorithm == 'ppo':
            advantages, returns = gae(rewards, values, terminals,
                bootstrap=episode['bootstrap'], gamma=gamma, lam=lam,
                discounts=discounts, lambdas=lambdas)
        else:
            advantages, returns = [grouped[index]] * len(values), [0.] * len(values)
        offset = 0
        for action in actions:
            n = len(action['tokens'])
            samples.append({**action, 'episode_index': index,
                'old_values': action.get('old_values', [0.] * n),
                'advantage': grouped[index] if algorithm == 'grpo' else advantages[offset:offset+n],
                'return': returns[offset:offset+n],
                'weight': (sum(action.get('loss_mask', [1]*n)) if index in kept else 0)
                          / sum(sum(a.get('loss_mask', [1]*len(a['tokens']))) for a in actions) / max(1, len(kept))})
            offset += n
    return samples
