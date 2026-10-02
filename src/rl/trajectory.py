"""Validated assistant-token trajectories with explicit environment time boundaries."""
import math

from .advantages import gae, group_advantages


def assign_episode_reward(actions, result):
    """A completed rollout is scored once; only its final action receives reward."""
    reward = float(result['final_reward'])
    if not actions or not math.isfinite(reward):
        raise ValueError('terminal scoring requires actions and a finite reward')
    for action in actions:
        action['reward'] = 0.0
    actions[-1]['reward'] = reward


def make_samples(group, algorithm, gamma=1.0, lam=.95, discount_unit='action'):
    if algorithm not in {'ppo', 'grpo'}:
        raise ValueError('unsupported trajectory objective')
    if discount_unit not in {'action', 'token'}:
        raise ValueError('discount_unit must be action or token')
    episodes = group['episodes']
    if not episodes or len({(e.get('task_id', group.get('task_id')), e['seed']) for e in episodes}) != 1:
        raise ValueError('a group must share the same initial task/seed')
    grouped = group_advantages([e['final_reward'] for e in episodes]) if algorithm == 'grpo' else None
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
                'weight': sum(action.get('loss_mask', [1]*n))
                          / sum(sum(a.get('loss_mask', [1]*len(a['tokens']))) for a in actions) / len(episodes)})
            offset += n
    return samples
