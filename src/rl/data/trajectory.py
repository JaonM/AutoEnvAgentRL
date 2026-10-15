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


def tree_advantages(episodes, kept, scale=None, mode='segment'):
    """Per-action advantages of a tree group (episode['tree']: role trunk, or role branch with trunk and fork_step;
    a branch's actions start after the trunk's first fork_step actions). Untrainable episodes get 0 and count nowhere.

    segment (TreeRL / SPO): a state's value V is the mean final reward of the kept trajectories through it (a fork
    point: its branches and the trunk itself; the root: the kept trunks). A trunk is cut at its fork points; the actions
    of a segment get local + global credit, (V_end - V_start) + (V_end - V_root), with V_end the next fork point's value
    or the trunk's own reward. A branch's actions get (R - V_fork) + (R - V_root).
    tree_grpo (Tree-GRPO): every kept trajectory scores intra-tree (R - mean of its trunk's tree) plus inter-tree
    (R - mean of all) credit; a trunk action gets the mean credit of the trajectories that contain it.
    Divided by `scale` (the task's reward spread) or else by the spread of the group's kept rewards."""
    reward = {i: episodes[i]['final_reward'] for i in kept}
    if not reward:
        return [[0.0] * len(e['actions']) for e in episodes]
    spread = scale
    if spread is None:
        values = list(reward.values())
        mean = sum(values) / len(values)
        spread = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
    if spread < 1e-8:
        return [[0.0] * len(e['actions']) for e in episodes]
    role = lambda i: (episodes[i].get('tree') or {}).get('role', 'trunk')
    trunks = [i for i in range(len(episodes)) if role(i) == 'trunk']
    branches = {}
    for i in range(len(episodes)):
        if role(i) == 'branch':
            meta = episodes[i]['tree']
            branches.setdefault((meta['trunk'], meta['fork_step']), []).append(i)
    mean = lambda items: sum(items) / len(items)
    root = mean([reward[i] for i in trunks if i in reward] or list(reward.values()))
    out = [[0.0] * len(e['actions']) for e in episodes]
    if mode == 'tree_grpo':
        overall = mean(list(reward.values()))
        tree_of = lambda i: i if role(i) == 'trunk' else episodes[i]['tree']['trunk']
        leaf = {}
        for i in reward:
            members = [j for j in reward if tree_of(j) == tree_of(i)]
            leaf[i] = (reward[i] - mean([reward[j] for j in members])) + (reward[i] - overall)
        for i in reward:
            if role(i) == 'branch':
                out[i] = [leaf[i] / spread] * len(episodes[i]['actions'])
            else:
                for j in range(len(episodes[i]['actions'])):
                    through = [leaf[i]] + [leaf[b] for (t, k), items in branches.items() if t == i and k > j
                                           for b in items if b in leaf]
                    out[i][j] = mean(through) / spread
        return out
    for i in trunks:
        if i not in reward:
            continue
        forks = sorted(k for (t, k) in branches if t == i)
        value = {k: mean([reward[b] for b in branches[(i, k)] if b in reward] + [reward[i]]) for k in forks}
        bounds = [0] + forks + [len(episodes[i]['actions'])]
        start_value = [root] + [value[k] for k in forks]
        end_value = [value[k] for k in forks] + [reward[i]]
        for s in range(len(bounds) - 1):
            credit = (end_value[s] - start_value[s]) + (end_value[s] - root)
            for j in range(bounds[s], bounds[s + 1]):
                out[i][j] = credit / spread
    for (t, k), items in branches.items():
        fork_value = mean([reward[b] for b in items if b in reward] + ([reward[t]] if t in reward else []))
        for b in items:
            if b in reward:
                out[b] = [((reward[b] - fork_value) + (reward[b] - root)) / spread] * len(episodes[b]['actions'])
    return out


def make_samples(group, algorithm, gamma=1.0, lam=.95, discount_unit='action', scale=None, aggregation='sequence',
                 tree_advantage='segment'):
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
    per_action = (tree_advantages(episodes, kept, scale, tree_advantage)
                  if algorithm == 'grpo' and any(e.get('tree') for e in episodes) else None)
    samples = []
    # aggregation="token": every assistant token of the group's trainable episodes weighs the same (DAPO); "sequence":
    # each trainable episode weighs the same and its tokens share it (original GRPO).
    group_tokens = sum(sum(a.get('loss_mask', [1] * len(a['tokens']))) for i in kept for a in episodes[i]['actions'])
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
        for step, action in enumerate(actions):
            n = len(action['tokens'])
            advantage = (per_action[index][step] if per_action is not None else
                         grouped[index] if algorithm == 'grpo' else advantages[offset:offset+n])
            samples.append({**action, 'episode_index': index,
                'old_values': action.get('old_values', [0.] * n),
                'advantage': advantage,
                'return': returns[offset:offset+n],
                'weight': (sum(action.get('loss_mask', [1]*n)) if index in kept else 0)
                          / (max(1, group_tokens) if aggregation == 'token' else
                             sum(sum(a.get('loss_mask', [1]*len(a['tokens']))) for a in actions) * max(1, len(kept)))})
            offset += n
    return samples
