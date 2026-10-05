"""Explicit handling of finite episodes without a usable policy gradient."""


def zero_variance_kind(episodes):
    if all(e['terminated'] and e['final_reward'] >= 1. - 1e-9 for e in episodes):
        return 'all_success'
    if all(e['final_reward'] <= 0 for e in episodes):
        return 'all_failure'
    return 'equal_reward'


def rejection_action(reason, policy, attempt, max_attempts):
    if reason == 'context_limit_empty':
        return 'skip'
    if reason != 'zero_reward_variance':
        return 'retry' if attempt + 1 < max_attempts else 'fail'
    if policy in ('skip', 'fail'):
        return policy
    if attempt + 1 < max_attempts:
        return 'retry'
    return 'skip' if policy == 'retry_skip' else 'fail'
