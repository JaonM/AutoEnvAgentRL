"""Explicit handling of finite episodes without a usable policy gradient."""


def zero_variance_kind(episodes):
    episodes = [e for e in episodes if e.get('trainable', True) is not False]
    if not episodes:
        return 'no_valid_episodes'
    if all(e['terminated'] and e['final_reward'] >= 1. - 1e-9 for e in episodes):
        return 'all_success'
    if all(e['final_reward'] <= 0 for e in episodes):
        return 'all_failure'
    return 'equal_reward'


def rejection_action(reason, policy, attempt, max_attempts, batch_policy="allow_short"):
    if batch_policy == "strict":
        return "retry" if attempt + 1 < max_attempts else "fail"
    if reason == 'context_limit_empty':
        return 'skip'
    if reason not in ('zero_reward_variance', 'insufficient_valid_episodes'):
        return 'retry' if attempt + 1 < max_attempts else 'fail'
    if policy in ('skip', 'fail'):
        return policy
    if attempt + 1 < max_attempts:
        return 'retry'
    return 'skip' if policy == 'retry_skip' else 'fail'


def check_sampling_budget(config, counts, started_at, *, now=None):
    import time
    if config.sampling_max_tokens and counts.get('generated_tokens', 0) > config.sampling_max_tokens:
        raise RuntimeError('sampling token budget exhausted before completing dataset batch')
    elapsed = (time.time() if now is None else now) - started_at
    if config.sampling_timeout_seconds and elapsed > config.sampling_timeout_seconds:
        raise RuntimeError('sampling time budget exhausted before completing dataset batch')
