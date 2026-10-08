"""Summaries from recorded rollout evidence, separate from optimizer/eval claims."""
from collections import Counter


def summarize_learning_signals(records):
    totals = Counter()
    tasks = {}
    rejections = Counter()
    seen = set()
    for record in records:
        if record.get('event') == 'actor_optimizer_step':
            totals['optimizer_steps_observed'] += 1
        if record.get('skipped') and 'mini_batch' in record:
            totals['optimizer_steps_skipped'] += 1
        if record.get('skipped'):
            rejections[record['skipped']] += 1
        if record.get('event') not in {'rollout_received', 'environment_quarantined'}:
            continue
        key = (record.get('task_id'), record.get('dataset_index'), record.get('rollout_attempt', 0))
        if key in seen:
            continue
        seen.add(key)
        task = tasks.setdefault(str(record.get('task_id')), Counter())
        if record['event'] == 'environment_quarantined':
            totals['quarantined_jobs'] += 1
            task['quarantined_jobs'] += 1
            continue
        signal = record.get('learning_signal')
        if not signal:
            totals['unmeasured_legacy_groups'] += 1
            continue
        values = {
            'measured_groups': 1, 'episodes': signal['episodes'],
            'reward_successes': signal['reward_successes'],
            'reward_non_successes': signal['episodes'] - signal['reward_successes'],
            'grpo_eligible_groups': signal.get('grpo_group_eligible', 0),
            'zero_advantage_groups': signal['grpo_zero_advantage_group'],
            'all_success_groups': signal['all_success_group'],
            'all_failure_groups': signal['all_failure_group'],
            'high_reward_unterminated': signal['high_reward_unterminated'],
        }
        totals.update(values)
        task.update(values)
    def rates(counts):
        result = dict(counts)
        for label, numerator, denominator in (
            ('reward_success_rate', 'reward_successes', 'episodes'),
            ('grpo_zero_advantage_rate', 'zero_advantage_groups', 'grpo_eligible_groups'),
            ('all_success_group_rate', 'all_success_groups', 'measured_groups'),
        ):
            result[label] = counts[numerator] / counts[denominator] if counts[denominator] else None
        optimizer_attempts = counts['optimizer_steps_observed'] + counts['optimizer_steps_skipped']
        result['optimizer_step_execution_rate'] = counts['optimizer_steps_observed'] / optimizer_attempts if optimizer_attempts else None
        attempts = counts['measured_groups'] + counts['quarantined_jobs']
        result['quarantined_job_rate'] = counts['quarantined_jobs'] / attempts if attempts else None
        return result
    return {'version': '1.0', 'scope': 'observed_training_rollouts_not_independent_evaluation',
            'totals': rates(totals), 'by_task': {name: rates(values) for name, values in tasks.items()},
            'admission_rejections': dict(rejections),
            'reward_accuracy': 'not_inferred_without_independent_labels',
            'limitations': ['GRPO zero advantages do not imply PPO has no learning signal.',
                            'High reward without termination is a diagnostic, not proof of reward exploitation.',
                            'Quarantined jobs have no fabricated episodes or rewards.',
                            'Independent eval and optimizer progress remain separate reports.']}
