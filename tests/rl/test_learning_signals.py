from rl.observability.learning_signals import summarize_learning_signals
from rl.observability.telemetry import group_metrics


def group(rewards):
    return {'episodes': [{'final_reward': r, 'terminated': r == 1, 'actions': [], 'trace': []} for r in rewards]}


def test_reward_saturation_and_zero_advantages_are_distinct_from_success():
    success = group_metrics(group([1, 1]))
    failure = group_metrics(group([0, 0]))
    varied = group_metrics(group([0, 1]))
    assert success['grpo_zero_advantage_group'] == failure['grpo_zero_advantage_group'] == 1
    assert success['all_success_group'] == 1 and failure['all_failure_group'] == 1
    assert varied['grpo_zero_advantage_group'] == 0 and varied['reward_non_success_rate'] == .5


def test_summary_separates_environment_jobs_policy_results_and_legacy_missing_data():
    record = {'event': 'rollout_received', 'task_id': 'a', 'dataset_index': 1,
              'learning_signal': group_metrics(group([0, 1]))}
    summary = summarize_learning_signals([record, record,
        {'event': 'environment_quarantined', 'task_id': 'b', 'dataset_index': 2},
        {'event': 'rollout_received', 'task_id': 'c', 'dataset_index': 3},
        {'skipped': 'zero_reward_variance'}])
    assert summary['totals']['episodes'] == 2
    assert summary['totals']['reward_success_rate'] == .5
    assert summary['totals']['quarantined_jobs'] == 1
    assert summary['totals']['unmeasured_legacy_groups'] == 1
    assert summary['by_task']['b']['reward_success_rate'] is None
    assert summary['admission_rejections']['zero_reward_variance'] == 1
    assert summary['reward_accuracy'] == 'not_inferred_without_independent_labels'


def test_single_trajectory_ppo_group_does_not_claim_grpo_eligibility():
    signal = group_metrics(group([1]))
    assert signal['grpo_group_eligible'] == 0
    summary = summarize_learning_signals([{'event': 'rollout_received', 'task_id': 'ppo',
        'dataset_index': 1, 'learning_signal': signal}])
    assert summary['totals']['grpo_zero_advantage_rate'] is None


def test_optimizer_attempt_rate_does_not_claim_policy_improvement():
    result = summarize_learning_signals([{'event': 'actor_optimizer_step'},
                                         {'mini_batch': 1, 'skipped': 'kl_threshold'}])
    assert result['totals']['optimizer_step_execution_rate'] == .5
    assert result['totals']['reward_success_rate'] is None
