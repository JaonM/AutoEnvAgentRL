import json
from pathlib import Path

import pytest

from rl.telemetry import EventLogger, RolloutMonitor, TrainingMonitor, group_metrics
from rl.train import Config


def accumulator(path):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    return EventAccumulator(str(path), size_guidance={'scalars': 0, 'tensors': 0}).Reload()


def group():
    return {'task_id': 'task', 'dataset_index': 3, 'policy_version': 2,
        'started_at': 100., 'finished_at': 104., 'episodes': [
            {'final_reward': reward, 'terminated': bool(reward), 'truncated': not reward,
             'trace': [{'path': '/v1/tools/lookup', 'status': 200 if reward else 400}],
             'actions': [{'tokens': [1, 2], 'generation_compute_seconds': .5,
                          'generation_finish': 'eos' if reward else 'token_limit'}]}
            for reward in [0., 1.]]}


def test_group_metrics_distinguish_reward_termination_and_token_limit():
    values = group_metrics(group())
    assert values['reward_mean'] == values['reward_std'] == .5
    assert values['success_rate'] == values['truncation_rate'] == .5
    assert values['tool_http_error_rate'] == values['token_limit_rate'] == .5
    assert values['generated_tokens'] == 4
    assert values['tokens_per_wall_second'] == 1
    assert values['tokens_per_compute_second'] == 4


def test_tensorboard_scalars_histograms_and_jsonl_match_training_values(tmp_path):
    config = Config(output=str(tmp_path), sandbox='unused')
    monitor = TrainingMonitor(config, 'session')
    monitor.restore(0, 0)
    monitor.optimizer({'optimizer_step': 1, 'loss': -.3, 'policy_loss': -.4,
                       'gradient_norm': 1.2, 'assistant_tokens': 4})
    monitor.group(group(), 1, 3)
    monitor.evaluation('after', {'mean_reward': .5, 'by_task': {'task': {'success_rate': .5}},
        'by_split': {'eval': {'mean_reward': .5, 'success_rate': .5}}}, 1)
    monitor.close()
    train = accumulator(tmp_path / 'tensorboard/train')
    assert train.Scalars('train/loss')[0].value == pytest.approx(-.3)
    assert train.Scalars('train/gradient_norm')[0].step == 1
    assert train.Scalars('eval/after/eval/success_rate')[0].value == .5
    groups = accumulator(tmp_path / 'tensorboard/rollout_groups')
    assert groups.Scalars('rollout/reward_mean')[0].value == .5
    assert groups.Scalars('rollout/policy_lag')[0].value == 1
    assert groups.Histograms('rollout/rewards')[0].histogram_value.num == 2
    events = [json.loads(line) for line in (tmp_path / 'logs/train/events.jsonl').read_text().splitlines()]
    assert events[0]['metrics']['train/loss'] == -.3
    assert events[0]['step'] == 1


def test_resume_purges_only_uncommitted_curve_points_and_keeps_jsonl(tmp_path):
    config = Config(output=str(tmp_path), sandbox='unused')
    first = TrainingMonitor(config, 'first')
    first.restore(0, 0)
    for step in (1, 2, 3):
        first.optimizer({'optimizer_step': step, 'loss': step})
        first.group(group(), step, 2)
    first.close()
    config.resume = True
    resumed = TrainingMonitor(config, 'second')
    resumed.restore(2, 2)  # Step/group 3 was logged but not checkpointed.
    resumed.optimizer({'optimizer_step': 3, 'loss': 30.})
    resumed.group(group(), 3, 4)
    resumed.close()
    values = accumulator(tmp_path / 'tensorboard/train').Scalars('train/loss')
    assert [(v.step, v.value) for v in values] == [(1, 1.), (2, 2.), (3, 30.)]
    groups = accumulator(tmp_path / 'tensorboard/rollout_groups').Scalars('rollout/policy_lag')
    assert [(v.step, v.value) for v in groups] == [(1, 0.), (2, 0.), (3, 2.)]
    assert len((tmp_path / 'logs/train/events.jsonl').read_text().splitlines()) == 4


def test_live_rollout_text_is_visible_before_episode_finishes(tmp_path):
    logger = EventLogger(tmp_path, 'workers/test', flush_secs=1)
    observer = RolloutMonitor(logger, {'task_id': '订单', 'policy_version': 3}, samples=1)
    messages = [{'role': 'user', 'content': '查询订单'}]
    observer('reset', rollout_index=0, step=0, messages=messages)
    sample = {'tokens': [1, 2], 'text': 'tool call', 'action': {'role': 'assistant', 'content': '',
        'tool_calls': [{'id': 'call_1', 'type': 'function',
                        'function': {'name': 'lookup', 'arguments': '{}'}}]}}
    observer('action', rollout_index=0, step=1, sample=sample)
    messages += [sample['action'], {'role': 'tool', 'tool_call_id': 'call_1',
        'content': '{"status":400,"tool_result":{"error":"未找到订单"}}'}]
    observer('step', rollout_index=0, step=1, messages=messages, seconds=.2)
    logger.flush()
    live = accumulator(tmp_path / 'tensorboard/workers/test')
    text = live.Tensors('rollout/trajectory/text_summary')[-1].tensor_proto.string_val[0].decode()
    assert '未找到订单' in text and 'call_1' in text and '订单' in text
    assert live.Scalars('environment/tool_errors')[-1].value == 1
    assert 'episode/reward' not in live.Tags()['scalars']
    observer('finish', rollout_index=0, step=1, messages=messages,
             result={'final_reward': .2, 'terminated': True})
    logger.close()
    assert live.Reload().Scalars('episode/reward')[-1].value == pytest.approx(.2)


def test_trace_sampling_limits_text_and_no_tensorboard_preserves_events(tmp_path):
    logger = EventLogger(tmp_path, 'test', enabled=False)
    observer = RolloutMonitor(logger, {'task_id': 'task'}, samples=1, max_chars=256)
    messages = [{'role': 'user', 'content': 'x' * 10000}]
    observer('reset', rollout_index=0, step=0, messages=messages)
    observer('reset', rollout_index=1, step=0, messages=messages)
    logger.resources(force=True)
    logger.close()
    assert not (tmp_path / 'tensorboard').exists()
    events = [json.loads(line) for line in (tmp_path / 'logs/test/events.jsonl').read_text().splitlines()]
    assert 'display truncated' in events[0]['text'] and len(events[0]['text']) < 1000
    assert 'text' not in events[1]
    assert events[2]['metrics']['system/process_rss_gb'] > 0


@pytest.mark.parametrize('settings', [
    {'log_flush_seconds': 0}, {'rollout_trace_samples': -1}, {'rollout_trace_max_chars': 100},
])
def test_logging_configuration_is_validated(settings):
    with pytest.raises(ValueError):
        Config(sandbox='unused', output='unused', **settings).validate()


def test_fast_restarts_keep_event_files_in_commit_order(tmp_path, monkeypatch):
    import rl.telemetry as telemetry
    monkeypatch.setattr(telemetry.time, 'time', lambda: 1000.)
    for step in range(1, 14):
        logger = EventLogger(tmp_path, 'train', purge_step=step)
        logger.emit('step', step=step, metrics={'train/loss': step})
        logger.close()
    assert [p.step for p in accumulator(tmp_path / 'tensorboard/train').Scalars('train/loss')] == list(range(1, 14))
