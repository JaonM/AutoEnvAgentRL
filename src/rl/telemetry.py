"""Append-only training events and process-local TensorBoard writers."""
import html
import json
import math
import os
from pathlib import Path
import statistics
import time
import uuid


class TensorBoardWriter:
    """Use TensorBoard's native async writer; flush drains its pending events."""
    def __init__(self, path, flush_secs, purge_step):
        from tensorboard.summary.writer.event_file_writer import _AsyncWriter
        from tensorboard.summary.writer.record_writer import RecordWriter
        from tensorboard.compat.proto.event_pb2 import Event, SessionLog
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        # TensorBoard reads event files lexicographically. Its default UID suffix
        # orders 10 before 9 on fast same-second restarts, corrupting purge order.
        previous = [int(p.name.split('.')[3]) for p in path.glob('events.out.tfevents.*')
                    if len(p.name.split('.')) > 3 and p.name.split('.')[3].isdigit()]
        sequence = max(int(time.time()), max(previous, default=0) + 1)
        filename = path / f'events.out.tfevents.{sequence:010d}.{uuid.uuid4().hex}'
        # Reuse TensorBoard's CRC record format and error-propagating async queue.
        self.writer = _AsyncWriter(RecordWriter(filename.open('wb')), max_queue_size=128, flush_secs=flush_secs)
        self.writer.write(Event(wall_time=time.time(), file_version='brain.Event:2').SerializeToString())
        if purge_step is not None:
            self.writer.write(Event(wall_time=time.time(), step=purge_step,
                                    session_log=SessionLog(status=SessionLog.START)).SerializeToString())
        self.writer.flush()

    def add(self, values, step, walltime=None):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        self.writer.write(Event(wall_time=time.time() if walltime is None else walltime,
                                step=step, summary=Summary(value=values)).SerializeToString())

    def add_scalar(self, tag, value, step, walltime=None):
        from tensorboard.compat.proto.summary_pb2 import Summary
        self.add([Summary.Value(tag=tag, simple_value=value)], step, walltime)

    def add_text(self, tag, text, step, walltime=None):
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.plugins.text import metadata
        from tensorboard.util import tensor_util
        self.add([Summary.Value(tag=tag + '/text_summary',
            tensor=tensor_util.make_tensor_proto(text, dtype='string'),
            metadata=metadata.create_summary_metadata(display_name=tag, description=''))], step, walltime)

    def add_histogram(self, tag, values, step):
        import numpy as np
        from tensorboard.compat.proto.summary_pb2 import HistogramProto, Summary
        values = np.asarray(values, dtype=float)
        counts, edges = np.histogram(values, bins=min(20, len(values)))
        histogram = HistogramProto(min=float(values.min()), max=float(values.max()), num=len(values),
            sum=float(values.sum()), sum_squares=float((values * values).sum()),
            bucket_limit=edges[1:].tolist(), bucket=counts.tolist())
        self.add([Summary.Value(tag=tag, histo=histogram)], step)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()


def group_metrics(group):
    episodes = group['episodes']
    actions = [a for e in episodes for a in e['actions']]
    rewards = [e['final_reward'] for e in episodes]
    tokens = sum(len(a['tokens']) for a in actions)
    compute = sum(a.get('generation_compute_seconds', 0.) for a in actions)
    tools = [t for e in episodes for t in e.get('trace', []) if t['path'].startswith('/v1/tools/')]
    values = {
        'reward_mean': statistics.mean(rewards), 'reward_std': statistics.pstdev(rewards),
        'reward_min': min(rewards), 'reward_max': max(rewards),
        'zero_reward_variance': int(max(rewards) == min(rewards)),
        'context_limit_rate': sum(e.get('finish_reason') == 'context_limit' for e in episodes) / len(episodes),
        'success_rate': sum(e['terminated'] and e['final_reward'] >= 1.-1e-9 for e in episodes) / len(episodes),
        'termination_rate': sum(e['terminated'] for e in episodes) / len(episodes),
        'truncation_rate': sum(e.get('truncated', not e['terminated']) for e in episodes) / len(episodes),
        'episodes': len(episodes), 'actions_mean': len(actions) / len(episodes),
        'generated_tokens': tokens, 'tokens_per_episode': tokens / len(episodes),
        'tool_calls': len(tools),
        'protocol_errors': sum(bool(a.get('action', {}).get('protocol_error')) for a in actions),
        'token_limit_rate': sum(a.get('generation_finish') == 'token_limit' for a in actions) / max(1, len(actions)),
    }
    if tools:
        values['tool_http_error_rate'] = sum(t.get('status', 200) >= 400 for t in tools) / len(tools)
    if compute > 0:
        values['tokens_per_compute_second'] = tokens / compute
    wall = group.get('finished_at', 0) - group.get('started_at', 0)
    if wall > 0:
        values['wall_seconds'] = wall
        values['tokens_per_wall_second'] = tokens / wall
    return values


class EventLogger:
    """One writer per process/source; no shared queues in the training hot path."""
    def __init__(self, output, source, *, enabled=True, flush_secs=5, purge_step=None):
        self.output, self.source = Path(output), source
        self.writer = None
        if enabled:
            self.writer = TensorBoardWriter(self.output / 'tensorboard' / source, flush_secs, purge_step)
        path = self.output / 'logs' / source / 'events.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open('a', encoding='utf-8', buffering=1)
        self.sequence = 0
        self.last_resources = -math.inf
        self.flush_secs = flush_secs

    def emit(self, event, *, step=None, metrics=None, text=None, **fields):
        self.sequence += 1
        step = self.sequence if step is None else step
        now = time.time()
        metrics = {k: float(v) for k, v in (metrics or {}).items()
                   if isinstance(v, (int, float)) and math.isfinite(v)}
        record = {'time': now, 'source': self.source, 'pid': os.getpid(), 'event': event,
                  'step': step, 'metrics': metrics, **fields}
        if text is not None:
            record['text'] = text
        self.stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        if self.writer is not None:
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, step, walltime=now)
            if text is not None:
                self.writer.add_text(event, '<pre>' + html.escape(text) + '</pre>', step, walltime=now)

    def resources(self, mx=None, *, force=False, **fields):
        if not force and time.monotonic() - self.last_resources < self.flush_secs:
            return
        self.last_resources = time.monotonic()
        import psutil
        process = psutil.Process()
        memory = psutil.virtual_memory()
        cpu = process.cpu_times()
        metrics = {'system/process_rss_gb': process.memory_info().rss / 1e9,
                   'system/process_cpu_seconds': cpu.user + cpu.system,
                   'system/available_memory_gb': memory.available / 1e9,
                   'system/memory_used_percent': memory.percent, 'system/swap_used_gb': psutil.swap_memory().used / 1e9}
        if mx is not None:
            metrics.update({'metal/active_gb': mx.get_active_memory() / 1e9,
                            'metal/cache_gb': mx.get_cache_memory() / 1e9,
                            'metal/peak_gb': mx.get_peak_memory() / 1e9})
        metrics.update(fields)
        self.emit('resources', metrics=metrics)

    def flush(self):
        self.stream.flush()
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if not self.stream.closed:
            self.stream.close()
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class TrainingMonitor:
    def __init__(self, config, run_id):
        self.config, self.run_id = config, run_id
        self.options = {'enabled': config.tensorboard, 'flush_secs': config.log_flush_seconds}
        self.runtime = EventLogger(config.output, f'runtime/actor-{run_id}', **self.options)
        self.train = self.groups = None
        self.step = 0
        self.runtime.emit('run/config', text=json.dumps(
            {k: v for k, v in vars(config).items() if not k.startswith('_')}, ensure_ascii=False, indent=2))
        self.runtime.emit('run/state', text='running')

    def restore(self, optimizer_step, groups):
        self.step = optimizer_step
        self.train = EventLogger(self.config.output, 'train',
            purge_step=optimizer_step + 1 if self.config.resume else None, **self.options)
        self.groups = EventLogger(self.config.output, 'rollout_groups',
            purge_step=groups + 1 if self.config.resume else None, **self.options)
        self.runtime.emit('run/resume' if self.config.resume else 'run/start',
                          optimizer_step=optimizer_step, consumed_groups=groups)

    def optimizer(self, row):
        self.step = row['optimizer_step']
        metrics = {f'train/{key}': value for key, value in row.items() if isinstance(value, (int, float))}
        metrics['train/learning_rate'] = self.config.learning_rate
        self.train.emit('optimizer_step', step=self.step, metrics=metrics)

    def group(self, group, step, current_version):
        metrics = {f'rollout/{key}': value for key, value in group_metrics(group).items()}
        metrics['rollout/policy_lag'] = current_version - group['policy_version']
        if group.get('finished_at'):
            metrics['rollout/queue_wait_seconds'] = max(0., time.time() - group['finished_at'])
        self.groups.emit('rollout_group', step=step, metrics=metrics,
                         task_id=group['task_id'], policy_version=group['policy_version'],
                         dataset_index=group.get('dataset_index'), attempt=group.get('rollout_attempt', 0))
        if self.groups.writer is not None:
            self.groups.writer.add_histogram('rollout/rewards',
                [e['final_reward'] for e in group['episodes']], step)

    def evaluation(self, name, result, step):
        from urllib.parse import quote
        metrics = {f'eval/{name}/reward_mean': result['mean_reward']}
        for task, values in result.get('by_task', {}).items():
            metrics.update({f'eval/{name}/tasks/{quote(str(task), safe="")}/{k}': v for k, v in values.items()})
        for split, values in result.get('by_split', {}).items():
            metrics.update({f'eval/{name}/{split}/{k}': v for k, v in values.items()})
        self.train.emit('evaluation', step=step, metrics=metrics, evaluation=name,
                        by_task=result.get('by_task', {}), by_split=result.get('by_split', {}))
        self.train.emit(f'eval/{name}/by_task', step=step,
                        text=json.dumps(result.get('by_task', {}), ensure_ascii=False, indent=2))

    def close(self):
        for logger in (self.train, self.groups, self.runtime):
            if logger is not None:
                logger.close()


class RolloutMonitor:
    def __init__(self, logger, context, *, samples=1, max_chars=16000):
        self.logger, self.context = logger, context
        self.samples, self.max_chars = samples, max_chars
        self.messages = {}

    def __call__(self, event, *, rollout_index, step, messages=None, sample=None, result=None, seconds=None):
        fields = {**self.context, 'rollout_index': rollout_index, 'action_step': step}
        metrics = {}
        if seconds is not None:
            metrics[f'latency/{event}_seconds'] = seconds
        if sample is not None:
            metrics['generation/tokens'] = len(sample['tokens'])
            metrics['generation/seconds'] = sample.get('generation_seconds', 0.)
        if event == 'step' and messages:
            feedback = []
            for message in reversed(messages):
                if message['role'] == 'assistant':
                    break
                feedback.append(message)
            tool_results, protocol_errors, tool_errors = 0, 0, 0
            for message in feedback:
                try:
                    value = json.loads(message.get('content', ''))
                except (ValueError, TypeError):
                    continue
                if not isinstance(value, dict):
                    continue
                protocol_errors += int('protocol_error' in value)
                if message['role'] == 'tool':
                    tool_results += 1
                    tool_errors += int(value.get('status', 200) >= 400)
            metrics.update({'environment/tool_results': tool_results,
                            'environment/tool_errors': tool_errors,
                            'environment/protocol_errors': protocol_errors})
        if result is not None:
            fields['finish_reason'] = result.get('finish_reason')
            metrics.update({'episode/reward': result['final_reward'],
                            'episode/terminated': result['terminated'],
                            'episode/truncated': result.get('truncated', not result['terminated'])})
        text = None
        if rollout_index < self.samples:
            if messages is not None:
                self.messages[rollout_index] = [m for m in messages if m['role'] != 'system']
            visible = list(self.messages.get(rollout_index, []))
            if sample is not None:
                visible.append(sample.get('action', {'role': 'assistant', 'content': sample['text']}))
            header = json.dumps({**fields, 'phase': event, **metrics}, ensure_ascii=False, indent=2)
            body = json.dumps(visible, ensure_ascii=False, indent=2)
            if len(body) > self.max_chars:
                first = self.max_chars // 3
                body = body[:first] + '\n… [display truncated; full data in rollout artifacts] …\n' + body[-(self.max_chars-first):]
            text = header + '\n\n' + body
        self.logger.emit('rollout/trajectory' if text is not None else 'rollout/event',
                         metrics=metrics, text=text, phase=event, **fields)
        if result is not None:
            self.messages.pop(rollout_index, None)


def worker_logger(config, index):
    return EventLogger(config['output'], f'workers/worker-{index}-{uuid.uuid4().hex[:12]}',
        enabled=config.get('tensorboard', True), flush_secs=config.get('log_flush_seconds', 5))
