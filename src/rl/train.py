"""Bounded asynchronous PPO/GRPO training on qualified local sandboxes."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
from dataclasses import asdict, dataclass
import json
import math
import multiprocessing as mp
import os
import platform
from pathlib import Path
import time
import traceback
import uuid

from .trajectory import make_samples
from .metrics import WeightedMetrics
from .batching import combine_samples, minibatches
from .dataset import DatasetSchedule, DatasetResults
from .model_options import policy_options
from .checkpoint import save_checkpoint, restore_checkpoint, atomic_json


@dataclass
class Config:
    sandbox: str = ""
    output: str = ""
    tasks: str = ""
    eval_episodes: int = 3
    eval_interval_steps: int = 100
    metrics_export_interval: int = 0
    group_artifacts_keep: int = 256
    zero_variance_policy: str = "retry_skip"
    context_limit_policy: str = "finish"
    numerical_check_mode: str = "periodic"
    numerical_check_interval: int = 20
    reference_cache_tokens: int = 1000000
    heartbeat_interval: float = 1.0
    model_load_timeout: float = 600.0
    group_timeout: float = 3600.0
    train_progress_timeout: float = 3600.0
    checkpoint_keep: int = 3
    replay_min_ess: float = .01
    replay_max_log_ratio: float = 60.
    model: str = 'models/Qwen3-4B-Instruct-2507-4bit'
    algorithm: str = 'grpo'
    replay_source: str = ''
    replay_capacity: int = 32
    replay_max_uses: int = 4
    replay_max_age: int = 32
    replay_batches_per_batch: int = 0
    tuning: str = 'qat'
    thinking_mode: str = 'auto'
    qat_scope: str = 'projections'
    lora_targets: str = 'self_attn.q_proj,self_attn.v_proj'
    lora_scale: float = 16.
    lora_dropout: float = 0.
    lora_merge_export: bool = False
    lora_requantize_export: bool = False
    gradient_checkpointing: bool = False
    logits_chunk_size: int = 128
    prefill_chunk_size: int = 512
    prefix_cache_tokens: int = 0
    packed_inference: bool = False
    profile_memory: bool = False
    checkpoint_interval_steps: int = 1
    policy_publish_interval_steps: int = 1
    updates: int = 0
    rollout_workers: int = 1
    rollout_concurrency: int = 2
    watchdog_interval: float = 1.0
    micro_batch_size: int = 4
    max_tokens_per_micro_batch: int = 8192
    batch_logp_tolerance: float = .01
    sandbox_backend: str = "local"
    sandbox_images: str = ""
    sandbox_max_active: int = 4
    sandbox_startup_workers: int = 2
    sandbox_idle_timeout: int = 300
    sandbox_cpus: str = "1"
    sandbox_memory: str = "1g"
    sandbox_services: str = ""
    environment_factory: str = "rl.environment:SandboxEpisode"
    rollout_group: int = 4
    over_sampling_batch_size: int = 0
    rollout_max_attempts: int = 3
    batch_size: int = 1
    mini_batch_size: int = 1
    queue_size: int = 2
    max_policy_lag: int = 1
    max_groups: int = 0
    max_steps: int = 6
    max_tokens: int = 256
    max_context: int = 4096
    layers: int = 1
    rank: int = 8
    bits: int = 4
    temperature: float = 1.0
    learning_rate: float = 1e-6
    clip: float = .2
    beta: float = .01
    gamma: float = 1.0
    discount_unit: str = "action"
    target_kl: float = .05
    gae_lambda: float = .95
    value_coefficient: float = .5
    max_grad_norm: float = 1.0
    epochs: int = 2
    optimization_passes: int = 1
    seed: int = 42
    rollout_timeout: int = 1200
    max_rollout_restarts: int = 2
    tensorboard: bool = True
    log_flush_seconds: int = 5
    rollout_trace_samples: int = 1
    rollout_trace_max_chars: int = 16000
    resume: bool = False

    @property
    def sampling_pool_size(self):
        return self.over_sampling_batch_size or 2 * self.batch_size

    @property
    def training_batch_size(self):
        return self.batch_size

    @property
    def training_mini_batch_size(self):
        return self.mini_batch_size

    def validate(self):
        if min(self.logits_chunk_size, self.prefill_chunk_size, self.checkpoint_interval_steps, self.policy_publish_interval_steps) < 1 or self.prefix_cache_tokens < 0:
            raise ValueError('invalid memory/scheduling budgets')
        if not math.isfinite(self.lora_scale) or self.lora_scale <= 0 or not self.lora_targets.strip():
            raise ValueError('invalid LoRA scale or targets')
        if self.lora_requantize_export and not self.lora_merge_export:
            raise ValueError('LoRA requantization requires merged export')
        if self.lora_dropout != 0:
            raise ValueError('RL requires lora_dropout=0 to preserve behavior/target probability consistency')
        for name in ('heartbeat_interval', 'model_load_timeout', 'group_timeout', 'train_progress_timeout'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.heartbeat_interval >= self.rollout_timeout:
            raise ValueError('heartbeat_interval must be below rollout_timeout')
        if min(self.eval_interval_steps, self.metrics_export_interval, self.reference_cache_tokens) < 0:
            raise ValueError('evaluation/export intervals and reference cache budget must be nonnegative')
        if self.group_artifacts_keep < 1 or self.numerical_check_interval < 1:
            raise ValueError('artifact retention and numerical check interval must be positive')
        if self.zero_variance_policy not in {'retry_skip', 'retry_fail', 'skip', 'fail'}:
            raise ValueError('invalid zero_variance_policy')
        if self.context_limit_policy not in {'finish', 'fail'}:
            raise ValueError('invalid context_limit_policy')
        if self.numerical_check_mode not in {'strict', 'periodic'}:
            raise ValueError('invalid numerical_check_mode')
        if self.log_flush_seconds < 1 or self.rollout_trace_max_chars < 256 or self.rollout_trace_samples < 0:
            raise ValueError('require log_flush_seconds >= 1, rollout_trace_max_chars >= 256, rollout_trace_samples >= 0')
        if bool(self.sandbox) == bool(self.tasks) or not self.output:
            raise ValueError("provide output and exactly one of sandbox or tasks")
        for name in ('rollout_workers', 'rollout_group', 'queue_size', 'batch_size', 'mini_batch_size', 'optimization_passes',
                     'max_steps', 'max_tokens', 'max_context', 'layers', 'rank',
                     'micro_batch_size', 'max_tokens_per_micro_batch', 'rollout_concurrency', 'epochs', 'rollout_timeout', 'eval_episodes', 'replay_capacity', 'replay_max_uses', 'replay_max_age', 'checkpoint_keep'):
            if getattr(self, name) < 1:
                raise ValueError(f'{name} must be positive')
        if self.sandbox_backend not in ('local', 'docker'):
            raise ValueError('sandbox_backend must be local or docker')
        if min(self.sandbox_max_active, self.sandbox_startup_workers, self.sandbox_idle_timeout) <= 0:
            raise ValueError('sandbox manager limits must be positive')
        if ':' not in self.environment_factory:
            raise ValueError('environment_factory must be module:callable')
        if not math.isfinite(self.batch_logp_tolerance) or self.batch_logp_tolerance <= 0:
            raise ValueError('batch_logp_tolerance must be positive and finite')
        if self.rollout_max_attempts < 1 or self.over_sampling_batch_size < 0:
            raise ValueError('rollout_max_attempts must be positive and over_sampling_batch_size nonnegative')
        if self.sampling_pool_size < self.batch_size:
            raise ValueError('over_sampling_batch_size must be at least batch_size')
        if self.mini_batch_size > self.batch_size:
            raise ValueError('mini_batch_size cannot exceed batch_size')
        if min(self.updates, self.max_groups, self.replay_batches_per_batch) < 0:
            raise ValueError('optional budgets and replay batches must be nonnegative')
        if self.replay_source and not self.replay_batches_per_batch:
            raise ValueError('replay_source requires replay_batches_per_batch > 0')
        if not math.isfinite(self.watchdog_interval) or self.watchdog_interval <= 0:
            raise ValueError('watchdog_interval must be positive and finite')
        if self.algorithm not in {'ppo', 'grpo'} or self.tuning not in {'qat', 'lora', 'full'}:
            raise ValueError('unsupported algorithm or tuning mode')
        if self.qat_scope not in {'projections', 'full'} or (self.qat_scope == 'full' and self.tuning != 'qat'):
            raise ValueError('qat_scope full requires tuning qat')
        if self.thinking_mode not in {'auto', 'thinking', 'no-thinking'}:
            raise ValueError('unsupported thinking mode')
        if not 0 <= self.replay_min_ess <= 1 or not 0 < self.replay_max_log_ratio <= 80:
            raise ValueError('replay ESS must be in [0,1] and log ratio bound in (0,80]')
        if self.algorithm == 'grpo' and self.rollout_group < 2:
            raise ValueError('GRPO requires rollout_group >= 2')
        if self.max_rollout_restarts < 0:
            raise ValueError('max_rollout_restarts must be nonnegative')
        if self.max_policy_lag < 0:
            raise ValueError('require max_policy_lag >= 0')
        if self.max_context <= self.max_tokens or self.bits not in {4, 8}:
            raise ValueError('require max_context > max_tokens and bits in {4, 8}')
        for name in ('learning_rate', 'clip', 'beta', 'gamma', 'gae_lambda',
                     'value_coefficient', 'max_grad_norm', 'temperature', 'target_kl'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.temperature == 0 or self.learning_rate == 0 or self.max_grad_norm == 0 or not 0 < self.clip < 1:
            raise ValueError('require positive learning rate/gradient norm and clip in (0, 1)')
        if self.discount_unit not in {'action', 'token'}:
            raise ValueError('discount_unit must be action or token')
        if self.gamma > 1 or self.gae_lambda > 1:
            raise ValueError('discounts must be in [0, 1]')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def train(config: Config):
    config.validate()
    manager = None
    try:
        if config.sandbox_backend == 'docker' and not config.sandbox_services:
            from dotenv import load_dotenv
            from .docker_manager import DockerManager
            from .tasks import load_tasks
            load_dotenv(Path(__file__).resolve().parents[2] / '.env')
            if not os.environ.get('SANDBOX_TRAINER_API_KEY'):
                os.environ['SANDBOX_TRAINER_API_KEY'] = uuid.uuid4().hex
            tasks = load_tasks(sandbox=config.sandbox, manifest=config.tasks)
            images = json.loads(Path(config.sandbox_images).read_text()) if config.sandbox_images else None
            manager = DockerManager(tasks, Path(config.output) / 'docker_services', images=images,
                                    max_active=config.sandbox_max_active,
                                    startup_workers=config.sandbox_startup_workers,
                                    idle_timeout=config.sandbox_idle_timeout,
                                    cpus=config.sandbox_cpus, memory=config.sandbox_memory)
            config.sandbox_services = manager.start()
        return _train_with_status(config)
    finally:
        if manager is not None:
            manager.close()


def _train_with_status(config: Config):
    run_id = uuid.uuid4().hex
    status_path = Path(config.output).resolve() / 'run_status.json'
    try:
        report = _train(config, run_id)
        if not report['completed']:
            raise RuntimeError('training execution did not satisfy completion evidence')
    except BaseException as error:
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get('run_id') == run_id:
                atomic_json(status_path, {**status, 'state':'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                         'error_type':type(error).__name__, 'finished_at':time.time()})
        monitor = getattr(config, '_monitor', None)
        if monitor is not None:
            monitor.runtime.emit('run/error', text=traceback.format_exc(),
                state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed')
        raise
    finally:
        for journal in getattr(config, '_journals', ()):
            journal.close()
        if hasattr(config, '_journals'):
            del config._journals
        monitor = getattr(config, '_monitor', None)
        if monitor is not None:
            monitor.close()
            del config._monitor
    atomic_json(status_path, {'run_id':run_id, 'state':'completed', 'finished_at':time.time()})
    return report


def _train(config: Config, run_id):
    config.validate()
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from dotenv import load_dotenv
    from .tasks import load_tasks, tasks_state
    from .evaluation import evaluate, PeriodicEvaluation, preserve_mlx_rng
    from .replay import validate_group, ReplayBuffer, import_replay, importance_diagnostics, validate_ppo_behavior
    from .retention import prune_checkpoints, prune_live_artifacts
    from .provenance import model_identity
    from .runtime import RolloutPool
    from .process_control import StopSignal, PolicyVersion
    from .parallel_rollout import environment_overlap
    from .model import Policy
    from .actor import ActorTrainer
    from .timing import StageTimes
    from .contracts import attach_contract
    from .checkpoint import sha256
    from .task_queue import TaskQueue

    load_dotenv(Path(__file__).resolve().parents[2] / '.env')
    # Training never uses fixture reward/user responses, even if the invoking
    # shell previously ran offline validation.
    os.environ.pop('SANDBOX_EVALUATOR_MOCK', None)
    os.environ['SANDBOX_MUTATION_MODE'] = 'disabled'
    if config.sandbox_services:
        config.sandbox_services = str(Path(config.sandbox_services).resolve())
        if not os.environ.get('SANDBOX_TRAINER_API_KEY'):
            raise ValueError('remote sandbox requires SANDBOX_TRAINER_API_KEY matching the sandbox service')
        if config.environment_factory not in ('rl.environment:SandboxEpisode', 'rl.remote_environment:RemoteSandboxEpisode'):
            raise ValueError('sandbox_services conflicts with custom environment_factory')
        config.environment_factory = 'rl.remote_environment:RemoteSandboxEpisode'
        os.environ['RL_SANDBOX_SERVICES'] = config.sandbox_services
    else:
        os.environ['SANDBOX_TRAINER_API_KEY'] = uuid.uuid4().hex
    tasks = load_tasks(sandbox=config.sandbox, manifest=config.tasks)
    schedule = DatasetSchedule(tasks, epochs=config.epochs, batch_size=config.batch_size, seed=config.seed)
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config.output = str(output)
    if config.sandbox:
        config.sandbox = str(Path(config.sandbox).resolve())
    if config.replay_source:
        config.replay_source = str(Path(config.replay_source).resolve())
    if config.tasks:
        config.tasks = str(Path(config.tasks).resolve())
    if Path(config.model).exists():
        config.model = str(Path(config.model).resolve())
    config.model, base_identity, model_files = model_identity(config.model)
    if config.tuning == 'full' or (config.tuning == 'qat' and config.qat_scope == 'full'):
        from .model_memory import check_full_training_memory
        write_json(output / 'model_memory_estimate.json',
                   check_full_training_memory(config.model, config.rollout_workers, qat=config.tuning == 'qat', packed_inference=config.packed_inference, bits=config.bits))
    if config.resume:
        previous = json.loads((output / 'config.json').read_text())
        mutable = {'resume', 'updates', 'max_groups', 'rollout_timeout', 'epochs',
                   'tensorboard', 'log_flush_seconds', 'rollout_trace_samples', 'rollout_trace_max_chars'}
        for key, value in asdict(config).items():
            if key not in mutable and previous.get(key) != value:
                raise ValueError(f'resume configuration changed: {key}')
        if not (output / 'checkpoints/latest.json').exists():
            raise ValueError('no committed resumable checkpoint')
    elif (output / 'config.json').exists():
        raise ValueError('use --resume or a new output; existing run is never overwritten')
    policy_identity = {"base": base_identity, "tuning": config.tuning,
                       "layers": config.layers, "rank": config.rank, "bits": config.bits,
                       "model_options": policy_options(config), "qat_scope": config.qat_scope, "temperature": config.temperature, "thinking_mode": config.thinking_mode, "chat_protocol": "native_tools_v1"}
    provenance = {
        'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted(Path(__file__).parent.glob('*.py'))},
        'packages': {name: importlib.metadata.version(name) for name in ('mlx', 'mlx-lm', 'numpy')},
        'platform': platform.platform(), 'machine': platform.machine(),
        'tasks': tasks_state(tasks), 'policy_identity': policy_identity, 'model_files': model_files,
    }
    if config.resume:
        original = json.loads((output / 'execution_provenance.json').read_text())
        if any(original[key] != provenance[key] for key in ('source_sha256', 'packages', 'tasks', 'policy_identity', 'model_files')):
            raise ValueError('resume code, dependency or sandbox identity changed')
    else:
        write_json(output / 'execution_provenance.json', provenance)
    write_json(output / 'config.json', asdict(config))
    atomic_json(output / 'run_status.json', {'run_id':run_id, 'state':'running', 'started_at':time.time()})
    from .telemetry import TrainingMonitor
    monitor = config._monitor = TrainingMonitor(config, run_id)
    print(f'Training dashboard logs: {output / "tensorboard"}', flush=True)
    mx.set_default_device(mx.gpu)
    if not mx.metal.is_available():
        raise RuntimeError('Apple Metal GPU is required')
    mx.random.seed(config.seed)
    policy = Policy(config.model, config.tuning, config.layers, config.rank, config.bits, temperature=config.temperature, critic=config.algorithm == 'ppo', **policy_options(config))
    initial_digest = policy.digest()
    initial_policy_digest = policy.digest(policy_only=True)
    initial_effective_digest = policy.digest(effective=True) if config.tuning == 'qat' else None
    initial = str(output / 'snapshots/policy-000000.safetensors')
    if not config.resume:
        policy.save_snapshot(output / 'snapshots', 0)
    reference = Policy(config.model, config.tuning, config.layers, config.rank, config.bits, temperature=config.temperature, critic=False, **policy_options(config, inference=True))
    reference.restore(initial, policy_only=True)
    reference.freeze()
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    optimizer.init(policy.trainable_parameters())
    restored = restore_checkpoint(output / 'checkpoints', policy, optimizer) if config.resume else {}
    times = StageTimes(restored.get('timings'), mlx=mx if config.profile_memory else None)
    inflight = restored.get('inflight_batch', {})
    actor = ActorTrainer(policy, reference, optimizer, config, times)
    actor.restore_state(restored.get('actor_numeric_state', {}))
    from .journal import MetricJournal
    journals = restored.get('metric_journals', {})
    config._journals = []
    for name in ('metrics', 'optimizer_metrics'):
        config._journals.append(MetricJournal(output, name, cursor=journals.get(name), legacy=restored.get(name, [])))
    metrics, optimizer_metrics = config._journals
    skipped_jobs = restored.get('skipped_jobs', {})
    updated_jobs = set(restored.get('updated_jobs', []))
    eval_state = restored.get('eval_state', {})
    updates, groups, dropped = (restored.get(key, 0) for key in ('updates', 'groups', 'dropped'))
    optimizer_step = restored.get('optimizer_step', 0)
    monitor.restore(optimizer_step, groups)
    periodic_eval = PeriodicEvaluation(output, tasks, config, monitor, eval_state)
    eval_state = periodic_eval.state

    def evaluate_periodically(*, force=False):
        with preserve_mlx_rng(), times.measure('actor_periodic_evaluation'):
            return periodic_eval.run(policy, optimizer_step, evaluator=evaluate, force=force)

    rollout_positions = restored.get('rollout_positions', {})
    initial_digest = restored.get('initial_digest', initial_digest)
    initial_policy_digest = restored.get('initial_policy_digest', initial_policy_digest)
    initial_effective_digest = restored.get('initial_effective_digest', initial_effective_digest)
    previous_seconds = restored.get('seconds', 0.)
    replay = ReplayBuffer(capacity=config.replay_capacity, max_uses=config.replay_max_uses,
                          max_age=config.replay_max_age, seed=config.seed, state=restored.get('replay'))
    dataset_cursor = restored.get('dataset_cursor', 0)
    consumed_jobs = set(restored.get('consumed_jobs', []))
    rollout_attempts = restored.get('rollout_attempts', {})
    task_queue = TaskQueue(output / 'task_queue', schedule.jobs, config.sampling_pool_size)
    task_queue.commit_consumed(consumed_jobs, rollout_attempts)
    if not 0 <= dataset_cursor <= len(schedule.batches):
        raise ValueError('resume dataset cursor exceeds epoch budget')
    if config.replay_source and not config.resume:
        imported = import_replay(config.replay_source, replay, tasks, policy_identity, current_version=updates)
        write_json(output / 'replay_import.json', {'source':config.replay_source, 'groups':imported})
    if (config.updates and updates > config.updates) or (config.max_groups and groups > config.max_groups):
        raise ValueError('resume budgets are behind the committed checkpoint')
    if config.resume:
        # Ignore unfinished optimizer work; keep its files as forensic evidence.
        recovery = output / 'recovery' / uuid.uuid4().hex
        recovery.mkdir(parents=True)
        for path in output.glob('group-*.json'):
            if int(path.stem.split('-')[1]) > groups:
                path.rename(recovery / path.name)
        for name in ('metrics.json', 'optimizer_metrics.json', 'training_report.json'):
            path = output / name
            if path.exists():
                path.rename(recovery / name)
        metrics.export()
        optimizer_metrics.export()
        from .recovery_versions import quarantine
        quarantine(output, updates, recovery)
        published_path = output / 'snapshots' / f'policy-{updates:06d}.safetensors'
        if restored.get('published_snapshot_sha256') and sha256(published_path) != restored['published_snapshot_sha256']:
            raise ValueError('published snapshot changed since committed checkpoint')
    def measure_drift(group):
        actions = [action for episode in group['episodes'] for action in episode['actions']]
        current = actor.score(actions)
        return importance_diagnostics(
            [p for action in actions for p, mask in zip(current[id(action)], action.get('loss_mask', [1]*len(action['tokens']))) if mask],
            [p for action in actions for p, mask in zip(action['old_logp'], action.get('loss_mask', [1]*len(action['tokens']))) if mask], config.clip)

    ctx = mp.get_context('spawn')
    results = ctx.Queue(maxsize=config.queue_size)
    stop = StopSignal(ctx)
    version = PolicyVersion(output / 'policy_version.json',updates)
    rollout_config = {**asdict(config), '_rollout_positions': rollout_positions,
                    '_dataset_jobs': schedule.jobs, '_run_id': run_id,
                    '_tasks': tasks_state(tasks), '_policy_identity': policy_identity,
                    '_task_queue': str(task_queue.root), '_queue_capacity': task_queue.capacity}
    pool = RolloutPool(ctx, rollout_config, version, results, stop, rollout_positions,
                     restarts=restored.get('rollout_restarts'), events=restored.get('rollout_recovery', []))
    started = time.time()
    committed_version = updates
    checkpoint_step = restored.get('optimizer_step', 0)
    published_step = restored.get('published_optimizer_step', restored.get('optimizer_step', 0))
    def commit(*, force=False):
        nonlocal committed_version, checkpoint_step
        if not force and config.checkpoint_interval_steps > 1 and optimizer_step - checkpoint_step < config.checkpoint_interval_steps:
            # Volatile queue progress is reset from the durable checkpoint on resume.
            task_queue.commit_consumed(consumed_jobs, rollout_attempts)
            return
        checkpoint_started = time.perf_counter()
        save_checkpoint(output / 'checkpoints', policy, optimizer, {
            'updates': updates, 'groups': groups, 'dropped': dropped,
            'published_optimizer_step': published_step,
            'published_snapshot_sha256': sha256(output / 'snapshots' / f'policy-{updates:06d}.safetensors'),
            'optimizer_step': optimizer_step, 'metric_journals': {j.name: j.cursor() for j in config._journals},
            'skipped_jobs': skipped_jobs, 'updated_jobs': sorted(updated_jobs), 'eval_state': eval_state,
            **pool.checkpoint_state(),
            'replay': replay.state(), 'dataset_cursor': dataset_cursor,
            'inflight_batch': inflight, 'consumed_jobs': sorted(consumed_jobs), 'rollout_attempts': rollout_attempts, 'timings': times.state(), 'actor_numeric_state': actor.state(),
            'initial_policy_digest': initial_policy_digest, 'initial_effective_digest': initial_effective_digest,
            'initial_digest': initial_digest, 'seconds': previous_seconds + time.time() - started,
        })

        committed_version = updates
        checkpoint_step = optimizer_step
        periodic_eval.publish_best()
        task_queue.commit_consumed(consumed_jobs, rollout_attempts)
        prune_checkpoints(output / 'checkpoints', config.checkpoint_keep)
        removed = prune_live_artifacts(output, task_queue, updates, keep=config.checkpoint_keep,
                                      group_keep=config.group_artifacts_keep)
        if removed:
            monitor.runtime.emit('retention', metrics={'storage/artifacts_removed': len(removed)})
        times.add('actor_checkpoint', time.perf_counter() - checkpoint_started)
        write_json(output / 'timings.json', times.state())
        monitor.runtime.emit('checkpoint', optimizer_step=optimizer_step, policy_version=updates,
            metrics={'progress/optimizer_step': optimizer_step, 'progress/policy_version': updates,
                     'progress/dataset_batches_completed': dataset_cursor,
                     'progress/visited_jobs': len(consumed_jobs), 'progress/skipped_jobs': len(skipped_jobs),
                     'progress/updated_jobs': len(updated_jobs),
                     'progress/dataset_fraction': dataset_cursor / max(1, len(schedule.batches)),
                     **{f'timing/{name}_seconds': value for name, value in times.seconds.items()}})
        monitor.runtime.resources(mx)
        monitor.train.flush()
        monitor.groups.flush()

    from .admission import rejection_action, zero_variance_kind
    from .advantages import group_advantages

    def prepare(group, path, replay_entry=None):
        nonlocal dropped
        attach_contract(group)
        if len(group['episodes']) != config.rollout_group:
            raise ValueError('rollout group size does not match configured rollout_group')
        lag = updates - group['policy_version']
        age = updates - replay_entry['age_origin'] if replay_entry else lag
        max_age = config.replay_max_age if replay_entry else config.max_policy_lag
        reason = 'context_limit_empty' if any(not e['actions'] for e in group['episodes']) else None
        drift = {}
        started_prepare = time.monotonic()
        if reason:
            pass
        elif age < 0 or age > max_age:
            reason = 'policy_age'
        elif config.algorithm == 'grpo' and all(abs(a) < 1e-8 for a in group_advantages(
                [episode['final_reward'] for episode in group['episodes']])):
            reason = 'zero_reward_variance'
        else:
            if config.algorithm == 'ppo':
                validate_ppo_behavior(group)
            drift = measure_drift(group)
            if drift['effective_sample_fraction'] < config.replay_min_ess or drift['max_abs_log_ratio'] > config.replay_max_log_ratio:
                reason = 'importance_drift'
        samples = [] if reason else make_samples(group, config.algorithm, config.gamma, config.gae_lambda, config.discount_unit)
        if reason:
            dropped += 1
            metrics.append({'task_id': group['task_id'], 'dataset_index':group.get('dataset_index'),
                            'rollout_attempt':group.get('rollout_attempt', 0), 'dataset_batch': dataset_cursor,
                            'replay': replay_entry is not None, 'skipped': reason,
                            **({'zero_variance_kind': zero_variance_kind(group['episodes'])}
                               if reason == 'zero_reward_variance' else {}), **drift})
            monitor.runtime.emit('rollout/rejected', text=json.dumps(metrics[-1], ensure_ascii=False),
                metrics={'admission/dropped_groups': dropped, **{f'admission/{k}': v for k, v in drift.items()}},
                optimizer_step=optimizer_step, reason=reason)
            return {'rejected': reason}
        monitor.runtime.emit('rollout/accepted', task_id=group['task_id'],
            policy_version=group['policy_version'], optimizer_step=optimizer_step,
            replay=replay_entry is not None, metrics={f'admission/{k}': v for k, v in drift.items()})
        return {'group': group, 'samples': samples, 'path': str(path), 'replay_entry': replay_entry,
                'drift': drift, 'lag': lag, 'queue_seconds': 0.,
                'target_seconds': time.monotonic() - started_prepare}

    def optimize(batch_groups, dataset_epoch, replay_cycle, optimization_pass=1, mini_offset=0):
        nonlocal optimizer_step, dropped
        if not batch_groups:
            return False
        if config.updates and updates >= config.updates:
            raise RuntimeError('policy update budget reached before dataset schedule finished')
        samples = combine_samples([[{**sample, 'dataset_index': item['group'].get('dataset_index', -1)}
                                    for sample in item['samples']] for item in batch_groups], config.rollout_group)
        batch = [(sample, None, None) for sample in samples]
        exceeded = False
        for mini_index, mini in enumerate(minibatches(batch, config.mini_batch_size,
                seed=config.seed + dataset_cursor * config.optimization_passes + optimization_pass), 1 + mini_offset):
            started_update = time.time()
            selected = [sample for sample, _, _ in mini]
            def actor_notify():
                pool.raise_if_failed()
                monitor.runtime.resources(mx)
            measured = actor.step(selected, notify=actor_notify)
            common = {'epoch': dataset_epoch, 'dataset_batch': dataset_cursor, 'replay_cycle': replay_cycle,
                      'optimization_pass': optimization_pass, 'mini_batch': mini_index,
                      'post_kl_scope': 'ready_mini_batch',
                      'mini_batch_sandboxes': len({s['sandbox_index'] for s in selected}),
                      'mini_batch_rollouts': len({s['episode_index'] for s in selected})}
            if measured.get('skipped'):
                metrics.append({**common, **measured})
                monitor.runtime.emit('train/skipped', text=json.dumps(metrics[-1], ensure_ascii=False),
                                     optimizer_step=optimizer_step, reason=measured['skipped'])
                if measured['skipped'] == 'target_kl':
                    return True
                continue
            pool.raise_if_failed()
            optimizer_step += 1
            if replay_cycle == 0:
                updated_jobs.update(s['dataset_index'] for s in selected if s['dataset_index'] in consumed_jobs)
            measured.update(**common, optimizer_step=optimizer_step, policy_version=updates + 1)
            update_seconds = time.time() - started_update
            assistant_tokens = sum(sum(s.get('loss_mask', [1]*len(s['tokens']))) for s in selected)
            measured.update(update_seconds=update_seconds, assistant_tokens=assistant_tokens,
                            assistant_tokens_per_second=assistant_tokens / max(update_seconds, 1e-9))
            optimizer_metrics.append(measured)
            measured.update({f'memory_{name}_peak_gb': value['peak_bytes'] / 1e9 for name, value in times.memory.items()})
            monitor.optimizer(measured)
            metrics.append({**measured, 'event': 'actor_optimizer_step', 'update': updates + 1,
                'batch_groups': [{'task_id': item['group']['task_id'], 'path': item['path'],
                                  'behavior_version': item['group']['policy_version'],
                                  'replay': item['replay_entry'] is not None, **item['drift']} for item in batch_groups],
                'assistant_tokens': sum(sum(s.get('loss_mask', [1]*len(s['tokens']))) for s in selected),
                'batch_scoring_max_logp_error': actor.max_numerical_error,
                'loss_aggregation': 'equal_sandbox_equal_rollout_token_mean',
                'started_at': started_update, 'finished_at': time.time()})
            if config.metrics_export_interval and optimizer_step % config.metrics_export_interval == 0:
                metrics.export()
                optimizer_metrics.export()
            print(json.dumps(metrics[-1]), flush=True)
            exceeded = measured['post_kl_exceeded']
            if exceeded:
                break
        return exceeded

    def prepared_records(*, only_untrained=False):
        items = []
        for record in inflight['records']:
            if sha256(record['path']) != record['sha256']:
                raise ValueError('in-flight rollout artifact checksum mismatch')
            if not record['eligible'] or (only_untrained and record.get('trained', True)):
                continue
            group = json.loads(Path(record['path']).read_text())
            validate_actor_group(group)
            attach_contract(group)
            items.append({'group': group, 'samples': make_samples(group, config.algorithm,
                config.gamma, config.gae_lambda, config.discount_unit), 'path': record['path'],
                'replay_entry': None, 'drift': record['drift'], 'lag': record['lag']})
        return items

    def validate_actor_group(group):
        validate_group(group, tasks)
        if group.get('policy_identity') != policy_identity:
            raise ValueError('trajectory policy identity mismatch')

    loader = DatasetResults(schedule, results, pool, rollout_positions, rollout_workers=config.rollout_workers,
                            timeout=config.rollout_timeout, progress_timeout=config.train_progress_timeout, validate=validate_actor_group, next_index=0,
                            completed=consumed_jobs, task_queue=task_queue,
                            on_poll=lambda: monitor.runtime.resources(mx, **{'queue/pending_groups': len(loader.pending)}))
    for record in inflight.get('records', []):
        if sha256(record['path']) != record['sha256']:
            raise ValueError('in-flight rollout artifact checksum mismatch')
    try:
        if not config.resume:
            evaluate_periodically(force=True)
            commit(force=True)
        if dataset_cursor < len(schedule.batches):
            pool.start_all()
        while dataset_cursor < len(schedule.batches):
            jobs = schedule.batches[dataset_cursor]
            if not inflight:
                inflight = {'records': [], 'next_mini': 0, 'steps_start': optimizer_step,
                            'passes_completed': 0, 'replay_completed': 0, 'stop_fresh': False}
            remaining = len(jobs) - sum(record.get('trained', True) for record in inflight['records'])
            if config.max_groups and groups + remaining - sum(not record.get('trained', True) for record in inflight['records']) > config.max_groups:
                raise RuntimeError('fresh group budget cannot cover next dataset batch')
            while remaining:
                wanted = min(config.mini_batch_size, remaining)
                fresh = prepared_records(only_untrained=True)
                # Only accepted complete groups count toward this optimizer mini-batch.
                while len(fresh) < wanted:
                    if config.max_groups and groups >= config.max_groups:
                        raise RuntimeError('oversampling exhausted max_groups before filling qualified batch')
                    with times.measure('actor_wait_ready_minibatch'):
                        ready = next(loader.ready_count(jobs[0]['dataset_epoch'], 1, 1))
                    group = ready[0]
                    groups += 1
                    index = group['dataset_index']
                    path = output / f'group-{groups:04d}.json'
                    attach_contract(group)
                    with times.measure('actor_rollout_io'):
                        write_json(path, group)
                    times.ingest_rollout(group)
                    monitor.group(group, groups, updates)
                    metrics.append({'event': 'rollout_received', 'task_id': group['task_id'],
                        'dataset_index': index, 'group': groups, 'started_at': group.get('started_at'),
                        'finished_at': group.get('finished_at'),
                        'environment_overlap': environment_overlap([group])})
                    with times.measure('actor_prepare_targets'):
                        item = prepare(group, path)
                    if 'rejected' in item:
                        attempt = group.get('rollout_attempt', 0)
                        disposition = rejection_action(item['rejected'], config.zero_variance_policy,
                                                       attempt, config.rollout_max_attempts)
                        if disposition == 'fail':
                            raise RuntimeError(f'oversampling exhausted {attempt + 1} attempts for sandbox {group["task_id"]}: {item["rejected"]}')
                        if disposition == 'skip':
                            consumed_jobs.add(index)
                            skipped_jobs[str(index)] = item['rejected']
                            inflight['records'].append({'dataset_index': index, 'path': str(path),
                                'sha256': sha256(path), 'eligible': False, 'trained': True})
                            remaining -= 1
                            wanted = min(config.mini_batch_size, remaining)
                            monitor.runtime.emit('task/skipped', text=item['rejected'], dataset_index=index,
                                task_id=group['task_id'], metrics={'admission/skipped_jobs': len(skipped_jobs)})
                        else:
                            rollout_attempts[str(index)] = attempt + 1
                            loader.completed.discard(index)
                        commit()
                        continue
                    consumed_jobs.add(index)
                    replay.add(path, group, updates)
                    inflight['records'].append({'dataset_index':index, 'path':str(path),
                        'sha256':sha256(path), 'eligible':True, 'drift':item['drift'], 'lag':item['lag'],
                        'trained':False})
                    fresh.append(item)
                if not fresh:
                    break
                if not inflight['stop_fresh']:
                    inflight['stop_fresh'] = optimize(fresh, jobs[0]['dataset_epoch'], 0,
                                                      mini_offset=inflight['next_mini'])
                for record in inflight['records']:
                    record['trained'] = True
                inflight['next_mini'] += 1
                remaining -= wanted
                commit()
            inflight['passes_completed'] = max(1, inflight['passes_completed'])
            while inflight['passes_completed'] < config.optimization_passes and not inflight['stop_fresh']:
                number = inflight['passes_completed'] + 1
                inflight['stop_fresh'] = optimize(prepared_records(), jobs[0]['dataset_epoch'], 0, number)
                inflight['passes_completed'] = number
                commit()
            while inflight['replay_completed'] < config.replay_batches_per_batch:
                cycle = inflight['replay_completed'] + 1
                replay_batch = []
                for record in inflight['records']:
                    if not record['eligible']:
                        continue
                    job = schedule.jobs[record['dataset_index']]
                    selected = replay.sample(updates, tasks, task_id=job['task_id'])
                    if selected:
                        group, entry = selected
                        item = prepare(group, entry['path'], entry)
                        if 'rejected' not in item:
                            replay_batch.append(item)
                for number in range(1, config.optimization_passes + 1):
                    if optimize(replay_batch, jobs[0]['dataset_epoch'], cycle, number):
                        break
                inflight['replay_completed'] = cycle
                commit()
            if optimizer_step > published_step and (optimizer_step - published_step >= config.policy_publish_interval_steps or dataset_cursor + 1 == len(schedule.batches)):
                with times.measure('actor_weight_publish'):
                    updates += 1
                    policy.save_snapshot(output / 'snapshots', updates)
                    published_step = optimizer_step
                    version.value = updates
            metrics.append({'event': 'dataset_batch_completed', 'epoch': jobs[0]['dataset_epoch'],
                            'dataset_batch': dataset_cursor, 'sandbox_ids': [schedule.jobs[record['dataset_index']]['task_id'] for record in inflight['records']],
                            'fresh_rollouts': len(jobs) * config.rollout_group,
                            'group_paths': [record['path'] for record in inflight['records']]})
            dataset_cursor += 1
            inflight = {}
            evaluate_periodically()
            commit()
            mx.clear_cache()
        evaluate_periodically(force=True)
        commit(force=True)
    finally:
        pool.close()
        prune_live_artifacts(output, task_queue, committed_version, keep=config.checkpoint_keep,
                             group_keep=config.group_artifacts_keep)
        metrics.export()
        optimizer_metrics.export()
    if dataset_cursor != len(schedule.batches):
        raise RuntimeError('dataset traversal incomplete; no completion claim')
    final_digest = policy.digest()
    final_policy_digest = policy.digest(policy_only=True)
    final_effective_digest = policy.digest(effective=True) if config.tuning == 'qat' else None
    policy_changed = initial_policy_digest != final_policy_digest
    effective_changed = initial_effective_digest != final_effective_digest if config.tuning == 'qat' else None
    if optimizer_step and not policy_changed:
        raise ValueError('only critic or no parameters changed; policy training is unproven')
    full_export = policy.export_full(output / 'full_model') if config.tuning == 'full' else None
    qat_layers = policy.export_qat(output)
    lora_export = None
    lora_validation = None
    if config.tuning == 'lora':
        from .lora_export import export, validate
        lora_export = export(policy, output, merge=config.lora_merge_export, requantize=config.lora_requantize_export)
        lora_validation = validate(policy, lora_export, requantize=config.lora_requantize_export)
        write_json(output / 'lora_export_validation.json', lora_validation)
    policy.restore(initial)
    before = evaluate(policy, tasks, config)
    monitor.evaluation('before', before, optimizer_step)
    policy.restore(output / 'snapshots' / f'policy-{updates:06d}.safetensors')
    after = evaluate(policy, tasks, config)
    monitor.evaluation('after', after, optimizer_step)
    export_validation = None
    if config.tuning == 'qat':
        probes = []
        for episode in after['episodes']:
            for action in episode['actions']:
                ids = mx.array([action['prompt'] + action['tokens']])
                logp, _ = policy.token_stats(ids, len(action['prompt']))
                mx.eval(logp)
                probes.append((ids, len(action['prompt']), logp))
        deployed_model = output / 'qat_model' if config.qat_scope == 'full' else config.model
        deployed_policy = Policy(str(deployed_model), 'inference', temperature=config.temperature,
                                 thinking_mode=config.thinking_mode, logits_chunk_size=config.logits_chunk_size)
        if config.qat_scope != 'full':
            deployed_policy.load_qat_export(output)
        max_error = max((float(mx.max(mx.abs(deployed_policy.token_stats(ids, length)[0] - logp)))
                        for ids, length, logp in probes), default=0.)
        if not math.isfinite(max_error) or max_error > .1:
            raise ValueError(f'packed QAT export log-probability mismatch: {max_error}')
        deployed = evaluate(deployed_policy, tasks, config)
        monitor.evaluation('deployed', deployed, optimizer_step)
        write_json(output / 'evaluation_deployed.json', deployed)
        write_json(output / 'deployment_identity.json', {'policy_identity':policy_identity,
                                                        'model_files':model_files})
        export_validation = {'packed_reload_verified': bool(probes), 'probe_count': len(probes), 'fresh_model_instance': True,
            'max_assistant_logp_error': max_error if probes else None, 'tolerance': .1,
            'deployed_reward': deployed['mean_reward']}
    write_json(output / 'evaluation_before.json', before)
    write_json(output / 'evaluation_after.json', after)
    from .timing import rollout_training_overlap
    overlap = rollout_training_overlap(metrics)
    report = {'async_overlap_observed': overlap,
              'rollout_concurrency': config.rollout_concurrency,
              'over_sampling_batch_size': config.sampling_pool_size, 'rollout_max_attempts': config.rollout_max_attempts,
              'sampling_attempts': groups, 'accepted_groups': len(consumed_jobs) - len(skipped_jobs),
              'visited_jobs': len(consumed_jobs), 'skipped_jobs': len(skipped_jobs),
              'updated_jobs': len(updated_jobs), 'training_updates_observed': optimizer_step > 0,
              'batch_size': config.training_batch_size, 'mini_batch_size': config.training_mini_batch_size,
              'critic_enabled': config.algorithm == 'ppo',
              'role_names': {'actor': 'policy trainer', 'rollout_worker': 'sampling process'},
              'micro_batch_size': config.micro_batch_size, 'max_physical_batch_observed': actor.max_physical_batch,
              'batch_scoring_max_logp_error': actor.max_numerical_error,
              'batch_numeric_fallbacks': actor.numeric_fallbacks,
              'batched_gradient_microbatches': actor.batched_gradient_microbatches,
              'cached_gradient_microbatches': actor.cached_gradient_microbatches,
              'timings': times.state(),
              'epochs_completed': config.epochs, 'dataset_batches_completed': dataset_cursor,
              'sandboxes_per_epoch': len([task for task in tasks if task.split == 'train']),
              'fresh_groups': groups, 'rollout_group': config.rollout_group,
              'environment_rpc_overlap_observed': any(m.get('environment_overlap') for m in metrics), 'queue_capacity': config.queue_size,
              'completed': dataset_cursor == len(schedule.batches) and (optimizer_step == 0 or (policy_changed and effective_changed is not False)),
              'policy_weights_changed': policy_changed, 'effective_qat_weights_changed': effective_changed,
              'initial_policy_sha256': initial_policy_digest, 'final_policy_sha256': final_policy_digest,
              'initial_effective_sha256': initial_effective_digest, 'final_effective_sha256': final_effective_digest,
              'algorithm': config.algorithm, 'tuning': config.tuning, 'qat_scope': config.qat_scope, 'thinking_mode': config.thinking_mode,
              'full_model_export': full_export, 'lora_export': lora_export,
              'lora_export_validation': lora_validation, 'updates': updates,
              'optimizer_steps': optimizer_step, 'discount_unit': config.discount_unit,
              'initial_parameters_sha256': initial_digest, 'final_parameters_sha256': final_digest,
              'tasks': tasks_state(tasks), 'independent_eval': after['independent_eval'],
              'evaluation_by_task': after['by_task'], 'evaluation_by_split': after['by_split'],
              'before_reward': before['mean_reward'], 'after_reward': after['mean_reward'],
              'qat_export_layers': qat_layers, 'qat_export_validation': export_validation,
              'dropped_groups': dropped,
              'dropped_stale_groups': sum(m.get('skipped') == 'policy_age' for m in metrics),
              'seconds': previous_seconds + time.time() - started, 'device': str(mx.default_device()),
              'peak_metal_gb': max([mx.get_peak_memory()] + [r['peak_bytes'] for r in times.memory.values()]) / 1e9,
              'sandbox_transport': 'http' if config.sandbox_services else 'in_process',
              'best_heldout_policy': eval_state.get('best'),
              'scope': 'dataset traversal completed; inspect updated_jobs and evaluation for learning evidence'}
    write_json(output / 'training_report.json', report)
    monitor.runtime.emit('run/report', text=json.dumps(report, ensure_ascii=False, indent=2))
    monitor.runtime.emit('run/state', text='completed' if report['completed'] else 'failed')
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--sandbox', default='' , help='qualified sandbox directory; training_ready must be true')
    parser.add_argument('--output', required=True, help='new training output directory')
    for name, field in Config.__dataclass_fields__.items():
        if name in {'sandbox', 'output'}:
            continue
        default = "docker" if name == "sandbox_backend" else field.default
        if name == 'tensorboard':
            parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, default=default,
                                help='write live TensorBoard scalars and rollout text')
            continue
        if isinstance(default, bool):
            parser.add_argument('--' + name.replace('_', '-'), action='store_true', help=name.replace('_', ' '))
            continue
        choices = {'sandbox_backend': ['local', 'docker'], 'algorithm': ['ppo', 'grpo'], 'tuning': ['qat', 'lora', 'full'], 'thinking_mode': ['auto', 'thinking', 'no-thinking'], 'qat_scope': ['projections', 'full'], 'bits': [4, 8], 'discount_unit': ['action', 'token'],
                   'zero_variance_policy': ['retry_skip', 'retry_fail', 'skip', 'fail'],
                   'context_limit_policy': ['finish', 'fail'],
                   'numerical_check_mode': ['strict', 'periodic']}.get(name)
        parser.add_argument('--' + name.replace('_', '-'), type=type(default), default=default, choices=choices, help={'over_sampling_batch_size': 'candidate pool size; 0 means twice batch_size',
                  'rollout_max_attempts': 'maximum sampling attempts per sandbox visit, including initial attempt',
                  'batch_size': 'completed sandbox groups per batch within an epoch (keeps final partial batch)',
                  'rollout_workers': 'sampling processes; actor means policy trainer',
                  'micro_batch_size': 'maximum action segments per physical tensor batch',
                  'max_tokens_per_micro_batch': 'maximum padded input tokens per physical tensor batch',
                  'batch_logp_tolerance': 'maximum batch/cached logprob discrepancy; sampled logprobs are never replaced',
                  'environment_factory': 'AgentRuntime factory as module:callable',
                  'sandbox_backend': 'docker auto-manages local containers; local uses in-process runtime',
                  'sandbox_images': 'optional task-ID to Docker image JSON; defaults to qualified image metadata',
                  'sandbox_services': 'existing Docker/Kubernetes services.json; skips auto-management',
                  'mini_batch_size': 'sandboxes per optimizer step; all their rollouts stay together',
                  'rollout_group': 'rollout trajectories sampled per sandbox visit',
                  'epochs': 'complete passes through the training sandbox dataset',
                  'optimization_passes': 'optimization passes over a collected batch; separate from dataset epochs',
                  'updates': 'optional policy update cap; 0 means no extra cap',
                  'max_groups': 'optional fresh sandbox-visit cap; 0 means no extra cap',
                  'replay_batches_per_batch': 'extra replay batches after each fresh dataset batch; does not advance epoch'}.get(name, name.replace('_', ' ')))
    config = Config(**vars(parser.parse_args()))
    try:
        config.validate()
    except ValueError as error:
        parser.error(str(error))
    train(config)


if __name__ == '__main__':
    main()
