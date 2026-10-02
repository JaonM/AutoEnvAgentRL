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
import uuid

from .trajectory import make_samples
from .metrics import WeightedMetrics
from .batching import combine_samples, minibatches
from .dataset import DatasetSchedule, DatasetResults
from .checkpoint import save_checkpoint, restore_checkpoint, atomic_json


@dataclass
class Config:
    sandbox: str = ""
    output: str = ""
    tasks: str = ""
    eval_episodes: int = 3
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
        if self.algorithm not in {'ppo', 'grpo'} or self.tuning not in {'qat', 'lora'}:
            raise ValueError('unsupported algorithm or tuning mode')
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
        raise
    atomic_json(status_path, {'run_id':run_id, 'state':'completed', 'finished_at':time.time()})
    return report


def _train(config: Config, run_id):
    config.validate()
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from dotenv import load_dotenv
    from .tasks import load_tasks, tasks_state
    from .evaluation import evaluate
    from .replay import validate_group, ReplayBuffer, import_replay, importance_diagnostics, validate_ppo_behavior
    from .retention import prune_checkpoints, prune_snapshots_after_shutdown
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
    if config.resume:
        previous = json.loads((output / 'config.json').read_text())
        mutable = {'resume', 'updates', 'max_groups', 'rollout_timeout', 'epochs'}
        for key, value in asdict(config).items():
            if key not in mutable and previous.get(key) != value:
                raise ValueError(f'resume configuration changed: {key}')
        if not (output / 'checkpoints/latest.json').exists():
            raise ValueError('no committed resumable checkpoint')
    elif (output / 'config.json').exists():
        raise ValueError('use --resume or a new output; existing run is never overwritten')
    policy_identity = {"base": base_identity, "tuning": config.tuning,
                       "layers": config.layers, "rank": config.rank, "bits": config.bits,
                       "temperature": config.temperature}
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
    mx.set_default_device(mx.gpu)
    if not mx.metal.is_available():
        raise RuntimeError('Apple Metal GPU is required')
    mx.random.seed(config.seed)
    policy = Policy(config.model, config.tuning, config.layers, config.rank, config.bits, temperature=config.temperature, critic=config.algorithm == 'ppo')
    initial_digest = policy.digest()
    initial_policy_digest = policy.digest(policy_only=True)
    initial_effective_digest = policy.digest(effective=True) if config.tuning == 'qat' else None
    initial = str(output / 'snapshots/policy-000000.safetensors')
    if not config.resume:
        policy.save_snapshot(output / 'snapshots', 0)
    reference = Policy(config.model, config.tuning, config.layers, config.rank, config.bits, temperature=config.temperature, critic=False)
    reference.restore(initial, policy_only=True)
    reference.freeze()
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    optimizer.init(policy.trainable_parameters())
    restored = restore_checkpoint(output / 'checkpoints', policy, optimizer) if config.resume else {}
    times = StageTimes(restored.get('timings'))
    inflight = restored.get('inflight_batch', {})
    actor = ActorTrainer(policy, reference, optimizer, config, times)
    actor.restore_state(restored.get('actor_numeric_state', {}))
    metrics = restored.get('metrics', [])
    optimizer_metrics = restored.get('optimizer_metrics', [])
    updates, groups, dropped = (restored.get(key, 0) for key in ('updates', 'groups', 'dropped'))
    optimizer_step = restored.get('optimizer_step', 0)
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
        write_json(output / 'metrics.json', metrics)
        write_json(output / 'optimizer_metrics.json', optimizer_metrics)
        if not inflight:
            policy.save_snapshot(output / 'snapshots', updates)
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
    def commit():
        nonlocal committed_version
        checkpoint_started = time.perf_counter()
        save_checkpoint(output / 'checkpoints', policy, optimizer, {
            'updates': updates, 'groups': groups, 'dropped': dropped,
            'optimizer_step': optimizer_step, 'metrics': metrics,
            'optimizer_metrics': optimizer_metrics, **pool.checkpoint_state(),
            'replay': replay.state(), 'dataset_cursor': dataset_cursor,
            'inflight_batch': inflight, 'consumed_jobs': sorted(consumed_jobs), 'rollout_attempts': rollout_attempts, 'timings': times.state(), 'actor_numeric_state': actor.state(),
            'initial_policy_digest': initial_policy_digest, 'initial_effective_digest': initial_effective_digest,
            'initial_digest': initial_digest, 'seconds': previous_seconds + time.time() - started,
        })

        committed_version = updates
        task_queue.commit_consumed(consumed_jobs, rollout_attempts)
        prune_checkpoints(output / 'checkpoints', config.checkpoint_keep)
        times.add('actor_checkpoint', time.perf_counter() - checkpoint_started)
        write_json(output / 'timings.json', times.state())

    def prepare(group, path, replay_entry=None):
        nonlocal dropped
        attach_contract(group)
        if len(group['episodes']) != config.rollout_group:
            raise ValueError('rollout group size does not match configured rollout_group')
        lag = updates - group['policy_version']
        age = updates - replay_entry['age_origin'] if replay_entry else lag
        max_age = config.replay_max_age if replay_entry else config.max_policy_lag
        reason = None
        drift = {}
        started_prepare = time.monotonic()
        if age < 0 or age > max_age:
            reason = 'policy_age'
        else:
            if config.algorithm == 'ppo':
                validate_ppo_behavior(group)
            drift = measure_drift(group)
            if drift['effective_sample_fraction'] < config.replay_min_ess or drift['max_abs_log_ratio'] > config.replay_max_log_ratio:
                reason = 'importance_drift'
        samples = [] if reason else make_samples(group, config.algorithm, config.gamma, config.gae_lambda, config.discount_unit)
        if not reason and config.algorithm == 'grpo' and all(abs(s['advantage']) < 1e-8 for s in samples):
            reason = 'zero_reward_variance'
        if reason:
            dropped += 1
            metrics.append({'task_id': group['task_id'], 'dataset_index':group.get('dataset_index'),
                            'rollout_attempt':group.get('rollout_attempt', 0), 'dataset_batch': dataset_cursor,
                            'replay': replay_entry is not None, 'skipped': reason, **drift})
            write_json(output / 'metrics.json', metrics)
            return None
        return {'group': group, 'samples': samples, 'path': str(path), 'replay_entry': replay_entry,
                'drift': drift, 'lag': lag, 'queue_seconds': 0.,
                'target_seconds': time.monotonic() - started_prepare}

    def optimize(batch_groups, dataset_epoch, replay_cycle, optimization_pass=1, mini_offset=0):
        nonlocal optimizer_step, dropped
        if not batch_groups:
            return False
        if config.updates and updates >= config.updates:
            raise RuntimeError('policy update budget reached before dataset schedule finished')
        samples = combine_samples([item['samples'] for item in batch_groups], config.rollout_group)
        batch = [(sample, None, None) for sample in samples]
        exceeded = False
        for mini_index, mini in enumerate(minibatches(batch, config.mini_batch_size,
                seed=config.seed + dataset_cursor * config.optimization_passes + optimization_pass), 1 + mini_offset):
            started_update = time.time()
            selected = [sample for sample, _, _ in mini]
            measured = actor.step(selected, notify=pool.raise_if_failed)
            common = {'epoch': dataset_epoch, 'dataset_batch': dataset_cursor, 'replay_cycle': replay_cycle,
                      'optimization_pass': optimization_pass, 'mini_batch': mini_index,
                      'post_kl_scope': 'ready_mini_batch',
                      'mini_batch_sandboxes': len({s['sandbox_index'] for s in selected}),
                      'mini_batch_rollouts': len({s['episode_index'] for s in selected})}
            if measured.get('skipped'):
                metrics.append({**common, **measured})
                if measured['skipped'] == 'target_kl':
                    return True
                continue
            pool.raise_if_failed()
            optimizer_step += 1
            measured.update(**common, optimizer_step=optimizer_step, policy_version=updates + 1)
            optimizer_metrics.append(measured)
            metrics.append({**measured, 'event': 'actor_optimizer_step', 'update': updates + 1,
                'batch_groups': [{'task_id': item['group']['task_id'], 'path': item['path'],
                                  'behavior_version': item['group']['policy_version'],
                                  'replay': item['replay_entry'] is not None, **item['drift']} for item in batch_groups],
                'assistant_tokens': sum(sum(s.get('loss_mask', [1]*len(s['tokens']))) for s in selected),
                'batch_scoring_max_logp_error': actor.max_numerical_error,
                'loss_aggregation': 'equal_sandbox_equal_rollout_token_mean',
                'started_at': started_update, 'finished_at': time.time()})
            write_json(output / 'optimizer_metrics.json', optimizer_metrics)
            write_json(output / 'metrics.json', metrics)
            print(json.dumps(metrics[-1]), flush=True)
            exceeded = measured['post_kl_exceeded']
            if exceeded:
                break
        return exceeded

    def prepared_records():
        items = []
        for record in inflight['records']:
            if sha256(record['path']) != record['sha256']:
                raise ValueError('in-flight rollout artifact checksum mismatch')
            if not record['eligible']:
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
                            timeout=config.rollout_timeout, validate=validate_actor_group, next_index=0,
                            completed=consumed_jobs, task_queue=task_queue)
    for record in inflight.get('records', []):
        if sha256(record['path']) != record['sha256']:
            raise ValueError('in-flight rollout artifact checksum mismatch')
    if not config.resume:
        commit()
    try:
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
                fresh = ([item for item, record in zip(prepared_records(), inflight['records']) if not record.get('trained', True)]
                         if any(not record.get('trained', True) for record in inflight['records']) else [])
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
                    with times.measure('actor_prepare_targets'):
                        item = prepare(group, path)
                    if item is None:
                        attempt = group.get('rollout_attempt', 0)
                        if attempt + 1 >= config.rollout_max_attempts:
                            raise RuntimeError(f'oversampling exhausted {config.rollout_max_attempts} attempts for sandbox {group["task_id"]}')
                        rollout_attempts[str(index)] = attempt + 1
                        loader.completed.discard(index)
                        # Persist accepted-but-not-yet-trained groups as well as retry admission.
                        commit()
                        continue
                    consumed_jobs.add(index)
                    replay.add(path, group, updates)
                    inflight['records'].append({'dataset_index':index, 'path':str(path),
                        'sha256':sha256(path), 'eligible':True, 'drift':item['drift'], 'lag':item['lag'],
                        'trained':False})
                    fresh.append(item)
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
                    job = schedule.jobs[record['dataset_index']]
                    selected = replay.sample(updates, tasks, task_id=job['task_id'])
                    if selected:
                        group, entry = selected
                        item = prepare(group, entry['path'], entry)
                        if item:
                            replay_batch.append(item)
                for number in range(1, config.optimization_passes + 1):
                    if optimize(replay_batch, jobs[0]['dataset_epoch'], cycle, number):
                        break
                inflight['replay_completed'] = cycle
                commit()
            if optimizer_step > inflight['steps_start']:
                with times.measure('actor_weight_publish'):
                    updates += 1
                    policy.save_snapshot(output / 'snapshots', updates)
                    mx.save_safetensors(str(output / 'optimizer.safetensors'),
                        {key: mx.array(value) for key, value in tree_flatten(optimizer.state)})
                    version.value = updates
            metrics.append({'event': 'dataset_batch_completed', 'epoch': jobs[0]['dataset_epoch'],
                            'dataset_batch': dataset_cursor, 'sandbox_ids': [schedule.jobs[record['dataset_index']]['task_id'] for record in inflight['records']],
                            'fresh_rollouts': len(jobs) * config.rollout_group,
                            'group_paths': [record['path'] for record in inflight['records']]})
            dataset_cursor += 1
            inflight = {}
            write_json(output / 'metrics.json', metrics)
            commit()
            mx.clear_cache()
    finally:
        pool.close()
        prune_snapshots_after_shutdown(output / 'snapshots', committed_version, config.checkpoint_keep,
            protected_versions=[json.loads(path.read_text())['policy_version']
                for path in task_queue.root.glob('continuation-*/group.json')
                if int(path.parent.name.split('-')[1]) not in task_queue.consumed()])
    if dataset_cursor != len(schedule.batches):
        raise RuntimeError('dataset traversal incomplete; no completion claim')
    final_digest = policy.digest()
    final_policy_digest = policy.digest(policy_only=True)
    final_effective_digest = policy.digest(effective=True) if config.tuning == 'qat' else None
    policy_changed = initial_policy_digest != final_policy_digest
    effective_changed = initial_effective_digest != final_effective_digest if config.tuning == 'qat' else None
    if not policy_changed:
        raise ValueError('only critic or no parameters changed; policy training is unproven')
    qat_layers = policy.export_qat(output)
    policy.restore(initial)
    before = evaluate(policy, tasks, config)
    policy.restore(output / 'snapshots' / f'policy-{updates:06d}.safetensors')
    after = evaluate(policy, tasks, config)
    export_validation = None
    if config.tuning == 'qat':
        probes = []
        for episode in after['episodes']:
            for action in episode['actions']:
                ids = mx.array([action['prompt'] + action['tokens']])
                logp, _ = policy.token_stats(ids, len(action['prompt']))
                mx.eval(logp)
                probes.append((ids, len(action['prompt']), logp))
        deployed_policy = Policy(config.model, config.tuning, config.layers, config.rank,
                                 config.bits, temperature=config.temperature, critic=config.algorithm == 'ppo')
        deployed_policy.load_qat_export(output)
        max_error = max(float(mx.max(mx.abs(deployed_policy.token_stats(ids, length)[0] - logp)))
                        for ids, length, logp in probes)
        if not math.isfinite(max_error) or max_error > .1:
            raise ValueError(f'packed QAT export log-probability mismatch: {max_error}')
        deployed = evaluate(deployed_policy, tasks, config)
        write_json(output / 'evaluation_deployed.json', deployed)
        write_json(output / 'deployment_identity.json', {'policy_identity':policy_identity,
                                                        'model_files':model_files})
        export_validation = {'packed_reload_verified': True, 'fresh_model_instance': True,
            'max_assistant_logp_error': max_error, 'tolerance': .1,
            'deployed_reward': deployed['mean_reward']}
    write_json(output / 'evaluation_before.json', before)
    write_json(output / 'evaluation_after.json', after)
    collected = [json.loads(path.read_text()) for path in (output / 'rollout_groups').glob('*.json')]
    overlap = any(g['started_at'] < m['finished_at'] and g['finished_at'] > m['started_at']
                  for g in collected for m in metrics if 'update' in m)
    report = {'async_overlap_observed': overlap,
              'rollout_concurrency': config.rollout_concurrency,
              'over_sampling_batch_size': config.sampling_pool_size, 'rollout_max_attempts': config.rollout_max_attempts,
              'sampling_attempts': groups, 'accepted_groups': len(consumed_jobs),
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
              'environment_rpc_overlap_observed': environment_overlap(
                  [json.loads(path.read_text()) for path in output.glob('group-*.json')]), 'queue_capacity': config.queue_size,
              'completed': dataset_cursor == len(schedule.batches) and updates > 0 and policy_changed and (effective_changed is not False),
              'policy_weights_changed': policy_changed, 'effective_qat_weights_changed': effective_changed,
              'initial_policy_sha256': initial_policy_digest, 'final_policy_sha256': final_policy_digest,
              'initial_effective_sha256': initial_effective_digest, 'final_effective_sha256': final_effective_digest,
              'algorithm': config.algorithm, 'tuning': config.tuning, 'updates': updates,
              'optimizer_steps': optimizer_step, 'discount_unit': config.discount_unit,
              'initial_parameters_sha256': initial_digest, 'final_parameters_sha256': final_digest,
              'tasks': tasks_state(tasks), 'independent_eval': after['independent_eval'],
              'evaluation_by_task': after['by_task'], 'evaluation_by_split': after['by_split'],
              'before_reward': before['mean_reward'], 'after_reward': after['mean_reward'],
              'qat_export_layers': qat_layers, 'qat_export_validation': export_validation,
              'dropped_groups': dropped,
              'dropped_stale_groups': sum(m.get('skipped') == 'policy_age' for m in metrics),
              'seconds': previous_seconds + time.time() - started, 'device': str(mx.default_device()),
              'peak_metal_gb': mx.get_peak_memory() / 1e9,
              'sandbox_transport': 'http' if config.sandbox_services else 'in_process',
              'scope': 'sandbox RL execution; not statistical learning improvement'}
    write_json(output / 'training_report.json', report)
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
        if isinstance(default, bool):
            parser.add_argument('--' + name.replace('_', '-'), action='store_true', help=name.replace('_', ' '))
            continue
        choices = {'sandbox_backend': ['local', 'docker'], 'algorithm': ['ppo', 'grpo'], 'tuning': ['qat', 'lora'], 'bits': [4, 8], 'discount_unit': ['action', 'token']}.get(name)
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
