"""Independent rollout models, group-boundary task/version changes and heartbeats."""
from pathlib import Path
import queue
import json
import os
import time
import traceback
import uuid
from contextlib import ExitStack

from rl.checkpointing.checkpoint import atomic_json
from rl.rollout.process_control import pin_policy
from rl.errors import InfrastructureError, ContextBudgetExceeded, EnvironmentProtocolError
from rl.data.contracts import attach_contract
from rl.data.tasks import TaskSpec, load_tasks
from rl.data.task_queue import TaskQueue
from rl.rollout.scheduler import EnvironmentPool, parallel_rollouts
from rl.data.trajectory import assign_episode_reward


def rollout(policy, environment, config, seed, *, greedy=False, stop=None, notify=None, observe=None):
    notify = notify or (lambda stage: None)
    observe = observe or (lambda event, **fields: None)
    notify('reset')
    environment.reset(seed)
    observe('reset', rollout_index=0, step=0, messages=environment.messages)
    tool_options = {'tools': environment.tools} if hasattr(environment, 'tools') else {}
    from .budgets import admit, execution_budget
    budget_admission = admit(policy, environment.messages, tool_options, config, execution_budget(environment))
    if budget_admission is not None:
        observe('budget_admission', rollout_index=0, step=0, report=budget_admission)
    actions = []
    finish_reason = None
    for _ in range(config['max_steps']):
        if stop is not None and stop.is_set():
            raise InterruptedError('rollout cancelled')
        notify('policy')
        try:
            sample = policy.sample(environment.messages, config['max_tokens'], config['max_context'], greedy=greedy, **tool_options)
        except ContextBudgetExceeded:
            if config.get('context_limit_policy', 'finish') == 'fail':
                raise
            finish_reason = 'context_limit'
            break
        observe('action', rollout_index=0, step=len(actions)+1, sample=sample)
        notify('environment')
        step_started = time.monotonic()
        delta, terminated = environment.step(sample.get('action', sample['text']))
        sample.update(reward=delta, terminated=terminated)
        actions.append(sample)
        observe('step', rollout_index=0, step=len(actions), messages=environment.messages,
                seconds=time.monotonic()-step_started)
        if terminated:
            break
    bootstrap_prompt = [] if environment.terminated else policy.encode(environment.messages, **tool_options)
    notify('bootstrap')
    # Step-limit completion is scored as the end of this finite episode too.
    bootstrap = 0.0
    result = environment.finish()
    if actions:
        assign_episode_reward(actions, result)
    result.update(actions=actions, bootstrap=bootstrap, bootstrap_prompt=bootstrap_prompt, seed=seed,
                  truncated=not environment.terminated,
                  finish_reason=finish_reason or ('terminated' if environment.terminated else 'step_limit'))
    if budget_admission is not None:
        result['budget_admission'] = budget_admission
    observe('finish', rollout_index=0, step=len(actions), messages=environment.messages, result=result)
    return result


def rollout_worker_main(index, config, shared_version, results, stop):
    env = None
    current_task = None
    stage = 'starting'
    count = 0
    logger = None
    mx = None
    last_heartbeat = -float('inf')
    group_started_at = None
    heartbeat = Path(config['output']) / 'rollout_workers' / f'worker-{index}.json'

    def notify(value):
        nonlocal stage, last_heartbeat
        previous_stage = stage
        stage = value
        now = time.monotonic()
        if stage == previous_stage and now - last_heartbeat < config.get('heartbeat_interval', 1.):
            return
        last_heartbeat = now
        atomic_json(heartbeat, {'group_started_at': group_started_at, 'rollout_worker':index, 'pid':os.getpid(), 'stage':stage, 'at':time.time(), 'group_index':count,
                                'task_id':current_task.id if current_task else None})
        if logger is not None:
            if stage != previous_stage:
                logger.emit('worker/state', text=stage, dataset_index=count,
                            task_id=current_task.id if current_task else None)
            logger.resources(mx)

    try:
        import mlx.core as mx
        from rl.backends.mlx.policy import Policy
        from rl.backends.mlx.model_options import policy_options
        from rl.observability.telemetry import worker_logger, RolloutMonitor
        logger = worker_logger(config, index)
        mx.set_default_device(mx.gpu)
        mx.random.seed(config['seed'] + 1000 + index)
        notify('loading_model')
        policy = Policy(config['model'], config['tuning'], config['layers'], config['rank'], config['bits'], temperature=config['temperature'], critic=config['algorithm'] == 'ppo', **policy_options(config, inference=True))
        tasks = ([TaskSpec(**t) for t in config['_tasks']] if '_tasks' in config
                 else load_tasks(sandbox=config['sandbox']))
        task_queue = TaskQueue(config['_task_queue'], config['_dataset_jobs'], config['_queue_capacity'])
        snapshot_seconds = 0.
        version = -1
        from rl.rollout.multiplex import sample_groups
        from rl.rollout.curriculum import Curriculum

        def location(job):
            path = Path(config['_task_queue']) / f"continuation-{job['dataset_index']}-attempt-{job.get('rollout_attempt', 0)}"
            path.mkdir(exist_ok=True)
            return path

        while not stop.is_set():
            with ExitStack() as wave:
                job = wave.enter_context(task_queue.claim())
                if job is None:
                    group_started_at = None
                    notify('task_queue_wait')
                    stop.wait(.1)
                    continue
                group_started_at = time.time()
                count = job['dataset_index']
                metadata_path = location(job) / 'group.json'
                wave_id = uuid.uuid4().hex
                def choose_version():
                    nonlocal wave_id
                    if metadata_path.exists():
                        metadata = json.loads(metadata_path.read_text())
                        wave_id = metadata.get('wave_id', f'legacy-{count}')
                        return metadata['policy_version']
                    selected = shared_version.value
                    atomic_json(metadata_path, {'policy_version': selected, 'wave_id': wave_id})
                    return selected
                current = wave.enter_context(pin_policy(Path(config['output']) / 'snapshots', choose_version))
                if version != current:
                    notify('loading_snapshot')
                    sync_started = time.perf_counter()
                    policy.restore(Path(config['output']) / 'snapshots' / f'policy-{current:06d}.safetensors')
                    version = current
                    snapshot_seconds += time.perf_counter() - sync_started
                claimed = [job]
                def compatible(candidate):
                    path = location(candidate) / 'group.json'
                    if not path.exists():
                        return True
                    metadata = json.loads(path.read_text())
                    return metadata['policy_version'] == current and metadata.get('wave_id') == wave_id
                for _ in range(config.get('rollout_active_groups', 1) - 1):
                    extra = wave.enter_context(task_queue.claim(ready=compatible))
                    if extra is None:
                        break
                    atomic_json(location(extra) / 'group.json', {'policy_version': current, 'wave_id': wave_id})
                    claimed.append(extra)
                if env is None:
                    notify('loading_environment')
                    env = EnvironmentPool(None, config['rollout_concurrency'],
                                          factory=config.get('environment_factory', 'rl.environment:SandboxEpisode'))
                specs, metadata = [], {}
                for job in claimed:
                    task = next(task for task in tasks if task.id == job['task_id'])
                    current_task = task
                    key, attempt = job['dataset_index'], job.get('rollout_attempt', 0)
                    observer = RolloutMonitor(logger, {'run_id': config.get('_run_id'),
                        'dataset_index': key, 'attempt': attempt, 'task_id': task.id,
                        'policy_version': version, 'seed': job['seed']},
                        samples=config.get('rollout_trace_samples', 1), max_chars=config.get('rollout_trace_max_chars', 16000))
                    start, curriculum = None, None
                    start_path = location(job) / 'start.json'
                    if config.get('reverse_curriculum'):
                        curriculum = Curriculum(config['reverse_curriculum'], Path(config['output']) / 'curriculum.json',
                                                threshold=config.get('curriculum_threshold', .6), seed=config['seed'] + key)
                        if start_path.exists():
                            start = json.loads(start_path.read_text())['start']
                        else:
                            start = curriculum.start(task.id)
                            atomic_json(start_path, {'start': start})
                    metadata[key] = (job, task, start, curriculum)
                    specs.append({'key': key, 'sandbox': task.sandbox,
                                  'config': {**config, '_start': start,
                                             '_sampling_seed': config['seed'] + 1000 + key + attempt * 1000003},
                                  'seed': job['seed'], 'continuation': location(job), 'observe': observer})
                try:
                    for key, episodes in sample_groups(policy, env, specs, stop=stop, notify=notify):
                        job, task, start, curriculum = metadata[key]
                        if start:
                            curriculum.record(task.id, start['level'], episodes)
                        for episode in episodes:
                            episode['task_id'] = task.id
                        group = {**job, 'schema_version': 2, 'rollout_worker': index,
                                 'group_id': f"{config.get('_run_id', 'local')}/job-{key}-attempt-{job.get('rollout_attempt', 0)}",
                                 'group_index': key, 'task_identity': task.identity, 'policy_version': version,
                                 'rollout_rng_after': [k.tolist() for k in mx.random.state],
                                 'started_at': group_started_at, 'finished_at': time.time(), 'episodes': episodes,
                                 'policy_identity': config.get('_policy_identity'), 'temperature': config['temperature'],
                                 'shared_task': True, 'wave_id': wave_id, 'wave_groups': len(specs),
                                 'timings': {'weight_load': snapshot_seconds}}
                        attach_contract(group)
                        snapshot_seconds = 0.
                        task_queue.publish(group)
                        logger.flush()
                        try:
                            results.put_nowait(group)
                        except queue.Full:
                            pass
                except EnvironmentProtocolError as error:
                    job, task, _, _ = metadata[error.group_key]
                    task_queue.quarantine(task.id, task.identity, error)
                    logger.emit('task/quarantined', text=str(error), task_id=task.id,
                                dataset_index=job['dataset_index'])
                    # Other groups retain their durable journals. Close every in-flight
                    # connection before reusing slots after a protocol failure.
                    env.close()
                    env = None
    except InterruptedError:
        if not stop.is_set():
            raise
    except Exception as error:
        message = {'rollout_error':traceback.format_exc(), 'rollout_worker':index, 'pid':os.getpid(), 'stage':stage,
                   'retryable':isinstance(error,(InfrastructureError,TimeoutError,OSError)),
                   'error_type':type(error).__name__}
        atomic_json(heartbeat, {**message, 'at':time.time()})
        if logger is not None:
            logger.emit('worker/error', text=message['rollout_error'], stage=stage,
                        retryable=message['retryable'], dataset_index=count)
        try:
            results.put(message, timeout=3)
        except queue.Full:
            atomic_json(heartbeat, {**message, 'at':time.time()})
        raise
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            if logger is not None:
                logger.close()
