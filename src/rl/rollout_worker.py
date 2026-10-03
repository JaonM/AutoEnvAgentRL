"""Independent rollout models, group-boundary task/version changes and heartbeats."""
from pathlib import Path
import queue
import json
import os
import time
import traceback

from .checkpoint import atomic_json
from .process_control import pin_policy
from .errors import InfrastructureError, ContextBudgetExceeded
from .contracts import attach_contract
from .tasks import TaskSpec, load_tasks
from .task_queue import TaskQueue
from .parallel_rollout import EnvironmentPool, parallel_rollouts
from .trajectory import assign_episode_reward


def rollout(policy, environment, config, seed, *, greedy=False, stop=None, notify=None, observe=None):
    notify = notify or (lambda stage: None)
    observe = observe or (lambda event, **fields: None)
    notify('reset')
    environment.reset(seed)
    observe('reset', rollout_index=0, step=0, messages=environment.messages)
    tool_options = {'tools': environment.tools} if hasattr(environment, 'tools') else {}
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
        from .model import Policy
        from .model_options import policy_options
        from .telemetry import worker_logger, RolloutMonitor
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
        while not stop.is_set():
            ready = None
            if config.get('sandbox_services'):
                from .remote_environment import docker_task_ready
                ready = lambda job: docker_task_ready(config['sandbox_services'], job['task_id'])
            with task_queue.claim(ready=ready) as job:
                if job is None:
                    group_started_at = None
                    notify('task_queue_wait')
                    stop.wait(.1)
                    continue
                group_started_at = time.time()
                count = job['dataset_index']
                attempt = job.get('rollout_attempt', 0)
                continuation = Path(config['_task_queue']) / f'continuation-{count}-attempt-{attempt}'
                continuation.mkdir(exist_ok=True)
                metadata_path = continuation / 'group.json'
                def choose_version():
                    if metadata_path.exists():
                        return json.loads(metadata_path.read_text())['policy_version']
                    selected = shared_version.value
                    atomic_json(metadata_path, {'policy_version': selected})
                    return selected
                with pin_policy(Path(config['output']) / 'snapshots', choose_version) as current:
                    mx.random.seed(config['seed'] + 1000 + count + attempt * 1000003)
                    if version != current:
                        notify('loading_snapshot')
                        sync_started = time.perf_counter()
                        policy.restore(Path(config['output']) / 'snapshots' / f'policy-{current:06d}.safetensors')
                        version = current
                        snapshot_seconds += time.perf_counter() - sync_started
                    task = next(task for task in tasks if task.id == job['task_id'])
                    if current_task != task:
                        if env is not None:
                            env.close()
                        current_task = task
                        notify('loading_environment')
                        env = EnvironmentPool(task.sandbox,min(config['rollout_concurrency'],config['rollout_group']),
                                              factory=config.get('environment_factory', 'rl.environment:SandboxEpisode'))
                    started = time.time()
                    seed = job['seed']
                    observer = RolloutMonitor(logger, {'run_id': config.get('_run_id'),
                        'dataset_index': count, 'attempt': attempt, 'task_id': task.id,
                        'policy_version': version, 'seed': seed},
                        samples=config.get('rollout_trace_samples', 1), max_chars=config.get('rollout_trace_max_chars', 16000))
                    episodes = parallel_rollouts(policy,env,config,seed,stop=stop,notify=notify,
                                                 continuation=continuation, observe=observer)
                    for episode in episodes:
                        episode['task_id'] = task.id
                    group = {**job, 'schema_version':2, 'rollout_worker':index, 'group_id':f'{config.get("_run_id", "local")}/job-{count}-attempt-{attempt}', 'group_index':count,
                             'task_id':task.id, 'task_identity':task.identity,
                             'rollout_rng_after':[key.tolist() for key in mx.random.state], 'policy_version':version,
                             'started_at':started, 'finished_at':time.time(), 'episodes':episodes,
                             'policy_identity':config.get('_policy_identity'), 'temperature':config['temperature']}
                    attach_contract(group)
                    group['timings'] = {'weight_load': snapshot_seconds}
                    snapshot_seconds = 0.
                    group['shared_task'] = True
                    task_queue.publish(group)
                    logger.flush()
                if config.get('sandbox_services') and env is not None:
                    env.close()
                    env = None
                    current_task = None
                # The durable result is authoritative; a full notification queue cannot block workers.
                try:
                    results.put_nowait(group)
                except queue.Full:
                    pass
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
