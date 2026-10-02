"""Independent rollout models, group-boundary task/version changes and heartbeats."""
from pathlib import Path
import queue
import json
import os
import time
import traceback

from .checkpoint import atomic_json
from .errors import InfrastructureError
from .contracts import attach_contract
from .tasks import TaskSpec, load_tasks
from .task_queue import TaskQueue
from .parallel_rollout import EnvironmentPool, parallel_rollouts
from .trajectory import assign_episode_reward


def rollout(policy, environment, config, seed, *, greedy=False, stop=None, notify=None):
    notify = notify or (lambda stage: None)
    notify('reset')
    environment.reset(seed)
    actions = []
    for _ in range(config['max_steps']):
        if stop is not None and stop.is_set():
            raise InterruptedError('rollout cancelled')
        notify('policy')
        sample = policy.sample(environment.messages, config['max_tokens'], config['max_context'], greedy=greedy)
        notify('environment')
        delta, terminated = environment.step(sample['text'])
        sample.update(reward=delta, terminated=terminated)
        actions.append(sample)
        if terminated:
            break
    bootstrap_prompt = [] if environment.terminated else policy.encode(environment.messages)
    notify('bootstrap')
    # Step-limit completion is scored as the end of this finite episode too.
    bootstrap = 0.0
    result = environment.finish()
    assign_episode_reward(actions, result)
    result.update(actions=actions, bootstrap=bootstrap, bootstrap_prompt=bootstrap_prompt, seed=seed,
                  truncated=not environment.terminated,
                  finish_reason='terminated' if environment.terminated else 'step_limit')
    return result


def rollout_worker_main(index, config, shared_version, results, stop):
    env = None
    current_task = None
    stage = 'starting'
    count = 0
    heartbeat = Path(config['output']) / 'rollout_workers' / f'worker-{index}.json'

    def notify(value):
        nonlocal stage
        stage = value
        atomic_json(heartbeat, {'rollout_worker':index, 'pid':os.getpid(), 'stage':stage, 'at':time.time(), 'group_index':count,
                                'task_id':current_task.id if current_task else None})

    try:
        import mlx.core as mx
        from .model import Policy
        mx.set_default_device(mx.gpu)
        mx.random.seed(config['seed'] + 1000 + index)
        notify('loading_model')
        policy = Policy(config['model'], config['tuning'], config['layers'], config['rank'], config['bits'], temperature=config['temperature'], critic=config['algorithm'] == 'ppo')
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
                    notify('task_queue_wait')
                    stop.wait(.1)
                    continue
                count = job['dataset_index']
                attempt = job.get('rollout_attempt', 0)
                continuation = Path(config['_task_queue']) / f'continuation-{count}-attempt-{attempt}'
                continuation.mkdir(exist_ok=True)
                metadata_path = continuation / 'group.json'
                if metadata_path.exists():
                    metadata = json.loads(metadata_path.read_text())
                    current = metadata['policy_version']
                else:
                    current = shared_version.value
                    atomic_json(metadata_path, {'policy_version': current})
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
                episodes = parallel_rollouts(policy,env,config,seed,stop=stop,notify=notify, continuation=continuation)
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
        try:
            results.put(message, timeout=3)
        except queue.Full:
            atomic_json(heartbeat, {**message, 'at':time.time()})
        raise
    finally:
        if env is not None:
            env.close()
