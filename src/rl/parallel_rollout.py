"""Concurrent episodes with isolated environment processes and rollout-worker-owned inference.

Only the rollout worker thread touches MLX. Environment RPCs overlap while that thread
serves other ready episodes; each worker owns its imports, SQLite DB and history.
"""
import multiprocessing as mp
import json
import fcntl
from contextlib import ExitStack
from pathlib import Path
from .checkpoint import atomic_json
import os
import threading
import time
import traceback

from .errors import InfrastructureError
from .trajectory import assign_episode_reward


def environment_worker(connection, sandbox, parent_pid, factory="rl.environment:SandboxEpisode"):
    # A forcibly restarted rollout worker must not leave simulator workers running.
    def watch_parent():
        while os.getppid() == parent_pid:
            time.sleep(.5)
        os._exit(0)
    threading.Thread(target=watch_parent, daemon=True).start()
    environment = None
    loaded_path = None
    try:
        from .contracts import create_runtime
        environment = create_runtime(factory, sandbox)
        while True:
            command, argument = connection.recv()
            if command == 'close':
                break
            with ExitStack() as resources:
                durable = None
                if command == 'durable':
                    durable = argument
                    path = Path(durable['path'])
                    lease = resources.enter_context(path.with_suffix('.lock').open('a'))
                    fcntl.flock(lease, fcntl.LOCK_EX)
                    if not all(callable(getattr(environment, name, None)) for name in ('snapshot', 'restore')):
                        raise TypeError('resumable runtime requires snapshot/restore')
                    saved = json.loads(path.read_text()) if path.exists() else None
                    if loaded_path != str(path):
                        if saved:
                            environment.restore(saved['environment'])
                        loaded_path = str(path)
                    command, argument = durable['operation'], durable['argument']
                    if saved and (command == 'resume' or saved['sequence'] == durable['sequence']):
                        if command != 'resume' and (command != saved['operation'] or argument != saved['argument']):
                            raise ValueError('environment continuation request mismatch')
                        connection.send(saved['reply'])
                        continue
                    if command == 'resume':
                        raise ValueError('missing environment continuation')
                    if saved and durable['sequence'] != saved['sequence'] + 1:
                        raise ValueError('environment continuation sequence mismatch')
                started = time.time()
                if command == 'reset':
                    value = environment.reset(argument)
                elif command == 'step':
                    value = environment.step(argument)
                elif command == 'finish':
                    value = environment.finish()
                else:
                    raise ValueError(f'unknown environment RPC: {command}')
                reply = {'value':value, 'messages':environment.messages,
                         'terminated':environment.terminated,
                         'started_at':started, 'finished_at':time.time()}
                if durable is not None:
                    atomic_json(path, {'sequence':durable['sequence'], 'operation':command, 'argument':argument,
                                       'environment':environment.snapshot(), 'reply':reply})
                connection.send(reply)
    except EOFError:
        pass
    except Exception as error:
        try:
            connection.send({'error':traceback.format_exc(),
                             'retryable':isinstance(error,(InfrastructureError,TimeoutError,OSError))})
        except (BrokenPipeError, EOFError):
            pass
    finally:
        if environment is not None:
            environment.close()
        connection.close()


class EnvironmentPool:
    def __init__(self, sandbox, workers, *, target=environment_worker, factory="rl.environment:SandboxEpisode"):
        self.workers = []
        context = mp.get_context('spawn')
        try:
            for _ in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=target, args=(child,sandbox,os.getpid(),factory) if target is environment_worker else (child,sandbox,os.getpid()))
                process.start()
                child.close()
                self.workers.append((parent,process))
        except BaseException:
            self.close()
            raise

    def close(self):
        for connection, process in self.workers:
            try:
                connection.send(('close',None))
            except (BrokenPipeError,EOFError,OSError):
                pass
        for connection, process in self.workers:
            process.join(timeout=1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            connection.close()
        self.workers = []


def parallel_rollouts(policy, pool, config, seed, *, stop=None, notify=None, continuation=None):
    notify = notify or (lambda stage: None)
    count = config['rollout_group']
    completed = [None] * count
    active = {}
    next_episode = 0
    decoder = policy.batch_decoder() if hasattr(policy, 'batch_decoder') else None

    journal = Path(continuation) / 'progress.json' if continuation is not None else None

    def persist():
        if journal is not None:
            atomic_json(journal, {'completed':completed, 'next_episode':next_episode,
                'active':{str(slot):{k:v for k,v in state.items() if k != 'generator'}
                          for slot,state in active.items()}})

    def send(slot, operation, argument):
        state = active[slot]
        state['operation'], state['sent_at'] = operation,time.monotonic()
        state['argument'] = argument
        if operation != 'resume':
            state['sequence'] = state.get('sequence', -1) + 1
        persist()
        if journal is None:
            pool.workers[slot][0].send((operation,argument))
        else:
            pool.workers[slot][0].send(('durable', {'operation':operation, 'argument':argument,
                'sequence':state['sequence'], 'path':str(Path(continuation) / f"episode-{state['index']}.json")}))

    def start(slot):
        nonlocal next_episode
        active[slot] = {'index':next_episode,'actions':[], 'rpc_intervals':[], 'started_at':time.time()}
        next_episode += 1
        send(slot,'reset',seed)

    if journal is not None and journal.exists():
        saved = json.loads(journal.read_text())
        completed, next_episode = saved['completed'], saved['next_episode']
        active = {int(slot):state for slot,state in saved['active'].items()}
        for slot in list(active):
            state = active[slot]
            operation = state['operation']
            if operation in ('policy', 'resume'):
                send(slot, 'resume', None)
            else:
                state['sequence'] -= 1
                send(slot, operation, state['argument'])
    for slot in range(min(len(pool.workers),count)):
        if slot not in active and next_episode < count:
            start(slot)
    while active:
        if stop is not None and stop.is_set():
            raise InterruptedError('parallel rollouts cancelled')
        progressed = False
        for slot in list(active):
            state = active[slot]
            connection, process = pool.workers[slot]
            if state['operation'] == 'policy':
                if decoder is not None:
                    continue
                progressed = True
                try:
                    next(state['generator'])
                    state['generated_steps'] += 1
                    if state['generated_steps'] % 16 == 0:
                        notify(f'rollout-{state["index"]}-policy')
                except StopIteration as result:
                    sample = result.value
                    state['actions'].append(sample)
                    del state['generator']
                    send(slot,'step',sample['text'])
                continue
            if time.monotonic()-state['sent_at'] > config['rollout_timeout']:
                raise TimeoutError(f'rollout {state["index"]} environment {state["operation"]} timed out')
            if not connection.poll():
                if not process.is_alive():
                    raise InfrastructureError(f'environment worker exited: {process.exitcode}')
                continue
            progressed = True
            try:
                reply = connection.recv()
            except EOFError as error:
                raise InfrastructureError('environment worker disconnected') from error
            if 'error' in reply:
                error_type = InfrastructureError if reply['retryable'] else RuntimeError
                raise error_type(reply['error'])
            operation = state['operation']
            state['rpc_intervals'].append({'operation':operation,
                'started_at':reply['started_at'],'finished_at':reply['finished_at']})
            notify(f'rollout-{state["index"]}-{operation}')
            if operation == 'finish':
                result = reply['value']
                assign_episode_reward(state['actions'], result)
                result.update(actions=state['actions'],bootstrap=state['bootstrap'],
                              bootstrap_prompt=state['bootstrap_prompt'],seed=seed,
                              truncated=not reply['terminated'],
                              finish_reason='terminated' if reply['terminated'] else 'step_limit',
                              started_at=state['started_at'],finished_at=time.time(),
                              environment_intervals=state['rpc_intervals'])
                completed[state['index']] = result
                del active[slot]
                if next_episode < count:
                    start(slot)
                persist()
                continue
            if operation == 'step':
                delta, terminal = reply['value']
                state['actions'][-1].update(reward=delta,terminated=terminal)
            if reply['terminated'] or len(state['actions']) >= config['max_steps']:
                prompt = [] if reply['terminated'] else policy.encode(reply['messages'])
                state['bootstrap_prompt'] = prompt
                state['bootstrap'] = 0.0
                send(slot,'finish',None)
                continue
            notify(f'rollout-{state["index"]}-policy')
            state['operation'] = 'policy'
            state['generated_steps'] = 0
            if decoder is not None:
                decoder.add(slot, reply['messages'], config['max_tokens'], config['max_context'])
            else:
                state['generator'] = policy.iter_sample(reply['messages'],config['max_tokens'],config['max_context'])
        if decoder is not None and decoder.requests:
            progressed = True
            notify('batched-policy-decode')
            for slot, sample in decoder.tick().items():
                active[slot]['actions'].append(sample)
                send(slot, 'step', sample['text'])
        if not progressed:
            time.sleep(.01)
    return completed


def environment_overlap(groups):
    for group in groups:
        episodes = group['episodes']
        for index, episode in enumerate(episodes):
            first = [x for x in episode.get('environment_intervals',[]) if x['operation']=='step']
            for other in episodes[index+1:]:
                second = [x for x in other.get('environment_intervals',[]) if x['operation']=='step']
                if any(a['started_at'] < b['finished_at'] and b['started_at'] < a['finished_at']
                       for a in first for b in second):
                    return True
    return False
