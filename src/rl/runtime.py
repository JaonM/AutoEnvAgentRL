"""Bounded rollout worker recovery; models never receive fabricated rewards on failures."""
from copy import deepcopy
import json
from pathlib import Path
import time
import threading
from functools import wraps

from .rollout_worker import rollout_worker_main
from .checkpoint import atomic_json


def synchronized(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self.lock:
            return method(self,*args,**kwargs)
    return call


class RolloutPool:
    def __init__(self, ctx, config, version, results, stop, positions, *, restarts=None, events=None, target=rollout_worker_main):
        self.lock = threading.RLock()
        self.watchdog_stop = threading.Event()
        self.watchdog = None
        self.failure = None
        self.ctx,self.config,self.version,self.results,self.stop = ctx,config,version,results,stop
        self.positions,self.target = positions,target
        self.processes = {}
        self.restarts = dict(restarts or {})
        self.events = list(events or [])
        journal = Path(config['output']) / 'rollout_recovery.json'
        if config.get('resume') and journal.exists():
            # Recovery attempts after the last optimizer commit still consume budget.
            persisted = json.loads(journal.read_text())
            known = {(event['rollout_worker'], event['restart']) for event in self.events}
            self.events.extend(event for event in persisted if (event['rollout_worker'], event['restart']) not in known)
        for event in self.events:
            key = str(event['rollout_worker'])
            self.restarts[key] = max(self.restarts.get(key, 0), event['restart'])

    @synchronized
    def start(self, index):
        heartbeat = Path(self.config['output'])/'rollout_workers'/f'worker-{index}.json'
        atomic_json(heartbeat, {'rollout_worker':index,'stage':'starting','at':time.time()})
        runtime_config={**self.config,'_rollout_positions':deepcopy(self.positions)}
        process=self.ctx.Process(target=self.target,args=(index,runtime_config,self.version,self.results,self.stop),name=f'rl-rollout-{index}')
        self.processes[index]=process
        process.start()

    @synchronized
    def start_all(self):
        for index in range(self.config['rollout_workers']):
            self.start(index)
        self.watchdog = threading.Thread(target=self._watch, name='rl-rollout-watchdog', daemon=True)
        self.watchdog.start()

    @synchronized
    def restart(self, index, reason):
        used=self.restarts.get(str(index),0)
        if used>=self.config['max_rollout_restarts']:
            raise RuntimeError(f'rollout worker {index} exhausted restart budget: {reason}')
        process=self.processes[index]
        if process.pid is not None:
            process.join(timeout=.2)
            if process.is_alive():
                process.terminate();process.join(timeout=5)
            if process.is_alive():
                raise RuntimeError(f'rollout worker {index} did not terminate')
        self.restarts[str(index)]=used+1
        self.events.append({'rollout_worker':index,'restart':used+1,'reason':reason,'at':time.time()})
        atomic_json(Path(self.config['output'])/'rollout_recovery.json',self.events)
        self.start(index)

    @synchronized
    def handle_error(self, message):
        process = self.processes[message['rollout_worker']]
        if message.get('pid') is not None and message['pid'] != process.pid:
            return  # Delayed failure notification from an already replaced process.
        if not message.get('retryable'):
            raise RuntimeError(message['rollout_error'])
        self.restart(message['rollout_worker'],message.get('error_type','infrastructure_error'))

    @synchronized
    def check(self, *, queue_empty=False):
        for index,process in list(self.processes.items()):
            if not process.is_alive():
                path = Path(self.config['output'])/'rollout_workers'/f'worker-{index}.json'
                heartbeat = json.loads(path.read_text()) if path.exists() else {}
                if heartbeat.get('rollout_error') and heartbeat.get('pid') == process.pid:
                    self.handle_error(heartbeat)
                    continue
                self.restart(index,f'exit_code={process.exitcode}')
                continue
            path=Path(self.config['output'])/'rollout_workers'/f'worker-{index}.json'
            if path.exists():
                heartbeat=json.loads(path.read_text())
                if time.time()-heartbeat['at'] > self.config['rollout_timeout']:
                    self.restart(index,'progress_deadline_exceeded:'+heartbeat.get('stage','unknown'))

    def _watch(self):
        while not self.watchdog_stop.wait(self.config.get('watchdog_interval',1.)):
            try:
                self.check()
            except Exception as error:
                self.failure = error
                return

    def raise_if_failed(self):
        if self.failure is not None:
            raise RuntimeError(f'rollout watchdog failed: {self.failure}') from self.failure

    @synchronized
    def record_position(self, index, position):
        self.positions[str(index)] = deepcopy(position)

    @synchronized
    def checkpoint_state(self):
        return {'rollout_positions':deepcopy(self.positions), 'rollout_restarts':dict(self.restarts),
                'rollout_recovery':deepcopy(self.events)}

    def close(self):
        self.watchdog_stop.set()
        if self.watchdog is not None:
            self.watchdog.join()
        self.stop.set()
        for process in self.processes.values():
            if process.pid is None:
                continue
            process.join(timeout=5)
            if process.is_alive():
                process.terminate();process.join(timeout=5)
        self.results.cancel_join_thread()
        self.results.close()
