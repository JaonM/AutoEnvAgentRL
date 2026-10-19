"""Bounded scoring workers consume immutable episode snapshots, never live slots."""
import atexit
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
from concurrent.futures import ProcessPoolExecutor

from rl.checkpointing.checkpoint import atomic_json
from rl.data.contracts import create_runtime
from rl.errors import InfrastructureError

_runtime = None
_binding = None


def _close():
    global _runtime
    if _runtime is not None:
        _runtime.close()
        _runtime = None


def _initialize(parent):
    atexit.register(_close)
    def watch():
        while os.getppid() == parent:
            time.sleep(.5)
        os._exit(0)
    threading.Thread(target=watch, daemon=True).start()


def score_snapshot(sandbox, factory, snapshot, output=None):
    global _runtime, _binding
    fingerprint = hashlib.sha256(json.dumps([sandbox, factory, snapshot], sort_keys=True).encode()).hexdigest()
    if output and Path(output).exists():
        saved = json.loads(Path(output).read_text())
        if saved['fingerprint'] != fingerprint:
            raise ValueError('scoring snapshot identity mismatch')
        return saved['reply']
    started = time.time()
    if _runtime is None or _binding != (sandbox, factory):
        _close()
        _runtime = create_runtime(factory, sandbox)
        _binding = sandbox, factory
    _runtime.restore(snapshot)
    result = _runtime.finish()
    reply = {'value': result, 'messages': _runtime.messages, 'terminated': _runtime.terminated,
             'started_at': started, 'finished_at': time.time()}
    if output:
        atomic_json(output, {'fingerprint': fingerprint, 'reply': reply})
    return reply


class ScoringPool:
    def __init__(self, workers, capacity):
        if workers < 1 or capacity < workers:
            raise ValueError('scoring capacity must cover positive worker count')
        self.capacity, self.pending = capacity, set()
        self.executor = ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'),
                                            initializer=_initialize, initargs=(os.getpid(),))

    def submit(self, *args):
        self.pending = {f for f in self.pending if not f.done()}
        if len(self.pending) >= self.capacity:
            return None
        try:
            future = self.executor.submit(score_snapshot, *args)
        except RuntimeError as error:
            raise InfrastructureError('scoring worker unavailable') from error
        self.pending.add(future)
        return future

    def close(self):
        if any(not future.done() for future in self.pending):
            self.executor.terminate_workers()
        else:
            self.executor.shutdown(wait=True, cancel_futures=True)
