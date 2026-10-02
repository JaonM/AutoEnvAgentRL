"""Actor control channels that cannot leave acquired locks after termination."""
import json
from pathlib import Path
import time
from .checkpoint import atomic_json


class StopSignal:
    """Single-writer shared byte; no process-shared semaphore to poison."""
    def __init__(self, context):
        self.flag = context.RawValue('b',0)

    def is_set(self):
        return bool(self.flag.value)

    def set(self):
        self.flag.value = 1

    def wait(self, timeout):
        end=time.monotonic()+timeout
        while not self.is_set() and time.monotonic()<end:
            time.sleep(min(.01,max(0.,end-time.monotonic())))
        return self.is_set()


class PolicyVersion:
    """Single learner publishes an atomic file after writing the snapshot."""
    def __init__(self,path,initial):
        self.path=Path(path)
        self.value=initial

    @property
    def value(self):
        return json.loads(self.path.read_text())['version']

    @value.setter
    def value(self,version):
        atomic_json(self.path,{'version':int(version)})
