"""Append-only validation timings; never log answers, credentials or judge payloads."""
from contextlib import contextmanager
import json
from pathlib import Path
import time
import uuid


class ExecutionProgress:
    def __init__(self, root):
        self.path = Path(root) / "execution_progress.jsonl"
        self.run_id = uuid.uuid4().hex

    @contextmanager
    def measure(self, phase, **identity):
        operation_id = uuid.uuid4().hex
        started = time.monotonic()
        common = {"run_id": self.run_id, "operation_id": operation_id,
                  "phase": phase, **identity}

        def emit(**event):
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as stream:
                stream.write(json.dumps({**common, "timestamp": time.time(), **event}) + "\n")

        emit(status="running")
        try:
            yield
        except BaseException as exc:
            emit(status="failed", seconds=time.monotonic() - started,
                 error_type=type(exc).__name__)
            raise
        else:
            emit(status="completed", seconds=time.monotonic() - started)
