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
        self.started = time.monotonic()

    def feedback(self, phase, **identity):
        """Timestamp each actionable finding as soon as it is available."""
        availability = (identity.get('error_type') in {'SemanticCalibrationUnavailable', 'SemanticReviewUnavailable'}
                        or identity.get('code') in {'SEMANTIC_CALIBRATION_UNAVAILABLE', 'CODE_AGENT_CAPACITY'})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a') as stream:
            stream.write(json.dumps({'run_id': self.run_id, 'timestamp': time.time(),
                'elapsed_seconds': time.monotonic() - self.started,
                'phase': phase, 'status': 'availability_feedback' if availability else 'actionable_feedback', **identity}) + '\n')

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
