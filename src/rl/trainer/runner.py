"""Training entry point, sandbox ownership and run lifecycle."""

import json
from pathlib import Path
import time
import traceback
import uuid
from rl.checkpointing.checkpoint import save_checkpoint, atomic_json
from rl.config import Config
from rl.data.trajectory import make_samples  # noqa: F401 - legacy rl.train export
from rl.paths import PROJECT_ROOT
from rl.trainer.cli import parse_args


def train(config: Config):
    """Train on certified sandbox bundles; --sandbox-backend docker runs each bundle's verified container."""
    config.validate()
    factory = ("rl.environments.sandbox:DockerSandboxEpisode" if config.sandbox_backend == "docker"
               else "rl.environments.sandbox:SandboxEpisode")
    if config.environment_factory == "rl.environment:SandboxEpisode":
        config.environment_factory = factory
    return _train_with_status(config)


def _train_with_status(config: Config):
    run_id = uuid.uuid4().hex
    status_path = Path(config.output).resolve() / "run_status.json"
    try:
        report = _train(config, run_id)
        if not report["completed"]:
            raise RuntimeError("training execution did not satisfy completion evidence")
    except BaseException as error:
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("run_id") == run_id:
                atomic_json(
                    status_path,
                    {
                        **status,
                        "state": "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed",
                        "error_type": type(error).__name__,
                        "finished_at": time.time(),
                    },
                )
        monitor = getattr(config, "_monitor", None)
        if monitor is not None:
            monitor.runtime.emit(
                "run/error",
                text=traceback.format_exc(),
                state="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            )
        raise
    finally:
        for journal in getattr(config, "_journals", ()):
            journal.close()
        if hasattr(config, "_journals"):
            del config._journals
        monitor = getattr(config, "_monitor", None)
        if monitor is not None:
            monitor.close()
            del config._monitor
    atomic_json(status_path, {"run_id": run_id, "state": "completed", "finished_at": time.time()})
    return report


def _train(config: Config, run_id):
    from rl.trainer.loop import TrainingSession

    return TrainingSession(config, run_id, save_checkpoint).run()


def main():
    train(parse_args())
