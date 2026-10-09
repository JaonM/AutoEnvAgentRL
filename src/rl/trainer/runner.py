"""Training entry point, sandbox ownership and run lifecycle."""

import json
import os
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
    config.validate()
    candidates = [Path(config.sandbox)] if config.sandbox else []
    if config.tasks and Path(config.tasks).is_file():
        manifest_path = Path(config.tasks).resolve()
        entries = json.loads(manifest_path.read_text()).get('tasks', [])
        candidates = [manifest_path.parent / entry['sandbox'] for entry in entries]
    if any((root / 'contract.json').is_file() and (root / 'delivery.json').is_file() for root in candidates):
        from rl.data.tasks import load_tasks
        tasks = load_tasks(sandbox=config.sandbox, manifest=config.tasks)
        v2_tasks = [(Path(task.sandbox) / 'contract.json').is_file() for task in tasks]
        if not all(v2_tasks) or config.sandbox_backend != 'local' or config.sandbox_services:
            raise ValueError('V2 currently requires an all-V2 task set and --sandbox-backend local')
        if config.environment_factory == 'rl.environment:SandboxEpisode':
            config.environment_factory = 'rl.environments.v2:V2Episode'
        if config.environment_factory != 'rl.environments.v2:V2Episode':
            raise ValueError('V2 tasks require rl.environments.v2:V2Episode')
    manager = None
    try:
        if config.sandbox_backend == "docker" and (not config.sandbox_services):
            from dotenv import load_dotenv
            from rl.environments.services.docker_manager import DockerManager
            from rl.data.tasks import load_tasks

            load_dotenv(PROJECT_ROOT / ".env")
            if not os.environ.get("SANDBOX_TRAINER_API_KEY"):
                os.environ["SANDBOX_TRAINER_API_KEY"] = uuid.uuid4().hex
            tasks = load_tasks(sandbox=config.sandbox, manifest=config.tasks)
            images = (
                json.loads(Path(config.sandbox_images).read_text())
                if config.sandbox_images
                else None
            )
            manager = DockerManager(
                tasks,
                Path(config.output) / "docker_services",
                images=images,
                max_active=config.sandbox_max_active,
                startup_workers=config.sandbox_startup_workers,
                idle_timeout=config.sandbox_idle_timeout,
                cpus=config.sandbox_cpus,
                memory=config.sandbox_memory,
            )
            config.sandbox_services = manager.start()
        return _train_with_status(config)
    finally:
        if manager is not None:
            manager.close()


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
