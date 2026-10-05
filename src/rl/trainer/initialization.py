"""Initialize models, services, queues and verified recovery state."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.trainer.loop import TrainingSession
import importlib.metadata
from dataclasses import asdict
import json
import multiprocessing as mp
import os
import platform
from pathlib import Path
import time
import uuid
from rl.data.dataset import DatasetSchedule, DatasetResults
from rl.backends.mlx.model_options import policy_options
from rl.checkpointing.checkpoint import restore_checkpoint, atomic_json
from rl.trainer.io import write_json
from rl.paths import PROJECT_ROOT
from rl.trainer.state import TrainingState
from rl.checkpointing.provenance import source_identity


def initialize(session: TrainingSession, config, run_id, checkpoint_saver):
    session.config = config
    session.run_id = run_id
    session.save_checkpoint = checkpoint_saver
    session.state = TrainingState()
    session.config.validate()
    import mlx.core as mx

    session.mx = mx
    import mlx.optimizers as optim
    from dotenv import load_dotenv
    from rl.data.tasks import load_tasks, tasks_state
    from rl.trainer.evaluation import evaluate, PeriodicEvaluation

    session.evaluate = evaluate
    from rl.data.replay import ReplayBuffer, import_replay
    from rl.checkpointing.provenance import model_identity
    from rl.rollout.manager import RolloutPool
    from rl.rollout.process_control import StopSignal, PolicyVersion
    from rl.backends.mlx.policy import Policy
    from rl.backends.mlx.actor import ActorTrainer
    from rl.observability.timing import StageTimes
    from rl.checkpointing.checkpoint import sha256
    from rl.data.task_queue import TaskQueue

    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.pop("SANDBOX_EVALUATOR_MOCK", None)
    os.environ["SANDBOX_MUTATION_MODE"] = "disabled"
    if session.config.sandbox_services:
        session.config.sandbox_services = str(Path(session.config.sandbox_services).resolve())
        if not os.environ.get("SANDBOX_TRAINER_API_KEY"):
            raise ValueError(
                "remote sandbox requires SANDBOX_TRAINER_API_KEY matching the sandbox service"
            )
        if session.config.environment_factory not in (
            "rl.environment:SandboxEpisode",
            "rl.remote_environment:RemoteSandboxEpisode",
        ):
            raise ValueError("sandbox_services conflicts with custom environment_factory")
        session.config.environment_factory = "rl.remote_environment:RemoteSandboxEpisode"
        os.environ["RL_SANDBOX_SERVICES"] = session.config.sandbox_services
    else:
        os.environ["SANDBOX_TRAINER_API_KEY"] = uuid.uuid4().hex
    session.tasks = load_tasks(sandbox=session.config.sandbox, manifest=session.config.tasks)
    session.schedule = DatasetSchedule(
        session.tasks,
        epochs=session.config.epochs,
        batch_size=session.config.batch_size,
        seed=session.config.seed,
    )
    session.output = Path(session.config.output).resolve()
    session.output.mkdir(parents=True, exist_ok=True)
    session.config.output = str(session.output)
    if session.config.sandbox:
        session.config.sandbox = str(Path(session.config.sandbox).resolve())
    if session.config.replay_source:
        session.config.replay_source = str(Path(session.config.replay_source).resolve())
    if session.config.tasks:
        session.config.tasks = str(Path(session.config.tasks).resolve())
    if Path(session.config.model).exists():
        session.config.model = str(Path(session.config.model).resolve())
    session.config.model, base_identity, session.model_files = model_identity(session.config.model)
    if session.config.tuning == "full" or (
        session.config.tuning == "qat" and session.config.qat_scope == "full"
    ):
        from rl.backends.mlx.model_memory import check_full_training_memory

        write_json(
            session.output / "model_memory_estimate.json",
            check_full_training_memory(
                session.config.model,
                session.config.rollout_workers,
                qat=session.config.tuning == "qat",
                packed_inference=session.config.packed_inference,
                bits=session.config.bits,
            ),
        )
    if session.config.resume:
        previous = json.loads((session.output / "config.json").read_text())
        mutable = {
            "resume",
            "updates",
            "max_groups",
            "rollout_timeout",
            "epochs",
            "tensorboard",
            "log_flush_seconds",
            "rollout_trace_samples",
            "rollout_trace_max_chars",
        }
        for key, value in asdict(session.config).items():
            if key not in mutable and previous.get(key) != value:
                raise ValueError(f"resume configuration changed: {key}")
        if not (session.output / "checkpoints/latest.json").exists():
            raise ValueError("no committed resumable checkpoint")
    elif (session.output / "config.json").exists():
        raise ValueError("use --resume or a new output; existing run is never overwritten")
    session.policy_identity = {
        "base": base_identity,
        "tuning": session.config.tuning,
        "layers": session.config.layers,
        "rank": session.config.rank,
        "bits": session.config.bits,
        "model_options": policy_options(session.config),
        "qat_scope": session.config.qat_scope,
        "temperature": session.config.temperature,
        "thinking_mode": session.config.thinking_mode,
        "chat_protocol": "native_tools_v1",
    }
    provenance = {
        "source_sha256": source_identity(),
        "packages": {name: importlib.metadata.version(name) for name in ("mlx", "mlx-lm", "numpy")},
        "platform": platform.platform(),
        "machine": platform.machine(),
        "tasks": tasks_state(session.tasks),
        "policy_identity": session.policy_identity,
        "model_files": session.model_files,
    }
    if session.config.resume:
        original = json.loads((session.output / "execution_provenance.json").read_text())
        if any(
            (
                original[key] != provenance[key]
                for key in ("source_sha256", "packages", "tasks", "policy_identity", "model_files")
            )
        ):
            raise ValueError("resume code, dependency or sandbox identity changed")
    else:
        write_json(session.output / "execution_provenance.json", provenance)
    write_json(session.output / "config.json", asdict(session.config))
    atomic_json(
        session.output / "run_status.json",
        {"run_id": session.run_id, "state": "running", "started_at": time.time()},
    )
    from rl.observability.telemetry import TrainingMonitor

    session.monitor = session.config._monitor = TrainingMonitor(session.config, session.run_id)
    print(f"Training dashboard logs: {session.output / 'tensorboard'}", flush=True)
    session.mx.set_default_device(session.mx.gpu)
    if not session.mx.metal.is_available():
        raise RuntimeError("Apple Metal GPU is required")
    session.mx.random.seed(session.config.seed)
    session.policy = Policy(
        session.config.model,
        session.config.tuning,
        session.config.layers,
        session.config.rank,
        session.config.bits,
        temperature=session.config.temperature,
        critic=session.config.algorithm == "ppo",
        **policy_options(session.config),
    )
    session.state.initial_digest = session.policy.digest()
    session.state.initial_policy_digest = session.policy.digest(policy_only=True)
    session.state.initial_effective_digest = (
        session.policy.digest(effective=True) if session.config.tuning == "qat" else None
    )
    session.initial = str(session.output / "snapshots/policy-000000.safetensors")
    if not session.config.resume:
        session.policy.save_snapshot(session.output / "snapshots", 0)
    reference = Policy(
        session.config.model,
        session.config.tuning,
        session.config.layers,
        session.config.rank,
        session.config.bits,
        temperature=session.config.temperature,
        critic=False,
        **policy_options(session.config, inference=True),
    )
    reference.restore(session.initial, policy_only=True)
    reference.freeze()
    session.optimizer = optim.Adam(learning_rate=session.config.learning_rate)
    session.optimizer.init(session.policy.trainable_parameters())
    restored = (
        restore_checkpoint(session.output / "checkpoints", session.policy, session.optimizer)
        if session.config.resume
        else {}
    )
    session.times = StageTimes(
        restored.get("timings"), mlx=session.mx if session.config.profile_memory else None
    )
    session.state.inflight = restored.get("inflight_batch", {})
    session.actor = ActorTrainer(
        session.policy, reference, session.optimizer, session.config, session.times
    )
    session.actor.restore_state(restored.get("actor_numeric_state", {}))
    from rl.observability.journal import MetricJournal

    journals = restored.get("metric_journals", {})
    session.config._journals = []
    for name in ("metrics", "optimizer_metrics"):
        session.config._journals.append(
            MetricJournal(
                session.output, name, cursor=journals.get(name), legacy=restored.get(name, [])
            )
        )
    session.metrics, session.optimizer_metrics = session.config._journals
    session.state.skipped_jobs = restored.get("skipped_jobs", {})
    session.state.updated_jobs = set(restored.get("updated_jobs", []))
    session.state.eval_state = restored.get("eval_state", {})
    session.state.updates, session.state.groups, session.state.dropped = (
        restored.get(key, 0) for key in ("updates", "groups", "dropped")
    )
    session.state.optimizer_step = restored.get("optimizer_step", 0)
    session.monitor.restore(session.state.optimizer_step, session.state.groups)
    session.periodic_eval = PeriodicEvaluation(
        session.output, session.tasks, session.config, session.monitor, session.state.eval_state
    )
    session.state.eval_state = session.periodic_eval.state
    session.state.rollout_positions = restored.get("rollout_positions", {})
    session.state.initial_digest = restored.get("initial_digest", session.state.initial_digest)
    session.state.initial_policy_digest = restored.get(
        "initial_policy_digest", session.state.initial_policy_digest
    )
    session.state.initial_effective_digest = restored.get(
        "initial_effective_digest", session.state.initial_effective_digest
    )
    session.state.previous_seconds = restored.get("seconds", 0.0)
    session.replay = ReplayBuffer(
        capacity=session.config.replay_capacity,
        max_uses=session.config.replay_max_uses,
        max_age=session.config.replay_max_age,
        seed=session.config.seed,
        state=restored.get("replay"),
    )
    session.state.dataset_cursor = restored.get("dataset_cursor", 0)
    session.state.consumed_jobs = set(restored.get("consumed_jobs", []))
    session.state.rollout_attempts = restored.get("rollout_attempts", {})
    session.task_queue = TaskQueue(
        session.output / "task_queue", session.schedule.jobs, session.config.sampling_pool_size
    )
    session.task_queue.commit_consumed(session.state.consumed_jobs, session.state.rollout_attempts)
    if not 0 <= session.state.dataset_cursor <= len(session.schedule.batches):
        raise ValueError("resume dataset cursor exceeds epoch budget")
    if session.config.replay_source and (not session.config.resume):
        imported = import_replay(
            session.config.replay_source,
            session.replay,
            session.tasks,
            session.policy_identity,
            current_version=session.state.updates,
        )
        write_json(
            session.output / "replay_import.json",
            {"source": session.config.replay_source, "groups": imported},
        )
    if (
        session.config.updates
        and session.state.updates > session.config.updates
        or (session.config.max_groups and session.state.groups > session.config.max_groups)
    ):
        raise ValueError("resume budgets are behind the committed checkpoint")
    if session.config.resume:
        recovery = session.output / "recovery" / uuid.uuid4().hex
        recovery.mkdir(parents=True)
        for path in session.output.glob("group-*.json"):
            if int(path.stem.split("-")[1]) > session.state.groups:
                path.rename(recovery / path.name)
        for name in ("metrics.json", "optimizer_metrics.json", "training_report.json"):
            path = session.output / name
            if path.exists():
                path.rename(recovery / name)
        session.metrics.export()
        session.optimizer_metrics.export()
        from rl.checkpointing.recovery import quarantine

        quarantine(session.output, session.state.updates, recovery)
        published_path = (
            session.output / "snapshots" / f"policy-{session.state.updates:06d}.safetensors"
        )
        if (
            restored.get("published_snapshot_sha256")
            and sha256(published_path) != restored["published_snapshot_sha256"]
        ):
            raise ValueError("published snapshot changed since committed checkpoint")
    ctx = mp.get_context("spawn")
    results = ctx.Queue(maxsize=session.config.queue_size)
    stop = StopSignal(ctx)
    session.version = PolicyVersion(session.output / "policy_version.json", session.state.updates)
    rollout_config = {
        **asdict(session.config),
        "_rollout_positions": session.state.rollout_positions,
        "_dataset_jobs": session.schedule.jobs,
        "_run_id": session.run_id,
        "_tasks": tasks_state(session.tasks),
        "_policy_identity": session.policy_identity,
        "_task_queue": str(session.task_queue.root),
        "_queue_capacity": session.task_queue.capacity,
    }
    session.pool = RolloutPool(
        ctx,
        rollout_config,
        session.version,
        results,
        stop,
        session.state.rollout_positions,
        restarts=restored.get("rollout_restarts"),
        events=restored.get("rollout_recovery", []),
    )
    session.started = time.time()
    session.state.committed_version = session.state.updates
    session.state.checkpoint_step = restored.get("optimizer_step", 0)
    session.state.published_step = restored.get(
        "published_optimizer_step", restored.get("optimizer_step", 0)
    )
    session.loader = DatasetResults(
        session.schedule,
        results,
        session.pool,
        session.state.rollout_positions,
        rollout_workers=session.config.rollout_workers,
        timeout=session.config.rollout_timeout,
        progress_timeout=session.config.train_progress_timeout,
        validate=session.validate_actor_group,
        next_index=0,
        completed=session.state.consumed_jobs,
        task_queue=session.task_queue,
        on_poll=lambda: session.monitor.runtime.resources(
            session.mx, **{"queue/pending_groups": len(session.loader.pending)}
        ),
    )
    for record in session.state.inflight.get("records", []):
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("in-flight rollout artifact checksum mismatch")
