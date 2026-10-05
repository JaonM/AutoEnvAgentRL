"""Trainer persistence: explicit session resources and progress."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.trainer.loop import TrainingSession
import time
from rl.trainer.io import write_json


def commit(session: TrainingSession, *, force=False):
    from rl.checkpointing.retention import prune_checkpoints
    from rl.checkpointing.retention import prune_live_artifacts
    from rl.checkpointing.checkpoint import sha256

    if (
        not force
        and session.config.checkpoint_interval_steps > 1
        and (
            session.state.optimizer_step - session.state.checkpoint_step
            < session.config.checkpoint_interval_steps
        )
    ):
        session.task_queue.commit_consumed(
            session.state.consumed_jobs, session.state.rollout_attempts
        )
        return
    checkpoint_started = time.perf_counter()
    session.save_checkpoint(
        session.output / "checkpoints",
        session.policy,
        session.optimizer,
        {
            "updates": session.state.updates,
            "groups": session.state.groups,
            "dropped": session.state.dropped,
            "published_optimizer_step": session.state.published_step,
            "published_snapshot_sha256": sha256(
                session.output / "snapshots" / f"policy-{session.state.updates:06d}.safetensors"
            ),
            "optimizer_step": session.state.optimizer_step,
            "metric_journals": {j.name: j.cursor() for j in session.config._journals},
            "skipped_jobs": session.state.skipped_jobs,
            "updated_jobs": sorted(session.state.updated_jobs),
            "eval_state": session.state.eval_state,
            **session.pool.checkpoint_state(),
            "replay": session.replay.state(),
            "dataset_cursor": session.state.dataset_cursor,
            "inflight_batch": session.state.inflight,
            "consumed_jobs": sorted(session.state.consumed_jobs),
            "rollout_attempts": session.state.rollout_attempts,
            "timings": session.times.state(),
            "actor_numeric_state": session.actor.state(),
            "initial_policy_digest": session.state.initial_policy_digest,
            "initial_effective_digest": session.state.initial_effective_digest,
            "initial_digest": session.state.initial_digest,
            "seconds": session.state.previous_seconds + time.time() - session.started,
        },
    )
    session.state.committed_version = session.state.updates
    session.state.checkpoint_step = session.state.optimizer_step
    session.periodic_eval.publish_best()
    session.task_queue.commit_consumed(session.state.consumed_jobs, session.state.rollout_attempts)
    prune_checkpoints(session.output / "checkpoints", session.config.checkpoint_keep)
    removed = prune_live_artifacts(
        session.output,
        session.task_queue,
        session.state.updates,
        keep=session.config.checkpoint_keep,
        group_keep=session.config.group_artifacts_keep,
    )
    if removed:
        session.monitor.runtime.emit(
            "retention", metrics={"storage/artifacts_removed": len(removed)}
        )
    session.times.add("actor_checkpoint", time.perf_counter() - checkpoint_started)
    write_json(session.output / "timings.json", session.times.state())
    session.monitor.runtime.emit(
        "checkpoint",
        optimizer_step=session.state.optimizer_step,
        policy_version=session.state.updates,
        metrics={
            "progress/optimizer_step": session.state.optimizer_step,
            "progress/policy_version": session.state.updates,
            "progress/dataset_batches_completed": session.state.dataset_cursor,
            "progress/visited_jobs": len(session.state.consumed_jobs),
            "progress/skipped_jobs": len(session.state.skipped_jobs),
            "progress/updated_jobs": len(session.state.updated_jobs),
            "progress/dataset_fraction": session.state.dataset_cursor
            / max(1, len(session.schedule.batches)),
            **{f"timing/{name}_seconds": value for name, value in session.times.seconds.items()},
        },
    )
    session.monitor.runtime.resources(session.mx)
    session.monitor.train.flush()
    session.monitor.groups.flush()
