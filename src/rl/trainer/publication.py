"""Trainer publication: explicit session resources and progress."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.trainer.loop import TrainingSession


def publish_policy(session: TrainingSession):
    if session.state.optimizer_step > session.state.published_step and (
        session.state.optimizer_step - session.state.published_step
        >= session.config.policy_publish_interval_steps
        or session.state.dataset_cursor + 1 == len(session.schedule.batches)
    ):
        with session.times.measure("actor_weight_publish"):
            session.state.updates += 1
            session.policy.save_snapshot(session.output / "snapshots", session.state.updates)
            session.state.published_step = session.state.optimizer_step
            session.version.value = session.state.updates
