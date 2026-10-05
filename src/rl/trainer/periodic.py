"""Trainer periodic: explicit session resources and progress."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.trainer.loop import TrainingSession


def evaluate_periodically(session: TrainingSession, *, force=False):
    from rl.trainer.evaluation import preserve_mlx_rng

    with preserve_mlx_rng(), session.times.measure("actor_periodic_evaluation"):
        return session.periodic_eval.run(
            session.policy, session.state.optimizer_step, evaluator=session.evaluate, force=force
        )
