"""Trainer optimization: explicit session resources and progress."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.trainer.loop import TrainingSession
import json
from pathlib import Path
import time
from rl.data.trajectory import make_samples
from rl.data.batching import combine_samples, minibatches


def measure_drift(session: TrainingSession, group):
    from rl.data.replay import importance_diagnostics

    actions = [action for episode in group["episodes"] for action in episode["actions"]]
    current = session.actor.score(actions)
    return importance_diagnostics(
        [
            p
            for action in actions
            for p, mask in zip(
                current[id(action)], action.get("loss_mask", [1] * len(action["tokens"]))
            )
            if mask
        ],
        [
            p
            for action in actions
            for p, mask in zip(
                action["old_logp"], action.get("loss_mask", [1] * len(action["tokens"]))
            )
            if mask
        ],
        session.config.clip,
    )


def prepare(session: TrainingSession, group, path, replay_entry=None):
    from rl.data.contracts import attach_contract
    from rl.algorithms.advantages import group_advantages
    from rl.data.replay import validate_ppo_behavior
    from rl.rollout.admission import zero_variance_kind

    attach_contract(group)
    if len(group["episodes"]) != session.config.rollout_group:
        raise ValueError("rollout group size does not match configured rollout_group")
    lag = session.state.updates - group["policy_version"]
    age = session.state.updates - replay_entry["age_origin"] if replay_entry else lag
    max_age = session.config.replay_max_age if replay_entry else session.config.max_policy_lag
    reason = "context_limit_empty" if any((not e["actions"] for e in group["episodes"])) else None
    drift = {}
    started_prepare = time.monotonic()
    if reason:
        pass
    elif age < 0 or age > max_age:
        reason = "policy_age"
    elif session.config.algorithm == "grpo" and all(
        (
            abs(a) < 1e-08
            for a in group_advantages([episode["final_reward"] for episode in group["episodes"]])
        )
    ):
        reason = "zero_reward_variance"
    else:
        if session.config.algorithm == "ppo":
            validate_ppo_behavior(group)
        drift = session.measure_drift(group)
        if (
            drift["effective_sample_fraction"] < session.config.replay_min_ess
            or drift["max_abs_log_ratio"] > session.config.replay_max_log_ratio
        ):
            reason = "importance_drift"
    samples = (
        []
        if reason
        else make_samples(
            group,
            session.config.algorithm,
            session.config.gamma,
            session.config.gae_lambda,
            session.config.discount_unit,
        )
    )
    if reason:
        session.state.dropped += 1
        session.metrics.append(
            {
                "task_id": group["task_id"],
                "dataset_index": group.get("dataset_index"),
                "rollout_attempt": group.get("rollout_attempt", 0),
                "dataset_batch": session.state.dataset_cursor,
                "replay": replay_entry is not None,
                "skipped": reason,
                **(
                    {"zero_variance_kind": zero_variance_kind(group["episodes"])}
                    if reason == "zero_reward_variance"
                    else {}
                ),
                **drift,
            }
        )
        session.monitor.runtime.emit(
            "rollout/rejected",
            text=json.dumps(session.metrics[-1], ensure_ascii=False),
            metrics={
                "admission/dropped_groups": session.state.dropped,
                **{f"admission/{k}": v for k, v in drift.items()},
            },
            optimizer_step=session.state.optimizer_step,
            reason=reason,
        )
        return {"rejected": reason}
    session.monitor.runtime.emit(
        "rollout/accepted",
        task_id=group["task_id"],
        policy_version=group["policy_version"],
        optimizer_step=session.state.optimizer_step,
        replay=replay_entry is not None,
        metrics={f"admission/{k}": v for k, v in drift.items()},
    )
    return {
        "group": group,
        "samples": samples,
        "path": str(path),
        "replay_entry": replay_entry,
        "drift": drift,
        "lag": lag,
        "queue_seconds": 0.0,
        "target_seconds": time.monotonic() - started_prepare,
    }


def optimize(
    session: TrainingSession,
    batch_groups,
    dataset_epoch,
    replay_cycle,
    optimization_pass=1,
    mini_offset=0,
):
    if not batch_groups:
        return False
    if session.config.updates and session.state.updates >= session.config.updates:
        raise RuntimeError("policy update budget reached before dataset schedule finished")
    samples = combine_samples(
        [
            [
                {**sample, "dataset_index": item["group"].get("dataset_index", -1)}
                for sample in item["samples"]
            ]
            for item in batch_groups
        ],
        session.config.rollout_group,
    )
    batch = [(sample, None, None) for sample in samples]
    exceeded = False
    for mini_index, mini in enumerate(
        minibatches(
            batch,
            session.config.mini_batch_size,
            seed=session.config.seed
            + session.state.dataset_cursor * session.config.optimization_passes
            + optimization_pass,
        ),
        1 + mini_offset,
    ):
        started_update = time.time()
        selected = [sample for sample, _, _ in mini]

        def actor_notify():
            session.pool.raise_if_failed()
            session.monitor.runtime.resources(session.mx)

        measured = session.actor.step(selected, notify=actor_notify)
        common = {
            "epoch": dataset_epoch,
            "dataset_batch": session.state.dataset_cursor,
            "replay_cycle": replay_cycle,
            "optimization_pass": optimization_pass,
            "mini_batch": mini_index,
            "post_kl_scope": "ready_mini_batch",
            "mini_batch_sandboxes": len({s["sandbox_index"] for s in selected}),
            "mini_batch_rollouts": len({s["episode_index"] for s in selected}),
        }
        if measured.get("skipped"):
            session.metrics.append({**common, **measured})
            session.monitor.runtime.emit(
                "train/skipped",
                text=json.dumps(session.metrics[-1], ensure_ascii=False),
                optimizer_step=session.state.optimizer_step,
                reason=measured["skipped"],
            )
            if measured["skipped"] == "target_kl":
                return True
            continue
        session.pool.raise_if_failed()
        session.state.optimizer_step += 1
        if replay_cycle == 0:
            session.state.updated_jobs.update(
                (
                    s["dataset_index"]
                    for s in selected
                    if s["dataset_index"] in session.state.consumed_jobs
                )
            )
        measured.update(
            **common,
            optimizer_step=session.state.optimizer_step,
            policy_version=session.state.updates + 1,
        )
        update_seconds = time.time() - started_update
        assistant_tokens = sum((sum(s.get("loss_mask", [1] * len(s["tokens"]))) for s in selected))
        measured.update(
            update_seconds=update_seconds,
            assistant_tokens=assistant_tokens,
            assistant_tokens_per_second=assistant_tokens / max(update_seconds, 1e-09),
        )
        session.optimizer_metrics.append(measured)
        measured.update(
            {
                f"memory_{name}_peak_gb": value["peak_bytes"] / 1000000000.0
                for name, value in session.times.memory.items()
            }
        )
        session.monitor.optimizer(measured)
        session.metrics.append(
            {
                **measured,
                "event": "actor_optimizer_step",
                "update": session.state.updates + 1,
                "batch_groups": [
                    {
                        "task_id": item["group"]["task_id"],
                        "path": item["path"],
                        "behavior_version": item["group"]["policy_version"],
                        "replay": item["replay_entry"] is not None,
                        **item["drift"],
                    }
                    for item in batch_groups
                ],
                "assistant_tokens": sum(
                    (sum(s.get("loss_mask", [1] * len(s["tokens"]))) for s in selected)
                ),
                "batch_scoring_max_logp_error": session.actor.max_numerical_error,
                "loss_aggregation": "equal_sandbox_equal_rollout_token_mean",
                "started_at": started_update,
                "finished_at": time.time(),
            }
        )
        if (
            session.config.metrics_export_interval
            and session.state.optimizer_step % session.config.metrics_export_interval == 0
        ):
            session.metrics.export()
            session.optimizer_metrics.export()
        print(json.dumps(session.metrics[-1]), flush=True)
        exceeded = measured["post_kl_exceeded"]
        if exceeded:
            break
    return exceeded


def prepared_records(session: TrainingSession, *, only_untrained=False):
    from rl.data.contracts import attach_contract
    from rl.checkpointing.checkpoint import sha256

    items = []
    for record in session.state.inflight["records"]:
        if sha256(record["path"]) != record["sha256"]:
            raise ValueError("in-flight rollout artifact checksum mismatch")
        if not record["eligible"] or (only_untrained and record.get("trained", True)):
            continue
        group = json.loads(Path(record["path"]).read_text())
        session.validate_actor_group(group)
        attach_contract(group)
        items.append(
            {
                "group": group,
                "samples": make_samples(
                    group,
                    session.config.algorithm,
                    session.config.gamma,
                    session.config.gae_lambda,
                    session.config.discount_unit,
                ),
                "path": record["path"],
                "replay_entry": None,
                "drift": record["drift"],
                "lag": record["lag"],
            }
        )
    return items


def validate_actor_group(session: TrainingSession, group):
    from rl.data.replay import validate_group

    validate_group(group, session.tasks)
    if group.get("policy_identity") != session.policy_identity:
        raise ValueError("trajectory policy identity mismatch")
