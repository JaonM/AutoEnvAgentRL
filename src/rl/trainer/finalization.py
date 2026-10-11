"""Trainer finalization: explicit session resources and progress."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl.trainer.loop import TrainingSession
import json
import math
import time
from rl.trainer.io import write_json


def finalize(session: TrainingSession):
    from rl.backends.mlx.policy import Policy
    from rl.data.tasks import tasks_state

    if session.state.dataset_cursor != len(session.schedule.batches):
        raise RuntimeError("dataset traversal incomplete; no completion claim")
    final_digest = session.policy.digest()
    final_policy_digest = session.policy.digest(policy_only=True)
    final_effective_digest = (
        session.policy.digest(effective=True) if session.config.tuning == "qat" else None
    )
    policy_changed = session.state.initial_policy_digest != final_policy_digest
    effective_changed = (
        session.state.initial_effective_digest != final_effective_digest
        if session.config.tuning == "qat"
        else None
    )
    if session.state.optimizer_step and (not policy_changed):
        raise ValueError("only critic or no parameters changed; policy training is unproven")
    full_export = (
        session.policy.export_full(session.output / "full_model")
        if session.config.tuning == "full"
        else None
    )
    qat_layers = session.policy.export_qat(session.output)
    lora_export = None
    lora_validation = None
    if session.config.tuning == "lora":
        from rl.export.lora import export, validate

        lora_export = export(
            session.policy,
            session.output,
            merge=session.config.lora_merge_export,
            requantize=session.config.lora_requantize_export,
        )
        lora_validation = validate(
            session.policy, lora_export, requantize=session.config.lora_requantize_export
        )
        write_json(session.output / "lora_export_validation.json", lora_validation)
    session.policy.restore(session.initial)
    before = session.evaluate(session.policy, session.tasks, session.config)
    session.monitor.evaluation("before", before, session.state.optimizer_step)
    session.policy.restore(
        session.output / "snapshots" / f"policy-{session.state.updates:06d}.safetensors"
    )
    after = session.evaluate(session.policy, session.tasks, session.config)
    session.monitor.evaluation("after", after, session.state.optimizer_step)
    export_validation = None
    if session.config.tuning == "qat":
        probes = []
        for episode in after["episodes"]:
            for action in episode["actions"]:
                ids = session.mx.array([action["prompt"] + action["tokens"]])
                logp, _ = session.policy.token_stats(ids, len(action["prompt"]))
                session.mx.eval(logp)
                probes.append((ids, len(action["prompt"]), logp))
        deployed_model = (
            session.output / "qat_model"
            if session.config.qat_scope == "full"
            else session.config.model
        )
        deployed_policy = Policy(
            str(deployed_model),
            "inference",
            temperature=session.config.temperature,
            thinking_mode=session.config.thinking_mode,
            logits_chunk_size=session.config.logits_chunk_size,
        )
        if session.config.qat_scope != "full":
            deployed_policy.load_qat_export(session.output)
        max_error = max(
            (
                float(
                    session.mx.max(
                        session.mx.abs(deployed_policy.token_stats(ids, length)[0] - logp)
                    )
                )
                for ids, length, logp in probes
            ),
            default=0.0,
        )
        if not math.isfinite(max_error) or max_error > 0.1:
            raise ValueError(f"packed QAT export log-probability mismatch: {max_error}")
        deployed = session.evaluate(deployed_policy, session.tasks, session.config)
        session.monitor.evaluation("deployed", deployed, session.state.optimizer_step)
        write_json(session.output / "evaluation_deployed.json", deployed)
        write_json(
            session.output / "deployment_identity.json",
            {"policy_identity": session.policy_identity, "model_files": session.model_files},
        )
        export_validation = {
            "packed_reload_verified": bool(probes),
            "probe_count": len(probes),
            "fresh_model_instance": True,
            "max_assistant_logp_error": max_error if probes else None,
            "tolerance": 0.1,
            "deployed_reward": deployed["mean_reward"],
        }
    write_json(session.output / "evaluation_before.json", before)
    write_json(session.output / "evaluation_after.json", after)
    from rl.observability.timing import rollout_training_overlap

    overlap = rollout_training_overlap(session.metrics)
    report = {
        "async_overlap_observed": overlap,
        "rollout_concurrency": session.config.rollout_concurrency,
        "over_sampling_batch_size": session.config.sampling_pool_size,
        "rollout_max_attempts": session.config.rollout_max_attempts,
        "sampling_attempts": session.state.groups,
        "accepted_groups": len(session.state.consumed_jobs) - len(session.state.skipped_jobs),
        "visited_jobs": len(session.state.consumed_jobs),
        "skipped_jobs": len(session.state.skipped_jobs),
        "updated_jobs": len(session.state.updated_jobs),
        "training_updates_observed": session.state.optimizer_step > 0,
        "batch_size": session.config.training_batch_size,
        "mini_batch_size": session.config.training_mini_batch_size,
        "critic_enabled": session.config.algorithm == "ppo",
        "role_names": {"actor": "policy trainer", "rollout_worker": "sampling process"},
        "micro_batch_size": session.config.micro_batch_size,
        "max_physical_batch_observed": session.actor.max_physical_batch,
        "batch_scoring_max_logp_error": session.actor.max_numerical_error,
        "batch_numeric_fallbacks": session.actor.numeric_fallbacks,
        "batched_gradient_microbatches": session.actor.batched_gradient_microbatches,
        "cached_gradient_microbatches": session.actor.cached_gradient_microbatches,
        "timings": session.times.state(),
        "epochs_completed": session.config.epochs,
        "dataset_batches_completed": session.state.dataset_cursor,
        "sandboxes_per_epoch": len([task for task in session.tasks if task.split == "train"]),
        "fresh_groups": session.state.groups,
        "rollout_group": session.config.rollout_group,
        "environment_rpc_overlap_observed": any(
            (m.get("environment_overlap") for m in session.metrics)
        ),
        "queue_capacity": session.config.queue_size,
        "completed": session.state.dataset_cursor == len(session.schedule.batches)
        and (
            session.state.optimizer_step == 0 or (policy_changed and effective_changed is not False)
        ),
        "policy_weights_changed": policy_changed,
        "effective_qat_weights_changed": effective_changed,
        "initial_policy_sha256": session.state.initial_policy_digest,
        "final_policy_sha256": final_policy_digest,
        "initial_effective_sha256": session.state.initial_effective_digest,
        "final_effective_sha256": final_effective_digest,
        "algorithm": session.config.algorithm,
        "tuning": session.config.tuning,
        "qat_scope": session.config.qat_scope,
        "thinking_mode": session.config.thinking_mode,
        "full_model_export": full_export,
        "lora_export": lora_export,
        "lora_export_validation": lora_validation,
        "updates": session.state.updates,
        "optimizer_steps": session.state.optimizer_step,
        "discount_unit": session.config.discount_unit,
        "initial_parameters_sha256": session.state.initial_digest,
        "final_parameters_sha256": final_digest,
        "tasks": tasks_state(session.tasks),
        "independent_eval": after["independent_eval"],
        "evaluation_by_task": after["by_task"],
        "evaluation_by_split": after["by_split"],
        "before_reward": before["mean_reward"],
        "after_reward": after["mean_reward"],
        "qat_export_layers": qat_layers,
        "qat_export_validation": export_validation,
        "dropped_groups": session.state.dropped,
        "dropped_stale_groups": sum((m.get("skipped") == "policy_age" for m in session.metrics)),
        "seconds": session.state.previous_seconds + time.time() - session.started,
        "device": str(session.mx.default_device()),
        "peak_metal_gb": max(
            [session.mx.get_peak_memory()]
            + [r["peak_bytes"] for r in session.times.memory.values()]
        )
        / 1000000000.0,
        "sandbox_transport": ("http" if session.config.sandbox_services
                              else "docker" if session.config.environment_factory.endswith(":V3DockerEpisode")
                              else "in_process"),
        "best_heldout_policy": session.state.eval_state.get("best"),
        "scope": "dataset traversal completed; inspect updated_jobs and evaluation for learning evidence",
    }
    write_json(session.output / "training_report.json", report)
    session.monitor.runtime.emit(
        "run/report", text=json.dumps(report, ensure_ascii=False, indent=2)
    )
    session.monitor.runtime.emit("run/state", text="completed" if report["completed"] else "failed")
    print(json.dumps(report), flush=True)
    return report
