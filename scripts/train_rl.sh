#!/usr/bin/env bash
# Local Apple Silicon RL: qualified sandbox -> rollout workers -> actor (PPO/GRPO).
# --tuning qat (default), lora, or full; full needs floating weights and enough RAM.
# --tuning qat --qat-scope full trains every LM parameter with weight QAT; needs FP32 master/Adam memory.
# --thinking-mode auto (default), thinking, or no-thinking controls Qwen3 chat templates.
# --lora-targets all-linear --layers 28 enables all Qwen3-0.6B transformer linear adapters.
# --gradient-checkpointing --logits-chunk-size 32 trades compute for activation memory.
# --packed-inference keeps QAT reference/workers packed; actor retains FP32 masters.
# --checkpoint-interval-steps and --policy-publish-interval-steps default to 1.
# --profile-memory measures synchronized stage peaks; see python -m rl.profile_memory.
# CUDA export: scripts/convert_to_hf.sh (install --extra cuda-export).
# Install once with: uv sync --extra rl
# Default sandbox backend: Docker Engine manager (async prewarm, HTTP rollout, auto cleanup).
# --sandbox-backend local explicitly uses the in-process development runtime.
# Required: --sandbox PATH --output NEW_PATH. See --help for algorithm budgets.
# --epochs (2) counts full passes through the training sandbox dataset.
# --batch-size (1) collects completed sandbox groups in completion order within an epoch.
# --mini-batch-size (1) counts sandboxes per optimizer step.
# Workers share recoverable task leases; action-boundary continuations live under output/task_queue/.
# --rollout-group (4) samples that many trajectories per sandbox visit.
# --over-sampling-batch-size (0 = 2 * batch-size) bounds the candidate pool.
# --rollout-max-attempts (3) caps attempts per sandbox visit; rejected groups are resampled.
# --zero-variance-policy retry_skip retries then skips zero-variance visits; retry_fail is strict.
# Only eligible groups enter optimization; skipped visits are reported separately.
# --context-limit-policy finish scores finite episodes without clipping history.
# --eval-interval-steps 100 evaluates held-out tasks at batch boundaries.
# JSONL metrics are live; JSON exports are written on exit (or --metrics-export-interval N).
# --optimization-passes (1) controls repeated optimization of a collected batch.
# Example: --epochs 2 --batch-size 4 --mini-batch-size 2 --rollout-group 8
# --rollout-workers (1) controls sampling processes; actor is the policy trainer.
# --rollout-concurrency (2) caps per-worker environments and dynamic GPU decode rows.
# Ready trajectories share one decode forward; tool-waiting trajectories leave the batch.
# --micro-batch-size (4) and --max-tokens-per-micro-batch (8192) bound tensor computation.
# --sandbox-services PATH reuses externally managed HTTP services (no auto cleanup).
# Optional standalone prewarm: scripts/sandbox_services.sh start; returns before readiness.
# Models, credentials and training artifacts are never embedded in this script.
# TensorBoard is enabled by default; scripts/train_dashboard.sh opens the viewer.
# --rollout-trace-samples N controls live traces; --no-tensorboard keeps JSONL only.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m rl.train --sandbox-backend docker "$@"
fi
exec uv run --extra rl python -m rl.train --sandbox-backend docker "$@"
