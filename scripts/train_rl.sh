#!/usr/bin/env bash
# Local Apple Silicon RL: qualified sandbox -> rollout workers -> actor (PPO/GRPO + QAT).
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
# Only eligible groups fill batch/mini-batch quotas; surplus results remain queued.
# --optimization-passes (1) controls repeated optimization of a collected batch.
# Example: --epochs 2 --batch-size 4 --mini-batch-size 2 --rollout-group 8
# --rollout-workers (1) controls sampling processes; actor is the policy trainer.
# --rollout-concurrency (2) caps per-worker environments and dynamic GPU decode rows.
# Ready trajectories share one decode forward; tool-waiting trajectories leave the batch.
# --micro-batch-size (4) and --max-tokens-per-micro-batch (8192) bound tensor computation.
# --sandbox-services PATH reuses externally managed HTTP services (no auto cleanup).
# Optional standalone prewarm: scripts/sandbox_services.sh start; returns before readiness.
# Models, credentials and training artifacts are never embedded in this script.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m rl.train --sandbox-backend docker "$@"
fi
exec uv run --extra rl python -m rl.train --sandbox-backend docker "$@"
