#!/usr/bin/env bash
# Live local training curves and sampled rollout text. Override defaults via args.
# Example: scripts/train_dashboard.sh --logdir output/rl_runs/my-run/tensorboard
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
args=(--logdir output/rl_runs --host 127.0.0.1 --port 6006 --reload_interval 5 --samples_per_plugin text=200)
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m tensorboard.main "${args[@]}" "$@"
fi
exec uv run --extra rl python -m tensorboard.main "${args[@]}" "$@"
