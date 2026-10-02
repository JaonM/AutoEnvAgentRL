#!/usr/bin/env bash
# Single-host Docker sandbox manager (default); no Kubernetes/registry required.
# start: detach manager, asynchronously prewarm and publish services.json.
# status: show readiness; stop: ask manager to reclaim its containers and save logs.
# Example: ./scripts/sandbox_services.sh start --sandbox output/sandbox/task-1 \
#   --output output/services --max-active 4 --startup-workers 2
# Requires Docker and SANDBOX_TRAINER_API_KEY (same key for standalone training).
# --images optionally maps task IDs to images; default uses qualified image metadata.
# Kubernetes remains available explicitly: python -m rl.kubernetes --help
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ -x .venv/bin/python ]]; then
  exec .venv/bin/python -m rl.docker_manager "$@"
fi
exec uv run python -m rl.docker_manager "$@"
