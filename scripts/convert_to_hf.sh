#!/usr/bin/env bash
# Install: uv sync --extra cuda-export (MLX is not required for conversion).
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
exec .venv/bin/python -m rl.convert_cuda "$@"
