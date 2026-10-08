#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run python -m env_factory.generate_task "$@"
