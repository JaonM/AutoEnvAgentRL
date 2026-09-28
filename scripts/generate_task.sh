#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec uv run python examples/generate_task.py "$@"
fi

if [[ ! -f .env ]]; then
  echo "未找到 .env，请先配置 LLM 参数；使用 --generation-source graph 时还需配置 Neo4j。" >&2
  exit 1
fi

exec uv run python examples/generate_task.py "$@"
