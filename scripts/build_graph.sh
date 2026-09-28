#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec uv run python examples/build_graph.py "$@"
fi

if [[ ! -f .env ]]; then
  echo "未找到 .env，请先复制 .env.example 并配置 Neo4j；完整 Scene 扩展还需 Wikipedia 和 LLM 参数。" >&2
  exit 1
fi

exec uv run python examples/build_graph.py "$@"
