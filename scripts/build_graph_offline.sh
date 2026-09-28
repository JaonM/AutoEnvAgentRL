#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  exec uv run python examples/build_graph.py "$@"
fi

if [[ ! -f .env ]]; then
  echo "未找到 .env；请配置 Neo4j、LLM 与 WIKIPEDIA_DUMP_DB。" >&2
  exit 1
fi

# Wikipedia and raw datasets stay local. The configured LLM endpoint may be remote.
exec uv run python examples/build_graph.py \
  --offline --incremental --local-dataset-links --skip-llm-dataset-links "$@"
