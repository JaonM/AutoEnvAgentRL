#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
用法: ./scripts/build_graph.sh [--offline | --online-wikipedia] [图谱扩展选项]

默认根据 .env 中的 WIKIPEDIA_DUMP_DB 使用本地 Wikipedia 索引，否则在线搜索。
图谱扩展通过 Neo4j 检查点增量续跑。其他参数传给 examples/build_graph.py。
EOF
  exit 0
fi
if [[ ! -f .env ]]; then
  echo "未找到 .env，请先配置 Neo4j、LLM 和 Wikipedia 来源。" >&2
  exit 1
fi
exec uv run python examples/build_graph.py "$@"
