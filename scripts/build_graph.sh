#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

usage() {
  cat <<'EOF'
用法: ./scripts/build_graph.sh [选项]

默认增量构建 Scene、目录及关系；自动使用 .env 中的本地 Wikipedia 索引。
已下载数据集会建立候选关系，已准入来源会运行 LLM 关系匹配。

常用选项:
  --data-only          只构建数据集目录和关系，不扩展 Scene
  --links-only         只更新关系，不重复同步目录或扩展 Scene
  --dataset KEY        只匹配指定的已下载数据集，可重复使用
  --max-datasets N     本轮最多审核 N 个已下载数据集，默认 20
  --offline            强制使用本地 Wikipedia 索引和本地原始文件
  --online             强制使用在线 Wikipedia 搜索
  --skip-approved-llm  跳过已准入来源的 LLM 关系匹配
  -h, --help           显示帮助

高级参数可直接透传给 examples/build_graph.py。
EOF
}

mode="all"
source_mode="auto"
skip_approved=false
mode_explicit=false
dataset_keys=()
max_datasets=""
extra=()
while (($#)); do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --data-only|--datasets-only)
      if [[ "$mode" != "links" ]]; then mode="data"; fi
      mode_explicit=true ;;
    --links-only) mode="links"; mode_explicit=true ;;
    --dataset|--local-dataset-key)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        echo "$1 需要数据集键，例如 kaggle:owner/slug" >&2; exit 2
      fi
      dataset_keys+=("$2"); shift ;;
    --max-datasets|--max-local-datasets)
      if (($# < 2)) || [[ "$2" == --* ]]; then
        echo "$1 需要正整数" >&2; exit 2
      fi
      max_datasets="$2"; shift ;;
    --offline) source_mode="offline" ;;
    --online) source_mode="online" ;;
    --skip-approved-llm|--skip-llm-dataset-links) skip_approved=true ;;
    *) extra+=("$1") ;;
  esac
  shift
done
if ((${#dataset_keys[@]} > 0)) && [[ "$mode_explicit" == false ]]; then
  mode="links"
fi

if [[ ! -f .env ]]; then
  echo "未找到 .env，请先配置 Neo4j、LLM 和 Wikipedia 来源。" >&2
  exit 1
fi

args=(--incremental --local-dataset-links)
if [[ "$mode" == "data" || "$mode" == "links" ]]; then
  args+=(--datasets-only)
fi
if [[ "$mode" == "links" ]]; then
  args+=(--links-only)
fi
if [[ "$source_mode" == "offline" ]]; then
  args+=(--offline)
elif [[ "$source_mode" == "online" ]]; then
  args+=(--online-wikipedia)
elif grep -Eq '^[[:space:]]*WIKIPEDIA_DUMP_DB[[:space:]]*=[[:space:]]*[^[:space:]#]+' .env; then
  args+=(--offline)
fi
if [[ "$skip_approved" == true ]]; then
  args+=(--skip-llm-dataset-links)
fi
if [[ -n "$max_datasets" ]]; then
  args+=(--max-local-datasets "$max_datasets")
fi
if ((${#dataset_keys[@]} > 0)); then
  for key in "${dataset_keys[@]}"; do
    args+=(--dataset-key "$key")
  done
fi
if ((${#extra[@]} > 0)); then
  args+=("${extra[@]}")
fi
exec uv run python examples/build_graph.py "${args[@]}"
