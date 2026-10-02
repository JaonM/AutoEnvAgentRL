#!/usr/bin/env bash
# Main entry: graph -> Code Agent task authoring -> sandbox -> training gates.
# Keep orchestration in loop_experiment.py; this script only supplies CLI defaults.
# Arguments are arrays, so paths containing spaces remain single arguments.
# Do not source .env here: Python loads it as data, without executing shell code.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'HELP'
用法：./scripts/run_pipeline.sh [参数]

生成并构建任务（默认一个）：Code Agent 生成 → 前置验收 → 构建/独立审查 → live 验收。
参数（路径相对项目根目录）：
  --count N                  生成并构建的任务数量，正整数（默认 1）；不可与 --task-ids 同用
  --output DIR               输出根目录（默认 output）
  --task-ids IDS             重建已有任务，如 1 或 1,2；跳过生成
  --code-agent NAME          codex（默认）、claude 或 opencode；非 codex 需指定模型
  --code-agent-model MODEL   作者、构建、审查模型（默认 gpt-6-luna）
  --language LANGUAGE        作者生成内容的语种（默认 zh-CN；例如 en、ja）
  --validation live|offline  验收方式（默认 live；offline 不代表可训练）
  --code-agent-timeout SEC   任务生成超时秒数（默认 600）
  --dry-run                  仅打印命令，不调用模型、图谱或 Docker
  -h, --help                 显示帮助

语种约束任务描述、工具描述、用户剧本与参考回答；平台日志和固定运行时提示不翻译。
模型参数不改变 live rollout 和 user simulator 的模型，后两者由 .env 配置。
前置条件：uv sync、所选 Agent CLI 已安装并认证、Neo4j 与 Docker 可用；配置项目 .env。

示例：
  ./scripts/run_pipeline.sh --code-agent-model gpt-6-luna --language en
  ./scripts/run_pipeline.sh --count 5 --output output/my_tasks
  ./scripts/run_pipeline.sh --task-ids 1

产物：<output>/task/task-N/ 和 <output>/sandbox/task-N/。
日志：<output>/logs/pipeline-*（每次运行独立保存，终端同步显示）。
阶段事件：<output>/pipeline_events.jsonl；生成详情见 generation.log，构建详情见 sandbox/task-N/build.log。
可训练标记：status.json.training_ready == true；success 仅表示构建成功。
新生成自动递增编号。退出码：0 通过当前验收；1 不合格；2 配置或基础设施失败。
实验参数不在本入口开放；实验请直接使用 scripts/loop_experiment.py。
HELP
}

forward=()
dry_run=false
task_ids=false
count=1
count_set=false
agent=codex
model_set=false
output_dir=output
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
    --count|--count=*|--code-agent|--code-agent=*|--output|--task-ids|--code-agent-model|--language|--validation|--code-agent-timeout|--output=*|--task-ids=*|--code-agent-model=*|--language=*|--validation=*|--code-agent-timeout=*)
      flag="${1%%=*}"
      if [[ "$1" == *=* ]]; then
        value="${1#*=}"; shift
      else
        if [[ $# -lt 2 || "$2" == --* ]]; then
          echo "缺少参数值：$1" >&2; exit 2
        fi
        value="$2"; shift 2
      fi
      if [[ -z "${value//[[:space:]]/}" ]]; then
        echo "参数值不能为空：$flag" >&2; exit 2
      fi
      [[ "$flag" != --output ]] || output_dir="$value"
      [[ "$flag" != --task-ids ]] || task_ids=true
      [[ "$flag" != --code-agent ]] || agent="$value"
      [[ "$flag" != --code-agent-model ]] || model_set=true
      if [[ "$flag" == --count ]]; then
        count="$value"; count_set=true
      else
        forward+=("$flag" "$value")
      fi ;;
    *) echo "不支持的参数：$1；请使用 --help 查看日常构建参数。" >&2; exit 2 ;;
  esac
done
if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
  echo "--count 必须是正整数。" >&2; exit 2
fi
if [[ "$count_set" == true && "$task_ids" == true ]]; then
  echo "--count 与 --task-ids 不能同时使用。" >&2; exit 2
fi
case "$agent" in codex|claude|opencode) ;; *) echo "不支持的 Code Agent：$agent" >&2; exit 2 ;; esac
if [[ "$agent" != codex && "$model_set" != true ]]; then
  echo "使用 $agent 时请指定 --code-agent-model。" >&2; exit 2
fi
cd "$project_dir"
if [[ -x "$project_dir/.venv/bin/python" ]]; then
  runner=("$project_dir/.venv/bin/python")
else
  runner=(uv run python)
fi
# Internal daily defaults; only the documented options may override them.
command=("${runner[@]}" "$project_dir/scripts/loop_experiment.py"
  --layout daily --generation-backend code_agent --generation-hops 2
  --max-rounds 1 --max-concurrency 1 --max-attempts 1 --max-total-repairs 1
  --certification-profile pilot --validation live --sandbox-runtime docker
  --code-agent-timeout 600 --sample-build-budget 0 --threshold 8
  --training-mix direct_response=0.20,simple_agentic=0.30,multi_step_agentic=0.50
  --output output --code-agent-model gpt-6-luna --language zh-CN)
if [[ "$task_ids" == false ]]; then
  command+=(--generate-count "$count")
fi
# Bash 3.2 treats an empty array as unset under nounset.
command+=(${forward[@]+"${forward[@]}"})
if [[ "$dry_run" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
# Each invocation gets a separate transcript, including startup/configuration errors.
# pipefail preserves engine/tee failures instead of reporting tee's success.
mkdir -p "$output_dir/logs"
log_file="$(mktemp "$output_dir/logs/pipeline-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
export PYTHONUNBUFFERED=1
set +e
{
  printf '[%s] 流水线启动；日志：%s\n' "$(date -u +%FT%TZ)" "$log_file"
  "${command[@]}"
  engine_status=$?
  printf '[%s] 流水线结束；退出码：%s\n' "$(date -u +%FT%TZ)" "$engine_status"
  exit "$engine_status"
} 2>&1 | tee -a "$log_file"
status=$?
exit "$status"
