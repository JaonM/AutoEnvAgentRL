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

Code Agent 主链路：图谱采样 → GPT-6-luna 生成任务 → 前置验收 →
GPT-6-luna 构建/独立审查沙箱 → 离线评分 → live 验收与奖励校准。

默认值（可用同名参数覆盖）：
  --generate-count 5          每轮生成 5 个样本
  --max-rounds 1              只运行一轮；0 表示持续运行至目标/总预算
  --generation-backend code_agent
  --generation-hops 2         图谱路径最大跳数
  --max-concurrency 4         调度并发；生成 worker 由调度器分配
  --max-attempts 1            构建尝试上限
  --max-total-repairs 1       全局修复预算
  --certification-profile pilot  候选验证；非 production 批次认证
  --validation live          真实模型验收；offline 不会标记可训练
  --sandbox-runtime docker   Docker 构建与执行
  --code-agent-timeout 600    单样本生成防卡死超时（秒）
  --sample-build-budget 0    不启用生成+构建硬时限；5 min 仅为观测目标
  --threshold 8              10 分制门槛
  --training-mix direct_response=0.20,simple_agentic=0.30,multi_step_agentic=0.50
  --output output/pipeline-<UTC时间>-<进程号>  新批次目录

脚本参数：
  -h, --help                 显示本说明，不启动任务
  --dry-run                  打印将执行的命令，不访问模型/图谱/Docker
  --engine-help              显示底层调度器所有参数
其余参数原样传递给 scripts/loop_experiment.py。路径相对项目根目录解析。
--task-ids 用于重建已有任务时，不添加默认 --generate-count。

前置条件：uv sync；codex CLI 已安装并认证；Neo4j 图谱可用；Docker 可用。
在项目 .env 配置 Neo4j、ROLLOUT_LLM_*、SANDBOX_LLM_*（可回退 LLM_*）。
任务作者/构建/审查使用 GPT-6-luna；live 的模型由 .env 决定。

示例：
  ./scripts/run_pipeline.sh
  ./scripts/run_pipeline.sh --output output/my_batch --experiment-seed 20260930
  ./scripts/run_pipeline.sh --generation-environment-mode stateful \
    --training-mix direct_response=0,simple_agentic=0,multi_step_agentic=1
  ./scripts/run_pipeline.sh --validation offline --dry-run

产物：<output>/round-01/generation/task/task-N/（任务生成交付物）
      <output>/round-01/sample-NNN/attempt-N/status.json（沙箱状态）
      <output>/history.json、round-01/round_report.json（汇总与恢复状态）
消费条件：status.json.training_ready == true；success 仅表示构建成功。
同一 --output 用于恢复同一配置的批次；新实验使用新目录。
退出码沿用调度器：0 达到实验目标；1 未达到目标/轮数耗尽；2 配置或基础设施失败。
单个样本是否可用请读 training_ready，不要仅依据批次退出码。
HELP
}

forward=()
dry_run=false
task_ids=false
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true ;;
    --engine-help) forward+=(--help) ;;
    --task-ids|--task-ids=*) task_ids=true; forward+=("$arg") ;;
    *) forward+=("$arg") ;;
  esac
done
cd "$project_dir"
if [[ -x "$project_dir/.venv/bin/python" ]]; then
  runner=("$project_dir/.venv/bin/python")
else
  runner=(uv run python)
fi
# Later occurrences override these argparse defaults. Explicit backend selection
# is retained for reproducible comparisons with spec and legacy experiments.
command=("${runner[@]}" "$project_dir/scripts/loop_experiment.py"
  --generation-backend code_agent --generation-hops 2
  --max-rounds 1 --max-concurrency 4 --max-attempts 1 --max-total-repairs 1
  --certification-profile pilot --validation live --sandbox-runtime docker
  --code-agent-timeout 600 --sample-build-budget 0 --threshold 8
  --training-mix direct_response=0.20,simple_agentic=0.30,multi_step_agentic=0.50
  --output "output/pipeline-$(date -u +%Y%m%dT%H%M%SZ)-$$")
if [[ "$task_ids" == false ]]; then
  command+=(--generate-count 5)
fi
# Bash 3.2 treats an empty array as unset under nounset.
command+=(${forward[@]+"${forward[@]}"})
if [[ "$dry_run" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
  exit 0
fi
exec "${command[@]}"
