#!/usr/bin/env bash
# Main entry (V3): domains -> requests -> sandboxes -> images -> policy gate -> tasks.json.
# Orchestration and validation live in src/env_factory_v3/production/run.py (`python -m env_factory_v3 run`).
# Arguments are arrays, so paths containing spaces remain single arguments.
# Do not source .env here: Python loads it as data, without executing shell code.
# The V1/V2 entry is scripts/run_pipeline_legacy.sh.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'HELP'
用法：./scripts/run_pipeline.sh [参数]

V3 端到端生产线：知识图谱采样业务领域 → codex 写任务（规则、数据、用户场景）→ deepseek 写用户开场白
→ codex 构建沙箱并通过全部 gate → 加固 Docker 镜像离线复验 → policy gate 真实对话验证 → tasks.json。
同一个 --output 再次运行会从中断处续跑（已完成的阶段和任务不重做）。

数量与来源：
  --count N              生成的任务数（默认 1）；类别比例 direct:simple:普通多步:复杂多步 = 2:2:2:4
  --output DIR           运行目录（默认 output/v3_run_<UTC 时间>）；同一目录再次运行即续跑
  --split NAME           从知识图谱该划分采样领域：dev（默认）、holdout-1..5；需要 Neo4j
                         holdout 划分留给泛化实验，日常生产用 dev
  --domains-file FILE    改用现成的领域列表 [{"name", "detail"}]，不访问图谱（与 --split 二选一）
  --domains N            采样的领域数（默认 min(count, 12)）
  --hops N               每个领域在场景图上的最大跳数（默认 2）
  --seed N               采样随机种子（默认 0），便于复现
  --splits FILE          固化的节点→划分映射（默认 experiments/v3/graph_splits.json），保证划分互不重叠
  --rubric-share X       主观任务（rubric 评分）占比，0..1（默认 0）；从多步名额中划出

构建：
  --parallel N           并发生成 / 构建数（默认 2）
  --repairs N            每个沙箱的修复轮数预算（默认 2）
  --model MODEL          codex 生成与构建模型（默认 gpt-6.1-sol）

可训练验证（默认全开，即良品标准）：
  --no-docker            跳过加固镜像构建与离线复验（不推荐：没有镜像的任务不进入 "tasks"）
  --policy-k N           policy gate 每个沙箱的 episode 数（默认 4；0 = 跳过）
                         N > 0 时，模型至少解出 1 次的沙箱才算可训练
  --policy-user-mode M   policy gate 的用户模式：natural（默认，自由对话，与训练一致）或 structured
  --policy-prefix P      policy gate 模型的环境变量前缀 <P>_API_KEY/_BASE_URL/_MODEL（默认 LLM）
  --eval-share X         可训练任务中按整领域划为评测集的比例，0..1（默认 0：全部训练）

其他：
  --dry-run              只打印将执行的命令，不调用模型、图谱或 Docker
  -h, --help             显示帮助

前置条件：uv sync；codex CLI 已安装并登录；Docker 可用；--split 需要 Neo4j（NEO4J_URI/USER/PASSWORD）；
项目 .env 配置 LLM_*（deepseek：开场白、policy gate、用户模拟、rubric 评审），可用 V3_OPENING_*、
V3_DOMAIN_*、V3_JUDGE_*、V3_USER_* 单独覆盖。

示例：
  ./scripts/run_pipeline.sh --count 10 --rubric-share 0.2 --parallel 3
  ./scripts/run_pipeline.sh --count 30 --output output/v3_batch_a --eval-share 0.2
  ./scripts/run_pipeline.sh --domains-file my_domains.json --count 5 --policy-user-mode natural
  ./scripts/run_pipeline.sh --output output/v3_batch_a      # 续跑（需与首次运行参数一致）

产物（<output>/）：
  tasks.json      "tasks"：可直接训练的任务（rl 格式，含 split train/eval）；"excluded"：其余任务及原因；
                  "details"：每个任务的类别、奖励类型、开场白、各阶段结果
  report.md       人可读汇总（良品数、各阶段失败分布）；report.json 为机器可读版本
  build/          每个任务的构建目录与认证 bundle；run_events.json 为阶段事件
日志：<output>/logs/run-*（每次运行独立保存，终端同步显示）。
训练：把 tasks.json 交给 rl 训练入口即可，bundle 使用 docker 后端时自动走加固容器。
退出码：0 至少一个可训练任务；1 没有可训练任务；2 配置或基础设施失败。
HELP
}

forward=()
dry_run=false
count=1
output_dir=""
domains=""
source_set=false
docker=true
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
    --no-docker) docker=false; shift ;;
    --count|--count=*|--output|--output=*|--split|--split=*|--domains-file|--domains-file=*|--domains|--domains=*|--hops|--hops=*|--seed|--seed=*|--splits|--splits=*|--rubric-share|--rubric-share=*|--parallel|--parallel=*|--repairs|--repairs=*|--model|--model=*|--policy-k|--policy-k=*|--policy-user-mode|--policy-user-mode=*|--policy-prefix|--policy-prefix=*|--eval-share|--eval-share=*)
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
      case "$flag" in
        --count) count="$value" ;;
        --output) output_dir="$value" ;;
        --domains) domains="$value" ;;
        *) [[ "$flag" != --split && "$flag" != --domains-file ]] || source_set=true
           forward+=("$flag" "$value") ;;
      esac ;;
    *) echo "不支持的参数：$1；请使用 --help 查看参数。" >&2; exit 2 ;;
  esac
done
if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
  echo "--count 必须是正整数。" >&2; exit 2
fi
if [[ -n "$domains" ]] && ! [[ "$domains" =~ ^[1-9][0-9]*$ ]]; then
  echo "--domains 必须是正整数。" >&2; exit 2
fi
[[ -n "$domains" ]] || domains=$(( count < 12 ? count : 12 ))
[[ -n "$output_dir" ]] || output_dir="output/v3_run_$(date -u +%Y%m%dT%H%M%SZ)"
cd "$project_dir"
if [[ -x "$project_dir/.venv/bin/python" ]]; then
  runner=("$project_dir/.venv/bin/python")
else
  runner=(uv run python)
fi
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
command=("${runner[@]}" -m env_factory_v3 run --output "$output_dir" --count "$count" --domains "$domains")
[[ "$source_set" == true ]] || command+=(--split dev)
[[ "$docker" != true ]] || command+=(--docker)
has_policy=false
for item in ${forward[@]+"${forward[@]}"}; do [[ "$item" != --policy-k ]] || has_policy=true; done
[[ "$has_policy" == true ]] || command+=(--policy-k 4)
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
log_file="$(mktemp "$output_dir/logs/run-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
export PYTHONUNBUFFERED=1
set +e
{
  printf '[%s] V3 生产线启动；运行目录：%s；日志：%s\n' "$(date -u +%FT%TZ)" "$output_dir" "$log_file"
  "${command[@]}"
  engine_status=$?
  printf '[%s] V3 生产线结束；退出码：%s；结果见 %s/report.md 与 tasks.json\n' \
    "$(date -u +%FT%TZ)" "$engine_status" "$output_dir"
  exit "$engine_status"
} 2>&1 | tee -a "$log_file"
status=$?
exit "$status"
