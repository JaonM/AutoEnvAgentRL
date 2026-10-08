#!/usr/bin/env bash
# Main entry: graph -> Code Agent task authoring -> sandbox -> training gates.
# Orchestration and defaults live in src/env_factory/task_pipeline.py.
# Arguments are arrays, so paths containing spaces remain single arguments.
# Do not source .env here: Python loads it as data, without executing shell code.
set -euo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'HELP'
用法：./scripts/run_pipeline.sh [参数]

生成并构建任务（默认一个）：Code Agent 生成 → 前置验收 → 构建/独立审查 → Docker 与可执行验收。
参数（路径相对项目根目录）：
  --count N                  生成并构建的任务数量，正整数（默认 1）
  --generation-workers N     并发任务生成数，正整数（默认 1）
  --build-workers N          并发沙箱构建数，正整数（默认 1）
  --output DIR               输出根目录（默认 output）
  --code-agent NAME          codex（默认）、claude 或 opencode；非 codex 需指定模型
  --code-agent-model MODEL   任务生成模型（默认 gpt-6.1-sol）
  --sandbox-model MODEL      沙箱构建与独立审查模型（Codex 默认 gpt-6.1-sol；其他 CLI 沿用任务模型）
  --generation-hops N        Scene 随机跳数上限 N，每次从 0..N 均匀抽取；N 为 0..20（默认 2）
  --training-category NAME  固定训练类别：direct_response、simple_agentic、multi_step_agentic
  --seed N                   固定批次随机种子，便于复现实验
  --task-type TYPES          QA、Event、Chat、Research；支持逗号分隔多选（默认全部随机）
  --environment-mode MODE    限定新任务环境模式（默认由 Agent 在可用且兼容的模式中选择）
  --user-script-count N      每个新任务的用户交互剧本（FSM）数量，正整数（默认 3）
  --language LANGUAGE        作者生成内容的语种（默认 zh-CN；例如 en、ja）
  --code-agent-timeout SEC   生成与修复模型进程累计秒数，不含验收（默认 600）
  --dry-run                  仅打印命令，不调用模型、图谱或 Docker
  -h, --help                 显示帮助

环境模式（控制业务形态，不改变 Docker 启动方式；生成结果不匹配会被前置验收拒绝）：
  stateless           无业务数据表，依据公开输入完成任务，如阅读材料回答问题
  reference_data      查询参考数据，无需修改业务状态，如查询库存、比较方案
  stateful            修改并持久化业务状态，必须有实际写入工具，如创建订单、修改预约
  external_capability 使用外部能力；需配置 SANDBOX_EXTERNAL_CAPABILITY_URL，
                      或 SANDBOX_EXTERNAL_FIXTURES 指向存在的 fixture 文件
                      fixture 是预先准备的测试数据或模拟响应，例如固定的物流查询结果，
                      用于可重复地模拟外部服务；文件存在仅代表允许选择该模式，内容仍需验收
不指定时默认允许前三种；外部能力已配置时额外允许 external_capability。
指定模式仍需与训练类别兼容，不会自动更改训练类别配比。

语种约束任务描述、工具描述、用户剧本与参考回答；平台日志和固定运行时提示不翻译。
构建不执行真实 rollout；训练使用的策略与 user simulator 模型由训练配置及 .env 决定。
前置条件：uv sync、所选 Agent CLI 已安装并认证、Neo4j 与 Docker 可用；配置项目 .env。

示例：
  ./scripts/run_pipeline.sh --code-agent-model gpt-6.1-sol --language en
  ./scripts/run_pipeline.sh --count 5 --user-script-count 3 --output output/my_tasks
  ./scripts/run_pipeline.sh --environment-mode stateful
  ./scripts/run_pipeline.sh --count 10 --generation-workers 3 --build-workers 2

产物：<output>/artifacts/artifact-N/task/ 和 <output>/artifacts/artifact-N/sandbox/。
日志：<output>/logs/pipeline-*（每次运行独立保存，终端同步显示）。
阶段事件：<output>/pipeline_events.jsonl；生成详情见 generation.log，构建详情见 artifacts/artifact-N/sandbox/build.log。
可训练标记：status.json.training_ready == true，表示通过可执行验收、允许进入训练，非真实策略验证。
success 仅表示构建成功；真实 rollout、奖励校准及轨迹隐私检查标为 not_run。
生成与构建交叠执行：每个任务验收完成后即可构建，两个阶段可分别设置并发数。
所有任务使用独立目录和容器标识；并发不会减少每个任务的修复预算或验收要求。
新生成自动递增 artifact 编号。旧目录不会自动迁移。
退出码：0 通过当前验收；1 不合格；2 配置或基础设施失败。
HELP
}

forward=()
dry_run=false
count=1
agent=codex
model_set=false
output_dir=output
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) dry_run=true; shift ;;
    --generation-workers|--generation-workers=*|--build-workers|--build-workers=*|--training-category|--training-category=*|--seed|--seed=*|--environment-mode|--environment-mode=*|--task-type|--task-type=*|--user-script-count|--user-script-count=*|--generation-hops|--generation-hops=*|--sandbox-model|--sandbox-model=*|--count|--count=*|--code-agent|--code-agent=*|--output|--code-agent-model|--language|--code-agent-timeout|--output=*|--code-agent-model=*|--language=*|--code-agent-timeout=*)
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
      [[ "$flag" != --code-agent ]] || agent="$value"
      [[ "$flag" != --code-agent-model ]] || model_set=true
      if [[ "$flag" == --count ]]; then
        count="$value"
      else
        forward+=("$flag" "$value")
      fi ;;
    *) echo "不支持的参数：$1；请使用 --help 查看日常构建参数。" >&2; exit 2 ;;
  esac
done
if ! [[ "$count" =~ ^[1-9][0-9]*$ ]]; then
  echo "--count 必须是正整数。" >&2; exit 2
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
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
command=("${runner[@]}" -m env_factory.task_pipeline)
command+=(--count "$count")
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
