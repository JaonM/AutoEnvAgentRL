#!/usr/bin/env bash

set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The workflow invokes project-owned Python entry points directly.  Make the
# src-layout package importable even when the caller did not activate the
# repository's virtual environment or install EnvFactory into site-packages.
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ -x "$project_dir/.venv/bin/python3" ]]; then
  export PATH="$project_dir/.venv/bin:$PATH"
fi
input="examples/clothing_materials_task.json"
output=""
output_auto="true"
agent="codex"
review_agent="codex"
model=""
review_model=""
runtime="none"
tag="env-factory-agent-sandbox"
start="false"
background="true"
max_concurrency="2"
max_attempts="3"
resume="false"
auto_score="true"
sandbox_port="8080"
env_file=""
current_phase="initializing"
current_node_id=""
current_attempt="0"
current_defect_ids=""
current_failure_category=""
current_failure_code=""

usage() {
  cat <<'EOF'
用法：scripts/develop_sandbox_with_agent.sh [选项]

调用 Code Agent 根据任务输入自主开发一个 RL 沙箱工程，然后可选构建容器。
默认后台运行，Agent 日志写入目标目录的 agent.log。
构建完成后默认不启动沙箱服务；只有显式传入 --start 才会启动容器。

选项：
  --input PATH       任务 JSON、JSON task list 或包含 task-N/task.json 的 artifacts 目录，默认：examples/clothing_materials_task.json
  --output DIR       单任务工作目录；输入为 list 时作为输出根目录，默认沿用 task-N 编号
  --agent NAME       开发 Agent：codex、claude 或 opencode，默认：codex
  --review-agent NAME 独立语义审查 Agent：codex、claude 或 opencode，默认：codex
  --model NAME       Codex 开发模型，默认 gpt-6-luna
  --review-model NAME Codex 审查模型；默认沿用 --model，均未指定时使用 Codex 配置
  --runtime NAME     none 或 docker，默认：none
  --tag NAME         镜像名称，默认：env-factory-agent-sandbox
  --max-concurrency N 并发任务数，默认：2
  --max-attempts N   每个开发节点及每个独立缺陷的最大修复次数，默认：3
  --resume           复用输出目录中的已有实现，跳过模块重开发并直接验收/修复
  --skip-auto-score  构建成功后不自动执行离线评分
  --sandbox-port N   Docker 宿主机映射端口，容器端口固定为 8000，默认：8080
  --env-file FILE    Docker 启动时注入的环境变量文件，默认：不使用
  --start            构建后立即启动容器，默认：不启动
  --foreground       前台等待 Agent 完成，默认：后台运行
  -h, --help         显示帮助
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input|--output|--agent|--review-agent|--model|--review-model|--runtime|--tag|--max-concurrency|--max-attempts|--sandbox-port|--env-file)
      if (( $# < 2 )); then
        echo "$1 需要提供参数值" >&2
        exit 2
      fi
      case "$1" in
        --input) input="$2" ;;
        --output) output="$2"; output_auto="false" ;;
        --agent) agent="$2" ;;
        --review-agent) review_agent="$2" ;;
        --model) model="$2" ;;
        --review-model) review_model="$2" ;;
        --runtime) runtime="$2" ;;
        --tag) tag="$2" ;;
        --max-concurrency) max_concurrency="$2" ;;
        --max-attempts) max_attempts="$2" ;;
        --sandbox-port) sandbox_port="$2" ;;
        --env-file) env_file="$2" ;;
      esac
      shift 2
      ;;
    --start) start="true"; shift ;;
    --resume) resume="true"; shift ;;
    --skip-auto-score) auto_score="false"; shift ;;
    --foreground) background="false"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数：$1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$max_concurrency" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-concurrency 必须是正整数：$max_concurrency" >&2
  exit 2
fi

if ! [[ "$max_attempts" =~ ^[1-9][0-9]*$ ]]; then
  echo "--max-attempts 必须是正整数：$max_attempts" >&2
  exit 2
fi
if ! [[ "$sandbox_port" =~ ^[1-9][0-9]*$ ]] || (( sandbox_port > 65535 )); then
  echo "--sandbox-port 必须是 1 到 65535：$sandbox_port" >&2
  exit 2
fi
case "$agent" in codex|claude|opencode) ;; *) echo "不支持的开发 Agent：$agent" >&2; exit 2 ;; esac
case "$review_agent" in codex|claude|opencode) ;; *) echo "不支持的审查 Agent：$review_agent" >&2; exit 2 ;; esac
if [[ -z "$model" && "$agent" == "codex" ]]; then model="gpt-6-luna"; fi
if [[ -z "$review_model" && "$review_agent" == "codex" ]]; then review_model="${model:-gpt-6-luna}"; fi
if [[ -n "$model" && "$agent" != "codex" ]]; then
  echo "--model 当前仅适用于 --agent codex" >&2
  exit 2
fi
if [[ -n "$review_model" && "$review_agent" != "codex" ]]; then
  echo "--review-model 当前仅适用于 --review-agent codex" >&2
  exit 2
fi
case "$runtime" in none|docker) ;; *) echo "不支持的 runtime：$runtime" >&2; exit 2 ;; esac
if [[ -z "$output" && "$output_auto" == "false" ]]; then
  echo "--output 不能为空" >&2
  exit 2
fi

if [[ "$start" == "true" && "$runtime" != "docker" ]]; then
  echo "--start 仅在 --runtime docker 时可用" >&2
  exit 2
fi

input_path="$input"
if [[ "$input_path" != /* ]]; then input_path="$project_dir/$input_path"; fi
input_origin="$input_path"
temporary_input=""
cleanup_temporary_input() {
  if [[ -n "$temporary_input" ]]; then
    rm -f "$temporary_input"
  fi
}
trap cleanup_temporary_input EXIT

if [[ -d "$input_path" ]]; then
  if [[ -f "$input_path/task.json" ]]; then
    input_path="$input_path/task.json"
  else
    task_json_candidates=()
    while IFS= read -r candidate; do task_json_candidates+=("$candidate"); done < <(find "$input_path" -mindepth 2 -maxdepth 2 -type f -name task.json -print | sort)
    if (( ${#task_json_candidates[@]} == 0 )); then
      echo "--input 目录必须直接包含 task.json，或包含一个或多个 task-N/task.json：$input_path" >&2
      exit 3
    elif (( ${#task_json_candidates[@]} == 1 )); then
      input_path="${task_json_candidates[0]}"
    else
      temporary_input="$(mktemp "${TMPDIR:-/tmp}/envfactory-task-list.XXXXXX")"
      python3 - "$temporary_input" "${task_json_candidates[@]}" <<'PY'
import json
import sys
from pathlib import Path

destination = Path(sys.argv[1])
tasks = [json.loads(Path(item).read_text(encoding="utf-8")) for item in sys.argv[2:]]
if not all(isinstance(task, dict) for task in tasks):
    raise SystemExit("任务目录中的 task.json 必须都是 JSON object")
destination.write_text(json.dumps(tasks, ensure_ascii=False) + "\n", encoding="utf-8")
PY
      input_path="$temporary_input"
    fi
  fi
fi
[[ -f "$input_path" ]] || { echo "任务输入不存在：$input_path" >&2; exit 3; }
task_count="$(python3 - "$input_path" <<'PY'
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if isinstance(value, list):
    if not value:
        raise SystemExit("任务 JSON list 不能为空")
    if not all(isinstance(task, dict) for task in value):
        raise SystemExit("任务 JSON list 的每一项必须是 object")
    print(len(value))
elif isinstance(value, dict):
    print(1)
else:
    raise SystemExit("任务输入必须是 JSON object 或 JSON list")
PY
)" || exit 3

if [[ "$output_auto" == "true" ]]; then
  output_name=""
  origin_name="$(basename "${input_origin%/}")"
  origin_parent="$(basename "$(dirname "${input_origin%/}")")"
  if [[ "$origin_name" =~ ^task[-_][0-9]+$ ]]; then
    output_name="$origin_name"
  elif [[ "$origin_name" == "task.json" && "$origin_parent" =~ ^task[-_][0-9]+$ ]]; then
    output_name="$origin_parent"
  fi
  if [[ -z "$output_name" ]]; then
    output_name="$(python3 - "$input_path" "$task_count" <<'PY'
import json
import re
import sys
import unicodedata
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
count = int(sys.argv[2])
task = value[0] if isinstance(value, list) else value
description = task.get("task") if isinstance(task, dict) else ""
description = unicodedata.normalize("NFKC", description if isinstance(description, str) else "")
name = re.sub(r"[^\w-]+", "_", description, flags=re.UNICODE).strip("._-")[:48]
name = name or "task"
print(f"{name}-batch" if count > 1 else name)
PY
    )"
  fi
  output="output/sandbox/$output_name"
fi

output_path="$output"
if [[ "$output_path" != /* ]]; then output_path="$project_dir/$output_path"; fi
# 规范化用户传入的目录，避免 --output xxx/ 与后续 /agent.log 等路径拼接产生双斜杠。
if [[ "$output_path" != "/" ]]; then output_path="${output_path%/}"; fi
if [[ "$output_path" == "$project_dir" || "$output_path" == "/" ]]; then
  echo "--output 不得指向项目根目录或文件系统根目录：$output_path" >&2
  exit 2
fi
if [[ -n "$env_file" && "$env_file" != /* ]]; then env_file="$project_dir/$env_file"; fi
if [[ -n "$env_file" && ! -f "$env_file" ]]; then
  echo "--env-file 文件不存在：$env_file" >&2
  exit 3
fi
root_output_path="$output_path"

prepare_task() {
  local task_index="$1"
  local task_output="$2"
  rm -f \
    "$task_output/status.json.tmp" \
    "$task_output/agent.log" \
    "$task_output/TASK_PROMPT.md" "$task_output/SPEC_TASK.md" "$task_output/AGENT_TASK.md" \
    "$task_output/BUILD_CONTRACT.json" \
    "$task_output/review_report.json" "$task_output/review_agent.stdout" "$task_output/review_agent.stderr" \
    "$task_output/defects.json" "$task_output/last_delivery_error.txt" "$task_output/acceptance_failure.log"
  if [[ "$resume" != "true" ]]; then
    rm -f \
      "$task_output/defect_attempts.json" \
      "$task_output/spec.md" "$task_output/action_plan.json" "$task_output/development_plan.json" "$task_output/tools.json" "$task_output/Dockerfile" \
      "$task_output/docker_build.sh" "$task_output/docker_run.sh" \
      "$task_output/acceptance.sh" "$task_output/IMPLEMENTATION_REPORT.md" \
      "$task_output/app.py" "$task_output/task_impl.py"
    rm -rf "$task_output/data" "$task_output/tests" "$task_output/.outer_conformance"
  fi
  mkdir -p "$task_output/data"
  # Provide the single reviewed runtime adapter to the Code Agent; the Agent
  # must use it for User Simulator and reward evaluator LLM calls.
  cp "$project_dir/src/env_factory/runtime_llm.py" "$task_output/runtime_llm.py"
  cp "$project_dir/src/env_factory/sandbox_runtime.py" "$task_output/sandbox_runtime.py"
  python3 - "$input_path" "$task_output" "$task_index" <<'PY'
import sys
from pathlib import Path

from env_factory.tasks.task_portability import prepare_sandbox_task

prepare_sandbox_task(
    Path(sys.argv[1]), Path(sys.argv[2]), index=int(sys.argv[3])
)
PY
  python3 - "$task_output/task.json" "$task_output/BUILD_CONTRACT.json" <<'PY'
import json
import sys
from pathlib import Path

task = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if not isinstance(task, dict):
    raise SystemExit("task.json 必须是 JSON object")

# BUILD_CONTRACT is a read-only projection of task.json.  Do not synthesize
# platform obligations, capabilities, endpoint rules, or evaluation fields:
# anything not present in the generated task must not become an implementation
# requirement merely because this outer script guessed it.
required_lists = ("metrics",)
for key in required_lists:
    if not isinstance(task.get(key), list):
        raise SystemExit(f"task.json.{key} 必须是 list")
if not isinstance(task.get("requirements", {}), dict):
    raise SystemExit("task.json.requirements 必须是 object")

# Actions remain available in task.json as task-generation context, but they
# are not part of the sandbox build contract. The sandbox exposes and executes
# the declared LLM tools; it does not build a second Trainer-action registry.
contract = {key: value for key, value in task.items() if key != "actions"}
Path(sys.argv[2]).write_text(
    json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY
  chmod 444 "$task_output/BUILD_CONTRACT.json"
  scaffold_args=(--root "$task_output")
  if [[ "$resume" == "true" ]]; then scaffold_args+=(--preserve-implementation); fi
  python3 "$project_dir/scripts/sandbox/generate_sandbox_scaffold.py" "${scaffold_args[@]}"
  write_task_prompt "$task_output"
}

write_task_prompt() {
  local task_output="$1"
  local validation_python="python3"
  if [[ -x "$project_dir/.venv/bin/python3" ]]; then
    validation_python="$project_dir/.venv/bin/python3"
  fi
  cp "$project_dir/docs/sandbox_spec_prompt.md" "$task_output/TASK_PROMPT.md"
  cat >> "$task_output/TASK_PROMPT.md" <<EOF

The complete task input is in ./task.json. The immutable outer-workflow contract is in ./BUILD_CONTRACT.json; it is the task.json projection with the top-level actions field removed. Treat it as authoritative for sandbox implementation. Implement the sandbox directly from BUILD_CONTRACT.json and the referenced artifacts. Do not create or depend on spec.md. EnvFactory generates an outer conformance plan after implementation; do not modify, replace, or treat the sandbox's acceptance.sh as authoritative.
The project output directory is: $task_output
For local validation use: $validation_python -m pytest -q. If a login shell resets PATH, use this interpreter directly; do not install dependencies from PyPI during sandbox development.
EOF
  # The rest is literal instruction text. Quoting the delimiter prevents
  # backticked examples and shell variables from executing during preparation.
  cat >> "$task_output/TASK_PROMPT.md" <<'EOF'

Reward implementation is contract-driven and must not be simplified. Load every metric.evaluator and top-level metric_implementations entry from BUILD_CONTRACT.json. Compiled process and rule-based metrics use DeclarativeMetricEvaluator; only metrics still declared model-based use the external evaluator boundary. Do not use fixed tool positions, keyword presence, successful-call coverage, or a boolean final-document shortcut as a substitute. Acceptance must test passing and failing cases for every evaluator family, including wrong tool arguments, wrong numeric results, missing business changes, repeated invalid calls, and user-intent deviation. The reward endpoint must use the declared metric scores and reward_formula exactly.

The sandbox is acceptable only if it teaches the tool policy declared by BUILD_CONTRACT.training_contract. For direct_response, the successful trajectory must not call tools and unnecessary tool use must not improve reward. For tool-requiring profiles, a canned final answer without required task-tool calls must receive reward <= 0.2. A trajectory with corrupted business arguments, placeholder values, empty required collections, empty business results, or skipped required steps must be rejected or receive reward <= 0.2. For dependent_tool_chain, downstream tools must consume upstream outputs. Valid tool-requiring trajectories must use real fixture-backed values and every required tool result must contain meaningful business evidence. Never derive an evaluator's expected arguments from the actual call being scored. Deterministic evaluator mocks must compile expected calls independently from task intent, reward_key_steps, tool bindings, and fixture values. Outcome reward must be gated by the process steps required by the declared profile.

The implementation must be contract-generic: do not copy generated metric IDs, metric weights, fixed expected tool-call maps, or fixed turn cutoffs into app.py. Load metric/evaluator definitions, reward weights, user_profiles, and user_scripts from BUILD_CONTRACT.json and its manifests. The runtime UserSimulator must reason from the current profile, FSM state, legal outgoing transitions, variables, and complete live conversation through RuntimeLLMClient. Implement a real ToolRegistry, RewardEvaluator, and UserSimulator boundary. The outer workflow will reject source that hard-codes generated metric IDs or a dialogue transcript.

## Implementation task

Follow the outer workflow's module development plan and implement one node at a time; do not treat this prompt as permission to skip the planned node boundaries.

Read BUILD_CONTRACT.json and the current development_plan node, then implement only that node's declared extension points. Use only fields and artifacts present in the contract; do not invent missing task semantics or add default termination, observation, platform, or evaluation rules. Implement the declared LLM tools and their complete execution logic; tool endpoints must only validate inputs, execute business behavior, persist changes, and return the business result. They must not calculate or return observation or reward. Implement the observation endpoint as the only endpoint for public environment observation. Implement reward calculation only behind the declared reward endpoint, and calculate it when that endpoint is called from the accumulated session trace and current business data. Do not create a separate task-action or Trainer-action registry. Use the platform-owned persistence, simulator, evaluation context and reward gate. Implement only missing business handlers and metric extensions, focused tests, acceptance.sh and IMPLEMENTATION_REPORT.md. Do not rewrite the outer-owned tools.json, requirements-dev.txt, Dockerfile, docker_build.sh, or docker_run.sh. The sandbox test suite must be executable with `python3 -m pytest -q`; do not report pytest as passed when it was skipped. Each defect repair must add or run a focused pytest case and report its test node in the defect-verification evidence. After acceptance, write acceptance_result.json with business_acceptance=passed and http_conformance=passed or skipped; when HTTP is skipped, include an explicit http_skip_reason. Do not create spec.md, action_plan.json, development_plan.json, topology files, or any other design handoff.

Production runtime requirements are mandatory: use the provided ./runtime_llm.py adapter and ./sandbox_runtime.py primitives for User Simulator, reward evaluator, authentication, episode storage, idempotency, and replay. Implement Trainer Bearer authentication using SANDBOX_TRAINER_API_KEY for reset, observation, state, user_simulator, reward, and replay; keep Agent tool endpoints separate and never expose Trainer endpoints through the Agent tool registry. `/v1/state` is trainer-only evidence for initial/final business-state verification and must never be copied into the policy observation. Implement per-episode storage isolation, POST /v1/reset with optional episode_id and seed, deterministic seeded replay, idempotency using the declared Idempotency-Key header, and GET /v1/replay with trace/data/version hashes. The shared adapter must use SANDBOX_LLM_API_KEY, SANDBOX_LLM_BASE_URL, SANDBOX_LLM_MODEL, SANDBOX_LLM_TIMEOUT_SECONDS, and SANDBOX_LLM_MAX_RETRIES. Never persist or log credentials. Implement SANDBOX_EVALUATOR_MOCK for deterministic evaluator tests; record evaluator calls and ensure mock and real paths use the same input/output schema. Add contract tests for unauthorized Trainer requests, cross-episode data leakage, reset/seed determinism, idempotent retries, replay integrity, LLM timeout/fallback, evaluator mock/real parity, and reward changes caused by real business-data changes. Run all checks and acceptance.sh before finishing.

The outer workflow has already generated app.py, task_impl.py, tools.json, requirements-dev.txt, Dockerfile, docker_build.sh, and docker_run.sh. Preserve these EnvFactory-owned composition and delivery assets; implement task-specific extension points in task_impl.py and add focused tests/acceptance evidence. Use sandbox_runtime.ManifestDataStore for manifest validation, baseline loading, data hashes, and per-episode business-data copies. Use sandbox_runtime.SandboxApplication as the HTTP/WSGI boundary; app.py must remain a thin composition and launcher, not a regenerated router. Use sandbox_runtime.ContractToolRegistry for tool schema validation, dispatch, tracing, and mutation hooks, sandbox_runtime.ContractUserSimulator for seeded FSM state, external-LLM reasoning, and fallback, sandbox_runtime.DeclarativeMetricEvaluator for metric_implementations, sandbox_runtime.ContractEvaluatorRuntime for cached and trace-visible external evaluator calls, and sandbox_runtime.ContractRewardAggregator for reward range validation, weighting, and clipping. Do not call RuntimeLLMClient directly from metric scoring. Do not inject a task-specific user renderer into ContractUserSimulator: the default runtime path must use its external LLM and its bounded non-advancing recovery when that boundary is unavailable. Pass BUILD_CONTRACT.noise_tools to ContractToolRegistry; do not write task-specific handlers for noise tools. Noise calls must be trace-visible but must not change task-critical state or satisfy rewarded key steps. Generate only task-specific business handlers, observation callbacks, and scoring logic for metrics not covered by metric_implementations; do not duplicate these generic runtime components.

Mutation testing is a mandatory executable acceptance gate, not documentation. Implement the declared SANDBOX_MUTATION_MODE environment variable with production default disabled and every declared mode: constant_tool_result, skip_business_write, constant_reward, ignore_tool_arguments, and bypass_trainer_auth. Each non-disabled mode must deliberately introduce the named defect in an externally observable way so the normal acceptance tests fail: constant_tool_result must make a valid tool case return an invalid/constant result; skip_business_write must suppress a business write that a success scenario requires; constant_reward must make /v1/reward disagree with the declared formula or omit valid components; ignore_tool_arguments must accept or process an invalid/different argument contrary to the tool schema; bypass_trainer_auth must allow an unauthenticated Trainer-only request. Do not expose a mutation endpoint and never enable a mutation mode in production. The outer workflow will run acceptance.sh and independent HTTP conformance checks once per mutation mode and will reject the build if any mutant survives. Ensure acceptance.sh itself starts the app with inherited environment variables and has assertions for every mutation mode.

The executable acceptance script must invoke python3 (or the active interpreter via "$PYTHON"), never the bare python command, because the outer environment may not provide a python alias. Its TCP-permission fallback must validate the same contract through the public business API and must not hard-code a task-specific field unless that field and expected value are present in BUILD_CONTRACT/task.json.

Tool handlers enforce the JSON schema, not stronger restrictions inferred from descriptive examples: wording such as "fixed" does not create an enum. A read/search/lookup tool must remain read-only. For deliverable-only tasks with no mutating business tool, POST /v1/agent_response is the persisted deliverable boundary; do not make a read tool write a synthetic outcome row. For environment_plan.mode=external_capability, route task tools through ExternalCapabilityClient. It supports a configured HTTP provider, deterministic training fixtures, and an explicit data_unavailable response; never fabricate live external facts.
Do not inspect replay/tool-call history inside a business handler to reject a call as PRECONDITION_FAILED or to enforce capability-DAG order. A downstream tool validates its declared arguments and current business state; shared ContractRewardGate and process metrics own causal ordering. This separation is mandatory so an Agent can recover from an early wrong-order call by later executing a valid dependency chain.

External evaluator failure must be conservative: never derive the expected tool or arguments from the actual call being scored, and never award a match merely because any call occurred. Use a contract-derived independent expectation when present, otherwise score zero. ContractUserSimulator calls the configured RuntimeLLMClient by default using only the sanitized payload: complete live messages, profile, current state, outgoing transitions, variables, and recovery policy. Task-specific user_renderer injection is forbidden. It must emit one of the fixed dialogue outcomes: goal_satisfied, information_required, user_correction, user_rejection, user_acceptance, agent_off_topic, agent_premature_completion, or unrecognized. Normal outcomes require match_status=matched and a current transition_id carrying the same outcome_category. Recovery outcomes must not select or advance a normal transition; unrecognized may be ambiguous, while the other recovery outcomes are unmatched. The shared runtime bounds recovery attempts and terminates them as unresolved_dialogue. The renderer must not receive future turns or hidden business truth. Do not replace profile-aware rendering with hard-coded state-name responses.

Preserve the scaffold's contract-generic observation fields: conversation, available_tools, tool_results, episode_id, and public_observation. Task-specific observation data may extend public_observation but must not replace the required top-level observation contract or expose hidden simulator/reward state.
EOF
}

run_code_agent() {
  local phase_prompt="$1"
  local capture code
  case "$agent" in
    codex)
      codex_args=(exec --approve-for-me)
      if [[ -n "$model" ]]; then codex_args+=(--model "$model"); fi
      codex_args+=("$phase_prompt")
      capture="$(mktemp "${TMPDIR:-/tmp}/envfactory-agent-start.XXXXXX")"
      (cd "$output_path" && codex "${codex_args[@]}") </dev/null 2>&1 | tee "$capture"
      code="${PIPESTATUS[0]}"
      if [[ "$code" -ne 0 ]] && grep -q 'failed to initialize in-process app-server client' "$capture"; then
        current_failure_category="infrastructure"
        current_failure_code="INFRA"
      fi
      rm -f "$capture"
      return "$code"
      ;;
    claude)
      (cd "$output_path" && claude --dangerously-skip-permissions --print "$phase_prompt") </dev/null
      ;;
    opencode)
      (cd "$output_path" && opencode run "$phase_prompt") </dev/null
      ;;
    *)
      echo "不支持的 agent：${agent}；可选值为 codex、claude、opencode"
      return 2
      ;;
  esac
}

write_status() {
  local status="$1"
  local exit_code="$2"
  local message="${3:-}"
  python3 - "$output_path/status.json" "$status" "$exit_code" "$message" "$agent" "$runtime" "$current_phase" "$current_node_id" "$current_attempt" "$current_defect_ids" "$current_failure_category" "$model" "$review_model" "${failed_phase:-}" "$current_failure_code" <<'PY'
import json
import sys
from datetime import datetime, timezone
import hashlib
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": sys.argv[2],
    "success": sys.argv[2] == "success",
    "exit_code": int(sys.argv[3]),
    "message": sys.argv[4],
    "agent": sys.argv[5],
    "runtime": sys.argv[6],
    "phase": sys.argv[7],
    "node_id": sys.argv[8] or None,
    "attempt": int(sys.argv[9]),
    "defect_ids": [item for item in sys.argv[10].split(",") if item],
    "failure_category": sys.argv[11] or None,
    "model": sys.argv[12] or None,
    "review_model": sys.argv[13] or None,
    "failed_phase": sys.argv[14] or None,
    "failure_code": sys.argv[15] or None,
    "finished_at": datetime.now(timezone.utc).isoformat(),
}

for filename in ("task.json", "BUILD_CONTRACT.json", "tools.json", "runtime_trace.jsonl", "review_report.json", "docker_image_metadata.json"):
    artifact = path.parent / filename
    if artifact.is_file():
        payload.setdefault("artifact_hashes", {})[filename] = hashlib.sha256(artifact.read_bytes()).hexdigest()

tmp = path.with_name(path.name + ".tmp")
tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
tmp.replace(path)
PY
}

set_build_phase() {
  current_phase="$1"
  current_node_id="${2:-}"
  current_attempt="${3:-0}"
  current_defect_ids="${4:-}"
  write_status "pending" 0 "${5:-沙箱环境正在构建}"
}

validate_contract_and_tools() {
  python3 - "$output_path/task.json" "$output_path/BUILD_CONTRACT.json" "$output_path/tools.json" <<'PY'
import json
import sys
from pathlib import Path

task_path, contract_path, tools_path = map(Path, sys.argv[1:])
task = json.loads(task_path.read_text(encoding="utf-8"))
contract = json.loads(contract_path.read_text(encoding="utf-8"))
tools = json.loads(tools_path.read_text(encoding="utf-8"))

expected_contract = {key: value for key, value in task.items() if key != "actions"}
if contract != expected_contract:
    raise SystemExit("BUILD_CONTRACT.json 与去除 actions 后的 task.json 深度不相等")
if not isinstance(tools, list) or not tools:
    raise SystemExit("tools.json 必须是非空的顶层 Function Tool 数组")

declared_tools = task.get("tools")
if isinstance(declared_tools, list) and tools != declared_tools:
    raise SystemExit("tools.json 与 task.json.tools 深度不相等")

names = set()
def check_schema(schema, location):
    if not isinstance(schema, dict):
        raise SystemExit(f"工具参数 {location} schema 必须是 object")
    if not isinstance(schema.get("description"), str) or not schema["description"].strip():
        raise SystemExit(f"工具参数 {location} 缺少 description")
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise SystemExit(f"工具参数 {location}.properties 必须是 object")
        required = schema.get("required", [])
        if not isinstance(required, list) or any(key not in properties for key in required):
            raise SystemExit(f"工具参数 {location}.required 无效")
        for key, value in properties.items():
            check_schema(value, f"{location}.{key}")
    elif schema.get("type") == "array" and "items" in schema:
        check_schema(schema["items"], f"{location}[]")

for index, tool in enumerate(tools, 1):
    if not isinstance(tool, dict) or tool.get("type") != "function":
        raise SystemExit(f"tools.json 第 {index} 项不是标准 function tool")
    function = tool.get("function")
    if not isinstance(function, dict):
        raise SystemExit(f"tools.json 第 {index} 项缺少 function")
    name = function.get("name")
    if not isinstance(name, str) or not name or name in names:
        raise SystemExit(f"tools.json 工具名无效或重复：{name!r}")
    if not isinstance(function.get("description"), str) or not function["description"].strip():
        raise SystemExit(f"工具 {name} 缺少 function.description")
    parameters = function.get("parameters")
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise SystemExit(f"工具 {name} 的 parameters 必须是 object schema")
    if not isinstance(parameters.get("properties"), dict):
        raise SystemExit(f"工具 {name} 缺少 parameters.properties")
    for key, value in parameters["properties"].items():
        check_schema(value, f"{name}.{key}")
    names.add(name)

noise_tools = contract.get("noise_tools", [])
if not isinstance(noise_tools, list):
    raise SystemExit("BUILD_CONTRACT.noise_tools 必须是数组")
noise_by_name = {}
for index, item in enumerate(noise_tools):
    if not isinstance(item, dict) or not isinstance(item.get("name"), str) or not item["name"]:
        raise SystemExit(f"noise_tools[{index}] 缺少 name")
    category = item.get("category")
    if category not in {"unrelated", "related_irrelevant"}:
        raise SystemExit(f"noise_tools[{index}] category 无效")
    if item["name"] in noise_by_name:
        raise SystemExit(f"noise_tools 工具名重复：{item['name']}")
    noise_by_name[item["name"]] = category
if not set(noise_by_name) <= names:
    raise SystemExit("BUILD_CONTRACT.noise_tools 中存在未暴露的工具")

interface = contract.get("requirements", {}).get("runtime_interface")
if not isinstance(interface, dict) or interface.get("protocol") != "http":
    raise SystemExit("BUILD_CONTRACT.requirements.runtime_interface 缺少 http 约定")
endpoints = interface.get("endpoints")
if not isinstance(endpoints, list):
    raise SystemExit("runtime_interface.endpoints 必须是数组")
from env_factory.contracts.runtime_contract import missing_system_endpoints
if missing_system_endpoints(interface):
    raise SystemExit("runtime_interface 缺少系统或 reward endpoint")
declared_tool_endpoints = [
    item for item in endpoints
    if isinstance(item, dict) and item.get("kind") == "llm_tool"
]
if {item.get("name") for item in declared_tool_endpoints} != names:
    raise SystemExit("runtime_interface 未逐一声明所有 LLM tool")
if "ask_user" in names:
    raise SystemExit("ask_user 不再是 LLM Tool，必须使用 user_simulator endpoint")

metrics = task.get("metrics")
if not isinstance(metrics, list) or not metrics:
    raise SystemExit("task.json.metrics 必须是非空数组")
key_steps = task.get("reward_key_steps")
if not isinstance(key_steps, list):
    raise SystemExit("task.json.reward_key_steps 必须是数组")
action_names = {
    str(item.get("name") or item.get("action"))
    for item in task.get("actions", [])
    if isinstance(item, dict)
}
key_step_ids = set()
key_action_names = set()
for index, step in enumerate(key_steps):
    if not isinstance(step, dict) or not isinstance(step.get("step_id"), str) or not step["step_id"]:
        raise SystemExit(f"reward_key_steps[{index}] 缺少有效 step_id")
    if step["step_id"] in key_step_ids:
        raise SystemExit(f"reward_key_steps[{index}] step_id 重复")
    if step.get("action_name") not in action_names:
        raise SystemExit(f"reward_key_steps[{index}] 引用了未知 action")
    if not isinstance(step.get("rationale"), str) or not step["rationale"].strip():
        raise SystemExit(f"reward_key_steps[{index}] 缺少 rationale")
    if not isinstance(step.get("required_for_goal"), bool):
        raise SystemExit(f"reward_key_steps[{index}] required_for_goal 必须是 boolean")
    key_step_ids.add(step["step_id"])
    key_action_names.add(step["action_name"])
for index, metric in enumerate(metrics):
    if not isinstance(metric, dict):
        raise SystemExit(f"metrics[{index}] 必须是 object")
    evaluator = metric.get("evaluator")
    if not isinstance(evaluator, dict):
        raise SystemExit(f"metrics[{index}] 缺少 executable evaluator")
    if not isinstance(evaluator.get("kind"), str) or not evaluator["kind"]:
        raise SystemExit(f"metrics[{index}].evaluator.kind 无效")
    if evaluator.get("source") not in {"runtime_rule", "external_llm"}:
        raise SystemExit(f"metrics[{index}].evaluator.source 无效")
    if not isinstance(evaluator.get("score_mapping"), dict) or not evaluator["score_mapping"]:
        raise SystemExit(f"metrics[{index}].evaluator.score_mapping 缺失")
    score_range = [-1, 0] if metric.get("category") == "penalty" else [0, 1]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not score_range[0] <= value <= score_range[1]
           for value in evaluator["score_mapping"].values()):
        raise SystemExit(f"metrics[{index}].evaluator.score_mapping 超出 {score_range}")
    category = metric.get("category")
    metric_type = metric.get("type")
    if category == "process":
        compiled_rule = (
            metric_type == "rule-based"
            and evaluator.get("kind") == "trajectory_rule"
            and evaluator.get("source") == "runtime_rule"
            and set(evaluator["score_mapping"]) == {"pass", "fail"}
        )
        model_rule = (
            metric_type == "hybrid"
            and evaluator.get("kind") == "hybrid_tool_call"
            and evaluator.get("source") == "external_llm"
            and evaluator.get("comparison") == "exact_tool_name_and_canonical_arguments"
            and set(evaluator["score_mapping"]) == {"match", "mismatch"}
        )
        if not (compiled_rule or model_rule):
            raise SystemExit(f"metrics[{index}] process evaluator 与声明式或外部模型契约不匹配")
        if metric.get("target_action") not in key_action_names:
            raise SystemExit(f"metrics[{index}] process target_action 必须属于 reward_key_steps")
    elif metric_type == "rule-based":
        if evaluator.get("kind") not in {"business_state_rule", "document_rule", "trajectory_rule"}:
            raise SystemExit(f"metrics[{index}] rule-based evaluator.kind 无效")
        if evaluator.get("source") != "runtime_rule" or not isinstance(evaluator.get("assertion"), str) or not evaluator["assertion"].strip():
            raise SystemExit(f"metrics[{index}] rule-based evaluator 缺少可执行 assertion")
    elif metric_type == "model-based":
        if evaluator.get("kind") != "external_llm_judge" or evaluator.get("source") != "external_llm":
            raise SystemExit(f"metrics[{index}] model-based evaluator 必须使用 external_llm_judge")
    elif metric_type == "hybrid":
        if evaluator.get("kind") != "hybrid_outcome" or evaluator.get("source") != "external_llm":
            raise SystemExit(f"metrics[{index}] hybrid evaluator.kind 无效")
        if not isinstance(evaluator.get("rule"), dict) or not isinstance(evaluator.get("external_llm"), dict):
            raise SystemExit(f"metrics[{index}] hybrid evaluator 缺少 rule/external_llm")
if interface.get("llm_tools") != [item.get("name") for item in declared_tool_endpoints]:
    raise SystemExit("runtime_interface.llm_tools 与 endpoint 顺序不一致")
mutation = interface.get("mutation_testing")
if (not isinstance(mutation, dict) or mutation.get("environment_variable") != "SANDBOX_MUTATION_MODE"
        or not isinstance(mutation.get("modes"), list) or not mutation["modes"]
        or mutation.get("production_default") != "disabled"):
    raise SystemExit("runtime_interface 缺少 mutation testing 约定")
if interface.get("launcher", {}).get("command") != ["python", "app.py", "--port", "{port}"]:
    raise SystemExit("runtime_interface launcher 必须使用 python app.py --port {port}")
reward_endpoints = [item for item in endpoints if isinstance(item, dict) and item.get("kind") == "reward_function"]
if len(reward_endpoints) != 1 or reward_endpoints[0].get("name") != "reward":
    raise SystemExit("runtime_interface 必须声明一个 reward function")
if reward_endpoints[0].get("access") != "rl_trainer_only":
    raise SystemExit("reward endpoint 必须仅允许 RL Trainer 调用")
user_simulator_endpoints = [item for item in endpoints if isinstance(item, dict) and item.get("kind") == "user_simulator"]
if len(user_simulator_endpoints) != 1 or user_simulator_endpoints[0].get("name") != "user_simulator":
    raise SystemExit("runtime_interface 必须声明一个 user_simulator endpoint")
if user_simulator_endpoints[0].get("access") != "rl_trainer_only":
    raise SystemExit("user_simulator endpoint 必须仅允许 RL Trainer 调用")
replay_endpoints = [item for item in endpoints if isinstance(item, dict) and item.get("kind") == "replay"]
if len(replay_endpoints) != 1 or replay_endpoints[0].get("name") != "replay" or replay_endpoints[0].get("access") != "rl_trainer_only":
    raise SystemExit("runtime_interface 必须声明 Trainer-only replay endpoint")
security = interface.get("security")
trainer_security = security.get("trainer") if isinstance(security, dict) else None
if not isinstance(trainer_security, dict) or trainer_security.get("scheme") != "bearer" or trainer_security.get("environment_variable") != "SANDBOX_TRAINER_API_KEY":
    raise SystemExit("runtime_interface 缺少 Trainer Bearer 鉴权约定")
episode = interface.get("episode")
if not isinstance(episode, dict) or episode.get("isolation") != "per_episode" or episode.get("reset_accepts_seed") is not True or episode.get("deterministic_replay") is not True:
    raise SystemExit("runtime_interface 缺少 episode 隔离/seed/replay 约定")
llm_runtime = interface.get("llm_runtime")
if not isinstance(llm_runtime, dict) or llm_runtime.get("api_key_environment_variable") != "SANDBOX_LLM_API_KEY":
    raise SystemExit("runtime_interface 缺少外部 LLM 凭据边界")
evaluator_runtime = interface.get("evaluator_runtime")
if not isinstance(evaluator_runtime, dict) or evaluator_runtime.get("mock_mode_environment_variable") != "SANDBOX_EVALUATOR_MOCK":
    raise SystemExit("runtime_interface 缺少 evaluator mock 约定")
errors = interface.get("errors")
if not isinstance(errors, dict) or errors.get("content_type") != "application/json" or not isinstance(errors.get("schema"), dict):
    raise SystemExit("runtime_interface 缺少统一错误协议")
observability = interface.get("observability")
if not isinstance(observability, dict) or observability.get("request_id_header") != "X-Request-ID" or observability.get("credential_redaction") is not True:
    raise SystemExit("runtime_interface 缺少可观测性/凭据脱敏约定")
if reward_endpoints[0].get("formula") != contract.get("reward_formula"):
    raise SystemExit("runtime_interface reward formula 与 task.json.reward_formula 不一致")
acceptance = task.get("acceptance_contract")
if not isinstance(acceptance, dict) or acceptance.get("authority") != "env_factory_outer_workflow":
    raise SystemExit("task.json 缺少 EnvFactory-owned acceptance_contract")
for key in ("fixtures", "scenarios", "tool_cases", "invariants", "mutations", "reward_cases", "mutation_tests"):
    if not isinstance(acceptance.get(key), (list, dict)):
        raise SystemExit(f"acceptance_contract.{key} 无效")
if not acceptance.get("invariants") or not acceptance.get("mutation_tests"):
    raise SystemExit("acceptance_contract 缺少业务不变量或 mutation tests")
print("BUILD_CONTRACT/task.json deep equality: ok")
print("tool schema: ok")
print("runtime HTTP interface: ok")
PY
}

validate_runtime_trace() {
  local contract_path="$output_path/BUILD_CONTRACT.json"
  local trace_path="$output_path/runtime_trace.jsonl"
  if [[ ! -s "$contract_path" ]]; then
    echo "运行时契约 trace 校验失败：缺少 BUILD_CONTRACT.json" >&2
    return 1
  fi
  if [[ ! -s "$trace_path" ]]; then
    echo "运行时契约 trace 校验失败：缺少或为空 runtime_trace.jsonl" >&2
    return 1
  fi
  python3 "$project_dir/scripts/sandbox/validate_runtime_trace.py" \
    --contract "$contract_path" \
    --trace "$trace_path"
}

validate_outer_conformance() {
  local outer_dir
  outer_dir="$(mktemp -d "${TMPDIR:-/tmp}/envfactory-outer-conformance.XXXXXX")"
  if ! python3 "$project_dir/scripts/sandbox/generate_outer_conformance.py" \
      --root "$output_path" --output "$outer_dir" --check; then
    rm -rf "$outer_dir"
    return 1
  fi
  rm -rf "$outer_dir"
}

validate_mutation_tests() {
  echo "执行自动 mutation testing：要求每个受控缺陷都被验收测试杀死"
  local mutation_log mutation_status
  mutation_log="$output_path/mutation_report.log"
  set +e
  python3 "$project_dir/scripts/sandbox/run_mutation_tests.py" --root "$output_path" \
    >"$mutation_log" 2>&1
  mutation_status=$?
  set -e
  cat "$mutation_log"
  return "$mutation_status"
}

validate_training_readiness() {
  echo "执行 RL 训练素材环境准备就绪硬门禁"
  SANDBOX_TRAINER_API_KEY="${SANDBOX_TRAINER_API_KEY:-envfactory-readiness-key}" \
    python3 "$project_dir/scripts/sandbox/validate_training_readiness.py" \
      --root "$output_path" --output "$output_path/training_readiness.json"
}

validate_agentic_training_value() {
  echo "执行 Agentic training value hard gate"
  SANDBOX_TRAINER_API_KEY="${SANDBOX_TRAINER_API_KEY:-envfactory-agentic-value-key}" \
  SANDBOX_EVALUATOR_MOCK="${SANDBOX_EVALUATOR_MOCK:-1}" \
    python3 "$project_dir/scripts/sandbox/validate_agentic_training_value.py" \
      --root "$output_path" --output "$output_path/agentic_training_value.json"
}

validate_runtime_genericity() {
  echo "执行运行时通用性/反硬编码校验"
  python3 "$project_dir/scripts/sandbox/validate_sandbox_runtime.py" --root "$output_path"
}

validate_semantic_review() {
  echo "执行独立 Code Agent 语义验收"
  local review_tmp review_stdout review_stderr review_stdout_tmp review_stderr_tmp review_status review_prompt
  review_tmp="$(mktemp "${TMPDIR:-/tmp}/envfactory-review.XXXXXX")"
  review_stdout="$output_path/review_agent.stdout"
  review_stderr="$output_path/review_agent.stderr"
  # Never stream reviewer logs into the directory being reviewed. Search
  # commands inside the reviewer would read their own growing transcript and
  # create a recursive context explosion. Publish logs only after completion.
  review_stdout_tmp="$(mktemp "${TMPDIR:-/tmp}/envfactory-review-stdout.XXXXXX")"
  review_stderr_tmp="$(mktemp "${TMPDIR:-/tmp}/envfactory-review-stderr.XXXXXX")"
  review_prompt="$(cat <<EOF
You are an independent read-only semantic reviewer for an RL sandbox.

Review the current sandbox directory and produce ONLY one JSON object in the
last message. Do not modify any file, do not weaken BUILD_CONTRACT.json, and
do not treat status.json or acceptance.sh as proof of semantic correctness.
Inspect files selectively; do not dump whole JSON or source files into the
transcript. Prefer targeted queries and executable evidence so the review
stays within the Luna context budget.
Compare task.json, BUILD_CONTRACT.json, the business data manifests, tools.json,
app.py, runtime_llm.py, sandbox_runtime.py, tests, runtime_trace.jsonl, and the
outer-conformance evidence.
Business data and user simulation files may be ignored by Git; verify their
presence with find or direct path checks. An empty rg --files result is not
evidence that an ignored runtime artifact is missing.
Treat defects.json and older review reports as non-authoritative hints only;
use the current acceptance output, current mutation output, and current source
files as evidence. If a finding contradicts a current executable result, do
not report it as a defect.

Mutation orchestration is owned by the outer workflow. acceptance.sh is a
single baseline/probe entry point which inherits SANDBOX_MUTATION_MODE; it is
not required to loop over mutation modes itself. Read mutation_report.log.
When it ends in "mutation testing: ok" and contains no surviving mutant,
accept killed and explicitly non-applicable mutations as authoritative. Do
not require an irrelevant mutation (for example skip_business_write in a
read-only or direct-response sandbox) to become artificially observable.

Review these modules independently:
1. business data and the real behavior of every task tool. Read the
   noise_tools metadata in BUILD_CONTRACT.json before reviewing tools. A
   noise tool is intentionally not part of the task business semantics:
   unrelated and related_irrelevant tools must not be reported as
   placeholder business implementations. For noise tools, check only schema
   validity, safe execution, no task-critical writes, no task-progress reward,
   and no hidden-truth leakage. Do not require a noise tool to have a complete
   data-backed domain implementation;
2. contract-driven ToolRegistry and schema/argument handling;
3. UserSimulator profile/script-tree selection, node transitions, complete
   messages input, should_end, persistence, and external LLM boundary;
4. RewardEvaluator: every metric.evaluator, process/outcome/penalty semantics,
   external evaluator calls, business-data changes, weights and [-1,1] formula;
5. HTTP authorization, episode isolation, replay, idempotency, observations,
   and tool/reward separation;
6. production completeness, failure paths, and whether the implementation is
   merely a fixed happy-path demo.

The Trainer protocol is reset-scoped: POST /v1/reset selects the active
episode for subsequent calls. Per-episode isolation requires independent
persisted state and no leakage when switching/resetting episode IDs; it does
not require simultaneous request routing to older episodes unless the
contract declares an episode-selection header. The runtime must retain
script/profile identity and declared state/variable transitions.

External-LLM failure cannot establish a normal user decision. A conservative
fallback that emits unrecognized, selects no normal transition, preserves
normal state/variables, and terminates only after the recovery bound is the
required safe behavior. Do not demand that fallback simulate acceptance,
rejection, correction, or goal completion.

SANDBOX_EVALUATOR_MOCK is an explicitly non-semantic acceptance fixture and
production defaults to the external evaluator. Its trace records
semantic_verification=false and readiness records live rollout as unverified.
Do not report exact fixture matching as a production reward shortcut unless
the same shortcut is reachable when mock mode is disabled.

Offline construction must not contact a real external evaluator or mark a
sandbox defective merely because training_readiness says offline_mock or
live_rollout_verified=false. The post-build live rollout stage owns real-model
User Simulator and evaluator verification. During this build review, verify
that the production default routes through RuntimeLLMClient, validates its
structured response, records trace evidence, and fails conservatively; do not
require external network evidence before the sandbox can reach that stage.

A task-specific business handler is allowed, but fixed generated metric IDs,
fixed metric weights, fixed expected-call maps, always selecting the first
session, fixed turn cutoffs, placeholder tools, or reward shortcuts are
critical findings only for task tools and declared runtime components. A
placeholder implementation finding must name a non-noise task tool. Check whether tests could pass while the real contract is
violated. Return exactly:
{
  "status": "pass" or "fail",
  "score": number between 0 and 1,
  "checked_modules": [string, ...],
  "findings": [
    {
      "severity": "critical"|"high"|"medium"|"low",
      "category": string,
      "tool_name": string or null,
      "tool_category": "task"|"unrelated"|"related_irrelevant" or null,
      "file": string,
      "line": number or null,
      "evidence": string,
      "contract_reference": string,
      "fix_required": string
    }
  ],
  "required_repairs": [string, ...]
}

Use status=fail for any critical/high finding. Do not mark pass merely because
local tests or mutation tests pass.
EOF
)"
  set +e
  case "$review_agent" in
    codex)
      review_codex_args=(exec --ephemeral --sandbox read-only --output-last-message "$review_tmp")
      if [[ -n "$review_model" ]]; then review_codex_args+=(--model "$review_model"); fi
      review_codex_args+=("$review_prompt")
      (cd "$output_path" && codex "${review_codex_args[@]}") \
          >"$review_stdout_tmp" 2>"$review_stderr_tmp"
      ;;
    claude)
      (cd "$output_path" && claude --print "$review_prompt") \
          >"$review_tmp" 2>"$review_stderr_tmp"
      ;;
    opencode)
      (cd "$output_path" && opencode run "$review_prompt") \
          >"$review_tmp" 2>"$review_stderr_tmp"
      ;;
  esac
  review_status=$?
  set -e
  cp "$review_stdout_tmp" "$review_stdout"
  cp "$review_stderr_tmp" "$review_stderr"
  rm -f "$review_stdout_tmp" "$review_stderr_tmp"
  if [[ "$review_status" -ne 0 || ! -s "$review_tmp" ]]; then
    echo "独立语义审查 Agent 执行失败：退出码=$review_status" >&2
    cat "$review_stderr" >&2 || true
    rm -f "$review_tmp"
    return 1
  fi
  cp "$review_tmp" "$output_path/review_report.json"
  python3 - "$output_path/review_report.json" "$output_path" <<'PY'
import hashlib
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

report_path = Path(sys.argv[1])
root = Path(sys.argv[2])
report = json.loads(report_path.read_text(encoding="utf-8"))
contract = json.loads((root / "BUILD_CONTRACT.json").read_text(encoding="utf-8"))
current_delivery_error = (root / "last_delivery_error.txt").read_text(encoding="utf-8", errors="replace") if (root / "last_delivery_error.txt").is_file() else ""
mutation_report = (root / "mutation_report.log").read_text(encoding="utf-8", errors="replace") if (root / "mutation_report.log").is_file() else ""
noise = {
    item.get("name")
    for item in contract.get("noise_tools", [])
    if isinstance(item, dict) and isinstance(item.get("name"), str)
}
# Defend against a reviewer applying the business-tool rule to a declared
# noise tool. This decision uses structured fields, never text matching.
filtered = []
for finding in report.get("findings", []):
    if isinstance(finding, dict):
        finding.setdefault("tool_name", None)
        finding.setdefault("tool_category", None)
    if (finding.get("category") == "mutation_testing"
            and "mutation testing: ok" in mutation_report
            and "survived" not in mutation_report.lower()):
        # The executable outer mutation runner is authoritative. Do not ask a
        # task agent to reimplement mutation orchestration or make an
        # archetype-irrelevant mutant artificially observable.
        continue
    finding_tool = finding.get("tool_name")
    finding_category = finding.get("tool_category")
    if (finding.get("category") in {"placeholder_tool", "placeholder_tools"}
            and ((finding_tool in noise) or finding_category in {"unrelated", "related_irrelevant"})):
        continue
    filtered.append(finding)
report["findings"] = filtered
report["review_run_id"] = uuid.uuid4().hex
report["reviewed_at"] = datetime.now(timezone.utc).isoformat()
report["noise_tool_names"] = sorted(noise)
report["source_hashes"] = {}
for filename in ("task.json", "BUILD_CONTRACT.json", "tools.json", "runtime_trace.jsonl", "last_delivery_error.txt"):
    path = root / filename
    if path.is_file():
        report["source_hashes"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
  rm -f "$review_tmp"
  python3 "$project_dir/scripts/sandbox/validate_review_report.py" \
    --report "$output_path/review_report.json"
}

validate_development_plan() {
  python3 "$project_dir/scripts/sandbox/validate_development_plan.py" "$output_path/development_plan.json"
}

validate_single_defect() {
  local defect_json_text="$1"
  local defect_id="$2"
  local review_tmp review_status prompt_text validation_status
  # BSD mktemp only replaces a trailing XXXXXX template.  A suffix after the
  # template can yield the same literal filename in concurrent builds.
  review_tmp="$(mktemp "${TMPDIR:-/tmp}/envfactory-defect-review.XXXXXX")"
  prompt_text="$(cat <<'EOF'
You are an independent read-only defect verifier for an RL sandbox.

Verify only defect __DEFECT_ID__ in the current sandbox. Read the current
BUILD_CONTRACT.json, current source files, current tests, and the defect below.
Git-ignored runtime data may be absent from rg --files. For missing-artifact
claims, check the manifest paths directly with find data -type f or ls and
inspect the declared files before deciding they are absent.
Do not trust older review reports or defects.json. Run or inspect the smallest
executable check that proves this exact defect is fixed. Do not modify files.
Your read-only sandbox may prohibit temporary files, TCP binding, or pytest
cache writes. Such an execution restriction is not evidence that the defect
remains. When execution is blocked, inspect the current regression test and
the fresh outer-workflow evidence files (pytest_*.log, acceptance_result.json,
.outer_conformance.json, mutation_report.json, runtime_trace.jsonl) and decide
from that evidence. Fail only for a remaining product defect or missing fresh
evidence, never solely because your own sandbox cannot rerun a check.

Defect:
__DEFECT_JSON__

Return only this JSON object:
{
  "status": "pass" or "fail",
  "defect_id": "__DEFECT_ID__",
  "evidence": "what was checked and why it proves the defect is fixed",
  "checks_run": ["pytest test path::test name", ...],
  "remaining_issue": "empty string when status is pass"
}
A pass requires the exact defect to be fixed, not merely a source-file change.
EOF
 )"
  prompt_text="${prompt_text//__DEFECT_ID__/$defect_id}"
  prompt_text="${prompt_text//__DEFECT_JSON__/$defect_json_text}"
  set +e
  case "$review_agent" in
    codex)
      defect_codex_args=(exec --ephemeral --sandbox read-only --output-last-message "$review_tmp")
      if [[ -n "$review_model" ]]; then defect_codex_args+=(--model "$review_model"); fi
      defect_codex_args+=("$prompt_text")
      (cd "$output_path" && codex "${defect_codex_args[@]}") \
          >"$output_path/defect_${defect_id}_review.stdout" 2>"$output_path/defect_${defect_id}_review.stderr"
      ;;
    claude)
      (cd "$output_path" && claude --print "$prompt_text") \
          >"$review_tmp" 2>"$output_path/defect_${defect_id}_review.stderr"
      ;;
    opencode)
      (cd "$output_path" && opencode run "$prompt_text") \
          >"$review_tmp" 2>"$output_path/defect_${defect_id}_review.stderr"
      ;;
  esac
  review_status=$?
  set -e
  if [[ "$review_status" -ne 0 || ! -s "$review_tmp" ]]; then
    rm -f "$review_tmp"
    echo "缺陷 $defect_id 独立验证 Agent 执行失败：退出码=$review_status" >&2
    return 1
  fi
  python3 - "$review_tmp" "$defect_id" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
defect_id = sys.argv[2]
report = json.loads(path.read_text(encoding="utf-8"))
if not isinstance(report, dict) or report.get("defect_id") != defect_id:
    raise SystemExit(f"缺陷验证报告 defect_id 不匹配：期望 {defect_id}")
if report.get("status") != "pass":
    raise SystemExit(json.dumps(report, ensure_ascii=False))
if not isinstance(report.get("evidence"), str) or not report["evidence"].strip():
    raise SystemExit("缺陷验证报告缺少 evidence")
if not isinstance(report.get("checks_run"), list) or not report["checks_run"] or any(not isinstance(item, str) or not item.strip() for item in report["checks_run"]):
    raise SystemExit("缺陷验证报告缺少 checks_run")
print(f"defect validation: {defect_id} ok")
PY
  validation_status=$?
  cp "$review_tmp" "$output_path/defect_${defect_id}_review.json"
  rm -f "$review_tmp"
  return "$validation_status"
}

extract_structured_defects() {
  python3 - "$output_path/review_report.json" "$output_path/defects.json" "$output_path/last_delivery_error.txt" <<'PY'
import json
import sys
from pathlib import Path

report_path, defects_path, error_path = map(Path, sys.argv[1:])
report = {}
findings = []
if report_path.is_file():
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        findings = report.get("findings", []) if isinstance(report, dict) else []
    except json.JSONDecodeError:
        findings = []
if not findings:
    error = error_path.read_text(encoding="utf-8", errors="replace") if error_path.is_file() else "验收失败"
    findings = [{"severity": "critical", "category": "delivery_failure", "file": None,
                 "line": None, "evidence": error[-12000:],
                 "contract_reference": "outer workflow validation",
                 "fix_required": "修复验收输出中的全部问题并重新运行相关检查"}]
normalized = []
review_run_id = report.get("review_run_id") if isinstance(report, dict) else None
source_hashes = report.get("source_hashes", {}) if isinstance(report, dict) else {}
for index, finding in enumerate(findings, 1):
    if isinstance(finding, dict):
        item = dict(finding)
        item["id"] = item.get("id") or f"DEF-{index:03d}"
        item["review_run_id"] = review_run_id
        item["source_hashes"] = source_hashes
        normalized.append(item)
defects_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(len(normalized))
PY
}

defect_json() {
  python3 - "$output_path/defects.json" "$1" <<'PY'
import json
import sys
items = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(json.dumps(items[int(sys.argv[2])], ensure_ascii=False, indent=2))
PY
}

defect_ids() {
  python3 - "$output_path/defects.json" <<'PY'
import json
import sys
items = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(",".join(item.get("id", "") for item in items if item.get("id")))
PY
}

implementation_hash() {
  python3 - "$output_path" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
included = []
for path in sorted(root.rglob("*")):
    if not path.is_file() or any(part in {".git", "__pycache__"} for part in path.parts):
        continue
    if path.name in {
        "agent.log", "review_agent.stdout", "review_agent.stderr",
        "review_report.json", "defects.json", "last_delivery_error.txt",
        "status.json", "runtime_trace.jsonl", "acceptance_failure.log",
    }:
        continue
    # Only implementation and executable test files count as repair progress.
    # Reports, plans, manifests and documentation cannot satisfy a defect.
    if path.suffix in {".py", ".sh"} or path.name in {"Dockerfile", "docker_build.sh", "docker_run.sh"}:
        included.append((str(path.relative_to(root)), hashlib.sha256(path.read_bytes()).hexdigest()))
payload = "\n".join(f"{name}:{digest}" for name, digest in included).encode()
print(hashlib.sha256(payload).hexdigest())
PY
}

validate_dockerfile_security() {
  local dockerfile_path="$output_path/Dockerfile"
  local user_line
  user_line="$(awk 'toupper($1) == "USER" {print $2; exit}' "$dockerfile_path")"
  if [[ -z "$user_line" || "$user_line" == "root" || "$user_line" == "0" ]]; then
    echo "Dockerfile 必须使用非 root USER" >&2
    return 1
  fi
  if grep -Eiq '^[[:space:]]*ENV[[:space:]].*(API_KEY|SECRET|TOKEN|PASSWORD)' "$dockerfile_path"; then
    echo "Dockerfile 不得写入 API key、secret、token 或 password" >&2
    return 1
  fi
}

validate_acceptance_portability() {
  local acceptance_path="$output_path/acceptance.sh"
  if grep -Eq '(^|[^[:alnum:]_])python([[:space:]]|$)' "$acceptance_path"; then
    echo "acceptance.sh 不得调用裸 python；请使用 python3 或明确的解释器路径"
    return 1
  fi
}

validate_pytest_setup() {
  local requirements_path="$output_path/requirements-dev.txt"
  if ! grep -Eiq '(^|[[:space:]])pytest([<>=!~[:space:]]|$)' "$requirements_path"; then
    echo "requirements-dev.txt 必须声明 pytest" >&2
    return 1
  fi
  if ! grep -Eq 'pytest|requirements-dev\.txt' "$output_path/Dockerfile"; then
    echo "Dockerfile 必须安装 requirements-dev.txt 或 pytest" >&2
    return 1
  fi
}

run_sandbox_pytest() {
  local label="${1:-full}"
  local report_path="$output_path/pytest_${label}.log"
  if ! PYTHONDONTWRITEBYTECODE=1 python3 -c 'import pytest' >/dev/null 2>&1; then
    echo "pytest 未安装，无法完成 ${label} 测试；requirements-dev.txt 已声明，构建不能将其记为通过" >&2
    return 1
  fi
  set +e
  (cd "$output_path" && PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q) >"$report_path" 2>&1
  local status=$?
  set -e
  cat "$report_path"
  return "$status"
}

validate_acceptance_result() {
  local result_path="$output_path/acceptance_result.json"
  if [[ ! -s "$result_path" ]]; then
    echo "缺少 acceptance_result.json，无法区分业务验收和 HTTP 验收状态" >&2
    return 1
  fi
  python3 - "$result_path" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("business_acceptance") != "passed":
    raise SystemExit("acceptance_result.business_acceptance 必须为 passed")
http_status = result.get("http_conformance")
if http_status not in {"passed", "skipped"}:
    raise SystemExit("acceptance_result.http_conformance 必须为 passed 或 skipped")
if http_status == "skipped" and not str(result.get("http_skip_reason", "")).strip():
    raise SystemExit("HTTP 验收跳过时必须提供 http_skip_reason")
print(f"acceptance result: business=passed http={http_status}")
PY
}

normalize_acceptance_result() {
  local result_path="$output_path/acceptance_result.json"
  python3 - "$result_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    result = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as exc:
    raise SystemExit(f"acceptance.sh 成功但 acceptance_result.json 无效: {exc}")
if not isinstance(result, dict):
    raise SystemExit("acceptance_result.json 必须是 object")
# Reaching this function means acceptance.sh returned zero.  Canonicalize the
# evidence envelope here so builders only own scenario execution, not an
# incidental outer-workflow serialization convention.
result.setdefault("business_acceptance", "passed")
result.setdefault("http_conformance", "skipped")
if result["http_conformance"] == "skipped":
    result.setdefault("http_skip_reason", "business acceptance completed through the in-process public application boundary")
path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

export_runtime_trace() {
  python3 - "$output_path/acceptance_result.json" "$output_path/runtime_trace.jsonl" <<'PY'
import json
import sys
import time
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
events = []
for scenario in result.get("scenarios", []):
    if not isinstance(scenario, dict):
        continue
    scenario_id = scenario.get("scenario_id")
    for item in scenario.get("history", []):
        if isinstance(item, dict) and isinstance(item.get("operation"), str):
            events.append({
                "event": item["operation"], "scenario_id": scenario_id,
                "status": item.get("status"), "timestamp": time.time(),
            })
if not events:
    events.append({"event": "business_acceptance", "status": result.get("business_acceptance"), "timestamp": time.time()})
Path(sys.argv[2]).write_text(
    "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in events),
    encoding="utf-8",
)
PY
}

restore_final_acceptance_evidence() {
  local acceptance_log acceptance_status
  acceptance_log="$(mktemp "${TMPDIR:-/tmp}/sandbox-final-acceptance.XXXXXX")"
  set +e
  (cd "$output_path" && SANDBOX_MUTATION_MODE=disabled bash ./acceptance.sh) >"$acceptance_log" 2>&1
  acceptance_status=$?
  set -e
  if [[ "$acceptance_status" -ne 0 ]]; then
    echo "mutation testing 后的最终基线验收失败，退出码：$acceptance_status" >&2
    cp "$acceptance_log" "$output_path/acceptance_failure.log"
    cat "$acceptance_log" >&2
    rm -f "$acceptance_log"
    return "$acceptance_status"
  fi
  rm -f "$acceptance_log"
  normalize_acceptance_result
  export_runtime_trace
  validate_acceptance_result
}

required_sandbox_files=(
  app.py task_impl.py tools.json development_plan.json runtime_llm.py sandbox_runtime.py Dockerfile .dockerignore docker_build.sh docker_run.sh
  requirements-dev.txt acceptance.sh IMPLEMENTATION_REPORT.md
)

# These files are generated or copied by EnvFactory and define the platform
# boundary.  A task implementation agent may only extend task_impl.py and
# focused tests; task-specific fixes must never fork the shared runtime.
platform_owned_files=(
  task.json BUILD_CONTRACT.json development_plan.json tools.json app.py
  runtime_llm.py sandbox_runtime.py acceptance_runner.py acceptance.sh
  Dockerfile .dockerignore docker_build.sh docker_run.sh requirements-dev.txt
)

snapshot_platform_assets() {
  local backup_dir file
  backup_dir="$(mktemp -d "${TMPDIR:-/tmp}/envfactory-platform.XXXXXX")"
  for file in "${platform_owned_files[@]}"; do
    [[ -e "$output_path/$file" ]] && cp -p "$output_path/$file" "$backup_dir/$file"
  done
  printf '%s\n' "$backup_dir"
}

restore_and_reject_platform_changes() {
  local backup_dir="$1" file changed=""
  for file in "${platform_owned_files[@]}"; do
    if [[ -e "$backup_dir/$file" ]] && ! cmp -s "$backup_dir/$file" "$output_path/$file"; then
      changed="${changed}${changed:+, }${file}"
      cp -p "$backup_dir/$file" "$output_path/$file"
    fi
  done
  case "$backup_dir" in
    "${TMPDIR:-/tmp}"/envfactory-platform.*) rm -rf -- "$backup_dir" ;;
    *) echo "拒绝清理非临时平台快照：$backup_dir" >&2; return 1 ;;
  esac
  if [[ -n "$changed" ]]; then
    echo "Code Agent 修改了平台资产，已恢复并拒绝本轮：$changed" >&2
    return 1
  fi
  return 0
}

validate_delivery() {
  local missing=()
  local file
  for file in "${required_sandbox_files[@]}"; do
    [[ -s "$output_path/$file" ]] || missing+=("$file")
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Code Agent 未生成必需文件：${missing[*]}"
    return 1
  fi

  local legacy=()
  for file in spec.md action_plan.json; do
    [[ -e "$output_path/$file" ]] && legacy+=("$file")
  done
  if (( ${#legacy[@]} > 0 )); then
    echo "生成了已废弃的拓扑产物：${legacy[*]}"
    return 1
  fi
  validate_acceptance_portability
  validate_pytest_setup

  local acceptance_log acceptance_status
  acceptance_log="$(mktemp "${TMPDIR:-/tmp}/sandbox-acceptance.XXXXXX")"
  set +e
  (cd "$output_path" && bash ./acceptance.sh) >"$acceptance_log" 2>&1
  acceptance_status=$?
  set -e
  if [[ "$acceptance_status" -ne 0 ]]; then
    echo "沙箱验收失败，退出码：$acceptance_status"
    cp "$acceptance_log" "$output_path/acceptance_failure.log"
    echo "验收失败日志：$output_path/acceptance_failure.log"
    cat "$acceptance_log"
    rm -f "$acceptance_log"
    return "$acceptance_status"
  fi
  rm -f "$acceptance_log"
  normalize_acceptance_result
  export_runtime_trace
  validate_acceptance_result
  if ! run_sandbox_pytest "acceptance"; then
    echo "pytest 业务测试失败" >&2
    return 1
  fi
  validate_contract_and_tools
  validate_runtime_genericity
  validate_outer_conformance
  validate_mutation_tests
  set_build_phase "training_readiness" "" "${current_attempt:-0}" "${current_defect_ids:-}" "正在验证 RL 训练素材环境准备就绪性"
  validate_training_readiness
  set_build_phase "agentic_training_value" "" "${current_attempt:-0}" "${current_defect_ids:-}" "正在验证 Agentic RL 训练素材价值"
  validate_agentic_training_value
  # Mutation runs execute acceptance.sh repeatedly and deliberately leave the
  # last mutant's failed evidence behind.  Re-run a clean baseline before the
  # semantic reviewer; merely normalizing the mutant envelope would preserve
  # false failure evidence and invite an unnecessary model repair.
  restore_final_acceptance_evidence
  validate_dockerfile_security
  set_build_phase "semantic_review" "" "${current_attempt:-0}" "${current_defect_ids:-}" "正在执行独立语义验收"
  validate_semantic_review
}

run_agent_and_finalize_impl() {
  set_build_phase "buildability" "" 0 "" "正在执行任务可构建性预检"
  if ! PYTHONDONTWRITEBYTECODE=1 python3 "$project_dir/scripts/sandbox/assess_task_buildability.py" \
      --root "$output_path" --output "$output_path/buildability.json"; then
    echo "任务契约未通过可构建性预检；不会消耗 Code Agent 修复预算" >&2
    return 4
  fi
  echo "Code Agent 模块化开发阶段开始：agent=$agent output=$output_path"
  implementation_prompt="$(<"$output_path/TASK_PROMPT.md")"

  if [[ "$resume" != "true" ]]; then
    set_build_phase "planning" "" 1 "" "正在生成确定性模块开发拓扑"
    python3 "$project_dir/scripts/sandbox/generate_development_plan.py" \
      --contract "$output_path/BUILD_CONTRACT.json" \
      --output "$output_path/development_plan.json"
    validate_development_plan

    # Bash 3.2 with `set -u` treats expansion of a declared-but-empty array as
    # an unbound variable.  Keep an inert sentinel so fully declarative tasks
    # can legitimately have a zero-node development plan.
    node_ids=("")
    while IFS= read -r node_id; do
      [[ -n "$node_id" ]] && node_ids+=("$node_id")
    done < <(python3 "$project_dir/scripts/sandbox/validate_development_plan.py" "$output_path/development_plan.json" --ids | tail -n +2)
    for node_id in "${node_ids[@]}"; do
    [[ -z "$node_id" ]] && continue
    node_done="false"
    node_error=""
    for (( node_attempt = 1; node_attempt <= max_attempts; node_attempt++ )); do
      node_json=$(python3 -c 'import json,sys; plan=json.load(open(sys.argv[1], encoding="utf-8")); print(json.dumps(next(node for node in plan["nodes"] if node["id"] == sys.argv[2]), ensure_ascii=False, indent=2))' "$output_path/development_plan.json" "$node_id")
      set_build_phase "node_development" "$node_id" "$node_attempt" "" "正在开发模块节点：$node_id"
      node_prompt="Read the relevant fields of BUILD_CONTRACT.json, TASK_PROMPT.md, and referenced artifacts with focused queries; avoid dumping complete runtime and contract files. Current work is limited to one node; do not redesign validated modules or modify task.json, BUILD_CONTRACT.json, or development_plan.json.
节点定义：
$node_json

Implement the task-specific behavior required by this node. Reuse runtime_llm.py and sandbox_runtime.py for platform behavior. Run every validation command declared by the node using the local validation interpreter named in TASK_PROMPT.md if bare python3 lacks pytest. Do not attempt a network package install. If validation fails, fix the first root cause before continuing."
      set +e
      platform_backup="$(snapshot_platform_assets)"
      run_code_agent "$node_prompt"
      node_status=$?
      if ! restore_and_reject_platform_changes "$platform_backup"; then
        node_status=10
      fi
      set -e
      if [[ "$node_status" -eq 0 ]]; then
        node_done="true"
        set_build_phase "node_completed" "$node_id" "$node_attempt" "" "模块节点完成：$node_id"
        break
      fi
      node_error="节点 $node_id 开发退出码：$node_status"
      echo "开发节点 $node_id 第 ${node_attempt}/${max_attempts} 次失败：$node_error" >&2
      if [[ "$current_failure_code" == "INFRA" ]]; then
        return 5
      fi
    done
    if [[ "$node_done" != "true" ]]; then
      echo "$node_error" >&2
      return 5
    fi
    done
  else
    echo "增量续建：保留已有实现并直接执行完整验收/缺陷修复"
    validate_development_plan
  fi

  implementation_succeeded="false"
  implementation_error=""
  repair_attempt=0
  while true; do
    set_build_phase "acceptance" "" "$repair_attempt" "" "正在执行完整业务验收"
    set +e
    delivery_error="$(validate_delivery 2>&1)"
    delivery_status=$?
    set -e
    if [[ "$delivery_status" -eq 0 ]]; then
      implementation_succeeded="true"
      break
    fi
    implementation_error="$delivery_error"
    printf '%s\n' "$implementation_error" > "$output_path/last_delivery_error.txt"
    extract_structured_defects >/dev/null
    current_defect_ids="$(defect_ids)"
    # The development agent repairs one evidenced root cause at a time. Feeding all findings
    # into one turn causes broad rewrites and regressions; remaining defects
    # are rediscovered against the repaired implementation in the next round.
    # The budget is persisted per defect, since extract_structured_defects may
    # reuse DEF-001 for a different root cause on the next validation pass.
    if ! budget_line="$(python3 "$project_dir/scripts/sandbox/defect_attempt_budget.py" \
      --state "$output_path/defect_attempts.json" --defects "$output_path/defects.json" \
      --max-attempts "$max_attempts")"; then
      echo "本次验收中的缺陷均达到各自 ${max_attempts} 次修复上限：$implementation_error" >&2
      return 5
    fi
    read -r defect_index budget_id repair_attempt <<<"$budget_line"
    defect="$(defect_json "$defect_index")"
    echo "针对缺陷 $budget_id 执行第 ${repair_attempt}/${max_attempts} 次修复"
    if [[ -n "$defect" ]]; then
      defect_id="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read()).get("id", "unknown"))' <<<"$defect")"
      set_build_phase "defect_repair" "$defect_id" "$repair_attempt" "$current_defect_ids" "正在修复结构化缺陷：$defect_id"
      before_defect_hash="$(implementation_hash)"
      repair_prompt="$implementation_prompt

这是一次针对性缺陷修复，不要重新生成 demo，也不要修改 BUILD_CONTRACT.json、task.json 或 development_plan.json。
当前缺陷：
$defect

请检查缺陷涉及的代码和契约，完成真实修复，并增加或运行能够证明该缺陷已修复的回归测试。修复后保留其他已完成模块的行为。"
      set +e
      platform_backup="$(snapshot_platform_assets)"
      run_code_agent "$repair_prompt"
      repair_status=$?
      if ! restore_and_reject_platform_changes "$platform_backup"; then
        repair_status=10
      fi
      set -e
      if [[ "$repair_status" -ne 0 ]]; then
        echo "缺陷 $defect_id 修复退出码：$repair_status" >&2
        if [[ "$current_failure_code" == "INFRA" ]]; then
          return 5
        fi
      fi
      after_defect_hash="$(implementation_hash)"
      if [[ "$before_defect_hash" == "$after_defect_hash" ]]; then
        echo "缺陷 $defect_id 修复没有产生生产代码或验收代码变化；拒绝将本轮标记为已修复" >&2
        repair_status=6
      fi
      if [[ "$repair_status" -eq 0 ]]; then
        set_build_phase "defect_validation" "$defect_id" "$repair_attempt" "$current_defect_ids" "正在校验缺陷修复产物：$defect_id"
        if ! PYTHONDONTWRITEBYTECODE=1 python3 - "$output_path" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
for path in root.rglob("*.py"):
    if "__pycache__" in path.parts:
        continue
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
PY
        then
          echo "缺陷 $defect_id 修复后 Python 语法检查失败" >&2
          repair_status=7
        fi
      fi
      if [[ "$repair_status" -eq 0 ]]; then
        if ! run_sandbox_pytest "defect_${defect_id}"; then
          echo "缺陷 $defect_id 的 pytest 回归测试未通过" >&2
          repair_status=9
        fi
      fi
      if [[ "$repair_status" -eq 0 ]]; then
        if ! validate_single_defect "$defect" "$defect_id"; then
          echo "缺陷 $defect_id 专项验证未通过；本缺陷保持未关闭" >&2
          repair_status=8
        fi
      fi
    fi
    echo "缺陷 $budget_id 第 ${repair_attempt}/${max_attempts} 次修复后重新验收" >&2
  done
  echo "Code Agent 模块化开发和缺陷修复完成：$output_path"
  echo "执行 BUILD_CONTRACT/task.json 深度相等和工具合规复核"
  set_build_phase "contract_validation" "" "$repair_attempt" "$current_defect_ids" "正在校验构建契约和工具 schema"
  if ! validate_contract_and_tools; then
    echo "BUILD_CONTRACT 或工具合规复核失败" >&2
    return 5
  fi
  echo "执行运行时契约 trace 校验：$output_path/runtime_trace.jsonl"
  set_build_phase "trace_validation" "" "$repair_attempt" "$current_defect_ids" "正在校验运行时 trace"
  if ! validate_runtime_trace; then
    echo "运行时契约 trace 校验失败" >&2
    return 5
  fi
  echo "执行 EnvFactory 独立外层 conformance 校验"
  echo "执行运行时通用性/反硬编码校验"
  set_build_phase "runtime_validation" "" "$repair_attempt" "$current_defect_ids" "正在执行运行时通用性校验"
  if ! validate_runtime_genericity; then
    echo "运行时通用性校验失败" >&2
    return 5
  fi
  if ! validate_outer_conformance; then
    echo "外层 conformance 校验失败" >&2
    return 5
  fi
  echo "执行自动 mutation testing"
  set_build_phase "mutation_testing" "" "$repair_attempt" "$current_defect_ids" "正在执行 mutation testing"
  if ! validate_mutation_tests; then
    echo "mutation testing 校验失败" >&2
    return 5
  fi
  # Mutation probes intentionally rerun acceptance.sh under broken modes and
  # therefore overwrite its evidence file. Always finish with a clean baseline
  # run so downstream scoring observes the real sandbox state.
  set_build_phase "acceptance" "" "$repair_attempt" "$current_defect_ids" "正在恢复最终基线验收证据"
  if ! restore_final_acceptance_evidence; then
    return 5
  fi
  # 留一份可读验收证据；最终判定仍来自上面的临时目录重生成结果。
  python3 "$project_dir/scripts/sandbox/generate_outer_conformance.py" \
    --root "$output_path" --output "$output_path/.outer_conformance" --check

  set_build_phase "docker_build" "" "$repair_attempt" "$current_defect_ids" "正在处理 Docker 构建"
  case "$runtime" in
    none)
      ;;
    docker)
      build_args=(--context "$output_path" --tag "$current_tag" --port "$sandbox_port")
      if [[ -n "$env_file" ]]; then build_args+=(--env-file "$env_file"); fi
      if [[ "$start" == "true" ]]; then build_args+=(--start); fi
      if "$project_dir/scripts/build_docker_sandbox_image.sh" "${build_args[@]}"; then
        :
      else
        return $?
      fi
      ;;
    *)
      echo "不支持的 runtime：${runtime}；可选值为 none、docker"
      return 2
      ;;
  esac
  echo "沙箱构建状态：$output_path/status.json"
  echo "Code Agent 已完成沙箱开发：$output_path"
}

run_agent_and_finalize() {
  local exit_code
  # A callee may legitimately toggle errexit while running a validation
  # command. Execute it in an `if` condition so Bash always returns control
  # here and we can persist a terminal status instead of leaving `pending`.
  if run_agent_and_finalize_impl; then
    exit_code=0
  else
    exit_code=$?
  fi
  if [[ "$exit_code" -eq 0 ]]; then
    failed_phase=""
    current_failure_category=""
    current_failure_code=""
    current_phase="completed"
    current_node_id=""
    current_defect_ids=""
    write_status "success" "$exit_code" "沙箱环境构建成功"
    if [[ "$auto_score" == "true" ]]; then
      echo "执行构建后离线评分：$output_path/offline_sandbox_score.json"
      set +e
      python3 "$project_dir/scripts/sandbox/score_sandbox_offline.py" \
        "$output_path" --project "$project_dir" \
        >"$output_path/offline_score.log" 2>&1
      score_status=$?
      set -e
      if [[ "$score_status" -eq 0 ]]; then
        echo "构建后离线评分通过：$output_path/offline_sandbox_score.json"
      else
        echo "构建成功，但离线评分未达到阈值；详情：$output_path/offline_score.log" >&2
      fi
    fi
  else
    failed_phase="$current_phase"
    if [[ "$current_failure_code" != "INFRA" ]]; then
      case "$current_phase" in
      buildability)
        current_failure_category="task_contract_failure"
        ;;
      node_development|acceptance|semantic_review|defect_repair|defect_validation|contract_validation|trace_validation|runtime_validation|mutation_testing|training_readiness|agentic_training_value)
        current_failure_category="sandbox_failure"
        ;;
      *)
        current_failure_category="workflow_error"
        ;;
      esac
    fi
    current_phase="failed"
    write_status "failed" "$exit_code" "沙箱环境构建失败，请查看 agent.log"
  fi
  return "$exit_code"
}

start_task() {
  local task_index="$1"
  local task_output="$2"
  output_path="$task_output"
  failed_phase=""
  current_failure_code=""
  current_tag="$tag"
  if (( task_count > 1 )); then
    current_tag="${tag}-task-$(printf '%03d' "$((task_index + 1))")"
  fi
  mkdir -p "$output_path"
  set_build_phase "preparing" "" 0 "" "正在准备沙箱环境"
  prepare_task "$task_index" "$output_path"
  prompt="$(<"$output_path/TASK_PROMPT.md")"
  export project_dir agent review_agent model review_model runtime start background input_path output_path prompt current_tag sandbox_port env_file max_attempts resume auto_score
  log_file="$output_path/agent.log"
  if [[ "$background" == "true" ]]; then
    (run_agent_and_finalize) >"$log_file" 2>&1 </dev/null &
    pid=$!
    task_pids+=("$pid")
    echo "Code Agent 已后台启动：task=$((task_index + 1)) PID=$pid"
    echo "日志文件：$log_file"
  else
    run_agent_and_finalize
  fi
}

task_pids=()
background_failures=0
wait_for_task_slot() {
  while (( ${#task_pids[@]} >= max_concurrency )); do
    pid="${task_pids[0]}"
    set +e
    wait "$pid"
    task_status=$?
    set -e
    task_pids=("${task_pids[@]:1}")
    if [[ "$task_status" -ne 0 ]]; then
      echo "后台任务结束但未成功：PID=${pid}，退出码=${task_status}" >&2
      background_failures=1
    fi
  done
}

wait_for_all_tasks() {
  local pid task_status
  for pid in "${task_pids[@]}"; do
    set +e
    wait "$pid"
    task_status=$?
    set -e
    if [[ "$task_status" -ne 0 ]]; then
      echo "后台任务结束但未成功：PID=${pid}，退出码=${task_status}" >&2
      background_failures=1
    fi
  done
  return "$background_failures"
}

if (( task_count == 1 )); then
  start_task 0 "$output_path"
else
  mkdir -p "$root_output_path"
  for (( index = 0; index < task_count; index++ )); do
    if [[ "$background" == "true" ]]; then
      wait_for_task_slot
    fi
    start_task "$index" "$root_output_path/task_$(printf '%03d' "$((index + 1))")"
  done
  if [[ "$background" == "true" ]]; then
    wait_for_all_tasks
  fi
fi
