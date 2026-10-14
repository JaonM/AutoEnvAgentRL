#!/usr/bin/env bash
# τ²-bench run against the local OpenAI-compatible server (serve_openai.py); the user simulator is
# LLM_* from .env (deepseek-flash), identical for every arm.
#   experiments/bench/run_tau2.sh ARM DOMAIN [extra tau2 args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
arm=$1; domain=$2; shift 2
set -a; . ./.env; set +a
export TAU2_DATA_DIR="$PWD/.bench/tau2-repo/data"
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
agent_args=$(printf '{"api_base": "http://127.0.0.1:%s/v1", "api_key": "local", "temperature": 0.0}' "${BENCH_PORT:-8765}")
user_args=$(printf '{"api_base": "%s", "api_key": "%s", "temperature": 0.0}' "$LLM_BASE_URL" "$LLM_API_KEY")
exec .bench/tau2/bin/python experiments/bench/tau2_run.py run --domain "$domain" --agent-llm "openai/qwen3-0.6b-$arm" --agent-llm-args "$agent_args" \
  --user-llm "openai/$LLM_MODEL" --user-llm-args "$user_args" \
  --save-to "v3bench/$arm-$domain" "$@"   # results: $TAU2_DATA_DIR/simulations/v3bench/ARM-DOMAIN
