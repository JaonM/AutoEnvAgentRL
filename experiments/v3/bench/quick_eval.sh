#!/usr/bin/env bash
# Quick τ²-bench evaluation for small-step iteration (target < 30 min): airline and retail run at the
# same time against one model server (serve_openai.py with its per-conversation prefix cache),
# TRIALS trials each (default 1), seed 300. Compare arms only with each other (same server, same
# settings); the full pre-registered evaluation stays eval_arm.sh.
#   experiments/v3/bench/quick_eval.sh ARM MODEL_PATH [TRIALS]
# THINK=1 serves the thinking template (reasoning before every reply, 4096-token budget per call).
set -uo pipefail
cd "$(dirname "$0")/../../.."
arm=$1; model=$2; trials=${3:-1}; port=${BENCH_PORT:-8766}
think=(); [ "${THINK:-0}" = 1 ] && think=(--thinking --max-tokens 4096)
mkdir -p output/tau2
started=$(date +%s)
PYTHONPATH=src .venv/bin/python experiments/v3/bench/serve_openai.py --model "$model" --port "$port" --cache-size 40 --cache-gb 22 ${think[@]+"${think[@]}"} \
  --log "output/tau2/server-q$arm.jsonl" > "output/tau2/server-q$arm.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null' EXIT
until curl -s "localhost:$port/health" >/dev/null 2>&1; do sleep 2; done
pids=()
for domain in airline retail; do
  rm -rf ".bench/tau2-repo/data/simulations/v3bench/q$arm-$domain"  # tau2 saves under its data directory
  BENCH_PORT=$port experiments/v3/bench/run_tau2.sh "q$arm" "$domain" --num-trials "$trials" --max-concurrency 14 \
    --seed 300 > "output/tau2/q$arm-$domain.log" 2>&1 < /dev/null &
  pids+=($!)
done
wait "${pids[@]}"
echo "{\"arm\": \"$arm\", \"model\": \"$model\", \"trials\": $trials, \"think\": ${THINK:-0}, \"seconds\": $(( $(date +%s) - started ))}" \
  > "output/tau2/q$arm.done"
cat "output/tau2/q$arm.done"
