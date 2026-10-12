#!/usr/bin/env bash
# One arm on τ²-bench (airline then retail, 4 trials each, seed 300): starts the model server, runs both
# domains, stops the server.   experiments/v3/bench/eval_arm.sh ARM MODEL_PATH
set -uo pipefail
cd "$(dirname "$0")/../../.."
arm=$1; model=$2; port=${BENCH_PORT:-8765}
PYTHONPATH=src .venv/bin/python experiments/v3/bench/serve_openai.py --model "$model" --port "$port" \
  --log "output/tau2/server-$arm.jsonl" > "output/tau2/server-$arm.log" 2>&1 &
server=$!
trap 'kill $server 2>/dev/null' EXIT
until curl -s "localhost:$port/health" >/dev/null 2>&1; do sleep 2; done
for domain in airline retail; do
  rm -rf ".bench/tau2-repo/data/simulations/v3bench/$arm-$domain"  # tau2 saves under its data directory
  BENCH_PORT=$port experiments/v3/bench/run_tau2.sh "$arm" "$domain" --num-trials 4 --max-concurrency 16 \
    > "output/tau2/$arm-$domain.log" 2>&1 < /dev/null
done
echo done > "output/tau2/$arm.done"
