#!/usr/bin/env bash
# Quick τ²-bench evaluation for small-step iteration: airline and retail run at the same time against
# SERVERS model processes (default 3; serve_openai.py with its per-conversation prefix cache, a sticky
# router in front), TRIALS trials each (default 1), seed 300, at most MAX_STEPS steps per simulation
# (default 100; τ²'s 200 only lets failing loops run on: the longest passing baseline dialogue had 36
# messages). Compare arms only with each other (same settings); the full evaluation stays eval_arm.sh.
#   experiments/bench/quick_eval.sh ARM MODEL_PATH [TRIALS]
# THINK=1 serves the thinking template (reasoning before every reply, THINK_BUDGET tokens, default 1024,
# then </think> is forced; 4096-token budget per call). BATCH decodes that many requests together
# (continuous batching with the prefix cache): default 4 with THINK=1, where decoding dominates, and 0
# (one at a time) otherwise, where batching was slower and ran out of memory on 30k-token prompts.
set -uo pipefail
cd "$(dirname "$0")/../.."
arm=$1; model=$2; trials=${3:-1}; port=${BENCH_PORT:-8766}
think=(); servers=${SERVERS:-3}; batch=${BATCH:-0}
if [ "${THINK:-0}" = 1 ]; then
  think=(--thinking --max-tokens 4096 --think-budget "${THINK_BUDGET:-1024}"); batch=${BATCH:-4}
fi
mkdir -p output/tau2
# The τ² user simulator and NL judge are DeepSeek: start in off-peak hours (ENVFACTORY_OFFPEAK=0 skips the wait).
PYTHONPATH=src .venv/bin/python -m env_factory.production.offpeak --wait
# At most BENCH_MAX_EVALS evaluations at once on this machine (default 2): each serves SERVERS model processes
# with up to 24 GB of prefix cache, and ten evaluations queued for the same off-peak start next to a training
# run would exhaust memory (the machine hung on 2026-10-15). A slot is a directory holding its owner's pid;
# slots of dead owners are reclaimed.
slots=output/tau2/.slots; mkdir -p "$slots"; slot=""
while [ -z "$slot" ]; do
  for i in $(seq 1 "${BENCH_MAX_EVALS:-2}"); do
    if [ -d "$slots/$i" ] && ! kill -0 "$(cat "$slots/$i/pid" 2>/dev/null)" 2>/dev/null; then rm -rf "$slots/$i"; fi
    if mkdir "$slots/$i" 2>/dev/null; then echo $$ > "$slots/$i/pid"; slot="$slots/$i"; break; fi
  done
  [ -z "$slot" ] && sleep 60
done
PYTHONPATH=src .venv/bin/python -m env_factory.production.offpeak --wait  # off-peak again after queueing
started=$(date +%s)
pids=(); backends=()
cache_gb=$(( 24 / servers ))
for i in $(seq 1 "$servers"); do
  p=$port; [ "$servers" -gt 1 ] && p=$(( port + 10 + i ))
  backends+=("$p")
  PYTHONPATH=src .venv/bin/python experiments/bench/serve_openai.py --model "$model" --port "$p" \
    --cache-size $(( 40 / servers + 4 )) --cache-gb "$cache_gb" --batch "$batch" ${think[@]+"${think[@]}"} \
    --log "output/tau2/server-q$arm-$i.jsonl" > "output/tau2/server-q$arm-$i.log" 2>&1 &
  pids+=($!)
done
if [ "$servers" -gt 1 ]; then
  .venv/bin/python experiments/bench/router.py --port "$port" --backends "${backends[@]}" \
    > "output/tau2/router-q$arm.log" 2>&1 &
  pids+=($!)
fi
trap 'kill ${pids[*]} 2>/dev/null; rm -rf "$slot"' EXIT
until curl -sf "localhost:$port/health" >/dev/null 2>&1; do sleep 2; done
runs=()
for domain in airline retail; do
  rm -rf ".bench/tau2-repo/data/simulations/v3bench/q$arm-$domain"  # tau2 saves under its data directory
  BENCH_PORT=$port experiments/bench/run_tau2.sh "q$arm" "$domain" --num-trials "$trials" --max-concurrency 14 \
    --seed 300 --max-steps "${MAX_STEPS:-100}" > "output/tau2/q$arm-$domain.log" 2>&1 < /dev/null &
  runs+=($!)
done
wait "${runs[@]}"
echo "{\"arm\": \"$arm\", \"model\": \"$model\", \"trials\": $trials, \"think\": ${THINK:-0}, \"servers\": $servers, \"batch\": $batch, \"max_steps\": ${MAX_STEPS:-100}, \"seconds\": $(( $(date +%s) - started ))}" \
  > "output/tau2/q$arm.done"
cat "output/tau2/q$arm.done"
