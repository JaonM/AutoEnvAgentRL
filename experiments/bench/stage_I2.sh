#!/usr/bin/env bash
# I2: full-parameter SFT with the anti-forgetting defaults, two seeds, each quick-evaluated (no WiSE: see below).
#   [TUNING=lora] [EPOCHS=2] [SFT_EXTRA="--example-values 0.5 --inject-errors 0.5"] experiments/bench/stage_I2.sh SELECTED.jsonl REPLAY.jsonl OUT_DIR [SEEDS]
# Defaults (plan, I2 and I2p-T): full fine-tuning; effective batch 16 episodes (gradient accumulation) at lr 2e-5
# (square-root scaling from 4 episodes at 1e-5); EPOCHS (default 1; use 2 below ~1000 episodes so there are enough
# updates); 8 validation passes per epoch; gradient norm clipped at 1.0; MLX seeded;
# 70% of episodes with renamed business tools; 60% with tool results as raw JSON / "Error: ..." text (τ²-bench's
# format); the base model's own successful episodes mixed in (REPLAY, sized to 10-20% of the set by the caller);
# failed and repeated-lookup turns kept as context but not trained; WiSE-FT alpha 0.5 (best in the probe).
# Two seeds per arm: the same data trained three times gave dev mean reward 0.274 / 0.345 / 0.292.
# Parallel: trainings run one after another (they share the GPU); each model's two quick evals start in the background
# as soon as it is trained, on their own ports (they mostly wait on the DeepSeek user and judge, and wait for
# off-peak hours themselves), so the next seed trains while the previous one is evaluated. A seed whose model
# exists is not trained again, and an eval whose log exists (running or finished) is not started again.
set -uo pipefail
cd "$(dirname "$0")/../.."
selected=$1; replay=$2; out=$3; seeds=${4:-"0 1"}; epochs=${EPOCHS:-1}
mkdir -p "$out"
name=$(basename "$out")
port=${BENCH_PORT_BASE:-8766}
# TUNING=lora: LoRA rank 64 at lr 2e-4 (the probe's 1e-4 at 4 episodes per update, square-root scaled to 16 like
# full fine-tuning's 1e-5 -> 2e-5), everything else unchanged, for the LoRA vs full comparison (I2c).
if [ "${TUNING:-full}" = lora ]; then tuning="--tuning lora --lora-rank 64 --learning-rate 2e-4"
else tuning="--tuning full --learning-rate 2e-5"; fi
evals=()
for seed in $seeds; do
  run="$out/seed$seed"; mkdir -p "$run"
  if [ ! -f "$run/sft/model/config.json" ]; then
    PYTHONPATH=src .venv/bin/python -u experiments/sft_train.py --teacher "$selected" --replay "$replay" \
      --model models/Qwen3-0.6B --output "$run/sft" $tuning --grad-accumulation 16 \
      --epochs $epochs --evals-per-epoch 8 --seed $seed \
      --rename-tools 0.7 --vary-results 0.6 --greet 0.5 --relayout 0.5 --no-error-turns --max-tokens 32768 \
      --render episode ${SFT_EXTRA:-} > "$run/sft.log" 2>&1 || { echo "seed $seed training failed"; continue; }
  fi
  # Only the trained model is evaluated: WiSE interpolation (alpha 0.5) was dropped on 2026-10-16 — its scores came
  # from the no-write tasks while tool calling collapsed (48-100% of simulations without a single call).
  if [ ! -f "output/tau2/q${name}-s$seed-full-airline.log" ]; then  # not started (or finished) already
    BENCH_PORT=$port experiments/v3/bench/quick_eval.sh "${name}-s$seed-full" "$run/sft/model" 1 &
    evals+=($!)
  fi
  port=$((port + 34))
done
for job in "${evals[@]}"; do wait $job; done
echo done > "$out/stage.done"
