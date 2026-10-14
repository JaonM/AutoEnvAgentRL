#!/usr/bin/env bash
# I2: full-parameter SFT with the anti-forgetting defaults, two seeds, each with WiSE-FT alpha 0.5, all quick-evaluated.
#   [EPOCHS=2] [SFT_EXTRA="--example-values 0.5 --inject-errors 0.5"] experiments/v3/bench/stage_I2.sh SELECTED.jsonl REPLAY.jsonl OUT_DIR [SEEDS]
# Defaults (plan, I2 and I2p-T): full fine-tuning; effective batch 16 episodes (gradient accumulation) at lr 2e-5
# (square-root scaling from 4 episodes at 1e-5); EPOCHS (default 1; use 2 below ~1000 episodes so there are enough
# updates); 8 validation passes per epoch; gradient norm clipped at 1.0; MLX seeded;
# 70% of episodes with renamed business tools; 60% with tool results as raw JSON / "Error: ..." text (τ²-bench's
# format); the base model's own successful episodes mixed in (REPLAY, sized to 10-20% of the set by the caller);
# failed and repeated-lookup turns kept as context but not trained; WiSE-FT alpha 0.5 (best in the probe).
# Two seeds per arm: the same data trained three times gave dev mean reward 0.274 / 0.345 / 0.292.
set -euo pipefail
cd "$(dirname "$0")/../../.."
selected=$1; replay=$2; out=$3; seeds=${4:-"0 1"}; epochs=${EPOCHS:-1}
mkdir -p "$out"
name=$(basename "$out")
for seed in $seeds; do
  run="$out/seed$seed"; mkdir -p "$run"
  PYTHONPATH=src .venv/bin/python -u experiments/v3/sft_train.py --teacher "$selected" --replay "$replay" \
    --model models/Qwen3-0.6B --output "$run/sft" --tuning full --learning-rate 2e-5 --grad-accumulation 16 \
    --epochs $epochs --evals-per-epoch 8 --seed $seed \
    --rename-tools 0.7 --vary-results 0.6 --greet 0.5 --relayout 0.5 --no-error-turns --max-tokens 32768 \
    --render episode ${SFT_EXTRA:-} > "$run/sft.log" 2>&1
  PYTHONPATH=src .venv/bin/python experiments/v3/interpolate.py --base models/Qwen3-0.6B --tuned "$run/sft/model" \
    --alpha 0.5 --output "$run/wise-0.5" >> "$run/sft.log" 2>&1
  experiments/v3/bench/quick_eval.sh "${name}-s$seed-full" "$run/sft/model" 1
  experiments/v3/bench/quick_eval.sh "${name}-s$seed-w0.5" "$run/wise-0.5" 1
done
echo done > "$out/stage.done"
