#!/usr/bin/env bash
# I2: SFT with the anti-forgetting defaults, then a weight-interpolation sweep, each quick-evaluated.
#   experiments/v3/bench/stage_I2.sh SELECTED.jsonl REPLAY.jsonl OUT_DIR
# Defaults (plan §2, I2): LoRA rank 64, lr 1e-4, 1 epoch; 70% of episodes with renamed business tools; 60% with
# tool results as raw JSON / "Error: ..." text (τ²-bench's format);
# the base model's own successful episodes mixed in (REPLAY, sized to 10-20% of the set by the caller);
# failed and repeated-lookup turns kept as context but not trained; WiSE-FT alpha in {0.3, 0.5, 0.7, 1.0}.
set -euo pipefail
cd "$(dirname "$0")/../../.."
selected=$1; replay=$2; out=$3
mkdir -p "$out"
PYTHONPATH=src .venv/bin/python -u experiments/v3/sft_train.py --teacher "$selected" --replay "$replay" \
  --model models/Qwen3-0.6B --output "$out/sft" --tuning lora --lora-rank 64 --learning-rate 1e-4 --epochs 1 \
  --rename-tools 0.7 --vary-results 0.6 --greet 0.5 --relayout 0.5 --no-error-turns --max-tokens 16384 > "$out/sft.log" 2>&1
for alpha in 0.3 0.5 0.7; do
  PYTHONPATH=src .venv/bin/python experiments/v3/interpolate.py --base models/Qwen3-0.6B --tuned "$out/sft/model" \
    --alpha $alpha --output "$out/wise-$alpha" >> "$out/sft.log" 2>&1
done
name=$(basename "$out")
experiments/v3/bench/quick_eval.sh "${name}-full" "$out/sft/model" 1
for alpha in 0.3 0.5 0.7; do
  experiments/v3/bench/quick_eval.sh "${name}-w$alpha" "$out/wise-$alpha" 1
done
echo done > "$out/stage.done"
