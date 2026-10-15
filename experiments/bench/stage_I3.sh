#!/usr/bin/env bash
# I3: GRPO from an SFT model with settings sized for multi-turn customer-service tasks (rl defaults are not).
#   experiments/bench/stage_I3.sh MODEL TASKS.json OUT_DIR [extra rl.train flags]
# rl defaults that do not fit, and the values used here (review 2026-10-15):
#   --max-steps 6 -> 30           episodes need 10-30 actions (τ²'s passing dialogues reach 36 messages)
#   --max-context 4096 -> 32768   system prompt and manual alone take several thousand tokens
#   --max-tokens 256 -> 1024      a reply with a summary or a long tool call was cut off
#   --max-tokens-per-micro-batch 8192 -> 32768   a later turn's segment exceeded it and the step raised
#   --mini-batch-size 1 -> 8      one task per optimizer step was a noisy, task-biased update
#   --rollout-group 4 -> 8        fewer all-zero groups on hard tasks
#   --advantage-std group -> task   group std is unstable at 8 rollouts
#   --tuning qat -> full          full-parameter training (decision 2026-10-15), lr 1e-6
set -euo pipefail
cd "$(dirname "$0")/../.."
model=$1; tasks=$2; out=$3; shift 3
exec ./scripts/train_rl.sh --model "$model" --tasks "$tasks" --output "$out" \
  --algorithm grpo --tuning full --learning-rate 1e-6 --thinking-mode no-thinking \
  --rollout-group 8 --rollout-concurrency 8 --mini-batch-size 8 --batch-size 8 \
  --max-steps 30 --max-context 32768 --max-tokens 1024 --max-tokens-per-micro-batch 32768 --micro-batch-size 1 \
  --advantage-std task --zero-variance-policy retry_skip --max-grad-norm 1.0 --beta 0.01 --clip 0.2 \
  --gradient-checkpointing --seed 0 "$@"
