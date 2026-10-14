#!/usr/bin/env bash
# After the pools are built: split, teacher demonstrations, full SFT (arm B), τ²-bench evaluation of B.
#   experiments/bench/stage_sft.sh POOL [POOL...]
set -euo pipefail
cd "$(dirname "$0")/../.."
out=output/v3_bench
PYTHONPATH=src .venv/bin/python experiments/bench/make_split.py --pools "$@" --output $out --heldout-share 0.15
PYTHONPATH=src .venv/bin/python -u experiments/collect_teacher.py --tasks $out/tasks_all.json --episodes 6 \
  --parallel 20 --output $out/teacher.jsonl > $out/collect.log 2>&1
PYTHONPATH=src .venv/bin/python -u experiments/sft_train.py --teacher $out/teacher.jsonl --model models/Qwen3-0.6B \
  --valid-ids $out/valid_ids.json --output $out/sft_B --tuning full --epochs 1.5 --learning-rate 1e-5 \
  --max-tokens 16384 > $out/sft_B.log 2>&1
experiments/bench/eval_arm.sh B $out/sft_B/model
echo done > $out/stage_sft.done
