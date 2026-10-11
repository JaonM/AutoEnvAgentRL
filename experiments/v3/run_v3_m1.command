#!/bin/zsh -l
# Double-click: V3 M1 pipeline on the 8 frozen complex cross-business samples, then truth evaluation.
cd "$(dirname "$0")/../.." || exit 1
OUT="output/v3_m1_$(date +%Y%m%d_%H%M%S)"
echo "V3 M1 batch -> $OUT"
command -v codex >/dev/null || { echo "codex CLI not found on PATH"; read -k1; exit 1; }
.venv/bin/python -m env_factory_v3 batch --frozen output/env_factory_v2_cross_business_frozen_01 \
  --samples 4,5,9,10,14,15,19,20 --output "$OUT" --parallel 2 2>&1 | tee "$OUT.log"
.venv/bin/python experiments/v3/evaluate_truth.py "$OUT" 2>&1 | tee -a "$OUT.log"
echo; echo "Done: $OUT/truth_evaluation.md  (press any key to close)"; read -k1
