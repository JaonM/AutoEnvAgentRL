#!/bin/zsh -l
# Retry the samples that hit OpenAI 403 infrastructure errors, one at a time.
cd "$(dirname "$0")/../.." || exit 1
OUT="output/v3_spike_retry_$(date +%Y%m%d_%H%M%S)"
echo "V3 spike retry -> $OUT"
.venv/bin/python experiments/v3_spike/run.py --output "$OUT" --samples 15,19,20 --parallel 1 2>&1 | tee "$OUT.log"
echo; echo "Done. Results: $OUT/summary.md  (press any key to close)"; read -k1
