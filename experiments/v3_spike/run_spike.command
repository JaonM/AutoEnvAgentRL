#!/bin/zsh -l
# Double-click to run the V3 spike on the 8 frozen complex samples (GPT-6.1-Sol via codex).
cd "$(dirname "$0")/../.." || exit 1
OUT="output/v3_spike_$(date +%Y%m%d_%H%M%S)"
echo "V3 spike -> $OUT"
command -v codex >/dev/null || { echo "codex CLI not found on PATH"; read -k1; exit 1; }
.venv/bin/python experiments/v3_spike/run.py --output "$OUT" --parallel 2 2>&1 | tee "$OUT.log"
echo; echo "Done. Results: $OUT/summary.md  (press any key to close)"; read -k1
