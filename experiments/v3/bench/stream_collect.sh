#!/usr/bin/env bash
# Collect teacher episodes on a pool's bundles as they are built, instead of after the whole pool (builds take
# hours). Every INTERVAL seconds the training-ready bundles so far go into a task list and collect_teacher runs on it
# (it resumes, so only new tasks are played, and it waits for DeepSeek's off-peak hours itself). Stops once the
# pool's run_until_built loop has exited and a last pass found nothing new.
#   experiments/v3/bench/stream_collect.sh POOL_DIR PREFIX OUTPUT.jsonl [EPISODES]
set -uo pipefail
cd "$(dirname "$0")/../../.."
pool=$1; prefix=$2; output=$3; episodes=${4:-8}; interval=${INTERVAL:-900}
tasks="${output%.jsonl}.tasks.json"
last=-1
while true; do
  count=$(python3 - "$pool" "$prefix" "$tasks" <<'PY'
import json, sys
from pathlib import Path
pool, prefix, out = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
tasks = []
for status in sorted(pool.glob('build/*/status.json')):
    s = json.loads(status.read_text())
    if s.get('training_ready') and (status.parent / 'bundle' / 'manifest.json').exists():
        tasks.append({'id': f"{prefix}-{status.parent.name}", 'sandbox': str(status.parent / 'bundle'), 'split': 'train',
                      'weight': 1.0})
Path(out).write_text(json.dumps({'tasks': tasks}, indent=1))
print(len(tasks))
PY
)
  finished=0; [ -f "$pool.exit" ] && [ "$pool.exit" -nt "$pool" ] && finished=1
  if [ "$count" != "$last" ]; then
    echo "[$(date '+%F %T')] $count ready bundles: collecting"
    PYTHONPATH=src .venv/bin/python experiments/v3/collect_teacher.py --tasks "$tasks" --episodes "$episodes" \
      --output "$output" >> "${output%.jsonl}.log" 2>&1
    last=$count
  elif [ $finished = 1 ]; then
    echo "[$(date '+%F %T')] pool finished and nothing new: done"; break
  fi
  sleep $interval
done
