#!/usr/bin/env bash
# Run (or resume) a pool with run_pipeline.sh, then resume again while any build failed for
# infrastructure reasons (codex rate limits, network); finished builds are always reused.
#   experiments/v3/bench/run_until_built.sh OUTPUT [run_pipeline args...]
set -uo pipefail
cd "$(dirname "$0")/../../.."
out=$1; shift
for round in 1 2 3 4; do
  bash scripts/run_pipeline.sh --output "$out" "$@" > /dev/null 2>&1
  status=$?
  infra=$(python3 -c "import json,sys;t=json.load(open('$out/tasks.json'));print(sum(1 for e in t['excluded'] if e.get('failure_class')=='infrastructure'))" 2>/dev/null || echo unknown)
  echo "round $round exit $status infrastructure_failures $infra" >> "$out.rounds"
  [ "$infra" = "0" ] && break
  sleep 300
done
echo "exit $status" > "$out.exit"
