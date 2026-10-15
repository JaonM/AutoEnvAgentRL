#!/usr/bin/env bash
# Run a resumable long job (RL training, a big teacher collection) only in DeepSeek's off-peak hours: start when
# off-peak, stop it (SIGINT, then SIGTERM) when peak hours begin, and start it again with RESUME_FLAG (default
# --resume) once they end, until it exits on its own.
#   experiments/v3/bench/offpeak_run.sh experiments/v3/bench/stage_I3.sh MODEL TASKS OUT
set -uo pipefail
cd "$(dirname "$0")/../../.."
offpeak() { PYTHONPATH=src .venv/bin/python -m env_factory_v3.production.offpeak "$@"; }
resume_flag=${RESUME_FLAG---resume}
first=1
while true; do
  offpeak --wait
  if [ $first = 1 ]; then "$@" & else "$@" $resume_flag & fi
  job=$!; first=0
  while kill -0 $job 2>/dev/null; do
    if [ "${ENVFACTORY_OFFPEAK:-1}" != 0 ] && [ "$(offpeak --status | cut -d' ' -f1)" = peak ]; then
      echo "[$(date '+%F %T')] peak hours: stopping the job, to resume when they end"
      kill -INT $job; sleep 120; kill -0 $job 2>/dev/null && kill -TERM $job; wait $job
      continue 2
    fi
    sleep 60
  done
  wait $job; exit $?
done
