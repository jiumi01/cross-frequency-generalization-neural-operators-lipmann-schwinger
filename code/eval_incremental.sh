#!/usr/bin/env bash
# Evaluate finished runs as they land, so the final pipeline's eval stage has
# (almost) nothing left to do.  evaluate.py --all skips runs that already have
# an eval_*.json, so this loop is idempotent and safe to run alongside training.
#
# It stops once the grid log has recorded TWO completed passes: the second pass
# is the one that picks up the specs added after the first pass started.  It
# deliberately does NOT build the paper -- wait_and_build.sh owns that, and two
# concurrent builds would fight over paper/<lang>/main.pdf.
#
# NOTE: `grep -c` prints 0 *and* exits 1 when there is no match, so
# `$(grep -c ... || echo 0)` yields "0\n0".  Use `|| true` instead.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY="conda run --no-capture-output -n wirenet python"

DEADLINE=$(( $(date +%s) + 4*3600 ))
while :; do
  N=$(( $(grep -c "grid pass finished" results/grid_progress.log 2>/dev/null || true) + 0 ))
  P=$(( $(ls results/eval_*.json 2>/dev/null | wc -l) + 0 ))
  R=$(( $(ls runs/*/checkpoint.pt 2>/dev/null | wc -l) + 0 ))
  echo "[eval-loop] $(date +%H:%M) passes=$N runs=$R evals=$P"
  if [ "$N" -ge 2 ] && [ "$P" -ge "$R" ]; then
    echo "[eval-loop] every finished run is evaluated; stopping"
    break
  fi
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    echo "[eval-loop] deadline reached; stopping (evals=$P runs=$R)"
    break
  fi
  $PY code/evaluate.py --all >/dev/null 2>&1
  sleep 90
done
echo "[eval-loop] DONE $(date +%H:%M) evals=$(ls results/eval_*.json 2>/dev/null | wc -l)"
