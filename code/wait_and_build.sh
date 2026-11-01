#!/usr/bin/env bash
# Wait for the running experiment grid, then run the whole paper pipeline.
#
#   1. wait until the current run_grid.py pass reports completion
#   2. re-run run_grid.py to pick up specs added after the pass started
#      (mode_m02_s0, which supplies the M < f_train evidence)
#   3. evaluate every checkpoint, emit macros/tables/figures, compile both PDFs,
#      and audit the deliverables
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY="conda run --no-capture-output -n wirenet python"

echo "[wait] start $(date +%F' '%H:%M)"

T0=$(date +%s)
while ! grep -q "grid pass finished" results/grid_progress.log 2>/dev/null; do
  N=$(ls runs/*/result.json 2>/dev/null | wc -l)
  EL=$(( $(date +%s) - T0 ))
  if [ "$EL" -gt 14400 ]; then
    echo "[wait] TIMEOUT after 4h at $(date +%H:%M) with $N runs done"; exit 2
  fi
  sleep 60
done
echo "[wait] main grid pass finished at $(date +%H:%M) ($(ls runs/*/result.json | wc -l) runs)"

echo "[step] extra specs (mode_m02_s0) $(date +%H:%M)"
$PY code/run_grid.py || { echo "[wait] extra-spec pass FAILED"; exit 3; }

echo "[step] full pipeline $(date +%H:%M)"
$PY code/make_paper.py --steps eval,tex,figs,build,audit

echo "[wait] PIPELINE DONE at $(date +%H:%M) exit=$?"
