#!/usr/bin/env bash
# Chain the seed-replication pass onto the end of the running pipeline.
#
# Why a chain rather than a second concurrent driver: this box has one GPU.
# Two training drivers at once would split throughput (and risk OOM), so the
# replication pass must start only after the current grid pass AND the paper
# pipeline that follows it have both finished.
#
# Ordering:
#   1. wait for the second "grid pass finished" line  (grid pass 2 done)
#   2. wait for results/.pipeline_done to be NEWER than the grid log
#      (i.e. the paper pipeline that wait_and_build.sh launched has completed)
#   3. run the extra seeds for every ablation arm  (run_grid.py --replicates)
#   4. re-run the full pipeline so the error bars in every figure are 3-seed
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY="conda run --no-capture-output -n wirenet python"
LOG=results/grid_progress.log
MARK=results/.pipeline_done

echo "[repl] armed $(date +%F' '%H:%M)"
T0=$(date +%s)

while [ "$(grep -c 'grid pass finished' "$LOG" 2>/dev/null || true)" -lt 2 ]; do
  if [ $(( $(date +%s) - T0 )) -gt 21600 ]; then
    echo "[repl] TIMEOUT waiting for grid pass 2"; exit 2
  fi
  sleep 60
done
echo "[repl] grid pass 2 done at $(date +%H:%M)"

while ! { [ -f "$MARK" ] && [ "$MARK" -nt "$LOG" ]; }; do
  if [ $(( $(date +%s) - T0 )) -gt 21600 ]; then
    echo "[repl] TIMEOUT waiting for the paper pipeline"; exit 2
  fi
  sleep 60
done
echo "[repl] paper pipeline done at $(date +%H:%M): $(cat "$MARK")"

echo "[step] seed replication $(date +%H:%M)"
$PY code/run_grid.py --replicates || { echo "[repl] replicate pass FAILED"; exit 3; }

echo "[step] rebuild with 3-seed error bars $(date +%H:%M)"
$PY code/make_paper.py --steps eval,tex,figs,build,audit

echo "[repl] ALL DONE at $(date +%H:%M)"
