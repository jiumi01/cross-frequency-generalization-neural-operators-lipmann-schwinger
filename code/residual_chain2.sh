#!/usr/bin/env bash
# Second residual pass: rerun the corrected diagnostic, then rebuild.
#
# The first pass crashed in part 3 (it partitioned the two populations by the
# residual the naive recurrence REPORTS, which is by construction the small
# quantity, so one population came out empty) and part 2 was asking a
# single-precision recomputation for a float64 answer.  Both are fixed:
#   * part 2 now measures the complex64 storage floor AND re-solves from the
#     shipped inputs at working precision, which is the quantity comparable to
#     the 1e-9 gate;
#   * part 3 partitions by the honest residual of each solution and reports the
#     silent-failure fraction.
#
# Strictly serial: waits for the last-mile chain (seed recovery + rebuild) to
# finish, because residual_dist.py part 2 does float64 solves on the same GPU.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY="conda run --no-capture-output -n wirenet python"
STATE=results/.lastmile_state
MINE=results/.residual2_state

say() { echo "[residual2] $*" | tee -a "$MINE"; }
: > "$MINE"
say "armed $(date +%F' '%H:%M:%S)"

T0=$(date +%s)
while ! grep -q "ALL DONE" "$STATE" 2>/dev/null; do
  if [ $(( $(date +%s) - T0 )) -gt 21600 ]; then say "TIMEOUT waiting for last mile"; exit 2; fi
  sleep 60
done
say "last mile done $(date +%H:%M); $(tail -1 "$STATE")"
sleep 60   # let the last training/eval process release the GPU

say "residual diagnostics (corrected) $(date +%H:%M)"
$PY code/residual_dist.py >> "$MINE" 2>&1 || say "residual_dist FAILED"

say "rebuilding both manuscripts $(date +%H:%M)"
$PY code/make_paper.py --steps tex,build,audit >> "$MINE" 2>&1
say "rebuild done $(date +%H:%M)"
say "ALL DONE $(date +%F' '%H:%M:%S)"
