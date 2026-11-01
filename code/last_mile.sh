#!/usr/bin/env bash
# Last mile: recover the one arm that a per-spec timeout cost, then deliver.
#
# State when this was written: the replication pass finished with 74 of 75
# declared specs on disk.  The missing arm is `pmatch_unet_a025_s1` (the
# width-matched U-Net, base=76) -- killed at 01:58 by a 3000 s ceiling while a
# 3-shard evaluation shared the GPU, then deleted as an incomplete directory.
#
# Why the earlier chain missed it: finish.sh computed "missing" by enumerating
# runs/*/ , so a spec whose directory had been removed was invisible.  The
# retry here is driven by the declared spec universe instead
# (`run_grid.py --list-missing` / `--all`), which is the robust form.
#
# Ordering is strictly serial -- nothing here overlaps the residual diagnostics
# or an evaluation, because concurrent GPU users are what caused the loss.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY="conda run --no-capture-output -n wirenet python"
STATE=results/.lastmile_state
RES=results/.residual_state

say() { echo "[lastmile] $*" | tee -a "$STATE"; }
: > "$STATE"
say "armed $(date +%F' '%H:%M:%S)"

T0=$(date +%s)
while ! grep -q "ALL DONE" "$RES" 2>/dev/null; do
  if [ $(( $(date +%s) - T0 )) -gt 21600 ]; then say "TIMEOUT waiting for residual chain"; exit 2; fi
  sleep 60
done
say "residual chain done $(date +%H:%M)"
sleep 60   # let the last GPU process release the device

say "missing before retry: [$( $PY code/run_grid.py --list-missing 2>/dev/null | tail -1 )]"
$PY code/run_grid.py --all >> "$STATE" 2>&1
say "missing after retry:  [$( $PY code/run_grid.py --list-missing 2>/dev/null | tail -1 )] $(date +%H:%M)"

say "final rebuild $(date +%H:%M)"
$PY code/make_paper.py --steps eval,tex,figs,build,audit >> "$STATE" 2>&1
say "rebuild done $(date +%H:%M)"
say "ALL DONE $(date +%F' '%H:%M:%S)"
