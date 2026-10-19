#!/usr/bin/env bash
# Chain the residual diagnostics onto the end of the campaign.
#
# The GPU part of code/residual_dist.py must not share the device with training
# (a concurrent campaign is what killed one seed earlier today), so this waits
# for finish.sh to write ALL DONE into results/.finish_state, then measures,
# then rebuilds the two manuscripts so the new numbers land in the deliverable.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY="conda run --no-capture-output -n wirenet python"
STATE=results/.finish_state
MINE=results/.residual_state

say() { echo "[residual] $*" | tee -a "$MINE"; }
: > "$MINE"
say "armed $(date +%F' '%H:%M:%S)"

T0=$(date +%s)
while ! grep -q "ALL DONE" "$STATE" 2>/dev/null; do
  if [ $(( $(date +%s) - T0 )) -gt 43200 ]; then say "TIMEOUT waiting for campaign"; exit 2; fi
  sleep 60
done
say "campaign done at $(date +%H:%M); $(tail -1 "$STATE")"
sleep 60   # let the last training/eval process release the GPU

say "measuring residual distribution + recomputation + gate sensitivity $(date +%H:%M)"
$PY code/residual_dist.py >> "$MINE" 2>&1 || say "residual_dist FAILED"
tail -6 "$MINE"

say "rebuilding both manuscripts $(date +%H:%M)"
$PY code/make_paper.py --steps tex,build,audit >> "$MINE" 2>&1
say "rebuild done $(date +%H:%M)"
say "ALL DONE $(date +%F' '%H:%M:%S)"
