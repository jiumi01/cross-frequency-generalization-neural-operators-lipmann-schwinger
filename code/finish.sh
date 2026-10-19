#!/usr/bin/env bash
# Final chain: take the campaign from "replication pass running" to the
# delivered 3-seed paper, without ever letting two GPU campaigns overlap.
#
# Why this exists (two failures observed on 2026-10-12):
#   * eval_incremental.sh hit its 4 h deadline while the seed-replication pass
#     was still running, so the last four finished runs had no evaluation.
#   * `pmatch_unet_a025_s1` was killed by the 3000 s per-spec ceiling because a
#     3-shard evaluation happened to be sharing the same GPU.  A killed spec is
#     simply skipped by the next pass (main() skips names that already have a
#     result.json), so it silently costs one of the three seeds -- exactly the
#     like-for-like uniformity the replication pass exists to buy.
#
# Ordering guarantees here are strictly serial:
#   A. wait for the replication pass to finish (3rd "grid pass finished");
#   B. wait for the rebuild that replicates.sh launches to finish
#      (its results/.pipeline_done is newer than the grid log) + settle;
#   C. retry every spec that still has no result.json, up to 3 attempts
#      (run_grid is idempotent: it skips specs that already have one);
#   D. one full rebuild -- eval, tex, figs, build, both gates.
set -u
cd "$(dirname "$0")/.."
export PYTHONIOENCODING=utf-8
PY="conda run --no-capture-output -n wirenet python"
LOG=results/grid_progress.log
MARK=results/.pipeline_done
STATE=results/.finish_state

say() { echo "[finish] $*" | tee -a "$STATE"; }
: > "$STATE"
say "armed $(date +%F' '%H:%M:%S)"

# Missing specs are derived from the DECLARED universe, not from the directory
# listing.  An earlier version enumerated runs/*/ , so a spec whose directory had
# been removed (e.g. after a per-spec timeout) was invisible and never retried --
# which is exactly how one width-matched U-Net seed was lost.
missing() { $PY code/run_grid.py --list-missing 2>/dev/null | tail -1; }

# ---------------------------------------------------------------- A
T0=$(date +%s)
while [ "$(grep -c 'grid pass finished' "$LOG" 2>/dev/null || true)" -lt 3 ]; do
  if [ $(( $(date +%s) - T0 )) -gt 21600 ]; then say "TIMEOUT waiting for replicate pass"; exit 2; fi
  sleep 60
done
say "A: replication pass done $(date +%H:%M) -- missing=[$(missing)]"

# ---------------------------------------------------------------- B
while ! { [ -f "$MARK" ] && [ "$MARK" -nt "$LOG" ]; }; do
  if [ $(( $(date +%s) - T0 )) -gt 28800 ]; then say "TIMEOUT waiting for repl rebuild"; exit 2; fi
  sleep 60
done
say "B: repl rebuild done $(date +%H:%M): $(cat "$MARK")"
sleep 90   # let any evaluation shard release the GPU before it is reused

# ---------------------------------------------------------------- C
for attempt in 1 2 3; do
  M=$(missing)
  [ -z "${M// /}" ] && break
  say "C$attempt: retrying [$M]"
  for n in $M; do rm -rf "runs/$n"; done
  $PY code/run_grid.py --replicates >> "$STATE" 2>&1
  $PY code/run_grid.py            >> "$STATE" 2>&1
  sleep 30
done
M=$(missing)
say "C: retries finished $(date +%H:%M) -- still missing=[$M]"

# ---------------------------------------------------------------- D
say "D: rebuild $(date +%H:%M)"
$PY code/make_paper.py --steps eval,tex,figs,build,audit >> "$STATE" 2>&1
say "D: rebuild done $(date +%H:%M)"
say "ALL DONE $(date +%F' '%H:%M:%S) missing=[$M]"
