# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""BA2-X: the anchored residual operator with the fit-depth/sum-depth defect
FIXED -- the analytic head now sums every term it was fitted against.

What was wrong.  run_ba2.py documents the construction as "the head still
sums all seven terms -- the coefficients are applied outside the network",
but train.py and evaluate.build truncated the Born term list to
`anchor_channels` BEFORE anchored_u, so every a050/a100 arm ever run summed
only the first 4 (resp. 2) terms of an 8-term (resp. 7-term) fit.  The first
few coefficients of a deep fit are tuned to cancel against the deep terms;
kept alone they are catastrophic.  Measured under the formal protocol with LS
coefficients (results/ls_head_alone.json, variant names truncated/full):

    head at its own contrast        a050 @ v=0.50        a100 @ v=1.00
    truncated (what the arms ran)   in 21.3  plat 98.3   in 10.4  plat 34.9
    full      (documented intent)   in 0.170 plat 0.385  in 0.538 plat 0.903
    shallow   (fit at channels)     in 0.329 plat 0.540  in 0.881 plat 1.182

The full LS head ALONE already beats the trained FNO out of band at both
contrasts (plateau 1.499 and 1.997), so the manuscript's "the construction
fails at v >= 0.50" boundary is an implementation artifact, not physics.
The zero-GPU probes also close the earlier hypotheses: the residual is not
the carrier (proj_precheck: projecting or zeroing it changes nothing), and
freezing the coefficients on top of the truncated head makes things WORSE
(baf arms, plateau 74-77) because the garbage head goes uncanceled.

This chain (strictly serial, one GPU job at a time):

  1. re-evaluate every archived anchored checkpoint (ba_*, baf_*) under the
     fixed head semantics -- their archives carry no head_semantics key, so
     the freshness gate forces it; the a025 arms are unaffected by the defect
     (channels=None, nothing truncated) and their re-evaluation must
     reproduce the archived numbers, which is the invariant check;
  2. retrain the corrected BA2 arm (fine-tuned coefficients, the construction
     as documented) at v=0.50 and v=1.00, three seeds each: bax_fno_*;
  3. evaluate the new arms.

Pre-registered reading for bax, per contrast, against the same-protocol
archives (paired across the three shared seeds):

  * PASS    : in-band <= trained FNO's AND plateau <= 0.5 x trained FNO's
              (0.749 at v=0.50, 0.999 at v=1.00) -- the construction, with
              the defect fixed, extends to full contrast;
  * PARTIAL : plateau <= trained FNO's but in-band worse;
  * FAIL    : plateau > 1.5 x the full LS head alone (0.578 at v=0.50,
              1.354 at v=1.00) -- even the honest deep head cannot protect
              the plateau once the residual is added.

    python code/run_bax.py
"""
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

ARMS = [("a050", "train_a050", 8, 4),
        ("a100", "train_a100", 7, 2)]
SEEDS = (0, 1, 2)
N_TRAIN = 12288


def sh(cmd):
    print("\n+ " + " ".join(cmd) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print("[%.0fs] exit=%d" % (time.time() - t0, r.returncode), flush=True)
    if r.returncode != 0:
        raise SystemExit("FAILED: " + " ".join(cmd))


def verify():
    head = os.path.join(C.RES, "ls_head_alone.json")
    if not os.path.exists(head):
        raise SystemExit("missing %s -- the go decision and the FAIL gate "
                         "rest on it" % head)
    d = C.load_json(head)
    if "variants" not in d["arms"]["0.50"]:
        raise SystemExit("%s lacks the variants block -- rerun "
                         "code/ls_head_eval.py" % head)
    print("head-alone archive present with truncated/full/shallow variants")


def reeval_old():
    # anchored archives scored under the truncated head have no
    # head_semantics key; _fresh refuses them and these patterns re-score
    # every one under the fixed operator.
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "*ba_fno_*_m*", "checkpoint.pt")])
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "*baf_fno_*_m*", "checkpoint.pt")])


def train(tag, cache, m, seed, channels=None):
    cmd = [PY, os.path.join(HERE, "train.py"),
           "--name", "bax_fno_%s_m%d_s%d" % (tag, m, seed), "--model", "fno",
           "--anchor", "born", "--anchor-order", str(m)]
    if channels is not None:
        cmd += ["--anchor-channels", str(channels)]
    cmd += [
        "--data", os.path.join(C.DATA, cache),
        "--epochs", "55", "--batch", "64", "--lr", "0.001",
        "--width", "40", "--modes", "20", "--layers", "4", "--proj", "128",
        "--norm", "layer", "--n-train", str(N_TRAIN), "--seed", str(seed)]
    sh(cmd)


def evaluate():
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "*bax_fno_*_m*", "checkpoint.pt")])


if __name__ == "__main__":
    verify()
    t0 = time.time()
    reeval_old()
    for tag, cache, m, ch in ARMS:
        for seed in SEEDS:
            train(tag, cache, m, seed, ch)
    evaluate()
    print("\nBA2-X chain complete.  Read results/eval_bax_fno_*.json against "
          "results/eval_ba_fno_*.json (re-scored under full semantics) and "
          "results/ls_head_alone.json (variants).")
    print("total %.0f min" % ((time.time() - t0) / 60))
