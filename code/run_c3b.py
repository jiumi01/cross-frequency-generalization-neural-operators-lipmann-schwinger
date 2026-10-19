# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C3b: the OUT-OF-BAND member of the C3 loss ablation.

The in-band energy penalty (run_c3.py) is inactive once the model is
energy-calibrated in band -- the relative-L2 optimum already sits at an
in-band energy ratio of ~1.015 -- so that arm reduces to the baseline and
cannot answer the review's question.  This arm adds a penalty on UNLABELLED
out-of-band samples (f ~ U[5,12], drawn fresh every step, never solved,
never labelled): the predicted correction energy is anchored on the first
Born field, computable at any frequency without labels.  Only the ENERGY is
constrained; the phase is free.  Consequently:

  * energy ratio recovers AND error drops  -> the pessimistic collapse was
    the mechanism; the manuscript's boundary claim must be softened;
  * energy ratio recovers, error stays ~1   -> the collapse is not the
    blocker: the model lacks out-of-band PHASE information, i.e. the
    boundary is informational, not a loss artefact;
  * energy ratio does not recover           -> the extrapolated operator
    ignores even a direct out-of-band penalty on the anchored quantity.

Same cache/prefix/hyperparameters/seeds as the baseline arm; lambda = 0.1.
"""
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CACHE = os.path.join(C.DATA, "train_a025")
N_TRAIN = 12288
SEEDS = (0, 1, 2)
LAM = 0.1


def sh(cmd):
    print("\n+ " + " ".join(cmd) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print("[%.0fs] exit=%d" % (time.time() - t0, r.returncode), flush=True)
    if r.returncode != 0:
        raise SystemExit("FAILED: " + " ".join(cmd))


def train(seed):
    sh([PY, os.path.join(HERE, "train.py"),
        "--name", "eb_fno_a025_s%d" % seed, "--model", "fno",
        "--ob-energy-lam", str(LAM),
        "--data", CACHE, "--epochs", "55", "--batch", "64",
        "--lr", "0.001", "--width", "40", "--modes", "20",
        "--layers", "4", "--proj", "128", "--norm", "layer",
        "--n-train", str(N_TRAIN),
        "--seed", str(seed)])


def evaluate():
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "eb_fno_a025_*", "checkpoint.pt")])


if __name__ == "__main__":
    for s in SEEDS:
        train(s)
    evaluate()
    print("\nC3b chain complete. Read: analyze_c3.py (energy_ob arm).", flush=True)
