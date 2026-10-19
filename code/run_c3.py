# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C3 discriminating experiment: loss-induced pessimistic collapse.

Question (external review round 4): out of the training band the error
saturates near 1, which is the signature of the model collapsing to the
pessimistic solution u_hat ~ 0 (psi_hat ~ psi_inc).  The relative-L2 loss is
minimised long before the output energy matches the target, so the collapse
could be a loss artefact rather than a physical generalisation boundary.
This experiment retrains the baseline arm with the SAME everything (cache
train_a025, prefix 12288, width 40 / modes 20 / layers 4 / proj 128 / norm
layer, batch 64, 55 epochs, 3 seeds) and one added loss term:

    L = mean rel-L2(psi_hat, psi)  +  lam * mean((log |u_hat| / |u|)^2),
    lam = 0.1  (--energy-lam; 0 reproduces the baseline exactly).

The log-ratio penalty is scale-free and symmetric in over/under-prediction;
it acts on TRAINING (in-band) samples only, so the training distribution is
unchanged and the only difference is the loss.

Reading:
  * out-of-band plateau drops / energy ratio recovers -> the collapse is
    loss-induced pessimism and the manuscript's boundary claim must be
    softened;
  * plateau and cliff unchanged -> the pessimistic-solution explanation is
    ruled out as well; the boundary survives both strongest confounds.

Strictly serial (one training at a time), per the project's GPU discipline.
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


def verify_baseline_protocol():
    """The energy arm must match main_fno_a025 in every recorded field."""
    import json
    cfgp = os.path.join(C.RUNS, "main_fno_a025_s0", "config.json")
    if not os.path.exists(cfgp):
        raise SystemExit("baseline config missing: %s" % cfgp)
    cfg = json.load(open(cfgp))
    if os.path.normpath(cfg["data"]) != os.path.normpath(CACHE):
        raise SystemExit("baseline arm cache moved: %s" % cfg["data"])
    keys = ("epochs", "batch", "lr", "width", "modes", "layers", "proj",
            "norm", "n_train")
    print("baseline protocol:", {k: cfg[k] for k in keys})


def train(seed):
    sh([PY, os.path.join(HERE, "train.py"),
        "--name", "en_fno_a025_s%d" % seed, "--model", "fno",
        "--energy-lam", str(LAM),
        "--data", CACHE, "--epochs", "55", "--batch", "64",
        "--lr", "0.001", "--width", "40", "--modes", "20",
        "--layers", "4", "--proj", "128", "--norm", "layer",
        "--n-train", str(N_TRAIN),
        "--seed", str(seed)])


def evaluate():
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "*en_fno_a025*", "checkpoint.pt")])


if __name__ == "__main__":
    verify_baseline_protocol()
    for s in SEEDS:
        train(s)
    evaluate()
    print("\nC3 chain complete. Read: out-of-band plateau and cliff in"
          " results/eval_en_fno_a025_s*.json vs the baseline arm; the energy"
          " probe is in analyze_c3.py.", flush=True)
