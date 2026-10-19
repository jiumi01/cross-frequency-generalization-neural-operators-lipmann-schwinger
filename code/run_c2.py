# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C2 discriminating experiment: input representation of the frequency.

Question (external review round 4): the baseline input encodes f as a single
normalised scalar channel, so the error step at the training-band edge could
in principle be an artefact of that low-bandwidth encoding -- the network may
simply lack the representational bandwidth to resolve frequencies it was not
trained on.  This experiment gives the model a strictly richer encoding of f:
the scalar channel is kept and K = 7 sin/cos Fourier-feature pairs at
geometrically spaced bands 2^0..2^6 are appended (code/data.py:FF_K).  Because
the scalar channel is retained, the ff model sees a strict superset of the
baseline's frequency information; if the cliff survives, representation
starvation is ruled out as the mechanism.

Protocol: identical to the baseline arm main_fno_a025 in every respect --
same cache (train_a025, seed0 = 4000), same deterministic prefix (n_train =
12288), same hyperparameters (width 40, modes 20, layers 4, proj 128,
norm layer, batch 64, 55 epochs), same 3 seeds.  The only difference is the
input encoding of f (parameter-count delta 560, 0.005%, from the wider first
1x1 convolution).

Reading:
  * cliff still at f = 4 -> the step is not an input-encoding artefact;
  * cliff moved or gone  -> the encoding explanation wins and the manuscript
    must credit the representation, not the physics, for the boundary.

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
CACHE = os.path.join(C.DATA, "train_a025")   # literally the baseline arm's cache
N_TRAIN = 12288
SEEDS = (0, 1, 2)


def sh(cmd):
    print("\n+ " + " ".join(cmd) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print("[%.0fs] exit=%d" % (time.time() - t0, r.returncode), flush=True)
    if r.returncode != 0:
        raise SystemExit("FAILED: " + " ".join(cmd))


def verify_pairing():
    """Paired by construction (the same cache directory), but assert it.

    The baseline arm trained on data/train_a025 with --n-train 12288; if that
    cache were ever regenerated with a different seed0 the pairing would break
    silently, so check the run config of the baseline actually points here.
    """
    import json
    cfgp = os.path.join(C.RUNS, "main_fno_a025_s0", "config.json")
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp))
        if os.path.normpath(cfg["data"]) != os.path.normpath(CACHE):
            raise SystemExit("baseline arm cache moved: %s" % cfg["data"])
    print("pairing: ff arm reads the baseline arm's cache %s" % CACHE)


def train(seed):
    sh([PY, os.path.join(HERE, "train.py"),
        "--name", "ff_fno_s%d" % seed, "--model", "fno",
        "--f-emb", "ff",
        "--data", CACHE, "--epochs", "55", "--batch", "64",
        "--lr", "0.001", "--width", "40", "--modes", "20",
        "--layers", "4", "--proj", "128", "--norm", "layer",
        "--n-train", str(N_TRAIN),
        "--seed", str(seed)])


def evaluate():
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "*ff_fno*", "checkpoint.pt")])


if __name__ == "__main__":
    verify_pairing()
    for s in SEEDS:
        train(s)
    evaluate()
    print("\nC2 chain complete. Read: cliff location in results/eval_ff_fno_s*.json"
          " vs results/eval_main_fno_a025_s*.json (same protocol, only the input"
          " encoding of f differs).", flush=True)
