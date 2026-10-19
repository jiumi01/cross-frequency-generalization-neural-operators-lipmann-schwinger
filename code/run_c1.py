# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C1 discriminating experiment: per-frequency loss-tangent compensation.

Question (external review round 4, item 2): the manuscript attributes the
error step at the band edge to frequency generalisation, but on the grid the
operator norm max|kappa^2 Ghat| rises 3.44x across the test frequencies at
fixed gamma = 0.05, so the conditioning is a live confound: the cliff sits
exactly where the norm stops rising.  This experiment removes the confound by
making gamma frequency-dependent, gamma(f), chosen so the attained norm is
CONSTANT over the whole test grid (code/gamma_comp.py; target 2.9075, reached
at every frequency to 4e-3, with gamma(f) inside the solver-convergent band
[0.02, 0.20]).  An FNO is then trained and tested under exactly the same
protocol as the baseline arm.

Reading:
  * cliff still at f = 4  -> the step is a property of the frequency boundary,
    not of the conditioning; the baseline claim survives its strongest
    alternative explanation;
  * cliff moved or gone    -> the conditioning explanation wins and the
    manuscript's mechanism claim must be rewritten.

Paired by construction: the cache uses the same seed0 as the baseline arm, so
the media are identical sample-for-sample and ONLY the regularisation differs.
Strictly serial (one training at a time), per the project's GPU discipline.
"""
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import gamma_comp as G

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CACHE = os.path.join(C.DATA, "train_gc")
N_TOTAL = 16384
CHUNK = 512
SEEDS = (0, 1, 2)
SEED0 = 4000          # same scatterer draws as train_a025 (gen_caches.py)
N_TRAIN = 12288       # deterministic prefix, as the baseline arm uses


def sh(cmd):
    print("\n+ " + " ".join(cmd) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print("[%.0fs] exit=%d" % (time.time() - t0, r.returncode), flush=True)
    if r.returncode != 0:
        raise SystemExit("FAILED: " + " ".join(cmd))


def build_cache():
    os.makedirs(CACHE, exist_ok=True)
    have = [f for f in os.listdir(CACHE) if f.endswith(".npz")]
    if len(have) * CHUNK >= N_TOTAL:
        print("cache present: %d chunks" % len(have))
        return
    # These parameters MUST match the train_gc entry of gen_caches.py:ARMS
    # (f=[1,4], vmax=0.25 fixed, seed0=4000).  A first run omitted them, fell
    # back to build_train_cache's mixed-contrast defaults, and produced a cache
    # with the right frequencies but the wrong contrast -- the pairing check
    # below is what caught it.  Pass them explicitly rather than trusting a
    # default that lives in another file.
    D.build_train_cache(N_TOTAL, CHUNK, seed0=SEED0, out_dir=CACHE,
                        f_lo=1.0, f_hi=4.0, vmax_lo=0.25, vmax_hi=0.25,
                        device="cuda" if __import__("torch").cuda.is_available()
                        else "cpu",
                        gamma=G.comp_gamma)


def verify_pairing():
    """The compensation arm must share the baseline's media, sample for sample.

    make_contrast draws from a generator seeded by (seed + 7777) independently
    of gamma, so if the seed0 and the (f, v_max) draw parameters match, chunk 0
    of the two caches holds the same contrast fields AND the same frequencies.
    Check both explicitly: a run that got the frequencies right but the
    contrast wrong (mixed instead of fixed 0.25) still produces identical f,
    so checking v alone catches it, but checking both makes the failure mode
    unambiguous.  If this fails the arm is not paired and every downstream
    comparison inherits the confound it was built to remove.
    """
    import numpy as np
    a = os.path.join(C.DATA, "train_a025", "chunk_0000.npz")
    b = os.path.join(CACHE, "chunk_0000.npz")
    if not (os.path.exists(a) and os.path.exists(b)):
        print("pairing check skipped (missing cache)")
        return
    za, zb = np.load(a), np.load(b)
    dv = float(np.abs(za["v"] - zb["v"]).max())
    df = float(np.abs(za["f"] - zb["f"]).max())
    same = dv == 0.0 and df == 0.0
    print("pairing vs train_a025 chunk 0: identical = %s (max|dv|=%.2e, "
          "max|df|=%.2e)" % (same, dv, df))
    if not same:
        raise SystemExit("C1 cache is not paired with the baseline arm")


def train(seed):
    sh([PY, os.path.join(HERE, "train.py"),
        "--name", "gc_fno_s%d" % seed, "--model", "fno",
        "--data", CACHE, "--epochs", "55", "--batch", "64",
        "--lr", "0.001", "--width", "40", "--modes", "20",
        "--layers", "4", "--proj", "128", "--norm", "layer",
        "--n-train", str(N_TRAIN),
        "--seed", str(seed)])


def evaluate():
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "*gc_fno*", "checkpoint.pt")])


if __name__ == "__main__":
    build_cache()
    verify_pairing()
    for s in SEEDS:
        train(s)
    evaluate()
    print("\nC1 chain complete. Read: cliff location in results/eval_gc_fno_s*.json"
          " vs results/eval_main_fno_a025_s*.json (same protocol, only gamma(f)"
          " differs).", flush=True)
