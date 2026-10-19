# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C4: transfer-learning FNO (TL-FNO protocol of wang2024tlfno) on this benchmark.

Question (external review round 4): TL-FNO pre-trains the FNO backbone at a
single source location and frequency and fine-tunes it for other sources and
frequencies, reporting gains over vanilla FNO in-distribution and
out-of-distribution.  Our manuscript reports that on this benchmark only
widening the training band helps, which reads as a contradiction.  This
experiment reproduces the TL-FNO protocol on the benchmark's own terms:

  stage 1 (pre-train)  FNO on a single-frequency cache: f = 2.5 fixed,
                        media paired with the baseline arm (seed0 = 4000, so
                        (v, theta) match train_a025 sample-for-sample; only
                        f differs), 4096 training samples, 55 epochs,
                        baseline hyperparameters, 3 seeds;
  stage 2 (fine-tune)   initialise from stage 1, train on the SAME wide-band
                        cache and prefix as the baseline arm (12288 samples,
                        55 epochs), lower learning rate 1e-4, fresh optimiser;
  evaluation            the standard zero-shot grid, scored exactly like the
                        baseline arm.

Reading:
  * TL arm beats the baseline OUT OF BAND (f >= 5) -> transfer genuinely
    buys frequency generalisation here and the manuscript's "only a wider
    band helps" must be qualified;
  * TL arm matches the baseline in band and does not move the cliff ->
    the protocols answer different questions (supervised adaptation vs.
    zero-shot extrapolation) and the apparent conflict dissolves; the
    manuscript states this explicitly.

Compute note (stated up front): the TL arm trains 55 epochs on 4096 PLUS 55
on 12288 samples, i.e. ~1.33x the baseline's compute.  A win for TL would
therefore need a compute-matched control before it is evidence of transfer;
a loss needs no such control (more compute failing is a fortiori).
"""
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
CACHE = os.path.join(C.DATA, "train_f25")     # single-frequency pre-train cache
WIDE = os.path.join(C.DATA, "train_a025")     # baseline arm's cache
N_TOTAL = 5120
CHUNK = 512
SEED0 = 4000
N_PRE = 4096
N_TRAIN = 12288
SEEDS = (0, 1, 2)


def sh(cmd):
    print("\n+ " + " ".join(cmd) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print("[%.0fs] exit=%d" % (time.time() - t0, r.returncode), flush=True)
    if r.returncode != 0:
        raise SystemExit("FAILED: " + " ".join(cmd))


def build_cache():
    import numpy as np
    os.makedirs(CACHE, exist_ok=True)
    have = [f for f in os.listdir(CACHE) if f.endswith(".npz")]
    if len(have) * CHUNK < N_TOTAL:
        import torch
        import data as D
        # sample_params draws (v, theta) BEFORE f, so with the same seed0 the
        # media sequence equals train_a025's -- the pre-train sees the same
        # scatterers as the baseline arm, at one fixed frequency.
        D.build_train_cache(N_TOTAL, CHUNK, seed0=SEED0, out_dir=CACHE,
                            f_lo=2.5, f_hi=2.5, vmax_lo=0.25, vmax_hi=0.25,
                            device="cuda" if torch.cuda.is_available() else "cpu")
    # pairing check: chunk 0 must match train_a025 chunk 0 in (v, theta)
    za = np.load(os.path.join(WIDE, "chunk_0000.npz"))
    zb = np.load(os.path.join(CACHE, "chunk_0000.npz"))
    dv = float(np.abs(za["v"] - zb["v"]).max())
    dt = float(np.abs(za["theta"] - zb["theta"]).max())
    df = float(np.abs(zb["f"] - 2.5).max())
    ok = dv == 0.0 and dt == 0.0 and df == 0.0
    print("pre-train cache pairing vs train_a025: identical=%s "
          "(max|dv|=%.1e max|dtheta|=%.1e max|f-2.5|=%.1e)" % (ok, dv, dt, df))
    if not ok:
        raise SystemExit("C4 pre-train cache is not paired with the baseline arm")


def pretrain(seed):
    sh([PY, os.path.join(HERE, "train.py"),
        "--name", "tlpre_fno_s%d" % seed, "--model", "fno",
        "--data", CACHE, "--epochs", "55", "--batch", "64",
        "--lr", "0.001", "--width", "40", "--modes", "20",
        "--layers", "4", "--proj", "128", "--norm", "layer",
        "--n-train", str(N_PRE),
        "--seed", str(seed)])


def finetune(seed):
    init = os.path.join(C.RUNS, "tlpre_fno_s%d" % seed, "checkpoint.pt")
    sh([PY, os.path.join(HERE, "train.py"),
        "--name", "tl_fno_a025_s%d" % seed, "--model", "fno",
        "--init", init,
        "--data", WIDE, "--epochs", "55", "--batch", "64",
        "--lr", "0.0001", "--width", "40", "--modes", "20",
        "--layers", "4", "--proj", "128", "--norm", "layer",
        "--n-train", str(N_TRAIN),
        "--seed", str(seed)])


def evaluate():
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "tl_fno_a025_*", "checkpoint.pt")])


if __name__ == "__main__":
    build_cache()
    for s in SEEDS:
        pretrain(s)
        finetune(s)
    evaluate()
    print("\nC4 chain complete. Read: results/eval_tl_fno_a025_s*.json vs"
          " results/eval_main_fno_a025_s*.json (out-of-band plateau and cliff).",
          flush=True)
