# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C4b: the STRONG version of the TL-FNO arm.

The lr = 1e-4 fine-tune (run_c4.py) leaves the model trapped near the
single-frequency pre-train minimum: its wide-band validation lands at ~1.5e-1
versus ~8.2e-2 for the from-scratch baseline, so a sceptic could attribute
the TL arm's failure to the low learning rate rather than to transfer.  This
arm removes that objection by giving transfer its best shot: the SAME
pre-trained checkpoint, fine-tuned at the baseline's own learning rate
(1e-3) -- i.e. a warm-started replica of the baseline run.  Warm-starting
cannot be worse than random initialisation in expectation, so this is the
favourable version of the protocol; if it still fails to move the out-of-band
cliff, the conclusion holds a fortiori.
"""
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
WIDE = os.path.join(C.DATA, "train_a025")
N_TRAIN = 12288
SEEDS = (0, 1, 2)


def sh(cmd):
    print("\n+ " + " ".join(cmd) + "\n", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    print("[%.0fs] exit=%d" % (time.time() - t0, r.returncode), flush=True)
    if r.returncode != 0:
        raise SystemExit("FAILED: " + " ".join(cmd))


def finetune_warm(seed):
    init = os.path.join(C.RUNS, "tlpre_fno_s%d" % seed, "checkpoint.pt")
    sh([PY, os.path.join(HERE, "train.py"),
        "--name", "tlw_fno_a025_s%d" % seed, "--model", "fno",
        "--init", init,
        "--data", WIDE, "--epochs", "55", "--batch", "64",
        "--lr", "0.001", "--width", "40", "--modes", "20",
        "--layers", "4", "--proj", "128", "--norm", "layer",
        "--n-train", str(N_TRAIN),
        "--seed", str(seed)])


def evaluate():
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "tlw_fno_a025_*", "checkpoint.pt")])


if __name__ == "__main__":
    for s in SEEDS:
        finetune_warm(s)
    evaluate()
    print("\nC4b chain complete. Read: analyze_c4.py (tl arm) and the tlw arm.",
          flush=True)
