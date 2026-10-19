# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""BA1: the Born-anchored residual operator.

Why this arm exists.  C3b (eb_fno) was the decisive measurement of the
campaign: forcing the out-of-band correction ENERGY onto the first Born field
recovers the energy ratio (0.122 -> 1.596 at f = 12) while the alignment
collapses to 0.012 and the error gets worse.  The trained operator does not
lack out-of-band energy; it lacks out-of-band PHASE.  The Born series is the
one object in this problem that is analytic at any frequency and carries the
phase by construction, so the natural remedy is to stop asking the network to
invent that part and give it instead:

    u_hat = a B1 + b B2 + r_theta(v, psi_inc, f, B1, B2)

with (a, b) two complex coefficients -- four real numbers -- trained on the
band only.  B1 and B2 are the first two Born corrections of the incident field
(code/data.py:anchor_terms), computable by FFT at any frequency with no solve
and no label.

The zero-GPU pre-screen (code/prescreen_anchor.py, code/anchor_baseline.py)
already fixed the arm's FLOOR, on the manuscript's own evaluation protocol:

    v_max=0.25   in-band 0.338 | e(4) 0.282 e(5) 0.280 | plateau 0.300

against the trained FNO's 0.0716 / 0.963 and the trivial baseline's 0.901.
Note the ratio e(5)/e(4) = 0.99: the anchor alone has no cliff at all, while
the trained operator carries r = 7.48.  The two are complementary -- the
network wins in band by 4.7x, the anchor wins out of band by 3.2x -- and the
additive form is what lets one arm hold both: if the learned residual collapses
out of band (the failure C3b measured), what remains is the anchor, not the
incident field.

Protocol: identical to the baseline arm main_fno_a025 in every respect -- same
cache (train_a025, seed0 = 4000), same deterministic prefix (n_train = 12288),
same hyperparameters (width 40, modes 20, layers 4, proj 128, norm layer,
batch 64, 55 epochs), same 3 seeds.  The differences are exactly two: four
extra input channels (Re/Im of B1 and B2) and four extra real parameters
(a, b), +0.000% of the parameter count.

Pre-registered reading:
  * PASS      : out-of-band mean error < 0.6 AND r = e(5)/e(4) < 2 -- the
                residual transfers, the remedy is a method;
  * PARTIAL   : out-of-band error at or below the anchor floor (0.300) but
                not better -- the residual neither transfers nor harms; the
                anchor is the remedy and the residual is a negative result;
  * FAIL      : out-of-band error WORSE than the anchor floor -- the residual
                adds out-of-band noise on top of a phase-correct anchor; report
                it as such and scope the claim to the anchor alone.

Strictly serial (one training at a time), per the project's GPU discipline.
"""
import os
import sys
import json
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
    """The baseline arm trained on data/train_a025 with --n-train 12288; the
    anchored arm must read the SAME prefix, or the comparison stops being
    paired and the whole reading is void."""
    cfgp = os.path.join(C.RUNS, "main_fno_a025_s0", "config.json")
    if os.path.exists(cfgp):
        cfg = json.load(open(cfgp))
        if os.path.normpath(cfg["data"]) != os.path.normpath(CACHE):
            raise SystemExit("baseline arm cache moved: %s" % cfg["data"])
        print("pairing: ba arm reads the baseline arm's cache %s (n_train=%d)"
              % (CACHE, cfg.get("n_train")))
    ab = os.path.join(C.RES, "anchor_baseline.json")
    if not os.path.exists(ab):
        raise SystemExit("missing %s -- run code/anchor_baseline.py first, "
                         "the arm's floor is scored against it" % ab)


def train(seed):
    sh([PY, os.path.join(HERE, "train.py"),
        "--name", "ba_fno_a025_s%d" % seed, "--model", "fno",
        "--anchor", "born",
        "--data", CACHE, "--epochs", "55", "--batch", "64",
        "--lr", "0.001", "--width", "40", "--modes", "20",
        "--layers", "4", "--proj", "128", "--norm", "layer",
        "--n-train", str(N_TRAIN),
        "--seed", str(seed)])


def evaluate():
    sh([PY, os.path.join(HERE, "evaluate.py"), "--all",
        "--pattern", os.path.join(C.RUNS, "*ba_fno*", "checkpoint.pt")])


if __name__ == "__main__":
    # Invoking this file starts a three-seed GPU chain immediately, which is
    # what it is for -- and also how it got started by accident once, on a GPU
    # another job already held (the run was killed mid-epoch and its partial
    # run directory had to be deleted).  Require an explicit token.
    if "--go" not in sys.argv:
        raise SystemExit("run_ba1.py starts a 3-seed GPU chain. Pass --go to "
                         "confirm the GPU is free and this is intentional.")
    verify_pairing()
    for s in SEEDS:
        train(s)
    evaluate()
    print("\nBA1 chain complete. Read: results/eval_ba_fno_a025_s*.json against "
          "results/eval_main_fno_a025_s*.json (same protocol) and against the "
          "anchor floor in results/anchor_baseline.json.", flush=True)
