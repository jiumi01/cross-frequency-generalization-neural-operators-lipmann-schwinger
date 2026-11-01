# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""BA2-FRZ: the anchored residual operator with the coefficients FROZEN at
their in-band least-squares values.

Why this arm exists.  P1-a was pre-registered as "residual band-limited
projection" -- u = u_head + Pi_R r_theta, six trainings -- to test whether the
residual carries the out-of-band failure of the anchored construction at
v >= 0.50.  Two zero-GPU probes falsified that hypothesis before any training:

  * code/proj_precheck.py -- projecting the trained residual to ANY radius,
    including zero, leaves the plateau at v=0.50 between 19.4 and 21.4
    (as trained: 18.6).  The residual is not the carrier.
  * code/anchor_freeze_check.py -- the LS-fitted head ALONE holds the plateau
    at v=0.50 to 0.51 (v=1.00: 1.23) while the FINE-TUNED head alone blows it
    to 21.3 in band 7.1.  The in-band fine-tune turns the analytic head into
    a band-specific cancellation the residual must then undo; out of band the
    cancellation fails.

So the single changed variable here is not the architecture but the
coefficients: same FNO, same head depth M_sel, same caches, same schedule as
the ba_fno_* arms -- only the coefficients stay at their LS values and the
network learns the residual against an HONEST head instead of against a
mis-fitted one it has to cancel.

Pre-registered reading, per contrast (v = 0.50 and v = 1.00), against the
formal eval archives of the same protocol:

  * PASS    : in-band <= the trained FNO's AND plateau <= 0.5 x the trained
              FNO's -- the construction, with frozen LS coefficients, extends
              to full contrast; the manuscript's BA2 boundary moves;
  * PARTIAL : plateau <= the trained FNO's but in-band worse -- the frozen
              head is safe out of band but the residual cannot recover the
              in-band fit the fine-tuned coefficients were buying;
  * FAIL    : plateau worse than 1.5 x the LS head alone (0.512 at v=0.50,
              1.228 at v=1.00, anchor_freeze_check variant C) -- even with an
              honest head the residual contaminates out of band.

Strictly serial (one training at a time), per the project's GPU discipline.

    python code/run_ba2_frz.py
"""
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# same arms as run_ba2.py: (tag, cache, head depth, channels to the network)
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
    sel = os.path.join(C.RES, "anchor_order_selection.json")
    if not os.path.exists(sel):
        raise SystemExit("missing %s -- the head depth is not a free dial"
                         % sel)
    pre = os.path.join(C.RES, "proj_precheck.json")
    frz = os.path.join(C.RES, "anchor_freeze_check.json")
    for p in (pre, frz):
        if not os.path.exists(p):
            raise SystemExit("missing %s -- the go decision for this chain "
                             "rests on the zero-GPU probes; refusing to run "
                             "without them" % p)
    print("probes present; head-depth archive present")


def train(tag, cache, m, seed, channels=None):
    cmd = [PY, os.path.join(HERE, "train.py"),
           "--name", "baf_fno_%s_m%d_s%d" % (tag, m, seed), "--model", "fno",
           "--anchor", "born", "--anchor-order", str(m),
           "--anchor-freeze"]
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
        "--pattern", os.path.join(C.RUNS, "*baf_fno_*_m*", "checkpoint.pt")])


if __name__ == "__main__":
    verify()
    t0 = time.time()
    for tag, cache, m, ch in ARMS:
        for seed in SEEDS:
            train(tag, cache, m, seed, ch)
    evaluate()
    print("\nBA2-FRZ chain complete.  Read results/eval_baf_fno_*.json "
          "against results/eval_ba_fno_*.json and the anchor-alone numbers "
          "in results/anchor_freeze_check.json.")
    print("total %.0f min" % ((time.time() - t0) / 60))
