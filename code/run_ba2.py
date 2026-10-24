# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""BA2: the anchored residual operator with a DEEP analytic head.

BA1 (code/run_ba1.py) answered its pre-registered question and the answer was
honest but not the one hoped for.  At v=0.25:

    trained FNO       in-band 0.0854   plateau 0.9631   r = 7.48
    BA1 (M=2 head)    in-band 0.0275   plateau 0.4829   r = 5.20
    anchor alone M=2                   plateau 0.3003

So the learned residual buys a factor 2.0 on the plateau against the trained
operator and a factor 3.1 in band, but it is still WORSE out of band than the
analytic head on its own (0.483 vs 0.300): the residual adds out-of-band error
on top of a phase-correct anchor.  Read as the manuscript reads its other
negative results, that is a real finding -- but it is also, in retrospect, an
artefact of a head that was too shallow.

code/anchor_higher_order.py changed the picture.  The anchor's plateau is Born
TRUNCATION error, and it falls geometrically with the head depth M:

    v=0.25   0.517 (M=1)  0.300 (M=2)  0.165 (M=3)  0.104 (M=4)
             0.064 (M=5)  0.044 (M=6)  0.029 (M=7)  0.022 (M=8)

against the trained operator's 0.963.  And the depth is not a dial to be
maximised blindly -- at v=1.00 the curve turns (0.516 at M=5, 0.517 at M=6)
because the Born contraction factor rho = ||B_{j+1}||/||B_j|| exceeds one
there and the fit becomes an asymptotic summation on a divergent series.
code/anchor_order_selection.py then showed that the depth can be chosen
IN BAND: fitting on one in-band prefix and validating on a disjoint in-band
prefix selects, at all five contrasts, exactly the order that is optimal out
of band.  M_sel = 8, 8, 8, 8, 7 for v = 0.05 ... 1.00.

BA2 is therefore the arm the reading actually points at: head depth M_sel,
coefficients fitted in band (and fine-tuned there), network learning only the
residual -- which is now a SMALL object, because the head has already
accounted for almost everything that is analytic in the frequency.

    u_hat = sum_{j=1..M} c_j B_j  +  r_theta(v, psi_inc, f, B_1..B_M)

If the residual is small, its out-of-band misbehaviour is proportionally
small, and the arm should keep the anchor's plateau while recovering the
in-band accuracy the anchor alone cannot reach at high contrast.

Pre-registered reading at each contrast (v=0.25 and v=1.00 are run here):
  * PASS    : in-band <= the trained FNO's AND plateau <= 0.5 x the trained
              FNO's -- the arm is simultaneously the best in band and far the
              best out of band; the construction is a method;
  * PARTIAL : plateau <= the trained FNO's but in-band worse -- the head
              dominates and the residual does not earn its parameters;
  * FAIL    : plateau worse than the anchor alone by more than 1.5x -- the
              residual still contaminates out of band even when small.

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

# (arm tag, cache, head depth) -- depth from results/anchor_order_selection.json
# (arm tag, cache, head depth, Born terms handed to the network as channels).
# The a100 arm carries M=7 in the head but only B_1, B_2 as input channels:
# at v_max = 1.00 the contraction factor is 4.5, so B_7 is about 8e3 times
# B_1, and feeding it raw sent the first training run to val = 19.  The head
# still sums all seven terms -- the coefficients are applied outside the
# network -- so the analytic content is unchanged.
# a050 was added after the a100 arm FAILED: the residual transfers across
# frequency at v=0.25 (PASS) and destroys the plateau at v=1.00, so the
# boundary of the construction is a contrast and it has to be located, not
# assumed.  At v=0.50 rho = 2.26, so B_8 is roughly 3e2 times B_1; four
# channels are handed to the network rather than eight, which keeps the
# largest channel at about 11 times B_1.
ARMS = [("a025", "train_a025", 8, None),
        ("a050", "train_a050", 8, 4),
        ("a100", "train_a100", 7, 2)]
CACHE_MSEL = {0.05: 8, 0.10: 8, 0.25: 8, 0.50: 8, 1.00: 7}
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
        raise SystemExit("missing %s -- run code/anchor_order_selection.py "
                         "first, the head depth is not a free dial" % sel)
    d = json.load(open(sel))
    got = {float(k): v["M_sel"] for k, v in d["arms"].items()
           if "M_sel" in v}
    for v, m in CACHE_MSEL.items():
        if got.get(v) != m:
            raise SystemExit("M_sel drifted at v=%.2f: recorded %s, script %d"
                             % (v, got.get(v), m))
    for tag, cache, m, _k in ARMS:
        cfgp = os.path.join(C.RUNS, "main_fno_%s_s0" % tag, "config.json")
        if os.path.exists(cfgp):
            cfg = json.load(open(cfgp))
            if os.path.normpath(cfg["data"]) != os.path.normpath(
                    os.path.join(C.DATA, cache)):
                raise SystemExit("baseline arm cache moved: %s" % cfg["data"])
    print("head depths verified against results/anchor_order_selection.json")


def train(tag, cache, m, seed, channels=None):
    cmd = [PY, os.path.join(HERE, "train.py"),
        "--name", "ba_fno_%s_m%d_s%d" % (tag, m, seed), "--model", "fno",
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
        "--pattern", os.path.join(C.RUNS, "*ba_fno_*_m*", "checkpoint.pt")])


if __name__ == "__main__":
    if "--go" not in sys.argv:
        raise SystemExit("run_ba2.py starts a GPU chain (%d runs). Pass --go "
                         "to confirm the GPU is free and this is intentional."
                         % (len(ARMS) * len(SEEDS)))
    verify()
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for tag, cache, m, k in ARMS:
        if only and tag != only:
            continue
        for s in SEEDS:
            train(tag, cache, m, s, k)
    evaluate()
    print("\nBA2 chain complete.", flush=True)
