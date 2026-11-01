# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""W1: extend the v=1.00 (a100) evidence from 3 to 6 seeds.

The a100 verdict is a PAIRED comparison across the shared seeds of
    main  = main_fno_a100_s<seed>          (baseline FNO, no anchor)
    bax  = bax_fno_a100_m7_s<seed>         (anchored residual, born order 7,
                                            anchor-channels 2 -- the corrected
                                            BA2-X construction)
At three seeds the plateau gain bax<main is b=-0.639 with a paired interval
[-1.411, 0.133] that still contains zero, so the manuscript calls it
"unresolved".  Six seeds (a) shrink the standard error by sqrt(2) and (b) move
the t multiplier from tcrit(2)=4.303 to tcrit(5)=2.571 -- both narrow the
interval, so the question W1 answers is whether it now excludes zero.

Strictly serial, one GPU job at a time (training pass, then evaluation pass) --
the project's discipline is that a concurrent evaluation does not merely run
slower, it can time out and silently drop a seed.  Idempotent: a training run
whose runs/<name>/result.json already exists is skipped, so an interrupted
chain resumes where it stopped.

    python code/run_w1_a100_seeds.py
"""
import os
import sys
import time
import glob
import json
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RES = C.RES
RUNS = C.RUNS

SEEDS = (3, 4, 5)          # the seeds to ADD (0,1,2 already exist)
CACHE = "train_a100"
EPOCHS = 55
N_TRAIN = 12288
# shared hyper-parameters, byte-for-byte from runs/{main_fno_a100_s0,bax_fno_a100_m7_s0}/config.json
COMMON = ["--model", "fno", "--data", os.path.join(C.DATA, CACHE),
          "--epochs", str(EPOCHS), "--batch", "64", "--lr", "0.001",
          "--width", "40", "--modes", "20", "--layers", "4", "--proj", "128",
          "--norm", "layer", "--n-train", str(N_TRAIN)]

ARMS = {
    "main": {"name": "main_fno_a100_s%d", "extra": []},
    "bax": {"name": "bax_fno_a100_m7_s%d",
            "extra": ["--anchor", "born", "--anchor-order", "7",
                      "--anchor-channels", "2"]},
}


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sh(cmd):
    log("+ " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd)
    log("[%.0fs] exit=%d" % (time.time() - t0, r.returncode))
    if r.returncode != 0:
        raise SystemExit("FAILED: " + " ".join(cmd))


def finished(name):
    return os.path.exists(os.path.join(RUNS, name, "result.json"))


def train_arm(arm):
    spec = ARMS[arm]
    for s in SEEDS:
        name = spec["name"] % s
        if finished(name):
            log("skip train %s (result.json present)" % name)
            continue
        cmd = [PY, os.path.join(HERE, "train.py"), "--name", name] + COMMON \
            + spec["extra"] + ["--seed", str(s)]
        sh(cmd)
        if not finished(name):
            raise SystemExit("train %s produced no result.json" % name)


def eval_arm(arm):
    spec = ARMS[arm]
    for s in SEEDS:
        name = spec["name"] % s
        ck = os.path.join(RUNS, name, "checkpoint.pt")
        if not os.path.exists(ck):
            raise SystemExit("no checkpoint for %s" % name)
        # single-run mode: --all absent -> names = [(name, ckpt)];
        # _fresh() skips only if a matching archive already exists.
        sh([PY, os.path.join(HERE, "evaluate.py"),
            "--ckpt", ck, "--name", name])


def _band_plateau(d, vmax=1.00):
    rows = d["sweep"]["%.2f" % vmax]
    return float(np.mean([r["mean"] for r in rows if r["f"] >= 6.0]))


def _band_inband(d, vmax=1.00):
    return next(float(r["mean"]) for r in d["in_band"]
                if abs(r["vmax"] - vmax) < 1e-6)


def paired(b, a):
    """Paired t interval on b - a; tcrit is df-dependent (scipy)."""
    from scipy import stats
    d = np.asarray(b) - np.asarray(a)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    t = float(stats.t.ppf(0.975, n - 1))
    half = t * se
    return mean, [mean - half, mean + half], n, t


def report():
    ALL = (0, 1, 2) + SEEDS
    main_pl, bax_pl, main_ib, bax_ib = [], [], [], []
    for s in ALL:
        m = json.load(open(os.path.join(RES, "eval_main_fno_a100_s%d.json" % s)))
        b = json.load(open(os.path.join(RES, "eval_bax_fno_a100_m7_s%d.json" % s)))
        main_pl.append(_band_plateau(m)); bax_pl.append(_band_plateau(b))
        main_ib.append(_band_inband(m)); bax_ib.append(_band_inband(b))
    log("a100 per-seed plateau  main=%s" % [round(x, 4) for x in main_pl])
    log("a100 per-seed plateau  bax =%s" % [round(x, 4) for x in bax_pl])
    log("a100 per-seed in-band  main=%s" % [round(x, 4) for x in main_ib])
    log("a100 per-seed in-band  bax =%s" % [round(x, 4) for x in bax_ib])
    dmean, ci, n, t = paired(bax_pl, main_pl)
    log("a100 PLATEAU  bax-main  n=%d  tcrit=%.3f  diff=%.4f  CI=[%.4f, %.4f]  %s"
        % (n, t, dmean, ci[0], ci[1],
           "EXCLUDES zero -> RESOLVED" if not (ci[0] <= 0 <= ci[1])
           else "contains zero -> still unresolved"))
    dmean2, ci2, _, _ = paired(bax_ib, main_ib)
    log("a100 IN-BAND  bax-main  diff=%.4f  CI=[%.4f, %.4f]  %s"
        % (dmean2, ci2[0], ci2[1],
           "excludes zero" if not (ci2[0] <= 0 <= ci2[1]) else "contains zero"))
    # verdict per the pre-registered rule (means over all seeds)
    m_pl, b_pl, m_ib, b_ib = np.mean(main_pl), np.mean(bax_pl), np.mean(main_ib), np.mean(bax_ib)
    if b_ib <= m_ib and b_pl <= 0.5 * m_pl:
        v = "PASS"
    elif b_pl <= m_pl:
        v = "PARTIAL"
    else:
        v = "FAIL/other"
    log("a100 verdict from means: bax_ib=%.4f vs main_ib=%.4f ; bax_pl=%.4f vs main_pl=%.4f -> %s"
        % (b_ib, m_ib, b_pl, m_pl, v))


if __name__ == "__main__":
    t0 = time.time()
    log("=== W1: a100 3 -> 6 seeds (train pass, serial) ===")
    for arm in ("main", "bax"):
        train_arm(arm)
    log("=== W1: evaluation pass (serial) ===")
    for arm in ("main", "bax"):
        eval_arm(arm)
    log("=== W1: 6-seed paired read ===")
    report()
    log("W1 done in %.0f min" % ((time.time() - t0) / 60))
