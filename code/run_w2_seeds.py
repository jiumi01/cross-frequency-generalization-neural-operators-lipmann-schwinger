# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""W2: extend every headline arm from three to six seeds.

Review round 10 (external pre-submission review) asks that the arms the core
conclusions rest on carry 5-10 seeds rather than three.  W1 already extended
the v=1.00 pair (main + bax); this chain completes the set, so that after it
every arm whose numbers the manuscript quotes as a claim has six seeds:

    main_fno_a005/a010/a025/a050   the paired FNO cliff arms (all five
                                   contrasts; a100 already at six from W1)
    ba_fno_a025_m8                the M=8 head+residual arm whose 0.005/0.026
                                   headline numbers are quoted in Section 5.4
    ba_fno_a025                   the M=2 arm of the "two-term head does not
                                   pay for its residual" negative result
    bax_fno_a050_m8               the corrected v=0.50 arm whose PASS verdict
                                   the abstract and contributions quote

The pre-correction arms (ba_fno_a050_m8, ba_fno_a100_m7) stay at three seeds
on purpose: they are the withdrawn as-run reading, quoted once as history in
the implementation-defect paragraph, and extending a withdrawn configuration
would spend GPU time on numbers no claim rests on.

Hyper-parameters are replicated byte-for-byte from the existing runs'
config.json (epochs 55, batch 64, lr 1e-3, width 40, modes 20, layers 4,
proj 128, layer norm, n_train 12288, p 128, base 24; anchor flags per arm).

Discipline, inherited from W1 and from the project's GPU rules:
  * strictly serial -- one training or evaluation job at a time.  A concurrent
    evaluation does not merely run slower; it can time out and silently drop a
    seed, which is how a three-seed reading gets mistaken for a six-seed one.
  * the chain WAITS for the GPU to be idle before it starts.  Another job
    (the user's wirenet campaign) held the device when this was written; the
    wait below is sustained-idle, so a momentary dip cannot start a collision.
  * idempotent -- a training run whose runs/<name>/result.json exists is
    skipped, and an evaluation whose results/eval_<name>.json exists is
    skipped, so an interrupted chain resumes where it stopped.

    python code/run_w2_seeds.py                 # wait for idle GPU, then run
    python code/run_w2_seeds.py --no-wait       # require an idle GPU now
    python code/run_w2_seeds.py --max-wait 0    # same as --no-wait
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
RES = C.RES
RUNS = C.RUNS
LOG = os.path.join(RES, "_w2_seeds_chain.log")

SEEDS = (3, 4, 5)          # the seeds to ADD (0,1,2 already exist)
EPOCHS = 55
N_TRAIN = 12288
COMMON = ["--epochs", str(EPOCHS), "--batch", "64", "--lr", "0.001",
          "--width", "40", "--modes", "20", "--layers", "4", "--proj", "128",
          "--norm", "layer", "--n-train", str(N_TRAIN), "--p", "128",
          "--base", "24"]

# name template -> (data cache, extra train.py flags)
ARMS = {
    "main_fno_a005_s%d": ("train_a005", []),
    "main_fno_a010_s%d": ("train_a010", []),
    "main_fno_a025_s%d": ("train_a025", []),
    "main_fno_a050_s%d": ("train_a050", []),
    "ba_fno_a025_s%d": ("train_a025", ["--anchor", "born"]),
    "ba_fno_a025_m8_s%d": ("train_a025",
                           ["--anchor", "born", "--anchor-order", "8"]),
    "bax_fno_a050_m8_s%d": ("train_a050", ["--anchor", "born",
                                           "--anchor-order", "8",
                                           "--anchor-channels", "4"]),
}
# the arms whose six-seed read the manuscript quotes, for the closing report
REPORT = [("main_fno_a025", None), ("main_fno_a050", None),
          ("ba_fno_a025_m8", None), ("bax_fno_a050_m8", 8)]


def log(msg):
    line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gpu_used_mib():
    """Total GPU memory in use by compute processes, or None if unknown."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().splitlines()[0].split()[0])
    except Exception:
        return None


def wait_for_idle_gpu(limit_mib=4000, sustain_checks=5, interval=120,
                      max_wait_s=8 * 3600):
    """Block until the GPU has been comfortably idle for a sustained period.

    Five consecutive two-minute readings below the limit is ten minutes of
    sustained idleness: long enough that a momentary dip between two jobs of
    the user's own campaign cannot be mistaken for a free device.
    """
    t0 = time.time()
    idle = 0
    while True:
        used = gpu_used_mib()
        if used is None:
            log("GPU wait: nvidia-smi unreadable; treating as busy")
            idle = 0
        elif used < limit_mib:
            idle += 1
            log("GPU wait: %d MiB used (%d/%d sustained-idle checks)"
                % (used, idle, sustain_checks))
            if idle >= sustain_checks:
                log("GPU idle for %d min; starting the chain"
                    % (sustain_checks * interval // 60))
                return True
        else:
            if idle or used >= limit_mib:
                log("GPU wait: %d MiB used (busy)" % used)
            idle = 0
        if time.time() - t0 > max_wait_s:
            log("GPU wait: still busy after %.1f h; aborting cleanly. "
                "Re-run this script when the device frees."
                % (max_wait_s / 3600.0))
            return False
        time.sleep(interval)


def sh(cmd):
    log("+ " + " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd)
    log("[%.0fs] exit=%d" % (time.time() - t0, r.returncode))
    if r.returncode != 0:
        raise SystemExit("FAILED: " + " ".join(cmd))


def finished_train(name):
    return os.path.exists(os.path.join(RUNS, name, "result.json"))


def finished_eval(name):
    return os.path.exists(os.path.join(RES, "eval_%s.json" % name))


def train_all():
    for tmpl, (cache, extra) in ARMS.items():
        for s in SEEDS:
            name = tmpl % s
            if finished_train(name):
                log("skip train %s (result.json present)" % name)
                continue
            cmd = [PY, os.path.join(HERE, "train.py"), "--name", name,
                   "--data", os.path.join(C.DATA, cache)] + COMMON + extra \
                + ["--seed", str(s)]
            sh(cmd)
            if not finished_train(name):
                raise SystemExit("train %s produced no result.json" % name)


def eval_all():
    for tmpl, _ in ARMS.items():
        for s in SEEDS:
            name = tmpl % s
            if finished_eval(name):
                log("skip eval %s (archive present)" % name)
                continue
            ck = os.path.join(RUNS, name, "checkpoint.pt")
            if not os.path.exists(ck):
                raise SystemExit("no checkpoint for %s" % name)
            sh([PY, os.path.join(HERE, "evaluate.py"),
                "--ckpt", ck, "--name", name])


def _plateau(d, vmax):
    rows = d["sweep"]["%.2f" % vmax]
    return float(np.mean([r["mean"] for r in rows if r["f"] >= 6.0]))


def _inband(d, vmax):
    return next(float(r["mean"]) for r in d["in_band"]
                if abs(r["vmax"] - vmax) < 1e-6)


def report():
    """Six-seed read of the arms the manuscript quotes, for the record."""
    ALL = (0, 1, 2) + SEEDS
    for tmpl, _ in REPORT:
        tag = tmpl.split("_")[2]          # "main_fno_a025_s%d" -> "a025"
        vmax = {"a005": 0.05, "a010": 0.10, "a025": 0.25, "a050": 0.50,
                "a100": 1.00}[tag]
        ib, pl = [], []
        for s in ALL:
            d = json.load(open(os.path.join(RES, "eval_%s.json" % (tmpl % s))))
            ib.append(_inband(d, vmax))
            pl.append(_plateau(d, vmax))
        log("%s six-seed in-band mean=%.4f (sd %.4f) plateau mean=%.4f "
            "(sd %.4f)" % (tmpl % "*", np.mean(ib), np.std(ib, ddof=1),
                           np.mean(pl), np.std(pl, ddof=1)))
        log("   per-seed in-band %s" % [round(x, 4) for x in ib])
        log("   per-seed plateau %s" % [round(x, 4) for x in pl])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true",
                    help="require an idle GPU now instead of waiting")
    ap.add_argument("--max-wait", type=float, default=8 * 3600,
                    help="seconds to wait for an idle GPU (0 = do not wait)")
    a = ap.parse_args()
    t0 = time.time()
    log("=== W2: headline arms 3 -> 6 seeds ===")
    if not a.no_wait and a.max_wait > 0:
        if not wait_for_idle_gpu(max_wait_s=a.max_wait):
            return
    elif gpu_used_mib() is not None and gpu_used_mib() >= 4000:
        raise SystemExit("GPU busy (%.0f MiB used); rerun with the wait "
                         "enabled" % gpu_used_mib())
    log("=== training pass (serial) ===")
    train_all()
    log("=== evaluation pass (serial) ===")
    eval_all()
    log("=== six-seed read ===")
    report()
    log("W2 done in %.0f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()
