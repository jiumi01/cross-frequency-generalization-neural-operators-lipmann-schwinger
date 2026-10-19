# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Generate every training cache needed by the experiment grid (resumable).

All caches share the SAME (v_max, theta, v) draws for a given seed; only the
frequency band differs, so band-ablation runs are paired.
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D

NPER = 16384

# Frequency band is the ONLY variable inside each arm; contrast arms share the
# same media draws (pairing), so arm-to-arm differences are contrast-only.
ARMS = [
    # (folder, f_lo, f_hi, vmax_lo, vmax_hi, n, seed0)
    # --- mixed-contrast main model -----------------------------------------
    ("train_b1_4", 1.0, 4.0, 0.05, 1.00, NPER, 1000),
    # --- fixed-contrast arms (fresh scatterers: seed0=4000) ----------------
    ("train_a005", 1.0, 4.0, 0.05, 0.05, NPER, 4000),
    ("train_a010", 1.0, 4.0, 0.10, 0.10, NPER, 4000),
    ("train_a025", 1.0, 4.0, 0.25, 0.25, NPER, 4000),
    ("train_a050", 1.0, 4.0, 0.50, 0.50, NPER, 4000),
    ("train_a100", 1.0, 4.0, 1.00, 1.00, NPER, 4000),
    # --- band-width ablation at fixed contrast v_max = 0.25 ----------------
    ("train_b025_2", 1.0, 2.0, 0.25, 0.25, NPER, 4000),
    ("train_b025_3", 1.0, 3.0, 0.25, 0.25, NPER, 4000),
    ("train_b025_6", 1.0, 6.0, 0.25, 0.25, NPER, 4000),
    ("train_b025_12", 1.0, 12.0, 0.25, 0.25, NPER, 4000),
    # --- propagator-loss robustness arms -----------------------------------
    ("train_g02", 1.0, 4.0, 0.25, 0.25, NPER, 4000),
    ("train_g10", 1.0, 4.0, 0.25, 0.25, NPER, 4000),
    # --- C1: per-frequency loss tangent (gamma(f) flattens the attained
    # operator norm over the whole test grid); paired with train_a025 ------
    ("train_gc", 1.0, 4.0, 0.25, 0.25, NPER, 4000),
]
GAMMA_ARM = {"train_g02": 0.02, "train_g10": 0.10}
# Arms whose regularisation is a function of the per-sample frequency rather
# than a constant.  Resolved lazily so this module stays importable without
# the compensation table present (gen_caches runs before gamma_comp does).
GAMMA_FN_ARM = {"train_gc": "comp_gamma"}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=460.0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import time
    t0 = time.time()
    for (name, flo, fhi, vlo, vhi, n, s0) in ARMS:
        folder = os.path.join(C.DATA, name)
        have = len([f for f in os.listdir(folder) if f.endswith(".npz")]) \
            if os.path.isdir(folder) else 0
        need = int((n + 511) // 512)
        if have >= need:
            print("skip %s (%d/%d chunks present)" % (name, have, need))
            continue
        left = a.budget - (time.time() - t0)
        if left <= 5:
            print("budget exhausted; rerun to continue")
            break
        print("--- %s  f=[%.1f,%.1f] vmax=[%.2f,%.2f] n=%d  (left %.0fs)"
              % (name, flo, fhi, vlo, vhi, n, left))
        gam = GAMMA_ARM.get(name, None)
        if name in GAMMA_FN_ARM:
            import gamma_comp
            gam = getattr(gamma_comp, GAMMA_FN_ARM[name])
        D.build_train_cache(n, 512, seed0=s0, out_dir=folder, device=dev,
                            f_lo=flo, f_hi=fhi, vmax_lo=vlo, vmax_hi=vhi,
                            budget=left, gamma=gam)
    print("elapsed %.0fs" % (time.time() - t0))
