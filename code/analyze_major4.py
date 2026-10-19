# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China
#
# Reproducible computation of the two minimal tables requested by the external
# reviewer (Major #4):
#   * Section 5.10 (robustness to the propagator loss tangent gamma):
#       gamma sweep at v_max = 0.25, three loss tangents {0.02, 0.05, 0.10}.
#   * Section 6.3 (mixed-contrast control):
#       mixed-contrast deployment model vs single-contrast baseline.
#
# Metric definitions mirror make_results_tex.py:
#   band-edge ratio  = e(f=5) / e(f=4) at v_max = 0.25  (cliff sharpness)
#   out-of-band plateau (own/reference arm) = mean error over f >= 6 at v_max = 0.25
#   out-of-band plateau across 5 contrasts  = mean over the five arms of the
#                                             f >= 6 plateau of each arm
# Every scalar is computed per seed and reported as mean +/- std with n = 3 seeds.

import json
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ARMS = [0.05, 0.10, 0.25, 0.50, 1.00]


def load_run(prefix, seed):
    p = os.path.join(RES, "eval_%s_s%d.json" % (prefix, seed))
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def mean_at(d, vmax, f):
    for r in d["sweep"]["%.2f" % vmax]:
        if abs(r["f"] - f) < 1e-9:
            return r["mean"]
    return np.nan


def plateau_at(d, vmax, fmin=6.0):
    vals = [r["mean"] for r in d["sweep"]["%.2f" % vmax] if r["f"] >= fmin]
    return float(np.mean(vals)) if vals else np.nan


def inband_at(d, vmax, fmax=4.0):
    vals = [r["mean"] for r in d["sweep"]["%.2f" % vmax] if r["f"] <= fmax]
    return float(np.mean(vals)) if vals else np.nan


def arm_plateaus(d, fmin=6.0):
    return np.array([plateau_at(d, vm, fmin) for vm in ARMS])


def per_seed_scalars(prefix):
    rows = []
    for s in (0, 1, 2):
        d = load_run(prefix, s)
        if d is None:
            rows.append(None)
            continue
        ratio25 = mean_at(d, 0.25, 5.0) / mean_at(d, 0.25, 4.0)
        plat25 = plateau_at(d, 0.25)
        plat5 = np.mean([plateau_at(d, vm) for vm in ARMS])
        inb = inband_at(d, 0.25)
        ap = arm_plateaus(d)
        rows.append((ratio25, plat25, plat5, inb, ap, d.get("gamma")))
    return rows


def agg(rows, idx):
    good = [r[idx] for r in rows if r is not None]
    return float(np.mean(good)), float(np.std(good, ddof=0))


def agg_arms(rows):
    # per-seed across-arm spread, aggregated across seeds
    good = [r[4] for r in rows if r is not None]
    mn = float(np.min([a.min() for a in good]))
    mx = float(np.max([a.max() for a in good]))
    return mn, mx


def summarize(rows):
    good = [r for r in rows if r is not None]
    if not good:
        return None
    arr = np.array([[r[0], r[1], r[2]] for r in good])
    return dict(
        n=len(good),
        ratio=(float(arr[:, 0].mean()), float(arr[:, 0].std(ddof=0))),
        plat25=(float(arr[:, 1].mean()), float(arr[:, 1].std(ddof=0))),
        plat5=(float(arr[:, 2].mean()), float(arr[:, 2].std(ddof=0))),
        gamma=good[0][3],
    )


def fmt_pair(p):
    return "%.3f +/- %.3f" % (p[0], p[1])


# ---- gamma robustness (Section 5.10) -------------------------------------
gamma_cfgs = [("gamma_g02", 0.02), ("main_fno_a025", 0.05), ("gamma_g10", 0.10)]
print("=" * 78)
print("TABLE for Section 5.10 -- robustness to propagator loss tangent gamma")
print("(v_max = 0.25; n = 3 seeds per gamma)")
print("-" * 78)
print("%-7s %-9s %-16s %-16s %-16s" %
      ("gamma", "norm", "in-band e", "plateau e", "band-edge ratio"))
print("%-7s %-9s %-16s %-16s %-16s" %
      ("", "(1+g^2)/2g", "(f<=4)", "(f>=6)", "e(5)/e(4)"))
gamma_table = []
for prefix, g in gamma_cfgs:
    rows = per_seed_scalars(prefix)
    norm = (1.0 + g * g) / (2.0 * g)
    ratio = agg(rows, 0)
    plat = agg(rows, 1)
    inb = agg(rows, 3)
    print("%-7.2f %-9.3f %-16s %-16s %-16s" %
          (g, norm, fmt_pair(inb), fmt_pair(plat), fmt_pair(ratio)))
    gamma_table.append(dict(gamma=g, norm=norm, inband=inb, plateau=plat,
                            ratio=ratio, n=3))

# ---- mixed-contrast control (Section 6.3) ---------------------------------
print()
print("=" * 78)
print("TABLE for Section 6.3 -- mixed-contrast (deployment) vs single-contrast baseline")
print("(n = 3 seeds per model)")
print("-" * 78)
print("%-26s %-16s %-16s %-16s" %
      ("configuration", "band-edge ratio", "plateau@v=0.25", "plateau x5 arms"))
print("%-26s %-16s %-16s %-16s" %
      ("", "e(5)/e(4)", "(f>=6)", "(f>=6, mean)"))
mixed_table = []
for label, prefix in [("single-contrast (baseline)", "main_fno_a025"),
                      ("mixed-contrast (deployment)", "mixed_fno")]:
    rows = per_seed_scalars(prefix)
    ratio = agg(rows, 0)
    plat25 = agg(rows, 1)
    plat5 = agg(rows, 2)
    amn, amx = agg_arms(rows)
    print("%-26s %-16s %-16s %-16s  [arm range %.3f-%.3f]" %
          (label, fmt_pair(ratio), fmt_pair(plat25), fmt_pair(plat5), amn, amx))
    mixed_table.append(dict(label=label, ratio=ratio, plateau25=plat25,
                            plateau5=plat5, arm_min=amn, arm_max=amx, n=3))

# ---- dump machine-readable summary ---------------------------------------
out = {"gamma_table": gamma_table, "mixed_table": mixed_table}
with open(os.path.join(RES, "major4_tables.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print()
print("wrote results/major4_tables.json")
