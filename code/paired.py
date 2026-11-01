# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Paired significance tests for the architecture / ablation comparisons.

Why this file exists
--------------------
The benchmark is *paired by construction*: the media drawn for a given
(v_max, frequency) cell are the same physical samples for every architecture,
because the draw is seeded only by the cell, not by the model.  Reporting two
means and two standard deviations therefore discards the very structure the
design was built to exploit.

Two consequences:

1. The natural test is on the *differences* d_i = e_i^A - e_i^B over the paired
   samples i, not on the two marginal distributions.  Pairing removes the
   medium-to-medium variance, which dominates here (the per-sample error spans
   more than a decade at fixed frequency), and is strictly more powerful.

2. The reported quantity should be a confidence interval on the *effect size*,
   not a bare "significant / not significant" verdict, because the useful
   question -- "is the U-Net plateau lower by more than seed noise?" -- is a
   magnitude question.

Implements a paired bootstrap over media (resampling the sample index, so the
same indices are used for both models in every resample) and a paired
sign-flip permutation test.  Reads the ``percell_*.npz`` written by
``evaluate.py``.

Usage
-----
    python code/paired.py --a base_unet_a025_s0 --b main_fno_a025_s0
    python code/paired.py --a base_unet_a100_s0 --b main_fno_a100_s0 --vmax 1.00
"""
import os
import sys
import glob
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RNG = np.random.default_rng(20261011)


def load_percell(name):
    p = os.path.join(C.RES, "percell_%s.npz" % name)
    if not os.path.exists(p):
        return None
    return dict(np.load(p))


def cell(d, vmax, f):
    """Per-sample error vector for one (vmax, f) cell."""
    vi = int(np.argmin(np.abs(d["vmax"] - vmax)))
    if f is None:
        return d["per_ib"][vi]
    fi = int(np.argmin(np.abs(d["f_test"] - f)))
    return d["per"][vi, fi]


def paired_bootstrap(a, b, n_boot=20000, alpha=0.05):
    """CI on mean(a - b) resampling the SAMPLE index (keeps the pairing)."""
    d = a - b
    n = d.size
    idx = RNG.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def paired_permutation(a, b, n_perm=20000):
    """Two-sided sign-flip permutation test on the paired differences."""
    d = a - b
    obs = abs(d.mean())
    signs = RNG.integers(0, 2, size=(n_perm, d.size)) * 2 - 1
    null = np.abs((signs * d).mean(axis=1))
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


def cluster_bootstrap(unit, n_boot=20000, alpha=0.05):
    """CI on the mean of ``unit``, resampling the CLUSTER index.

    ``unit`` holds one value per independent paired unit -- the per-medium mean
    of all (seed, cell) replicates of that medium.  Resampling the units is what
    makes the interval honest when several replicate rows share a unit; feeding
    the raw rows to ``paired_bootstrap`` instead treats near-duplicates as
    independent draws and returns an interval that is too narrow.
    """
    unit = np.asarray(unit, float)
    n = unit.size
    idx = RNG.integers(0, n, size=(n_boot, n))
    boots = unit[idx].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(unit.mean()), float(lo), float(hi)


def cluster_signflip(unit, n_perm=20000):
    """Two-sided sign-flip permutation test at the cluster level.

    The sign is drawn once per unit and applied to that unit's value, so an
    exchangeability assumption is made between units and not between the
    replicates that share one.
    """
    unit = np.asarray(unit, float)
    obs = abs(unit.mean())
    signs = RNG.integers(0, 2, size=(n_perm, unit.size)) * 2 - 1
    null = np.abs((signs * unit).mean(axis=1))
    p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
    return obs, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run name (model A)")
    ap.add_argument("--b", required=True, help="run name (model B)")
    ap.add_argument("--vmax", type=float, default=0.25)
    ap.add_argument("--freqs", default=None,
                    help="comma separated; default = all test frequencies")
    a = ap.parse_args()

    da, db = load_percell(a.a), load_percell(a.b)
    if da is None or db is None:
        print("missing percell file(s): run evaluate.py --all first")
        return 1
    if not np.allclose(da["f_test"], db["f_test"]):
        print("the two runs were scored on different grids -- refusing to pair")
        return 1

    freqs = ([float(x) for x in a.freqs.split(",")] if a.freqs
             else list(da["f_test"]))
    print("paired test  A=%s  B=%s  v_max=%.2f" % (a.a, a.b, a.vmax))
    print("  difference is mean(e_A - e_B); negative => A is better")
    print("  %8s %10s %10s %22s %10s" %
          ("f", "mean(A)", "mean(B)", "95% CI on mean diff", "p (sign-flip)"))

    verdict_better = 0
    for f in freqs:
        ea, eb = cell(da, a.vmax, f), cell(db, a.vmax, f)
        m, lo, hi = paired_bootstrap(ea, eb)
        _, p = paired_permutation(ea, eb)
        sig = "" if (lo <= 0 <= hi) else "  *"
        if hi < 0:
            verdict_better += 1
        print("  %8.2f %10.3f %10.3f   [%+8.4f, %+8.4f] %10.2e%s"
              % (f, ea.mean(), eb.mean(), lo, hi, p, sig))
    print("  A is significantly better (95%% CI entirely below 0) at "
          "%d/%d frequencies" % (verdict_better, len(freqs)))

    # in-band cell, the comparison the manuscript quotes most often
    ea, eb = cell(da, a.vmax, None), cell(db, a.vmax, None)
    m, lo, hi = paired_bootstrap(ea, eb)
    _, p = paired_permutation(ea, eb)
    print("  in-band  A=%.3f  B=%.3f  diff %+.4f [%+.4f, %+.4f]  p=%.2e"
          % (ea.mean(), eb.mean(), m, lo, hi, p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
