# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Drive the paired tests and archive them as a table the paper can print.

The benchmark is paired by construction, so every cross-arm comparison in the
manuscript is a comparison of paired differences and should carry a confidence
interval rather than two means side by side.  This script runs the tests that the
prose actually makes and writes the result to ``results/paired.json`` for the
table generator.

Each comparison holds one value per (seed, cell) replicate of each physical
medium, and the inference resamples the MEDIA, with the seeds treated as
replicates inside a medium.  Seeds are independent training runs, but they are
scored on identical media (``evaluate.py`` seeds the evaluation draw by the
contrast cell only), so they are not independent observations of the paired
difference.  See ``pool`` for the measurement that made this concrete.

Read-only with respect to every existing archive.  Run it after
``evaluate.py --all``.
"""
import os
import sys
import glob
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import paired as PT


def seeds_of(kind):
    """Run folder names for a model family, e.g. ``base_unet_a025``."""
    names = []
    for p in sorted(glob.glob(os.path.join(C.RES, "percell_%s_s*.npz" % kind))):
        names.append(os.path.basename(p)[len("percell_"):-len(".npz")])
    return names


def pool(kind, vmax, sel):
    """Per-MEDIUM values for one model family, one row per (seed, cell).

    Returns an array of shape (n_seed * n_cell, n_media), or None.  Columns
    are media and are therefore aligned between any two families: the evaluation
    draw is seeded by the contrast cell only (``evaluate.py``: ``seed=60000+vi``),
    never by the training seed, so every seed of every arm is scored on the same
    physical media.

    That alignment is why the columns, not the flattened entries, are the paired
    units.  An earlier revision concatenated the rows into one long vector and
    resampled from it, which treats the three seeds of an arm as three
    independent replications of each medium.  Measured 2026-10-12: the per-medium
    errors of two seeds of the same arm correlate at 0.98, so the 3x pooling
    inflates the effective sample size and narrows every interval; one row of the
    table (``capw64_vs_fno_a025_in``) then carried a star it does not deserve.
    """
    rows = []
    n_seed = 0
    for name in seeds_of(kind):
        d = PT.load_percell(name)
        if d is None:
            continue
        if not np.allclose(d["f_test"], np.array(C.F_TEST)):
            # A model scored on a different grid cannot be paired with the rest;
            # pairing it silently would compare different frequencies' errors.
            return None
        c = len(d["f_test"])
        vi = int(np.argmin(np.abs(d["vmax"] - vmax)))
        n_seed += 1
        if sel[0] == "in":
            rows.append(np.asarray(d["per_ib"][vi], float)[None, :])
        elif sel[0] == "out":
            keep = d["f_test"] >= 6.0
            rows.append(np.asarray(d["per"][vi, keep], float))
        else:
            fi = int(np.argmin(np.abs(d["f_test"] - sel[1])))
            rows.append(np.asarray(d["per"][vi, fi], float)[None, :])
    if not rows:
        return None
    # Every seed must carry the same number of media; a mismatch means the
    # evaluation protocol itself changed and the columns would not align.
    widths = {r.shape[1] for r in rows}
    if len(widths) != 1:
        return None
    return np.vstack(rows)


def media_mean(M):
    """Mean over the replicate rows of each medium: the paired unit's value.

    The rows within a column are (seed, cell) replicates of the same physical
    medium.  Every row carries the same number of media, so averaging the
    columns of ``M`` and then averaging those is the same point estimate as
    averaging every entry -- but it is the per-media mean that the cluster
    bootstrap and the cluster sign-flip test must resample, and reducing to it
    first keeps the resampling O(n_media) instead of O(n_row x n_media).
    """
    return M.mean(axis=0)


# (label, runA, runB, v_max, selA, selB); runA == runB means a within-model
# comparison, which is how the band-edge jump of a single arm is tested.
CASES = [
    ("unet_base_vs_fno_a025", "base_unet_a025", "main_fno_a025", 0.25,
     ("out",), ("out",)),
    ("unet_pmatch_vs_fno_a025", "pmatch_unet_a025", "main_fno_a025", 0.25,
     ("out",), ("out",)),
    ("unet_base_vs_fno_a100", "base_unet_a100", "main_fno_a100", 1.00,
     ("out",), ("out",)),
    ("unet_pmatch_vs_fno_a100", "pmatch_unet_a100", "main_fno_a100", 1.00,
     ("out",), ("out",)),
    ("capw64_vs_fno_a025", "cap_w64_a025", "main_fno_a025", 0.25,
     ("out",), ("out",)),
    ("capw64_vs_fno_a025_in", "cap_w64_a025", "main_fno_a025", 0.25,
     ("in",), ("in",)),
    ("mode_m02_vs_m20_in", "mode_m02", "main_fno_a025", 0.25,
     ("in",), ("in",)),
    ("jump_fno_a025", "main_fno_a025", "main_fno_a025", 0.25,
     ("f", 5.0), ("f", 4.0)),
    ("jump_fno_a100", "main_fno_a100", "main_fno_a100", 1.00,
     ("f", 5.0), ("f", 4.0)),
    ("jump_unet_base_a025", "base_unet_a025", "base_unet_a025", 0.25,
     ("f", 5.0), ("f", 4.0)),
    ("jump_unet_pmatch_a025", "pmatch_unet_a025", "pmatch_unet_a025", 0.25,
     ("f", 5.0), ("f", 4.0)),
    ("jump_don_a025", "base_don_a025", "base_don_a025", 0.25,
     ("f", 5.0), ("f", 4.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--out", default=os.path.join(C.RES, "paired.json"))
    a = ap.parse_args()

    res = []
    for label, ka, kb, vmax, sa, sb in CASES:
        Ma = pool(ka, vmax, sa)
        Mb = pool(kb, vmax, sb)
        if Ma is None or Mb is None or Ma.shape != Mb.shape:
            print("SKIP %-28s (missing or unpaired)" % label)
            continue
        # The paired unit is the medium; the rows inside a column are the
        # (seed, cell) replicates of it.
        ua, ub = media_mean(Ma), media_mean(Mb)
        m, lo, hi = PT.cluster_bootstrap(ua - ub, n_boot=a.n_boot)
        _, p = PT.cluster_signflip(ua - ub, n_perm=a.n_perm)
        res.append(dict(label=label, a=ka, b=kb, vmax=vmax,
                        sel_a=list(sa), sel_b=list(sb),
                        n=int(Ma.size), n_units=int(ua.size),
                        n_rows=int(Ma.shape[0]),
                        n_seeds=len(seeds_of(ka)),
                        mean_a=float(ua.mean()), mean_b=float(ub.mean()),
                        diff=m, lo=lo, hi=hi, p=p,
                        better=("A" if hi < 0 else ("B" if lo > 0 else "tie"))))
        print("%-28s media=%4d rows=%5d  A=%.4f B=%.4f  diff=%+.4f [%+.4f,%+.4f]"
              "  p=%.2e  %s"
              % (label, ua.size, Ma.shape[0], ua.mean(), ub.mean(), m, lo, hi,
                 p, res[-1]["better"]))
    C.save_json(dict(n_boot=a.n_boot, n_perm=a.n_perm,
                     unit="media", replicates="(seed, cell) within a medium",
                     cases=res), a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
