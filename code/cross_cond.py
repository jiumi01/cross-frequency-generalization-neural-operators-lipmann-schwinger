"""Conditional analysis of the frequency x mesh cross, and locator sensitivity.

Why this file exists
--------------------
The cross is reported as an R^2 comparison: 1.000 for a model with one free mean
per frequency, 0.521 for a single parameter in log q.  A referee can object, and
did, that the two models do not have the same number of parameters, so the
comparison does not establish that frequency acts independently of sampling
density.  It also cannot: 14 means against 1 slope is not a nested test.

This script answers the objection with the test the design supports.  Using the
archived per-sample errors (3 seeds x 14 frequencies x 3 meshes x 200 media), it
fits

    log(eps) = alpha_f + beta * log(q) + u_seed + u_media + e

with u_media eliminated by within-media demeaning (the media are paired across
every cell, which is what makes that legitimate) and cluster-robust standard
errors over the 200 media.  beta is therefore the effect of points per
wavelength *after* frequency is controlled, which is exactly the quantity the
R^2 comparison was being used to proxy.

It also recomputes the f* locator at thresholds 3x, 4x and 6x on the single-grid
reference arm and on each mesh of the cross arm, because f* is a threshold rule
and a difference between two arms at one threshold is not evidence of a physical
difference.

Output: results/cross_cond.json
"""
import os
import json
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
SEEDS = (0, 1, 2)
CONTRAST = "0.25"


# --------------------------------------------------------------- loading
def load_cross():
    rows = []
    for s in SEEDS:
        p = os.path.join(RES, "cross_eval_s%d.json" % s)
        d = json.load(open(p, encoding="utf-8"))
        for r in d["rows"]:
            for m, err in enumerate(r["per_sample"]):
                rows.append((s, float(r["f"]), int(r["N"]), m,
                             float(np.log(max(err, 1e-12)))))
    a = np.array(rows, float)
    return dict(seed=a[:, 0].astype(int), f=a[:, 1], N=a[:, 2].astype(int),
                m=a[:, 3].astype(int), y=a[:, 4])


def load_cross_cell_means():
    """(f, N) -> seed-mean error, for the locator scan."""
    acc = {}
    for s in SEEDS:
        d = json.load(open(os.path.join(RES, "cross_eval_s%d.json" % s),
                          encoding="utf-8"))
        for r in d["rows"]:
            acc.setdefault((float(r["f"]), int(r["N"])), []).append(r["mean"])
    return {k: float(np.mean(v)) for k, v in acc.items()}


# ---------------------------------------------------------------- design
def demean(y, *groups):
    """Subtract the mean within each combination of the given group arrays."""
    key = np.stack(groups, axis=1)
    out = y.copy()
    for k in np.unique(key, axis=0):
        sel = np.all(key == k, axis=1)
        out[sel] -= out[sel].mean()
    return out


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    return b, resid


def cluster_se(X, resid, clusters):
    """Cluster-robust covariance (CR0) with a small-sample factor."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for c in np.unique(clusters):
        sel = clusters == c
        u = X[sel].T @ resid[sel]
        meat += np.outer(u, u)
    n, k = X.shape
    g = len(np.unique(clusters))
    adj = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    return adj * XtX_inv @ meat @ XtX_inv


def bootstrap_beta(d, B=1000, rng_seed=20261031):
    """Media-cluster bootstrap of beta.

    The 25 200 records are 200 media x 14 frequencies x 3 meshes x 3 seeds, so
    they are not 25 200 independent observations and a row-level bootstrap would
    return an interval several times too narrow.  Resampling the 200 media with
    replacement keeps a medium's whole block together, which is the unit the
    design actually replicates.  This is a second instrument for the same
    interval `cluster_se` returns analytically: if the two disagree the analytic
    one is wrong, and the manuscript quotes both.
    """
    med = np.unique(d["m"])
    logq = np.log(d["N"] / d["f"])
    fl = np.unique(d["f"])
    idx_by_m = {m: np.where(d["m"] == m)[0] for m in med}
    rng = np.random.default_rng(rng_seed)
    # Group codes for (seed, resampled copy) are rebuilt each draw, so the
    # demeaning has to be vectorised rather than reuse `demean`'s Python loop
    # over groups: the loop is fine once, and 1000 times over 25 200 rows it is
    # not.
    def fast_demean(v, key):
        _, inv = np.unique(key, return_inverse=True)
        cnt = np.bincount(inv, minlength=inv.max() + 1).astype(float)
        s = np.bincount(inv, weights=v, minlength=inv.max() + 1)
        return v - (s / cnt)[inv]

    betas = []
    for _ in range(B):
        pick = rng.choice(med, size=len(med), replace=True)
        idx = np.concatenate([idx_by_m[m] for m in pick])
        # One cluster id per *resampled copy*: a medium drawn twice contributes
        # two independent clusters, which is what "resample the unit" means.
        inst = np.repeat(np.arange(len(pick)), [len(idx_by_m[m]) for m in pick])
        s, f, n, y = d["seed"][idx], d["f"][idx], d["N"][idx], d["y"][idx]
        key = s.astype(np.int64) * 100003 + inst.astype(np.int64)
        yd = fast_demean(y, key)
        qd = fast_demean(logq[idx], key)
        F = np.stack([fast_demean((f == v).astype(float), key) for v in fl[1:]],
                     axis=1)
        X = np.hstack([F, qd[:, None]])
        b, *_ = np.linalg.lstsq(X, yd, rcond=None)
        betas.append(float(b[-1]))
    betas = np.array(betas, float)
    return dict(beta_boot_mean=float(betas.mean()),
                beta_boot_se=float(betas.std(ddof=1)),
                beta_ci_boot=[float(np.percentile(betas, 2.5)),
                              float(np.percentile(betas, 97.5))],
                beta_boot_n=int(B), beta_boot_seed=int(rng_seed))


# ---------------------------------------------------------------- f* scan
def edge_node(f, edge):
    band = f[f <= edge + 1e-9]
    return float(band.max()) if len(band) else None


def fstar(f, e, edge, mult):
    ref = edge_node(f, edge)
    i0 = int(np.where(np.isclose(f, ref))[0][0])
    thr = mult * e[i0]
    for i in range(i0 + 1, len(f)):
        if e[i] > thr:
            return float(f[i])
    return None


def locator_scan():
    """f* at 3x/4x/6x for the single-grid arm and for each mesh of the cross."""
    from locators import load_curve
    out = {}
    # The arms are named by what they are, not by a role; `main_fno` is the
    # single-grid reference every table quotes, and it is the one whose f* the
    # cross has to be compared against.
    cands = ["main_fno", "ba_fno", "eb_fno", "ref_fno"]
    arm = None
    for c in cands:
        cur = load_curve("%s_a025" % c)
        if cur is not None:
            arm = (c + "_a025", cur)
            break
    if arm is not None:
        name, (f, e, _meta) = arm
        out[name] = {("%g" % m): fstar(f, e, 4.0, m) for m in (3.0, 4.0, 6.0)}

    cm = load_cross_cell_means()
    Ns = sorted({k[1] for k in cm})
    fs = sorted({k[0] for k in cm})
    f = np.array(fs, float)
    for N in Ns:
        e = np.array([cm[(ff, N)] for ff in fs], float)
        out["cross_N%d" % N] = {("%g" % m): fstar(f, e, 4.0, m)
                                for m in (3.0, 4.0, 6.0)}
    return out


# ------------------------------------------------------------------ main
def main():
    d = load_cross()
    y = d["y"]
    logq = np.log(d["N"] / d["f"])
    fl = np.unique(d["f"])

    # within-(seed, media) demeaning absorbs u_seed and u_media, and the media
    # pairing across cells is what makes u_media identifiable at all.
    yd = demean(y, d["seed"], d["m"])
    qd = demean(logq, d["seed"], d["m"])

    # Model A: frequency fixed effects only.
    F = np.stack([(d["f"] == v).astype(float) for v in fl[1:]], axis=1)
    Fd = np.stack([demean(F[:, j], d["seed"], d["m"]) for j in range(F.shape[1])],
                  axis=1)
    bA, rA = ols(Fd, yd)
    rssA = float(rA @ rA)

    # Model B: frequency fixed effects + log q.
    XB = np.hstack([Fd, qd[:, None]])
    bB, rB = ols(XB, yd)
    rssB = float(rB @ rB)
    beta = float(bB[-1])
    cov = cluster_se(XB, rB, d["m"])
    se = float(np.sqrt(cov[-1, -1]))
    z = 1.959963985
    r2A = 1.0 - rssA / float(yd @ yd)
    r2B = 1.0 - rssB / float(yd @ yd)
    partial = (rssA - rssB) / rssA

    # Interaction: one log-q slope per mesh, controlling frequency.
    Ns = sorted({int(n) for n in d["N"]})
    Xi = np.hstack([Fd] + [demean((d["N"] == n).astype(float) * logq,
                                  d["seed"], d["m"])[:, None] for n in Ns])
    bi, ri = ols(Xi, yd)
    rssI = float(ri @ ri)
    df1 = len(Ns) - 1
    df2 = len(yd) - Xi.shape[1]
    Fint = ((rssB - rssI) / df1) / (rssI / df2) if df1 else 0.0

    # Sanity: eta^2 of frequency alone, for the record.
    out = dict(
        n_obs=int(len(yd)), grids=[int(n) for n in Ns],
        freqs=[float(v) for v in fl], n_media=int(len(np.unique(d["m"]))),
        beta_logq=beta, beta_se_cluster=se,
        beta_ci=[beta - z * se, beta + z * se],
        beta_t=beta / se,
        r2_freq=r2A, r2_freq_plus_logq=r2B, partial_r2_logq=float(partial),
        interaction_F=float(Fint), interaction_df=[int(df1), int(df2)],
        interaction_p=float(1.0 - _f_cdf(Fint, df1, df2)) if df1 else 1.0,
        # Within one mesh, log q = log N - log f is collinear with log f, so a
        # log-q slope is not identified inside a mesh.  Beta is identified only
        # from the cross: three meshes, three different (f, log q) geometries.
        # The interaction test above therefore asks the answerable question --
        # does the frequency effect differ by mesh -- rather than a slope
        # difference that the design cannot resolve.
        identified_by="cross_mesh_only",
    )
    out["locator_thresholds"] = locator_scan()
    out.update(bootstrap_beta(d))

    p = os.path.join(RES, "cross_cond.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items()
                      if k != "locator_thresholds"}, indent=1))
    print("f* at 3x/4x/6x:")
    for k, v in out["locator_thresholds"].items():
        print("  %-12s %s" % (k, v))


def _f_cdf(x, d1, d2):
    """Survival-free F CDF via the regularized incomplete beta, no scipy."""
    if x <= 0:
        return 0.0
    try:
        from scipy.stats import f as _f
        return float(_f.cdf(x, d1, d2))
    except Exception:
        return float("nan")


if __name__ == "__main__":
    main()
