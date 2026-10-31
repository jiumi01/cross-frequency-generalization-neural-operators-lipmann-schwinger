# Author: Longgang Wang
"""Read the frequency x grid cross and decide what it licenses.

Pre-registered before the runs (this docstring is the record):

  S1  location invariance : the three locators (f*, onset, largest-step
      interval) return the SAME values at N = 64, 96, 128.
  S2  jump survival       : the band-edge jump factor eps(4.25)/eps(4.0) stays
      above 3 at every N, i.e. resolution does not remove the step.
  S3  frequency dominance : on log eps over the 14 x 3 cells, a model with one
      free mean per frequency explains at least 0.85 of the variance and beats
      a one-parameter model in log q = log(N/f) by at least 0.30.
  S4  direction          : if q governed the boundary, the onset in f would
      scale as 4.25 * N/64 (6.4 at N=96, 8.5 at N=128).  Observed onsets that
      track N falsify the paper's claim; observed onsets that do not, bound it.

  SUPPORT     = S1 and S2 and S3 -> the cross goes into the paper as a
                strengthening of the sampling-density paragraph.
  NULL        = some of S1-S3 hold and some do not -> the cross is reported in
                the supplement as a bounded check, and the main text keeps the
                weaker wording it already carries.
  CONTRADICT  = onset moves with N -> the paper's claim is wrong and must be
                rewritten, not hidden.  This is not a "do not include" outcome.

Usage
-----
    python code/cross_analyze.py
"""
import os
import sys
import json
import itertools

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from locators import loc_fstar, loc_onset, loc_step

EDGE = 4.0
MULT = 6.0


def load(seeds=(0, 1, 2)):
    out = {}
    for s in seeds:
        p = os.path.join(C.RES, "cross_eval_s%d.json" % s)
        if not os.path.exists(p):
            continue
        out[s] = json.load(open(p, encoding="utf-8"))
    return out


def curve(res):
    """f -> mean eps, plus the per-cell rows."""
    rows = res["rows"]
    f = np.array(sorted({r["f"] for r in rows}))
    N = np.array(sorted({r["N"] for r in rows}))
    e = np.zeros((len(N), len(f)))
    for r in rows:
        i = int(np.where(N == r["N"])[0][0])
        j = int(np.where(f == r["f"])[0][0])
        e[i, j] = r["mean"]
    return f, N, e


def r2(y, design):
    """Least-squares R^2 of y on a design matrix (with intercept)."""
    X = np.column_stack([np.ones(len(y))] + design)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return 1.0 - resid.var() / y.var(), X.shape[1]


def paired(a, b):
    """Paired mean difference with 95% CI and a two-sided paired t p-value."""
    d = np.asarray(a) - np.asarray(b)
    n = len(d)
    m, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    try:
        from scipy import stats
        t = m / se
        p = 2 * stats.t.sf(abs(t), n - 1)
        tc = stats.t.ppf(0.975, n - 1)
    except Exception:
        tc, p = 1.96, float("nan")
    return m, m - tc * se, m + tc * se, p


def main():
    data = load()
    if not data:
        print("no cross_eval_s*.json -- run code/cross_eval.py first")
        return
    seeds = sorted(data)
    print("seeds: %s" % seeds)

    # seed-mean curve per (N, f)
    f, N, e = curve(data[seeds[0]])
    acc = np.zeros_like(e)
    for s in seeds:
        _, _, es = curve(data[s])
        acc += es
    e = acc / len(seeds)

    print("\nseed-mean relative L2 error  eps(f, N)")
    print("   f   " + "".join("   N=%3d" % n for n in N))
    for j, fj in enumerate(f):
        print("%5.2f  %s" % (fj, "".join("  %6.4f" % e[i, j] for i in range(len(N)))))

    # Two ratios are read off every row and they are NOT the same quantity:
    #   r_edge = eps(4.25)/eps(4.00)  the first out-of-band node against the edge
    #   r_1Hz  = eps(5.00)/eps(4.00)  the whole first hertz beyond the edge
    # The manuscript quotes both, so both are emitted together with the seed
    # dispersion of each, and with the per-seed values so the paired mesh effect
    # can be quoted as a paired quantity rather than as two marginal means.
    i425 = int(np.where(np.isclose(f, 4.25))[0][0])
    i400 = int(np.where(np.isclose(f, 4.0))[0][0])
    i500 = int(np.where(np.isclose(f, 5.0))[0][0])

    def ratios(y):
        return (float(y[i425] / y[i400]), float(y[i500] / y[i400]))

    print("\nlocators (edge f=4, f* rule = %.0fx the edge node)" % MULT)
    loc, per_seed = {}, {}
    for i, n in enumerate(N):
        st = loc_step(f, e[i], EDGE)
        re, r1 = ratios(e[i])
        loc[int(n)] = dict(fstar=loc_fstar(f, e[i], EDGE),
                           onset=loc_onset(f, e[i], EDGE),
                           step=st,
                           inband=float(e[i][f < EDGE].mean()),
                           plateau=float(e[i][f > EDGE + 1].mean()),
                           jump=re, r1hz=r1)
        L = loc[int(n)]
        print("  N=%3d  f*=%s  onset=%s  step=%s  r_edge=%.2f  r_1Hz=%.2f"
              "  in=%0.4f  plat=%0.4f"
              % (n, L["fstar"], L["onset"], L["step"], L["jump"], L["r1hz"],
                 L["inband"], L["plateau"]))

    # Per-seed locators, same code path as above but on one seed's curve.  The
    # SD of the three seeds is what the table prints beside each mean: the three
    # meshes share their seeds, so a within-mesh SD would understate nothing and
    # would let a reader mistake a seed-level quantity for a mesh-level one.
    for s in seeds:
        _, _, es = curve(data[s])
        per_seed[s] = {}
        for i, n in enumerate(N):
            re, r1 = ratios(es[i])
            per_seed[s][int(n)] = dict(jump=re, r1hz=r1,
                                       inband=float(es[i][f < EDGE].mean()),
                                       plateau=float(es[i][f > EDGE + 1].mean()))
    for i, n in enumerate(N):
        for k in ("jump", "r1hz", "inband", "plateau"):
            vals = [per_seed[s][int(n)][k] for s in seeds]
            loc[int(n)]["sd_" + k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    # Paired mesh effect: same seed, 128^2 against 64^2.  Reported because the
    # seed-to-seed spread of the ratio is an order of magnitude larger than the
    # mesh-to-mesh one, so an unpaired reading of "0.6% across the ladder" would
    # be indistinguishable from seed noise while the paired one is not.
    d_edge = [per_seed[s][int(max(N))]["jump"] - per_seed[s][int(min(N))]["jump"]
              for s in seeds]
    d_1hz = [per_seed[s][int(max(N))]["r1hz"] - per_seed[s][int(min(N))]["r1hz"]
             for s in seeds]
    print("\npaired mesh effect %d^2 - %d^2 over seeds: r_edge %+.3f (%s),"
          " r_1Hz %+.3f" % (max(N), min(N), float(np.mean(d_edge)),
                            ", ".join("%+.3f" % v for v in d_edge),
                            float(np.mean(d_1hz))))
    loc_extra = dict(mesh_pair=[int(min(N)), int(max(N))],
                     d_edge=float(np.mean(d_edge)),
                     d_edge_seeds=[float(v) for v in d_edge],
                     d_1hz=float(np.mean(d_1hz)),
                     d_edge_lo=float(min(d_edge)), d_edge_hi=float(max(d_edge)))

    # ---- S1, S2 -----------------------------------------------------------
    s1 = len({(loc[int(n)]["fstar"], loc[int(n)]["onset"],
               tuple(loc[int(n)]["step"])) for n in N}) == 1
    s2 = all(loc[int(n)]["jump"] > 3.0 for n in N)

    # ---- S3: variance decomposition --------------------------------------
    y = np.log(e.reshape(-1))
    ff = np.repeat(f[None, :], len(N), axis=0).reshape(-1)
    qq = (N[:, None] / f[None, :]).reshape(-1)
    d_f = [(ff == v).astype(float) for v in f[1:]]
    d_q = [np.log(qq) - np.log(qq).mean()]
    d_N = [(N[:, None].repeat(len(f), 1).reshape(-1) == v).astype(float)
           for v in N[1:]]
    r2_f, k_f = r2(y, d_f)
    r2_q, k_q = r2(y, d_q)
    r2_fn, k_fn = r2(y, d_f + d_N)
    d_int = [a * b for a, b in itertools.product(d_f, d_N)]
    r2_full, k_full = r2(y, d_f + d_N + d_int)
    s3 = (r2_f >= 0.85) and (r2_f - r2_q >= 0.30)
    print("\nvariance of log eps over the %d cells" % len(y))
    print("  f only      R2=%.3f (%d params)" % (r2_f, k_f))
    print("  q only      R2=%.3f (%d param, linear in log N/f)" % (r2_q, k_q))
    print("  f + N       R2=%.3f (%d params)" % (r2_fn, k_fn))
    print("  saturated   R2=%.3f (%d params)" % (r2_full, k_full))

    # ---- S4: matched-q pairs ---------------------------------------------
    print("\nmatched-q pairs (same points per wavelength, different f and N)")
    pairs = []
    for i, n in enumerate(N):
        for j, fj in enumerate(f):
            for i2, n2 in enumerate(N):
                if n2 <= n:
                    continue
                f2 = fj * n2 / n
                if f2 in f:
                    j2 = int(np.where(f == f2)[0][0])
                    pairs.append((fj, n, f2, n2, e[i, j], e[i2, j2]))
    for p in pairs:
        print("  f=%5.2f N=%3d  eps=%.4f   vs  f=%5.2f N=%3d  eps=%.4f   ratio=%.2f"
              % (p[0], p[1], p[4], p[2], p[3], p[5], p[5] / p[4]))

    # ---- paired tests across N at fixed f --------------------------------
    print("\npaired across N at fixed f (same media, n=%d)"
          % data[seeds[0]]["n"])
    tests = {}
    for j, fj in enumerate(f):
        i64, i128 = int(np.where(N == 64)[0][0]), int(np.where(N == 128)[0][0])
        a = [r for r in data[seeds[0]]["rows"] if r["f"] == fj and r["N"] == 128][0]["per_sample"]
        b = [r for r in data[seeds[0]]["rows"] if r["f"] == fj and r["N"] == 64][0]["per_sample"]
        m, lo, hi, p = paired(a, b)
        tests[float(fj)] = dict(diff=m, lo=lo, hi=hi, p=p)
        print("  f=%5.2f  eps(128)-eps(64) = %+.4f  [%+.4f, %+.4f]  p=%.3g"
              % (fj, m, lo, hi, p))

    verdict = "SUPPORT" if (s1 and s2 and s3) else (
        "CONTRADICT" if not s1 else "NULL")
    print("\nS1 location invariance : %s" % ("pass" if s1 else "FAIL"))
    print("S2 jump survival       : %s" % ("pass" if s2 else "FAIL"))
    print("S3 frequency dominance : %s" % ("pass" if s3 else "FAIL"))
    print("VERDICT: %s" % verdict)

    out = dict(seeds=seeds, n=data[seeds[0]]["n"], f=f.tolist(),
               N=N.tolist(), eps=e.tolist(), loc={str(k): v for k, v in loc.items()},
               paired_mesh=loc_extra, per_seed={str(k): v for k, v in per_seed.items()},
               r2=dict(f=r2_f, q=r2_q, fN=r2_fn, full=r2_full),
               pairs=[list(p) for p in pairs], tests={str(k): v for k, v in tests.items()},
               s1=bool(s1), s2=bool(s2), s3=bool(s3), verdict=verdict)
    C.save_json(out, os.path.join(C.RES, "cross_analysis.json"))
    print("\nwrote results/cross_analysis.json")


if __name__ == "__main__":
    main()
