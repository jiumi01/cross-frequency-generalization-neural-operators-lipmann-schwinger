# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Interval checks on three readings that were reported as point values.

Why this file exists
--------------------
Three claims currently rest on a point estimate that has no interval attached:

  (1) The plateau is the same on 64^2 and 128^2 (a 0.03% difference).  The
      claim is "the step does not move with the mesh", but a difference of
      0.0003 with no interval cannot distinguish "not moving" from "moving by
      less than the seed noise".

  (2) The Marmousi jump orders the two families one way at 64^2 and the other
      way at 128^2, which the manuscript reads as "the jump is not a
      discriminator".  The 64^2 ordering is real > synthetic by 8.6%, but the
      three windows spread over 17%, so the ordering may never have been
      resolvable.

  (3) The synthetic control fields are band-limited by construction, so the
      control arm differs from Marmousi in spectral content as well as in
      geological structure.  Part (3) quantifies that difference so the two
      cannot be conflated in the text.

Method
------
Everything here is a re-reading of archived per-seed and per-window curves.
No model is evaluated, no reference solve is run, no GPU is touched.  The
paired t interval uses Student-t with df = n-1, which is the right interval
for n=3 and is deliberately wide; a bootstrap over three points would report
a narrower interval than the data can support, which is the error mode this
file exists to avoid.

Output: results/p0_checks.json
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RES = getattr(C, "RES", os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "results"))
RES_PATH = os.path.join(RES, "res_study.json")
OUT = os.path.join(RES, "p0_checks.json")

# f grid used by every arm in res_study.py; read from the archive, not retyped
F_GRID = None


def _t_crit_95(df):
    """Two-sided 95% Student-t critical value, df small."""
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
    return table.get(df, 1.96)


def _t_two_sided_p(t, df):
    """Two-sided p for Student-t, by Simpson integration of the density.

    scipy is not installed in every interpreter this archive is rebuilt with,
    and a p-value that cannot be recomputed is worse than no p-value, so the
    tail probability is integrated here from the density definition.
    """
    import math
    t = abs(float(t))
    if not math.isfinite(t):
        return 0.0

    def dens(u):
        c = math.gamma((df + 1) / 2.0) / (math.sqrt(df * math.pi)
                                          * math.gamma(df / 2.0))
        return c * (1.0 + u * u / df) ** (-(df + 1) / 2.0)

    # integrate [t, t+hi] with Simpson; the tail beyond t+hi is bounded by the
    # density at the cut times the remaining width and is below 1e-9 here
    hi = max(60.0, t + 40.0)
    n = 20000
    h = (hi - t) / n
    s = dens(t) + dens(hi)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * dens(t + i * h)
    tail = s * h / 3.0
    return float(min(1.0, 2.0 * tail))


def paired(a, b, label):
    """Paired difference b - a with a t interval; a, b are per-unit arrays."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    d = b - a
    n = len(d)
    if n < 2:
        return dict(label=label, n=n, a=float(a.mean()), b=float(b.mean()),
                    diff=float(d.mean()), ci=None, t=None, p=None,
                    note="n<2, no interval")
    m = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n))
    tc = _t_crit_95(n - 1)
    ci = [m - tc * se, m + tc * se]
    t = m / se if se > 0 else float("inf")
    return dict(label=label, n=n, a=float(a.mean()), b=float(b.mean()),
                diff=m, ci=ci, half=tc * se, t=float(t),
                p=_t_two_sided_p(t, n - 1),
                separates_zero=bool(ci[0] > 0 or ci[1] < 0))


def seed_matrix(rows):
    """rows: list of per-frequency dicts -> (seed, f) array of mean errors."""
    keys = sorted(k for k in rows[0] if k.startswith("s") and len(k) == 2)
    return np.array([[r[k]["mean"] for k in keys] for r in rows], float).T, keys


def part_grid():
    """(1) plateau and jump, 64^2 vs 128^2, paired over seeds."""
    d = json.load(open(RES_PATH, encoding="utf-8"))
    out = {}
    for part in ("part_a", "part_b"):
        if part not in d or "curves" not in d[part]:
            continue
        f = np.array(F_GRID, float)
        acc = {}
        for g in ("64", "128"):
            A, _ = seed_matrix(d[part]["curves"][g])
            i4 = int(np.where(f == 4.0)[0][0])
            i5 = int(np.where(f == 5.0)[0][0])
            acc[g] = dict(
                plateau5=A[:, f >= 5].mean(axis=1),
                plateau_gt5=A[:, f > 5].mean(axis=1),
                in_band=A[:, f < 4.0].mean(axis=1),
                jump=A[:, i5] / A[:, i4])
        out[part] = {
            "plateau5": paired(acc["64"]["plateau5"], acc["128"]["plateau5"],
                               "plateau f>=5, 128-64"),
            "plateau_gt5": paired(acc["64"]["plateau_gt5"],
                                  acc["128"]["plateau_gt5"],
                                  "plateau f>5, 128-64"),
            "in_band": paired(acc["64"]["in_band"], acc["128"]["in_band"],
                              "in band, 128-64"),
            "jump": paired(acc["64"]["jump"], acc["128"]["jump"],
                           "jump eps(5)/eps(4), 128-64"),
        }
    return out


def part_marmousi():
    """(2) real vs synthetic family, per window, at both grids."""
    d = json.load(open(RES_PATH, encoding="utf-8"))
    pc = d.get("part_c")
    if not pc:
        return {}
    out = {}
    for g in ("64", "128"):
        pg = pc["grids"][str(g)]["per_geometry"]
        real = ["geo_west", "geo_centre", "geo_east"]
        syn = ["syn_0", "syn_1", "syn_2"]
        rec = {}
        for key in ("jump", "plateau5", "plateau", "in_band"):
            a = [pg[k][key] for k in real]
            b = [pg[k][key] for k in syn]
            rec[key] = paired(a, b, "%s, syn-real" % key)
            rec[key]["real"] = [float(x) for x in a]
            rec[key]["syn"] = [float(x) for x in b]
        # overlap of the two families' observed ranges, same definition the
        # manuscript's ResGeoPlatGap uses: positive = separated by that gap,
        # negative = the ranges intersect by that amount
        rp = rec["plateau5"]["real"]
        sp = rec["plateau5"]["syn"]
        rec["plateau_gap"] = dict(
            real_range=[min(rp), max(rp)], syn_range=[min(sp), max(sp)],
            gap=float(min(sp) - max(rp)),
            note="min(syn)-max(real); positive means disjoint")
        rec["fstar"] = dict(
            real=[pg[k]["fstar"] for k in real],
            syn=[pg[k]["fstar"] for k in syn])
        out[str(g)] = rec
    return out


def part_bands():
    """(3) how far apart the two families are in spectral content."""
    d = json.load(open(RES_PATH, encoding="utf-8"))
    pc = d.get("part_c")
    if not pc:
        return {}
    out = {}
    for g in ("64", "128"):
        pg = pc["grids"][str(g)]["per_geometry"]
        real = ["geo_west", "geo_centre", "geo_east"]
        syn = ["syn_0", "syn_1", "syn_2"]
        modes = sorted(pg[real[0]]["bands"], key=int)
        out[str(g)] = {}
        for M in modes:
            r = float(np.mean([pg[k]["bands"][M] for k in real]))
            s = float(np.mean([pg[k]["bands"][M] for k in syn]))
            out[str(g)][M] = dict(real=r, syn=s,
                                  ratio=(r / s if s > 0 else None),
                                  log10_gap=(float(np.log10(r / s))
                                             if r > 0 and s > 0 else None))
    return out


def part_surrogate():
    """(4) real Marmousi windows against spectrum-matched surrogates.

    The surrogate is the IAAFT control built by geo_windows.py: amplitude
    spectrum reproduced to 0.1-10%, marginal distribution exact, phase structure
    randomised.  The comparison is paired inside a window, so the spectrum is
    held constant by construction and what is left is the phase structure, i.e.
    the layering.
    """
    path = os.path.join(RES, "geo_windows.json")
    if not os.path.exists(path):
        return {}
    d = json.load(open(path, encoding="utf-8")).get("by_grid", {})
    out = {}
    for g, blk in d.items():
        pairs = blk["pairs"]
        sub = [p for p in pairs if p["nonoverlap"]]
        rec = dict(n_windows=len(pairs), n_surrogates=blk["n_surrogates"],
                   grad_l1_ratio=float(np.mean([p["grad_l1_sur"] for p in pairs])
                                       / np.mean([p["grad_l1_real"] for p in pairs])))
        # how closely the surrogate reproduces the spectrum it is matched to
        meta = blk.get("meta", {})
        devs = []
        for p in pairs:
            w = p["window"]
            real = meta.get("real_" + w, {}).get("bands", {})
            for s in range(blk["n_surrogates"]):
                sur = meta.get("sur_%s_%d" % (w, s), {}).get("bands", {})
                for M, rv in real.items():
                    if M in sur and rv > 1e-12:
                        devs.append(abs(sur[M] / rv - 1.0))
        if devs:
            rec["band_match_max_dev"] = float(max(devs))
            rec["band_match_med_dev"] = float(np.median(devs))
        for key, name in (("delta_jump", "jump"), ("delta_plat", "plateau5")):
            for tag, sel in (("all", pairs), ("nonoverlap", sub)):
                real = [float(p["real_" + ("jump" if name == "jump" else "plat")])
                        for p in sel]
                sur = [float(p["sur_" + ("jump" if name == "jump" else "plat")])
                       for p in sel]
                rec["%s_%s" % (name, tag)] = paired(
                    real, sur, "%s real-surrogate (%s)" % (name, tag))
                rec["%s_%s" % (name, tag)]["real"] = real
                rec["%s_%s" % (name, tag)]["sur"] = sur
        out[g] = rec
    return out


def main():
    global F_GRID
    d = json.load(open(RES_PATH, encoding="utf-8"))
    # recover the frequency grid from a stored curve rather than retyping it
    F_GRID = [r["f"] for r in d["part_a"]["curves"]["64"]]
    res = dict(source="results/res_study.json", f_grid=F_GRID,
               n_note="intervals are Student-t with df=n-1; n=3 unless stated",
               grid=part_grid(), marmousi=part_marmousi(),
               bands=part_bands(), surrogate=part_surrogate())
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)

    print("=" * 78)
    for part, rec in res["grid"].items():
        print(part)
        for k, v in rec.items():
            print("  %-34s 64=%.5f 128=%.5f  d=%+.5f  CI[%+.5f,%+.5f]  p=%.3f  %s"
                  % (k, v["a"], v["b"], v["diff"], v["ci"][0], v["ci"][1],
                     v["p"] or 0.0,
                     "SEPARATES" if v["separates_zero"] else "crosses 0"))
    print("=" * 78)
    for g, rec in res["marmousi"].items():
        print("Marmousi grid %s" % g)
        for k in ("jump", "plateau5"):
            v = rec[k]
            print("  %-12s real=%.3f syn=%.3f d=%+.3f CI[%+.3f,%+.3f] p=%.3f  %s"
                  % (k, v["a"], v["b"], v["diff"], v["ci"][0], v["ci"][1],
                     v["p"] or 0.0,
                     "SEPARATES" if v["separates_zero"] else "crosses 0"))
        pg = rec["plateau_gap"]
        print("  plateau ranges real %s syn %s  gap=%+.3f"
              % (["%.3f" % x for x in pg["real_range"]],
                 ["%.3f" % x for x in pg["syn_range"]], pg["gap"]))
    print("=" * 78)
    for g, rec in res.get("surrogate", {}).items():
        print("surrogate (spectrum-matched) grid %s  win=%d  grad_L1 ratio %.2f"
              % (g, rec["n_windows"], rec["grad_l1_ratio"]))
        for k in ("jump_all", "jump_nonoverlap", "plateau5_all",
                  "plateau5_nonoverlap"):
            if k not in rec:
                continue
            v = rec[k]
            print("  %-20s real=%.3f sur=%.3f d=%+.3f CI[%+.3f,%+.3f] p=%.3f  %s"
                  % (k, v["a"], v["b"], v["diff"], v["ci"][0], v["ci"][1],
                     v["p"] or 0.0,
                     "SEPARATES" if v["separates_zero"] else "crosses 0"))
    print("=" * 78)
    for g, rec in res["bands"].items():
        print("bands grid %s (out-of-band fraction, real vs synthetic)" % g)
        for M, v in rec.items():
            print("  M=%-3s real=%.3e syn=%.3e  ratio=%.1e"
                  % (M, v["real"], v["syn"], v["ratio"] or 0))
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
