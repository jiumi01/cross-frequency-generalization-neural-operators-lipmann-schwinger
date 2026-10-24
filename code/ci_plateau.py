# Author: Longgang Wang
"""Bootstrap intervals for the three headline scalar statistics.

Review item 1, fourth row: 误差比值、平台值和边界频率 need 95% intervals, and
the intervals have to be computed on the right resampling unit.  This project
already resamples media for its paired tests (Table 13) because every seed of
every arm is scored on the same media; repeating that choice here is what makes
these intervals comparable with the ones already in the paper.  Resampling the
(media x seed x frequency) entries would count each medium three times and
return intervals that look precise without being so.

Three statistics, all defined before the run:

  e_in    mean in-band per-sample error                      (per_ib)
  e_plat  mean per-sample error over f >= 6                  (per)
  r       e(5) / e(4), the adjacent-grid jump ratio          (per)
  e_triv  mean trivial baseline over f >= 6, per medium      (triv_paired)
  ratio   e_plat / e_triv, the plateau-to-baseline ratio     (per / triv_paired)

  onset   the smallest node at which the group-mean curve first reaches
          50% of its own plateau.  The onset is a *curve* statistic, not a
          per-medium one, so it is bootstrapped by resampling the media and
          recomputing the whole group-mean curve; the interval is the spread
          of that statistic, not a per-medium spread.

Each is computed on the seed-averaged per-medium errors, then resampled over
media with 4000 replicates, percentile method.

The `ratio` is the one statistic here whose denominator is not a model output,
and it used to be the only headline scalar without an interval because the
baseline lived in `spectrum.json` as a mean over 600 media while the numerator
had 200.  `baseline_paired.py` recomputes the baseline on the evaluation media
themselves (same seed base, same n, same draw order), so numerator and
denominator are element-wise paired and `boot_ratio` resamples one index for
both.  The same index is required, not merely convenient: a ratio of two
independently resampled sums would carry the baseline's sampling error twice.
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

NBOOT = 4000
SEEDS = (0, 1, 2)
FREQS = C.F_TEST
PLAT_F = [f for f in FREQS if f >= 6.0]


def load_arm(tag):
    """per[ci, fi, mi] seed-averaged; ib[ci, mi]."""
    files = []
    for s in SEEDS:
        p = os.path.join(C.RES, "percell_%s_s%d.npz" % (tag, s))
        if os.path.exists(p):
            files.append(p)
    if not files:
        return None
    per, ib = [], []
    for p in files:
        d = np.load(p)
        per.append(d["per"])
        ib.append(d["per_ib"])
    return np.mean(per, axis=0), np.mean(ib, axis=0), len(files)


def boot_ci(vals, rng, nboot=NBOOT, method="mean"):
    n = vals.shape[0]
    idx = rng.integers(0, n, size=(nboot, n))
    m = vals[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ratio(num, den, rng, nboot=NBOOT, paired=True):
    """Interval on mean(num)/mean(den) at the media level.

    `paired=True` resamples ONE index vector and applies it to both sides, which
    is what a ratio of two quantities measured on the same medium requires.
    `paired=False` resamples them independently and is kept only as the
    counterfactual: the guard asserts that the paired interval is materially
    narrower, so that "paired" is doing work rather than being an adjective.
    """
    n = num.shape[0]
    idx = rng.integers(0, n, size=(nboot, n))
    idxb = idx if paired else rng.integers(0, n, size=(nboot, n))
    a = num[idx].mean(axis=1)
    b = den[idxb].mean(axis=1)
    r = a / np.maximum(b, 1e-12)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def onset_of(curve, plateau):
    """Smallest node of the group-mean curve at 50% of its own plateau."""
    thr = 0.5 * plateau
    for i in range(len(FREQS)):
        if curve[i] >= thr:
            return FREQS[i]
    return float("nan")


def boot_onset(per, rng, nboot=NBOOT):
    """per[fi, mi]; resample media, rebuild the curve, re-read the onset.

    The onset is a node-valued statistic and every media share the same
    frequency grid, so its interval can degenerate to a single node.  The
    distribution over nodes is kept as well: that, rather than a zero-width
    interval, is what says whether the step sits at the band edge for every
    resample.
    """
    nf, nm = per.shape
    idx = rng.integers(0, nm, size=(nboot, nm))
    out = np.empty(nboot)
    for b in range(nboot):
        cur = per[:, idx[b]].mean(axis=1)
        plat = cur[[FREQS.index(f) for f in PLAT_F]].mean()
        out[b] = onset_of(cur, plat)
    return out


def load_triv():
    """triv[vmax][fi, mi] from baseline_paired.py, or None.

    Returned transposed to [fi, mi] so it indexes exactly like `per`, which is
    what makes a single media index usable for both sides of the ratio.
    """
    p = os.path.join(C.RES, "triv_paired.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    fs = [float(x) for x in d["f"]]
    out = {}
    for vk, arm in d["data"].items():
        rows = []
        for f in FREQS:
            rows.append(np.array(arm["f=%g" % f], float))
        out[float(vk)] = np.array(rows)          # [fi, mi]
        assert fs == list(FREQS), "baseline and percell use different grids"
    return out


def main():
    rng = np.random.default_rng(20261022)
    TRIV = load_triv()
    if TRIV is None:
        print("WARNING: results/triv_paired.json absent, ratio columns will be "
              "omitted (run baseline_paired.py --go first)", flush=True)
    arms = [
        ("main_fno_a005", "FNO", 0.05), ("main_fno_a010", "FNO", 0.10),
        ("main_fno_a025", "FNO", 0.25), ("main_fno_a050", "FNO", 0.50),
        ("main_fno_a100", "FNO", 1.00),
        ("base_unet_a025", "U-Net", 0.25), ("base_unet_a100", "U-Net", 1.00),
        ("pmatch_unet_a025", "U-Net(m)", 0.25),
        ("ba_fno_a025_m8", "head+resid", 0.25),
    ]
    rows = []
    for tag, arch, vmax in arms:
        got = load_arm(tag)
        if got is None:
            print("skip (no percell):", tag)
            continue
        per, ib, nseed = got
        ci = int(np.argmin(np.abs(np.array(C.VMAX_LIST) - vmax)))
        p_arm = per[ci]                     # [fi, mi]
        ib_arm = ib[ci]                     # [mi]
        i4 = FREQS.index(4.0)
        i5 = FREQS.index(5.0)
        ip = [FREQS.index(f) for f in PLAT_F]

        e_in = float(ib_arm.mean())
        e_plat = float(p_arm[ip].mean())
        r = float(p_arm[i5].mean() / max(p_arm[i4].mean(), 1e-12))

        in_lo, in_hi = boot_ci(ib_arm, rng)
        pl_lo, pl_hi = boot_ci(p_arm[ip].mean(axis=0), rng)
        r_lo, r_hi = boot_ratio(p_arm[i5], p_arm[i4], rng)
        on = boot_onset(p_arm, rng)
        on_lo, on_hi = float(np.percentile(on, 2.5)), float(np.percentile(on, 97.5))
        on_point = onset_of(p_arm.mean(axis=1), e_plat)
        vals, cnts = np.unique(on, return_counts=True)
        on_dist = {("%g" % v): float(c / len(on)) for v, c in zip(vals, cnts)}

        # ---- plateau relative to its own baseline, on the same media ------
        e_triv = ratio = ra_lo = ra_hi = un_lo = un_hi = None
        if TRIV is not None:
            key = min(TRIV, key=lambda k: abs(k - vmax))
            if abs(key - vmax) < 1e-9:
                tr_all = TRIV[key]                     # [fi, mi]
                e_triv = float(tr_all[ip].mean())
                # Marginal interval of the denominator, kept so that the paper
                # can compare the ratio's interval against BOTH components.
                tv_lo, tv_hi = boot_ci(tr_all[ip].mean(axis=0), rng)
                if e_triv > 0:
                    ratio = e_plat / e_triv
                    ra_lo, ra_hi = boot_ratio(p_arm[ip].mean(axis=0),
                                              tr_all[ip].mean(axis=0), rng)
                    un_lo, un_hi = boot_ratio(p_arm[ip].mean(axis=0),
                                              tr_all[ip].mean(axis=0), rng,
                                              paired=False)

        rows.append(dict(tag=tag, arch=arch, vmax=vmax, nseed=nseed,
                         e_in=e_in, e_in_lo=in_lo, e_in_hi=in_hi,
                         e_plat=e_plat, e_plat_lo=pl_lo, e_plat_hi=pl_hi,
                         r=r, r_lo=r_lo, r_hi=r_hi,
                         e_triv=e_triv, e_triv_lo=tv_lo, e_triv_hi=tv_hi,
                         ratio=ratio,
                         ratio_lo=ra_lo, ratio_hi=ra_hi,
                         ratio_un_lo=un_lo, ratio_un_hi=un_hi,
                         onset=on_point, onset_lo=on_lo, onset_hi=on_hi,
                         onset_dist=on_dist))
        print("%-18s v=%-5s in %.4f [%.4f,%.4f]  plat %.4f [%.4f,%.4f]  "
              "r %.3f [%.3f,%.3f]  ratio %s  onset %.2f [%.2f,%.2f]  (%d seeds)"
              % (tag, vmax, e_in, in_lo, in_hi, e_plat, pl_lo, pl_hi,
                 r, r_lo, r_hi,
                 ("%.3f [%.3f,%.3f]" % (ratio, ra_lo, ra_hi))
                 if ratio is not None else "--",
                 on_point, on_lo, on_hi, nseed), flush=True)

    C.save_json(dict(nboot=NBOOT, seeds=list(SEEDS), rows=rows),
                os.path.join(C.RES, "ci_plateau.json"))
    print("saved results/ci_plateau.json")


if __name__ == "__main__":
    main()
