# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Does the Marmousi jump measure geology, or only spectral content?

Why this file exists
--------------------
The third evidence level compares three Marmousi windows against three
synthetic control fields drawn from the benchmark generator.  Two problems
with that comparison were found while re-reading the archive:

  * The control fields are band-limited by construction.  At mode 4 the
    out-of-band energy fraction is 6.8e-2 for the real windows and 7.3e-4 for
    the controls -- a factor 94, rising to 1e12 by mode 32.  So the two arms
    differ in spectral content at least as much as in geological structure,
    and a difference in the measured jump cannot be attributed to structure.

  * Three windows per arm cannot resolve anything: the window-to-window spread
    of the jump factor is 17%, the between-arm difference is 8.6%, and the
    paired t interval on 64^2 is [-1.87, +0.58] -- it crosses zero.

This file replaces both.  It builds a surrogate control that is matched to each
real window in the two quantities the old control got wrong:

    amplitude spectrum   matched by iterative amplitude-adjusted Fourier
                         transform (IAAFT), which alternates projecting onto
                         the target amplitude spectrum and rank-mapping onto
                         the target marginal distribution.  Measured residuals
                         on the archived fields: band fractions agree to
                         0.1-10% and the marginal distribution is exact by
                         construction.
    marginal distribution matched exactly by the rank map.

What it does NOT match is the direction in which the spectral energy is
concentrated.  Marmousi is layered, so its gradient L1 norm is lower than that
of an isotropic field with the same spectrum: 0.047 against 0.074-0.075 for the
surrogates, a factor 1.6.  That factor is reported, not hidden, and it is the
operational definition of "geological structure" used in the text.

Design
------
Eight window positions (3 km wide, 1 km stride, so neighbours overlap by 2/3),
each carrying one real field and two surrogates.  The comparison is paired
within a window: real against the mean of its own two surrogates, so the
spectrum is held constant inside every pair by construction.  Both grids.
Same checkpoints, same frequency grid, same n as every other rung.

The overlap means the eight windows are not independent, so the t interval with
df=7 is optimistic.  The three non-overlapping windows are therefore reported
separately as the conservative reading, and the paper quotes the wider one.

Usage
-----
    python code/geo_windows.py --grid 64 --device cpu
    python code/geo_windows.py --grid 128 --device cpu
Results merge into results/geo_windows.json.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
import evaluate as E
import locators as L
import make_realgeo as G

ARM = 0.25
VI = 2
EDGE = 4.0
SEEDS = (0, 1, 2)
WIN_M = G.WIN_M
STRIDE = 1000.0
N_WIN = 9
N_SUR = 2
OUT = os.path.join(C.RES, "geo_windows.json")


# ------------------------------------------------------------------ window pool
def window_pool(c, x, z):
    """Overlapping 3 km windows, 1 km stride, plus a non-overlap subset flag."""
    x_lo, x_hi = float(x.min()), float(x.max())
    z_lo, z_hi = float(z.min()), float(z.max())
    # linspace, not arange: the last legal start is x_hi - WIN_M = 6200 m, which
    # is not on the 1 km lattice rooted at x_lo = -200 m.  An arange pool would
    # silently drop it and with it one of the three non-overlapping windows the
    # archive is anchored on.
    starts = np.linspace(x_lo, x_hi - WIN_M, N_WIN)
    pool = []
    for i, s in enumerate(starts):
        # a window belongs to the conservative subset if it is one of the three
        # archived, non-overlapping positions
        nonoverlap = (abs(s - x_lo) < 1e-6 or
                      abs(s - (x_lo + (x_hi - x_lo - WIN_M) / 2.0)) < 1e-6 or
                      abs(s - (x_hi - WIN_M)) < 1e-6)
        pool.append(dict(name="w%02d" % i, x0=float(s), x1=float(s + WIN_M),
                         z0=z_lo, z1=z_hi, nonoverlap=bool(nonoverlap)))
    # fail fast: the three archived windows anchor the conservative reading, so
    # a pool that does not contain all of them must not silently produce an
    # interval over two points and call it a conservative bound.
    n_star = sum(p["nonoverlap"] for p in pool)
    assert n_star == 3, "pool has %d archived windows, expected 3" % n_star
    return pool


# ------------------------------------------------------------------- surrogates
def iaaft(g, rng, iters=200):
    """Iterative amplitude-adjusted FT surrogate.

    Alternates (i) imposing the target amplitude spectrum on the current field
    and (ii) rank-mapping the result onto the target empirical marginal.  At
    convergence the spectrum and the marginal are both reproduced; only the
    phase structure differs, which is the quantity a layered medium carries.
    """
    A = np.abs(np.fft.fft2(g))
    s = np.sort(g.ravel())
    h = np.real(np.fft.ifft2(A * np.exp(
        1j * np.angle(np.fft.fft2(rng.standard_normal(g.shape))))))
    for _ in range(iters):
        H = np.fft.fft2(h)
        h = np.real(np.fft.ifft2(A * np.exp(1j * np.angle(H))))
        order = np.argsort(np.argsort(h.ravel()))
        h = s[order].reshape(g.shape)
    return h


def grad_l1(f):
    gy, gx = np.gradient(f, 1.0, 1.0)
    return float(np.hypot(gx, gy).mean())


def bank(grid, c, x, z, pool):
    """{name: (field, kind, window)} for real and surrogate fields."""
    geo = {}
    for w in pool:
        box = (w["x0"], w["x1"], w["z0"], w["z1"])
        g = G.unit01(G.resample(G.crop(c, x, z, box), grid))
        geo["real_" + w["name"]] = (g, "real", w["name"])
        for s in range(N_SUR):
            rng = np.random.default_rng(20260 + 91 * int(w["name"][1:]) + s)
            geo["sur_%s_%d" % (w["name"], s)] = (iaaft(g, rng), "surrogate",
                                                 w["name"])
    return geo


# ------------------------------------------------------------------- evaluation
def ckpt(grid, seed):
    if grid == 64:
        return os.path.join(C.RUNS, "main_fno_a025_s%d" % seed, "checkpoint.pt")
    return os.path.join(C.RUNS, "main_fno_a025_128_s%d" % seed, "checkpoint.pt")


def rel_err(model, grid, k, theta, v, psi_ref, dev):
    xx, inc = E.build(grid, k, theta, v, dev)
    with torch.no_grad():
        o = model(xx)
    psi_hat = inc + torch.complex(o[:, 0], o[:, 1])
    num = torch.linalg.norm((psi_hat - psi_ref).reshape(v.shape[0], -1), dim=1)
    den = torch.linalg.norm(psi_ref.reshape(v.shape[0], -1), dim=1)
    return (num / den).cpu().numpy()


def loc_block(f, per_seed):
    f = np.asarray(f, float)
    per_seed = [np.asarray(c, float) for c in per_seed]
    mean = np.mean(np.stack(per_seed, axis=0), axis=0)
    out = dict(fstar=L.loc_fstar(f, mean, EDGE),
               in_band=float(mean[f < EDGE].mean()),
               plateau5=float(mean[f >= 5.0].mean()))
    i4 = int(np.where(f == 4.0)[0][0])
    i5 = int(np.where(f == 5.0)[0][0])
    out["jump"] = float(mean[i5] / mean[i4])
    out["curve"] = [float(v) for v in mean]
    out["jump_seeds"] = [float(c[i5] / c[i4]) for c in per_seed]
    return out


def run(grid, dev, n):
    c, x, z = G.load_marmousi()
    pool = window_pool(c, x, z)
    geo = bank(grid, c, x, z, pool)
    models = {}
    for s in SEEDS:
        p = ckpt(grid, s)
        if os.path.exists(p):
            models[s] = E.load_model(p, dev)
    if not models:
        raise SystemExit("no checkpoints for grid %d" % grid)
    print("[geo_windows] grid=%d  %d windows  %d fields  %d seeds"
          % (grid, len(pool), len(geo), len(models)), flush=True)

    meta = {}
    for name, (g, kind, win) in geo.items():
        meta[name] = dict(kind=kind, window=win, grad_l1=grad_l1(g),
                          std=float(g.std()), mean=float(g.mean()),
                          bands={str(M): G.band_frac_unit(g, M) for M in G.MODES})

    summary, t0 = {}, time.time()
    for name, (g, kind, win) in geo.items():
        vunit = torch.from_numpy(g).double()
        per_seed = {s: [] for s in models}
        for f in C.F_TEST:
            _, _, theta = D.sample_params(n, None, None, ARM, ARM,
                                          60000 + VI, dev, f_fixed=f)
            k = torch.full((n,), 2 * np.pi * f, device=dev)
            v = (vunit.to(dev) * ARM).unsqueeze(0).repeat(n, 1, 1)
            psi_ref, rres = P.solve_ls(v, k, theta, C.GAMMA, grid, tol=1e-9,
                                       maxiter=900, device=dev, return_info=True)
            for s, (m, ck) in models.items():
                per_seed[s].append(float(rel_err(m, grid, k, theta, v,
                                                 psi_ref, dev).mean()))
        lb = loc_block(C.F_TEST, [per_seed[s] for s in sorted(models)])
        lb["ref_res_max"] = float(rres.max())
        summary[name] = lb
        print("  %-12s %-9s jump=%.2f plateau5=%.3f f*=%s  (%.0fs)"
              % (name, kind, lb["jump"], lb["plateau5"],
                 -1 if lb["fstar"] is None else lb["fstar"],
                 time.time() - t0), flush=True)

    # paired within window: real minus mean of its surrogates
    pairs = []
    for w in pool:
        r = summary["real_" + w["name"]]
        su = [summary["sur_%s_%d" % (w["name"], s)] for s in range(N_SUR)]
        pairs.append(dict(
            window=w["name"], nonoverlap=w["nonoverlap"],
            real_jump=r["jump"],
            sur_jump=float(np.mean([s["jump"] for s in su])),
            delta_jump=float(r["jump"] - np.mean([s["jump"] for s in su])),
            real_plat=r["plateau5"],
            sur_plat=float(np.mean([s["plateau5"] for s in su])),
            delta_plat=float(r["plateau5"] -
                             np.mean([s["plateau5"] for s in su])),
            grad_l1_real=meta["real_" + w["name"]]["grad_l1"],
            grad_l1_sur=float(np.mean([meta["sur_%s_%d" % (w["name"], s)]
                                       ["grad_l1"] for s in range(N_SUR)]))))

    def interval(vals, label):
        v = np.asarray(vals, float)
        if len(v) < 2:
            return dict(label=label, n=len(v), mean=float(v.mean()), ci=None)
        se = v.std(ddof=1) / np.sqrt(len(v))
        tc = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
              8: 2.306}.get(len(v) - 1, 1.96)
        return dict(label=label, n=len(v), mean=float(v.mean()),
                    sd=float(v.std(ddof=1)), se=float(se), tcrit=tc,
                    ci=[float(v.mean() - tc * se), float(v.mean() + tc * se)],
                    separates_zero=bool(v.mean() - tc * se > 0
                                        or v.mean() + tc * se < 0))

    all_d = [p["delta_jump"] for p in pairs]
    sub_d = [p["delta_jump"] for p in pairs if p["nonoverlap"]]
    all_p = [p["delta_plat"] for p in pairs]
    sub_p = [p["delta_plat"] for p in pairs if p["nonoverlap"]]
    stats = dict(
        jump_all=interval(all_d, "jump real-surrogate, 8 windows"),
        jump_nonoverlap=interval(sub_d, "jump real-surrogate, 3 non-overlapping"),
        plat_all=interval(all_p, "plateau5 real-surrogate, 8 windows"),
        plat_nonoverlap=interval(sub_p, "plateau5, 3 non-overlapping"),
        grad_l1_ratio=float(np.mean([p["grad_l1_sur"] for p in pairs]) /
                            np.mean([p["grad_l1_real"] for p in pairs])))

    block = dict(geo_windows=dict(
        grid=grid, arm=ARM, n=n, n_windows=len(pool), n_surrogates=N_SUR,
        window_pool=pool, meta=meta, summary=summary, pairs=pairs,
        stats=stats, device=str(dev),
        date=time.strftime("%Y-%m-%d %H:%M:%S")))

    d = {}
    if os.path.exists(OUT):
        try:
            d = json.load(open(OUT, encoding="utf-8"))
        except Exception:
            d = {}
    d.setdefault("by_grid", {})[str(grid)] = block["geo_windows"]
    json.dump(d, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\n[geo_windows] wrote %s" % OUT, flush=True)
    return block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=64, choices=(64, 128))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    dev = torch.device(a.device)
    run(a.grid, dev, a.n)


if __name__ == "__main__":
    main()
