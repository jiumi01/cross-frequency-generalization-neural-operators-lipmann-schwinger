# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Anchor-only baseline, measured on the paper's own evaluation protocol.

code/prescreen_anchor.py established the effect on a 128-sample probe grid at
four contrasts.  That is enough for a go/no-go but not enough to put in a
table: every other arm in the manuscript is scored on n=200 samples at seeds
60000+vi over C.F_TEST and 70000+vi in band, per contrast.  This script
re-measures the anchor on exactly that protocol so its numbers can be compared
with the trained arms cell by cell.

Protocol, and why each piece is where it is:
  - The two complex coefficients (a, b) on (B1, B2) are fitted by least squares
    on a PREFIX OF THE TRAINING CACHE for that contrast arm -- no test sample,
    no out-of-band frequency, no label beyond f <= 4.  The anchor is therefore
    a zero-shot object in the same sense the trained arms are: whatever it
    knows, it learned inside the band.
  - Scoring uses the identical media and seeds as the trained arms, so the
    comparison is paired at the sample level (the pairing the whole manuscript
    rests on).
  - The trivial baseline (a predictor that returns the incident field) and the
    raw order-1 / order-2 Born series are scored on the same cells.  The
    manuscript already reports that the trained FNO's out-of-band plateau lands
    on the trivial baseline; the anchor has to be read against that.

CPU only.  Nothing here trains and nothing here may share the GPU with a
training job.
"""
import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import physics as P

DEV = "cpu"
N = 200                 # the protocol's sample count
N_FIT = 512             # cache prefix used to fit (a, b)
CACHE = {0.05: "train_a005", 0.10: "train_a010", 0.25: "train_a025",
         0.50: "train_a050", 1.00: "train_a100"}


def born_pair(v, k, th):
    return D.anchor_terms(v, k, th)


def rel(a, b):
    num = torch.linalg.norm((a - b).reshape(a.shape[0], -1), dim=1)
    den = torch.linalg.norm(a.reshape(a.shape[0], -1), dim=1).clamp_min(1e-12)
    return (num / den).cpu().numpy()


def fit_ab(v, k, th, u):
    """Least-squares (a, b) on [B1, B2]; returns the complex pair."""
    b1, b2 = born_pair(v, k, th)
    y = u.reshape(u.shape[0], -1)
    m = torch.stack([b1.reshape(u.shape[0], -1), b2.reshape(u.shape[0], -1)],
                    dim=1)
    gram = torch.einsum("njd,nkd->jk", m.conj(), m)
    rhs = torch.einsum("njd,nd->j", m.conj(), y)
    a, b = torch.linalg.solve(gram, rhs)
    return a, b


def cache_batch(cache, n_fit, device=DEV):
    """First n_fit samples of a training cache, as (v, k, theta, u)."""
    import glob
    zs = [np.load(p) for p in sorted(glob.glob(os.path.join(C.DATA, cache,
                                                            "*.npz")))]
    need = int(np.ceil(n_fit / zs[0]["v"].shape[0]))
    v = torch.from_numpy(np.concatenate([z["v"] for z in zs[:need]])).to(device)
    psi = torch.from_numpy(np.concatenate([z["psi"] for z in zs[:need]])).to(device)
    k = torch.from_numpy(np.concatenate([z["k"] for z in zs[:need]])).to(device)
    th = torch.from_numpy(np.concatenate([z["theta"] for z in zs[:need]])).to(device)
    v, psi, k, th = v[:n_fit], psi[:n_fit], k[:n_fit], th[:n_fit]
    pi = P.incident_plane_wave(C.N, k, th, device).to(torch.complex128)
    return v.float(), k, th, (psi - pi).to(torch.complex128)


def main():
    out = dict(n=N, n_fit=N_FIT, gamma=C.GAMMA, f_test=C.F_TEST,
               protocol="scored on the paper's own eval seeds (60000+vi per "
                        "frequency, 70000+vi in band), coefficients fitted on "
                        "a prefix of that contrast's own training cache",
               arms={})
    for vi, vmax in enumerate(C.VMAX_LIST):
        v, k, th, u = cache_batch(CACHE[vmax], N_FIT)
        a, b = fit_ab(v, k, th, u)
        print("v=%.2f  a=(%.4f,%.4f)  b=(%.4f,%.4f)" %
              (vmax, a.real, a.imag, b.real, b.imag), flush=True)

        rows, ib_rows, tv_rows, bn_rows = [], [], [], []
        # Per-media curves above 6 Hz, kept so the plateau can carry a spread.
        # A plateau is one number per media (that media's mean error over
        # f >= 6), and only then averaged over media; averaging the per-f means
        # first and attaching the spread of those means would report the
        # frequency-to-frequency wobble of a single curve, not the media
        # dispersion the anchor curve is supposed to be uncertain about.
        pl_vals, tv_vals = [], []
        # in band, same protocol and seeds as every trained arm
        d = D.sample_batch(N, 70000 + vi, DEV, f_lo=1.0, f_hi=4.0,
                           vmax_lo=vmax, vmax_hi=vmax)
        inc = P.incident_plane_wave(C.N, d["k"], d["theta"], DEV).to(
            torch.complex128)
        uu = (d["psi"] - inc).to(torch.complex128)
        b1, b2 = born_pair(d["v"], d["k"], d["theta"])
        e = rel(uu, a * b1 + b * b2)
        ib_rows.append(float(e.mean()))
        for fi, f in enumerate(C.F_TEST):
            d = D.sample_batch(N, 60000 + vi, DEV, vmax_lo=vmax, vmax_hi=vmax,
                               f_fixed=f)
            inc = P.incident_plane_wave(C.N, d["k"], d["theta"], DEV).to(
                torch.complex128)
            uu = (d["psi"] - inc).to(torch.complex128)
            b1, b2 = born_pair(d["v"], d["k"], d["theta"])
            psi = d["psi"].to(torch.complex128)
            anch = rel(uu, a * b1 + b * b2)                 # anchored
            # trivial: a predictor that returns the incident field scores
            # ||u|| / ||psi||, the error the trained plateau lands on
            nu = torch.linalg.norm(uu.reshape(N, -1), dim=1)
            npsi = torch.linalg.norm(psi.reshape(N, -1), dim=1).clamp_min(1e-12)
            triv = (nu / npsi).cpu().numpy()
            born1 = rel(uu, b1)                             # raw order-1 Born
            born2 = rel(uu, b1 + b2)                        # raw order-2 Born
            rows.append(dict(f=f, mean=float(anch.mean()),
                             std=float(anch.std()),
                             sem=float(anch.std() / np.sqrt(N))))
            tv_rows.append(float(triv.mean()))
            if f >= 6.0:
                pl_vals.append(anch)
                tv_vals.append(triv)
            bn_rows.append([float(born1.mean()), float(born2.mean())])
            if fi % 4 == 0:
                print("   f=%5.2f  anchor %.3f  trivial %.3f  born1 %.3f"
                      % (f, anch.mean(), tv_rows[-1], bn_rows[-1][0]),
                      flush=True)
        plateau = float(np.mean([r["mean"] for r in rows
                                 if r["f"] >= 6.0]))
        # Same plateau, resolved per media, so panel (b) of the anchor figure
        # can carry a bar at every contrast point.  The mean of the per-media
        # plateaus is by construction the mean of the per-f means; the assert
        # keeps the two bookkeeping routes from drifting apart.
        pl = np.mean(np.stack(pl_vals), axis=0)
        tv = np.mean(np.stack(tv_vals), axis=0)
        assert abs(float(pl.mean()) - plateau) < 1e-12, \
            "per-media plateau %.12f != per-f plateau %.12f" % (pl.mean(),
                                                                plateau)
        out["arms"]["%.2f" % vmax] = dict(
            a=[float(a.real), float(a.imag)], b=[float(b.real), float(b.imag)],
            in_band=float(np.mean(ib_rows)), sweep=rows,
            trivial=[float(x) for x in tv_rows],
            born1=[float(x[0]) for x in bn_rows],
            born2=[float(x[1]) for x in bn_rows],
            plateau=plateau,
            plateau_sd=float(pl.std(ddof=1)),
            plateau_sem=float(pl.std(ddof=1) / np.sqrt(N)),
            trivial_plateau=float(np.mean([tv_rows[i] for i, f in
                                           enumerate(C.F_TEST) if f >= 6.0])),
            trivial_plateau_sem=float(tv.std(ddof=1) / np.sqrt(N)),
            e4=float(rows[[r["f"] for r in rows].index(4.0)]["mean"]),
            e5=float(rows[[r["f"] for r in rows].index(5.0)]["mean"]))
        print("   -> in-band %.3f | e(4) %.3f e(5) %.3f | plateau %.3f "
              "(trivial %.3f)"
              % (out["arms"]["%.2f" % vmax]["in_band"],
                 out["arms"]["%.2f" % vmax]["e4"],
                 out["arms"]["%.2f" % vmax]["e5"], plateau,
                 out["arms"]["%.2f" % vmax]["trivial_plateau"]), flush=True)

    path = os.path.join(C.RES, "anchor_baseline.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
