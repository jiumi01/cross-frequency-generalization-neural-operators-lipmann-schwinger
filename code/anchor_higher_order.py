# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Anchor depth scan: how much of the out-of-band plateau is closed by B1..BM?

The manuscript's anchor (Table 7) is a four-parameter fit on (B1, B2).  It
already removes the cliff -- r = e(5)/e(4) lands in [0.97, 1.05] against the
trained operator's 7.19--8.30 -- but it is a FLOOR, not a method, and it is
inherently order-limited: the residual it cannot see is the multiple-scattering
tail that the truncated Born series leaves behind.  The tail shrinks as the
contrast falls, which is exactly why the anchor's own error grows monotonically
with contrast (0.023 at v=0.05 up to 0.703 at v=1.00) while staying flat in
frequency.

This script asks the next question without a GPU: does adding analytic orders
B3, B4 (still FFT-only, still label-free, still fitted on the same in-band
cache prefix) buy plateau?  Three outcomes are possible and all three are
reportable:

  * monotone gain  -> the plateau is a truncation artefact, the anchor's
    contrast scaling is a series-truncation statement, and the reading
    "contrast-controlled residual" is sharpened into a quantitative one;
  * no gain        -> the residual is not series-truncation error at all;
  * divergence     -> the Born series is being pushed past its radius of
    convergence and the fit is absorbing garbage (this MUST be detected rather
    than silently reported as a gain).

Divergence is the live risk and it is why the Gram matrix condition number is
recorded per (contrast, order): a least-squares fit on a basis whose Gram is
singular to working precision is not a measurement, it is a rounding artefact
dressed as one.  Per the project's instrument-discipline rule, the fit is
refused above a conditioning ceiling instead of being reported.

CPU only.  Nothing here trains and nothing here may share the GPU with a
training job.
"""
import os
import sys
import json
import numpy as np
import glob
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import physics as P

DEV = "cpu"
DT = torch.complex128
N = 200                 # the protocol's sample count
N_FIT = 512             # cache prefix used to fit the coefficients
ORDERS = (1, 2, 3, 4)
COND_CEIL = 1e12        # above this the normal equations are noise
# --max-order extends the scan; the point of going past 4 is to FIND the
# reversal.  Once rho = ||B_{j+1}||/||B_j|| exceeds one the series is being
# summed outside its convergence disk, and a least-squares fit over a
# divergent asymptotic series must eventually get worse, not better.  Where
# that turning point sits is a measurement, not an opinion, and reporting only
# the improving part of the curve would be selective reporting.
CACHE = {0.05: "train_a005", 0.10: "train_a010", 0.25: "train_a025",
         0.50: "train_a050", 1.00: "train_a100"}


def born_terms(v, k, theta, order, gamma=None, Ngrid=None):
    """(B1, ..., Border): successive Born corrections of the incident field.

    One pass over the Neumann recursion, no linear solve, no label:
    B1 = kappa^2 G (v psi_inc), B_{j+1} = kappa^2 G (v B_j).
    """
    gamma = C.GAMMA if gamma is None else gamma
    Ngrid = C.N if Ngrid is None else Ngrid
    dev = v.device
    vc = v.to(DT)
    Ghat = P.green_symbol(Ngrid, k, gamma, dev)
    kappa2 = ((k.to(DT) * (1.0 + 1j * gamma)) ** 2)[:, None, None]

    def op(psi):
        return kappa2 * torch.fft.ifft2(Ghat * torch.fft.fft2(vc * psi))

    psi_inc = P.incident_plane_wave(Ngrid, k, theta, dev).to(DT)
    out, cur = [], op(psi_inc)
    for _ in range(order):
        out.append(cur)
        cur = op(cur)
    return out


def flatten(x, n):
    return x.reshape(n, -1)


def fit(v, k, theta, u, order):
    """Complex least squares on [B1..Border]; returns (coeffs, cond, ok)."""
    terms = born_terms(v, k, theta, order)
    n = u.shape[0]
    y = flatten(u, n)
    m = torch.stack([flatten(t, n) for t in terms], dim=1)   # (n, d, order)
    gram = torch.einsum("njd,nkd->jk", m.conj(), m).cpu().numpy()
    rhs = torch.einsum("njd,nd->j", m.conj(), y).cpu().numpy()
    cond = float(np.linalg.cond(gram))
    if not np.isfinite(cond) or cond > COND_CEIL:
        return None, cond, False
    c = np.linalg.solve(gram, rhs)
    return c, cond, True


def apply_coeffs(terms, c):
    return sum(complex(ci) * t for ci, t in zip(c, terms))


def rel(a, b, n):
    num = torch.linalg.norm((a - b).reshape(n, -1), dim=1)
    den = torch.linalg.norm(a.reshape(n, -1), dim=1).clamp_min(1e-12)
    return (num / den).cpu().numpy()


def cache_batch(cache, n_fit, device=DEV):
    zs = [np.load(p) for p in sorted(glob.glob(os.path.join(C.DATA, cache,
                                                            "*.npz")))]
    need = int(np.ceil(n_fit / zs[0]["v"].shape[0]))
    v = torch.from_numpy(np.concatenate([z["v"] for z in zs[:need]])).to(device)
    psi = torch.from_numpy(np.concatenate(
        [z["psi"] for z in zs[:need]])).to(device)
    k = torch.from_numpy(np.concatenate([z["k"] for z in zs[:need]])).to(device)
    th = torch.from_numpy(np.concatenate(
        [z["theta"] for z in zs[:need]])).to(device)
    v, psi, k, th = v[:n_fit], psi[:n_fit], k[:n_fit], th[:n_fit]
    pi = P.incident_plane_wave(C.N, k, th, device).to(DT)
    return v.float(), k, th, (psi - pi).to(DT)


def main():
    out = dict(
        n=N, n_fit=N_FIT, grid=C.N, gamma=C.GAMMA, orders=list(ORDERS),
        cond_ceiling=COND_CEIL, f_test=list(C.F_TEST),
        protocol="identical to code/anchor_baseline.py: coefficients fitted by "
                 "least squares on a 512-sample prefix of that contrast's own "
                 "training cache (in-band only, no test sample, no out-of-band "
                 "label), scored on the manuscript's eval seeds (70000+vi in "
                 "band, 60000+vi per frequency) at n=200.",
        arms={})
    for vi, vmax in enumerate(C.VMAX_LIST):
        v, k, th, u = cache_batch(CACHE[vmax], N_FIT)
        key = "%.2f" % vmax
        out["arms"][key] = dict(orders={})

        # ---- evaluation media, drawn once, reused for every order ----
        cells = []
        d = D.sample_batch(N, 70000 + vi, DEV, f_lo=1.0, f_hi=4.0,
                           vmax_lo=vmax, vmax_hi=vmax)
        inc = P.incident_plane_wave(C.N, d["k"], d["theta"], DEV).to(DT)
        cells.append(("in_band", None, d, (d["psi"] - inc).to(DT)))
        for f in C.F_TEST:
            d = D.sample_batch(N, 60000 + vi, DEV, vmax_lo=vmax, vmax_hi=vmax,
                               f_fixed=f)
            inc = P.incident_plane_wave(C.N, d["k"], d["theta"], DEV).to(DT)
            cells.append(("sweep", f, d, (d["psi"] - inc).to(DT)))

        # ---- Born terms per cell, up to the deepest order, once ----
        term_cache = {}
        for tag, f, d, _ in cells:
            term_cache[(tag, f)] = born_terms(d["v"], d["k"], d["theta"],
                                              max(ORDERS))

        for order in ORDERS:
            c, cond, ok = fit(v, k, th, u, order)
            if not ok:
                out["arms"][key]["orders"][str(order)] = dict(
                    fitted=False, cond=cond,
                    note="Gram condition %.2e exceeds %.0e -- the fit would be "
                         "a rounding artefact, refused." % (cond, COND_CEIL))
                print("v=%.2f order=%d  REFUSED (cond=%.2e)" %
                      (vmax, order, cond), flush=True)
                continue
            sweep, ib = [], []
            for tag, f, d, uu in cells:
                terms = term_cache[(tag, f)][:order]
                e = rel(uu, apply_coeffs(terms, c), N)
                if tag == "in_band":
                    ib.append(float(e.mean()))
                else:
                    sweep.append(dict(f=f, mean=float(e.mean()),
                                      std=float(e.std()),
                                      sem=float(e.std() / np.sqrt(N)),
                                      finite=bool(np.isfinite(e.mean()))))
            fs = [r["f"] for r in sweep]
            e4 = float(sweep[fs.index(4.0)]["mean"])
            e5 = float(sweep[fs.index(5.0)]["mean"])
            plateau = float(np.mean([r["mean"] for r in sweep
                                     if r["f"] >= 6.0]))
            out["arms"][key]["orders"][str(order)] = dict(
                fitted=True, cond=cond,
                coeffs=[[float(z.real), float(z.imag)] for z in c],
                in_band=float(np.mean(ib)), sweep=sweep, e4=e4, e5=e5,
                r=float(e5 / e4), plateau=plateau)
            print("v=%.2f order=%d  cond=%.1e  in-band %.3f | e(4) %.3f "
                  "e(5) %.3f r=%.2f | plateau %.3f"
                  % (vmax, order, cond, np.mean(ib), e4, e5, e5 / e4, plateau),
                  flush=True)

    path = os.path.join(C.RES, "anchor_higher_order.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % path)


if __name__ == "__main__":
    if "--max-order" in sys.argv:
        m = int(sys.argv[sys.argv.index("--max-order") + 1])
        ORDERS = tuple(range(1, m + 1))
    main()
