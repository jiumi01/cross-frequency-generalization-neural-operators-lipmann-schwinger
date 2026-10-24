# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Measuring the Born contraction factor rho = ||B_{j+1}|| / ||B_j||.

Why this exists.  The factorization reading in the manuscript says the solution
operator's frequency dependence splits into an analytic part (the Born head,
whose spectral locus moves with k by construction and which is computable by
FFT at any frequency with no label) and a residual that is "contrast-controlled
and near-frequency-independent".  That second clause was, until now, an
inference from the anchor's error profile: the anchor's error is flat in
frequency and grows monotonically with contrast (0.023 at v=0.05 to 0.703 at
v=1.00 in the order-2 fit).  Flat-in-frequency error is evidence, but indirect
evidence -- it is compatible with several accounts.

The residual of a truncated Born series is governed by the contraction factor
of the Neumann operator T psi = kappa^2 G(v psi), and that factor is directly
measurable as the ratio of successive Born corrections:

    rho_j = ||B_{j+1}|| / ||B_j||.

If the reading is right, rho should depend strongly on the contrast v and only
weakly on the frequency f.  This script measures rho(v, f) on the manuscript's
own evaluation protocol (n=200, seeds 60000+vi per frequency, 70000+vi in
band) and reports the two numbers that matter:

  * rho_spread(v) = max_f rho / min_f rho  -- how much the residual's
    convergence rate actually moves with frequency;
  * rho_level(v)  = the geometric mean over f -- how it scales with contrast.

The comparison that decides the reading is rho_spread against the cliff the
trained operators carry (7.19--8.30x in e(5)/e(4)).  If rho moves by, say,
1.3x across f while the boundary error moves by 7--8x, then the frequency
dependence of the problem is NOT sitting in the residual's convergence rate:
it is sitting in the analytic part, which is exactly what the reading asserts
and exactly what the anchor exploits.

Companion to code/anchor_higher_order.py, which measured the CONSEQUENCE (the
anchor's plateau decays geometrically with Born order); this measures the
MECHANISM (the decay rate itself, and what it depends on).

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
DT = torch.complex128
N = 200
DEPTH = 6                 # B1..B6, so rho_1..rho_5
RHO_USE = (1, 2, 3)       # indices actually aggregated (early terms dominate
                          # the truncation error the anchor cannot see)
F_INBAND = (1.5, 2.5, 3.5)


def born_terms(v, k, theta, order, gamma=None, Ngrid=None):
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


def norms(t):
    return torch.linalg.norm(t.reshape(t.shape[0], -1), dim=1)


def rho_table(v, k, theta):
    """Geometric mean over samples of ||B_{j+1}||/||B_j||, for j = 1..DEPTH-1.

    Geometric (not arithmetic) mean: the ratio is a multiplicative quantity and
    a single near-singular sample would otherwise dominate the average.
    """
    B = born_terms(v, k, theta, DEPTH)
    n = v.shape[0]
    out = []
    for j in range(DEPTH - 1):
        r = (norms(B[j + 1]) / norms(B[j]).clamp_min(1e-300)).cpu().numpy()
        r = r[np.isfinite(r) & (r > 0)]
        if r.size == 0:
            out.append(float("nan"))
        else:
            out.append(float(np.exp(np.mean(np.log(r)))))
    return out


def cell_rho(vi, vmax, f=None, in_band=False):
    if in_band:
        d = D.sample_batch(N, 70000 + vi, DEV, f_lo=1.0, f_hi=4.0,
                           vmax_lo=vmax, vmax_hi=vmax)
    else:
        d = D.sample_batch(N, 60000 + vi, DEV, vmax_lo=vmax, vmax_hi=vmax,
                           f_fixed=f)
    return rho_table(d["v"], d["k"], d["theta"])


def main():
    out = dict(n=N, grid=C.N, gamma=C.GAMMA, depth=DEPTH,
               rho_indices=list(RHO_USE),
               protocol="the manuscript's own evaluation protocol: n=200 media "
                        "per cell, seeds 60000+vi per test frequency and "
                        "70000+vi in band, one contrast per arm. rho is the "
                        "geometric mean over samples of ||B_{j+1}||/||B_j||.",
               note="rho_j uses the un-normalised Born corrections, i.e. the "
                    "actual contraction realised by the Neumann operator on "
                    "this medium at this frequency; no coefficient fit and no "
                    "label enters.",
               arms={})
    for vi, vmax in enumerate(C.VMAX_LIST):
        rows = []
        ib = cell_rho(vi, vmax, in_band=True)
        rows.append(dict(f="in-band", rho=[float(x) for x in ib]))
        for f in C.F_TEST:
            rows.append(dict(f=float(f),
                             rho=[float(x) for x in cell_rho(vi, vmax, f=f)]))
        # aggregate over the early indices, across all test frequencies
        per_f = []
        for r in rows[1:]:
            vals = [r["rho"][j - 1] for j in RHO_USE]
            per_f.append(float(np.exp(np.mean(np.log(vals)))))
        lo, hi = float(np.min(per_f)), float(np.max(per_f))
        out["arms"]["%.2f" % vmax] = dict(
            rows=rows,
            in_band_rho=[float(x) for x in ib],
            rho_min_over_f=lo, rho_max_over_f=hi,
            rho_spread=float(hi / lo),
            rho_level=float(np.exp(np.mean(np.log(per_f)))),
            rho_at_f4=per_f[[float(r["f"]) for r in rows[1:]].index(4.0)],
            rho_at_f12=per_f[[float(r["f"]) for r in rows[1:]].index(12.0)])
        print("v=%.2f  rho(level)=%.3f  spread=%.2f  [min %.3f max %.3f]  "
              "rho(4)=%.3f rho(12)=%.3f  in-band rho=%s"
              % (vmax, out["arms"]["%.2f" % vmax]["rho_level"],
                 out["arms"]["%.2f" % vmax]["rho_spread"], lo, hi,
                 out["arms"]["%.2f" % vmax]["rho_at_f4"],
                 out["arms"]["%.2f" % vmax]["rho_at_f12"],
                 "[%s]" % ", ".join("%.3f" % x for x in ib[:3])), flush=True)

    path = os.path.join(C.RES, "anchor_contraction.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
