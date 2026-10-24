# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Order selection for the analytic head: can M* be chosen IN BAND?

code/anchor_higher_order.py established that the anchor's out-of-band plateau
is Born-truncation error and that it falls geometrically with the order M --
at v=0.25 from 0.517 (M=1) to 0.022 (M=8), against the trained operator's
0.963.  It also established that the improvement is not unbounded: at v=1.00
the curve turns (0.516 at M=5, 0.517 at M=6) and by M=8 the normal equations
are singular to working precision and the fit is refused.  So M is a
hyperparameter that matters and that cannot simply be maximised.

The question this script answers is whether M can be chosen WITHOUT looking
out of band -- because if the choice needs out-of-band labels, the whole
construction stops being a zero-shot object and the manuscript's central
distinction collapses.

The procedure is deliberately mundane: fit the M complex coefficients on one
in-band prefix, score them on a DISJOINT in-band prefix, take the argmin.  No
test sample, no out-of-band frequency, no label beyond f <= 4.  The
out-of-band consequence of that choice is then measured on the manuscript's
own protocol and reported -- including, honestly, alongside the order that
would have been best out of band had one been allowed to peek.

Reported per contrast:
  * M_sel      -- order chosen by in-band validation;
  * M_oracle   -- order with the lowest out-of-band plateau (peeking; reported
                  only as a reference for how much the honest rule gives up);
  * plateau(M_sel), plateau(M_oracle), plateau(M=2) -- the manuscript's
    existing four-parameter anchor -- and the trained operator's plateau.

CPU only.  Nothing here trains and nothing here may share the GPU with a
training job.
"""
import os
import sys
import json
import glob
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import physics as P

DEV = "cpu"
DT = torch.complex128
N_FIT = 512             # prefix used to fit the coefficients
N_VAL = 512             # DISJOINT prefix used to choose the order
N = 200                 # out-of-band scoring, the manuscript's protocol
MAX_M = 8
COND_CEIL = 1e12
CACHE = {0.05: "train_a005", 0.10: "train_a010", 0.25: "train_a025",
         0.50: "train_a050", 1.00: "train_a100"}


def born_terms(v, k, theta, order):
    vc = v.to(DT)
    Ghat = P.green_symbol(C.N, k, C.GAMMA, v.device)
    kappa2 = ((k.to(DT) * (1.0 + 1j * C.GAMMA)) ** 2)[:, None, None]

    def op(psi):
        return kappa2 * torch.fft.ifft2(Ghat * torch.fft.fft2(vc * psi))

    psi_inc = P.incident_plane_wave(C.N, k, theta, v.device).to(DT)
    out, cur = [], op(psi_inc)
    for _ in range(order):
        out.append(cur)
        cur = op(cur)
    return out


def flat(x, n):
    return x.reshape(n, -1)


def fit(v, k, theta, u, order):
    terms = born_terms(v, k, theta, order)
    n = u.shape[0]
    m = torch.stack([flat(t, n) for t in terms], dim=1)
    gram = torch.einsum("njd,nkd->jk", m.conj(), m).cpu().numpy()
    rhs = torch.einsum("njd,nd->j", m.conj(), flat(u, n)).cpu().numpy()
    cond = float(np.linalg.cond(gram))
    if not np.isfinite(cond) or cond > COND_CEIL:
        return None, cond
    return np.linalg.solve(gram, rhs), cond


def apply(terms, c):
    return sum(complex(ci) * t for ci, t in zip(c, terms))


def rel(a, b, n):
    num = torch.linalg.norm((a - b).reshape(n, -1), dim=1)
    den = torch.linalg.norm(a.reshape(n, -1), dim=1).clamp_min(1e-12)
    return (num / den).cpu().numpy()


def cache_batch(cache, need_total):
    zs = [np.load(p) for p in sorted(glob.glob(os.path.join(C.DATA, cache,
                                                            "*.npz")))]
    k0 = int(np.ceil(need_total / zs[0]["v"].shape[0]))
    v = torch.from_numpy(np.concatenate([z["v"] for z in zs[:k0]]))
    psi = torch.from_numpy(np.concatenate([z["psi"] for z in zs[:k0]]))
    k = torch.from_numpy(np.concatenate([z["k"] for z in zs[:k0]]))
    th = torch.from_numpy(np.concatenate([z["theta"] for z in zs[:k0]]))
    v, psi, k, th = (v[:need_total], psi[:need_total],
                     k[:need_total], th[:need_total])
    pi = P.incident_plane_wave(C.N, k, th, DEV).to(DT)
    return v.float(), k, th, (psi - pi).to(DT)


def main():
    out = dict(n_fit=N_FIT, n_val=N_VAL, n_eval=N, max_m=MAX_M,
               cond_ceiling=COND_CEIL, gamma=C.GAMMA,
               procedure="coefficients fitted on the first %d in-band cache "
                         "samples, order chosen by validation error on the "
                         "DISJOINT next %d in-band samples, out-of-band "
                         "consequence then measured on the manuscript's "
                         "protocol (n=200, seeds 60000+vi per frequency)."
                         % (N_FIT, N_VAL),
               arms={})
    for vi, vmax in enumerate(C.VMAX_LIST):
        vf, kf, thf, uf = cache_batch(CACHE[vmax], N_FIT)
        vv, kv, thv, uv = cache_batch(CACHE[vmax], N_FIT + N_VAL)
        # disjoint slice: the second block only
        vv, kv, thv, uv = (vv[N_FIT:], kv[N_FIT:], thv[N_FIT:], uv[N_FIT:])

        coeffs, conds, val_err = {}, {}, {}
        for M in range(1, MAX_M + 1):
            c, cond = fit(vf, kf, thf, uf, M)
            conds[M] = cond
            if c is None:
                val_err[M] = float("inf")
                continue
            coeffs[M] = c
            tv = born_terms(vv, kv, thv, M)
            val_err[M] = float(rel(uv, apply(tv, c), uv.shape[0]).mean())
        M_sel = min(range(1, MAX_M + 1), key=lambda m: (val_err[m], m))
        if not np.isfinite(val_err[M_sel]):
            print("v=%.2f  no order passed conditioning" % vmax, flush=True)
            continue

        # ---- out-of-band consequence, on the manuscript's protocol ----
        plateaus, ibs = {}, {}
        for f in C.F_TEST:
            d = D.sample_batch(N, 60000 + vi, DEV, vmax_lo=vmax, vmax_hi=vmax,
                               f_fixed=f)
            inc = P.incident_plane_wave(C.N, d["k"], d["theta"], DEV).to(DT)
            uu = (d["psi"] - inc).to(DT)
            tv = born_terms(d["v"], d["k"], d["theta"], MAX_M)
            for M in coeffs:
                plateaus.setdefault(M, []).append(
                    (f, float(rel(uu, apply(tv[:M], coeffs[M]), N).mean())))
        d = D.sample_batch(N, 70000 + vi, DEV, f_lo=1.0, f_hi=4.0,
                           vmax_lo=vmax, vmax_hi=vmax)
        inc = P.incident_plane_wave(C.N, d["k"], d["theta"], DEV).to(DT)
        uu = (d["psi"] - inc).to(DT)
        tv = born_terms(d["v"], d["k"], d["theta"], MAX_M)
        for M in coeffs:
            ibs[M] = float(rel(uu, apply(tv[:M], coeffs[M]), N).mean())

        def plat(M):
            return float(np.mean([e for f, e in plateaus[M] if f >= 6.0]))

        M_or = min(coeffs, key=lambda m: plat(m))
        out["arms"]["%.2f" % vmax] = dict(
            val_err={str(m): val_err[m] for m in sorted(val_err)},
            cond={str(m): conds[m] for m in sorted(conds)},
            in_band={str(m): ibs[m] for m in sorted(ibs)},
            plateau={str(m): plat(m) for m in sorted(coeffs)},
            M_sel=M_sel, M_oracle=M_or,
            plateau_sel=plat(M_sel), plateau_oracle=plat(M_or),
            plateau_m2=plat(2) if 2 in coeffs else None,
            in_band_sel=ibs[M_sel])
        print("v=%.2f  M_sel=%d (val %.4f) -> plateau %.4f | M_oracle=%d -> "
              "%.4f | M=2 -> %.4f | in-band(M_sel) %.4f"
              % (vmax, M_sel, val_err[M_sel], plat(M_sel), M_or, plat(M_or),
                 plat(2), ibs[M_sel]), flush=True)

    path = os.path.join(C.RES, "anchor_order_selection.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
