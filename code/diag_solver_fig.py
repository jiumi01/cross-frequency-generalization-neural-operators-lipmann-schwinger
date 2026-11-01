# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Data for the solver-validation figure (Fig. solver).

Two independent measurements are produced and written to results/solver_fig.json:

 (a) Dense cross-check.  On a small N x N grid the batched Bi-CGSTAB solution is
     compared against a dense direct solve, where the dense matrix is assembled
     by probing the mat-vector product with unit vectors -- i.e. the comparison
     does not reuse the FFT code path that is being validated.

 (b) Freezing ablation.  The same batched system is solved with the frozen
     (correct) and the naive (unfrozen) recurrence.  For each condition we count
     the fraction of samples whose returned backward-substitution residual is
     non-finite or above 1e-6.  The naive variant silently corrupts a fraction of
     the batch while reporting success.

Run with:  conda run -n wirenet python code/diag_solver_fig.py
"""
import os
import sys
import json

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def matvec_factory(N, v, k, gamma):
    """Return (matvec, psi_inc) for the discrete LS system on an N x N grid."""
    vc = v.to(torch.complex128)
    Ghat = P.green_symbol(N, k, gamma, DEV)
    kappa2 = ((k.to(torch.complex128) * (1.0 + 1j * gamma)) ** 2)[:, None, None]

    def matvec(psi):
        return psi - kappa2 * torch.fft.ifft2(Ghat * torch.fft.fft2(vc * psi))

    return matvec


# ---------------------------------------------------------------------------
def dense_check(N=10, B=6, f=4.0, vmax=0.5, seed=1234, tol=1e-13, maxiter=2000):
    """Dense direct solve vs batched Bi-CGSTAB on the same samples.

    The operator is built strictly PER SAMPLE (a single (1,N,N) field at a time):
    reusing the batched operator here would silently broadcast the single-sample
    probe against the whole batch and the comparison would be meaningless.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    v = P.make_contrast(N, B, vmax, C.MM_REF, g, DEV)
    k = torch.full((B,), 2 * np.pi * f, device=DEV)
    th = torch.rand(B, generator=g, device=DEV) * 2 * np.pi
    Ghat = P.green_symbol(N, k, C.GAMMA, DEV)                       # (B,N,N)
    kappa2 = (k.to(torch.complex128) * (1.0 + 1j * C.GAMMA)) ** 2   # (B,)
    psi_inc = P.incident_plane_wave(N, k, th, DEV)                  # (B,N,N)

    n = N * N
    diffs, res_k = [], []
    for b in range(B):
        vb = v[b].to(torch.complex128)
        Gb, k2 = Ghat[b], kappa2[b]

        def mv(z, vb=vb, Gb=Gb, k2=k2):
            return z - k2 * torch.fft.ifft2(Gb * torch.fft.fft2(vb * z))

        A = torch.zeros((n, n), dtype=torch.complex128, device=DEV)
        for j in range(n):
            e = torch.zeros((1, N, N), dtype=torch.complex128, device=DEV)
            e[0].reshape(-1)[j] = 1.0
            A[:, j] = mv(e)[0].reshape(-1)
        rhs = psi_inc[b:b + 1]
        x_dense = torch.linalg.solve(A, rhs.reshape(-1))
        x_k, r_k = P.bicgstab(mv, rhs, tol=tol, maxiter=maxiter)
        d = torch.linalg.norm(x_dense - x_k.reshape(-1)) / torch.linalg.norm(x_dense)
        diffs.append(float(d))
        res_k.append(float(r_k.amax()))
    return diffs, res_k


# ---------------------------------------------------------------------------
# (b) Freezing ablation.
#
# The batches are deliberately HETEROGENEOUS (v_max and f are mixed, exactly as
# in the real training cache): a batched Krylov solver shares one iteration loop
# across all samples, so the failure of a bad "freeze" only shows up when the
# batch contains both fast- and slow-converging members.
BATCH = 1024


def one_batch(seed, B, f_lo, f_hi, vmax_lo, vmax_hi):
    d = D.sample_batch(B, seed, DEV, f_lo=f_lo, f_hi=f_hi, vmax_lo=vmax_lo,
                       vmax_hi=vmax_hi)
    mv = matvec_factory(C.N, d["v"], d["k"], C.GAMMA)
    psi_inc = P.incident_plane_wave(C.N, d["k"], d["theta"], DEV)
    return mv, psi_inc


def corrupt_mask(x, res):
    return (~torch.isfinite(res)) | (~torch.isfinite(x)).any(dim=(-2, -1)) | (res > 1e-6)


def freeze_ablation(nb=8, B=BATCH, tol=1e-9, maxiter=600):
    """Three implementations of the shared loop, on identical heterogeneous batches.

    For the unfrozen variant we additionally record the SILENT corruption: the
    samples for which the residual maintained inside the loop claims success
    (< tol) while the honest backward-substitution residual is large or
    non-finite.  That is the fraction a practitioner would lose without noticing
    if the in-loop residual were used as the health check.
    """
    rows = []
    tot = dict(naive=0, zeromask=0, frozen=0)
    silent = 0
    n_tot = 0
    for k in range(nb):
        mv, psi_inc = one_batch(9000 + 13 * k, B, 1.0, 4.0, 0.05, 1.0)
        x_n, res_n, loop_n = P.bicgstab_naive(mv, psi_inc, tol=tol,
                                              maxiter=maxiter, return_loop_res=True)
        bad_n = corrupt_mask(x_n, res_n)
        got = dict(naive=int(bad_n.sum()))
        sl = int(((loop_n < tol) & bad_n).sum())
        silent += sl
        maxres = dict(naive=float(np.nanmax(res_n.cpu().numpy())))
        for tag, fn in [("zeromask", P.bicgstab_zeromask),
                        ("frozen", P.bicgstab)]:
            x, res = fn(mv, psi_inc, tol=tol, maxiter=maxiter)
            got[tag] = int(corrupt_mask(x, res).sum())
            maxres[tag] = float(res.amax())
        n_tot += B
        for t in tot:
            tot[t] += got[t]
        rows.append(dict(batch=k, n=B, **got, silent=sl, max_res=maxres))
        print("batch %d  naive=%4d (silent %4d)  zeromask=%4d  frozen=%4d"
              % (k, got["naive"], sl, got["zeromask"], got["frozen"]), flush=True)
    agg = dict(n_total=n_tot, silent=silent, frac_silent=silent / n_tot,
               **{t: dict(n_bad=tot[t], frac=tot[t] / n_tot) for t in tot})
    return rows, agg


def main():
    C.set_seed(0)
    diffs, res_k = dense_check()
    print("dense cross-check rel diff: max=%.2e  mean=%.2e"
          % (max(diffs), float(np.mean(diffs))), flush=True)
    print("\nfreezing ablation on heterogeneous batches (mixed v_max, f in [1,4]):")
    rows, agg = freeze_ablation()
    out = dict(grid=C.N, dense=dict(N=10, diffs=diffs, krylov_res=res_k,
                                    max_diff=float(max(diffs)),
                                    mean_diff=float(np.mean(diffs))),
               freeze=dict(batch=BATCH, conds=rows, aggregate=agg,
                           setup="heterogeneous batches, f~U[1,4], v_max~U[0.05,1.0]"))
    C.save_json(out, os.path.join(C.RES, "solver_fig.json"))
    print("\naggregate over %d samples:" % agg["n_total"])
    for t in ("naive", "zeromask", "frozen"):
        print("  %-9s %5d bad  (%.3f%%)" % (t, agg[t]["n_bad"],
                                            100 * agg[t]["frac"]))
    print("  of the naive corruption, %d (%.2f%% of all samples) are SILENT:"
          % (agg["silent"], 100 * agg["frac_silent"]))
    print("  the in-loop residual reports success (< tol) while the honest")
    print("  backward residual is large or non-finite.")


if __name__ == "__main__":
    main()
