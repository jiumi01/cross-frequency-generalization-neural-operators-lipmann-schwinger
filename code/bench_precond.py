# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Cost of the learned operator against a STRONG reference solver.

The paper's cost table times the learned forward pass against an
unpreconditioned batched Bi-CGSTAB solve and says so, framing the speed-up
as an upper bound.  An upper bound against a deliberately weak baseline is
not a cost comparison, so this script supplies the missing strong baseline:

  (i)   the unpreconditioned Bi-CGSTAB reference solve (the paper's number),
  (ii)  the same solve with a frozen-coefficient preconditioner: the operator
        with the contrast replaced by its per-sample spatial mean, which is
        diagonal in the Fourier basis and therefore inverted exactly by one
        FFT pair per application.

Both run on the identical 64x64 problem, the identical media draw, the
identical frequencies, tolerance and iteration cap as code/bench_cost.py,
so the two numbers are directly comparable.  The device is CPU by design:
the headline quantity is the iteration count, which is device-independent,
and a GPU timing run would contend with whatever training is sharing the
card.

The iteration loop is a faithful copy of physics.bicgstab (same frozen-sample
discipline, same honest backward-substitution residual) with per-sample
convergence-iteration bookkeeping added; it is validated against
physics.solve_ls on the identity preconditioner before any timing is taken.

Writes results/cost_precond.json.
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P


def pc_bicgstab(matvec, precond, b, tol=1e-10, maxiter=400):
    """Right-preconditioned batched Bi-CGSTAB with per-sample iteration counts.

    Solves A x = b via (A M^-1) y = b, x = M^-1 y.  `precond` applies M^-1.
    Returns x, honest residual per sample, iterations per sample.
    """
    B = b.shape[0]
    y = torch.zeros_like(b)
    r = b - matvec(precond(y))
    r0 = r.clone()
    bnorm = torch.linalg.norm(b.reshape(B, -1), dim=1).clamp_min(1e-30)
    res = torch.linalg.norm(r.reshape(B, -1), dim=1) / bnorm
    done = res < tol
    iters = torch.where(done, torch.zeros(B, dtype=torch.long),
                        torch.full((B,), maxiter, dtype=torch.long))

    rho_old = torch.ones(B, dtype=b.dtype, device=b.device)
    alpha = torch.ones(B, dtype=b.dtype, device=b.device)
    omega = torch.ones(B, dtype=b.dtype, device=b.device)
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)

    def ip(a, c):
        return torch.sum(torch.conj(a) * c, dim=(-2, -1))

    it_done = 0
    for it in range(maxiter):
        if bool(done.all()):
            break
        act = ~done
        rho = ip(r0, r)
        rho = torch.where(act, rho, torch.ones_like(rho))
        rho = torch.where(rho.abs() < 1e-300, torch.ones_like(rho), rho)
        beta = (rho / rho_old) * (alpha / omega)
        beta = torch.where(act, beta, torch.zeros_like(beta))
        p = r + beta[:, None, None] * (p - omega[:, None, None] * v)
        v = matvec(precond(p))
        denom = ip(r0, v)
        alpha = rho / torch.where(denom.abs() < 1e-300, denom + 1e-30, denom)
        alpha = torch.where(act, alpha, torch.zeros_like(alpha))
        s = r - alpha[:, None, None] * v
        t = matvec(precond(s))
        tt = ip(t, t)
        omega = ip(t, s) / torch.where(tt.abs() < 1e-300, tt + 1e-30, tt)
        omega_step = torch.where(act, omega, torch.zeros_like(omega))
        omega = torch.where(act, omega, torch.ones_like(omega))
        y = y + alpha[:, None, None] * p + omega_step[:, None, None] * s
        r = s - omega_step[:, None, None] * t
        rho_old = rho
        it_done = it + 1
        res = torch.linalg.norm(r.reshape(B, -1), dim=1) / bnorm
        fresh = act & (res < tol)
        iters = torch.where(fresh, torch.full_like(iters, it + 1), iters)
        done = done | (~torch.isfinite(res)) | (res < tol)

    x = precond(y)
    res = torch.linalg.norm((b - matvec(x)).reshape(B, -1), dim=1) / bnorm
    return x, res, iters


def make_matvec(v, k_arr, gamma, N, dev):
    kappa2 = ((k_arr.to(torch.complex128) * (1.0 + 1j * gamma)) ** 2)[:, None, None]
    Ghat = P.green_symbol(N, k_arr, gamma, dev)

    def matvec(psi):
        return psi - kappa2 * torch.fft.ifft2(Ghat * torch.fft.fft2(v * psi))
    return matvec


def make_frozen_precond(v, k_arr, gamma, N, dev):
    """M^-1 for the frozen-coefficient operator at the per-sample mean contrast."""
    kappa2 = ((k_arr.to(torch.complex128) * (1.0 + 1j * gamma)) ** 2)[:, None, None]
    Ghat = P.green_symbol(N, k_arr, gamma, dev)
    vbar = v.to(torch.complex128).mean(dim=(1, 2))[:, None, None]
    sym = 1.0 - kappa2 * vbar * Ghat

    def apply(r):
        return torch.fft.ifft2(torch.fft.fft2(r) / sym)
    return apply


def timeit(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=C.RES)
    a = ap.parse_args()
    dev = "cpu"
    dtype = torch.complex128

    g = torch.Generator(device=dev).manual_seed(4242)
    v = P.make_contrast(C.N, a.batch, 0.25, C.MM_REF, g, dev)
    freqs = [1.5, 4.0, 8.0, 12.0]

    # ---- validation: identity preconditioner must reproduce solve_ls -------
    k = torch.full((a.batch,), 2 * np.pi * 4.0, device=dev, dtype=torch.float64)
    th = torch.zeros(a.batch, device=dev, dtype=torch.float64)
    mv = make_matvec(v, k, C.GAMMA, C.N, dev)
    b = P.incident_plane_wave(C.N, k, th, dev)
    x_id, res_id, it_id = pc_bicgstab(mv, lambda r: r, b, tol=1e-9, maxiter=900)
    x_ref = P.solve_ls(v, k, th, C.GAMMA, C.N, tol=1e-9, maxiter=900, device=dev)
    diff = float((x_id - x_ref).abs().max())
    res_ref = float(torch.linalg.norm(
        (b - mv(x_ref)).reshape(a.batch, -1), dim=1).max()
        / torch.linalg.norm(b.reshape(a.batch, -1), dim=1).max())
    print("validation: max|x_pc - x_ref| = %.3e   honest res pc=%.3e ref=%.3e"
          % (diff, float(res_id.max()), res_ref))
    if diff > 1e-8:
        print("VALIDATION FAILED -- pc_bicgstab does not reproduce solve_ls")
        return 1

    res = {"batch": a.batch, "reps": a.reps, "device": dev,
           "validation_max_abs_diff": diff, "solver": {}}
    print("device=%s batch=%d (CPU by design; iteration counts are the "
          "device-independent quantity)" % (dev, a.batch))

    for f in freqs:
        k = torch.full((a.batch,), 2 * np.pi * f, device=dev, dtype=torch.float64)
        th = torch.zeros(a.batch, device=dev, dtype=torch.float64)
        mv = make_matvec(v, k, C.GAMMA, C.N, dev)
        pre = make_frozen_precond(v, k, C.GAMMA, C.N, dev)
        b = P.incident_plane_wave(C.N, k, th, dev)

        def once_plain():
            return P.solve_ls(v, k, th, C.GAMMA, C.N, tol=1e-9, maxiter=900,
                              device=dev)

        def once_pc():
            x, _, _ = pc_bicgstab(mv, pre, b, tol=1e-9, maxiter=900)
            return x

        t_plain = timeit(once_plain, reps=a.reps)
        t_pc = timeit(once_pc, reps=a.reps)

        _, res_plain, _ = pc_bicgstab(mv, lambda r: r, b, tol=1e-9, maxiter=900)
        _, res_pc, it_pc = pc_bicgstab(mv, pre, b, tol=1e-9, maxiter=900)
        # iteration count of the plain solve, from the same validated loop
        _, _, it_plain = pc_bicgstab(mv, lambda r: r, b, tol=1e-9, maxiter=900)

        res["solver"]["%.1f" % f] = dict(
            plain_ms=t_plain * 1e3, pc_ms=t_pc * 1e3,
            plain_per_sample_ms=t_plain * 1e3 / a.batch,
            pc_per_sample_ms=t_pc * 1e3 / a.batch,
            plain_iters_med=float(np.median(it_plain.numpy())),
            plain_iters_max=int(it_plain.max()),
            pc_iters_med=float(np.median(it_pc.numpy())),
            pc_iters_max=int(it_pc.max()),
            plain_max_res=float(res_plain.max()), pc_max_res=float(res_pc.max()),
            k=float(k[0]))
        s = res["solver"]["%.1f" % f]
        print("  f=%5.1f  plain %7.2f ms (%3d it, med %2d)   "
              "precond %7.2f ms (%3d it, med %2d)   res %.1e/%.1e"
              % (f, s["plain_ms"], s["plain_iters_max"], s["plain_iters_med"],
                 s["pc_ms"], s["pc_iters_max"], s["pc_iters_med"],
                 s["plain_max_res"], s["pc_max_res"]))

    fmax = max(res["solver"], key=float)
    s = res["solver"][fmax]
    res["headline"] = dict(
        f=fmax,
        iter_reduction=s["plain_iters_med"] / max(s["pc_iters_med"], 1e-9),
        time_ratio=s["plain_ms"] / s["pc_ms"],
        plain_iters_med=s["plain_iters_med"], pc_iters_med=s["pc_iters_med"])
    C.save_json(res, os.path.join(a.out, "cost_precond.json"))
    print("wrote %s" % os.path.join(a.out, "cost_precond.json"))
    print("headline: at f=%s the frozen-coefficient preconditioner cuts the "
          "median iteration count from %.0f to %.0f (%.1fx) and wall time by "
          "%.2fx" % (fmax, s["plain_iters_med"], s["pc_iters_med"],
                     res["headline"]["iter_reduction"],
                     res["headline"]["time_ratio"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
