# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""
Lippmann-Schwinger forward solver for 2D scalar Helmholtz scattering.

Reference problem (periodic unit cell, small absorbing loss):

    (Delta + kappa^2 (1 + v(x))) psi(x) = kappa^2 (1 - k^2/kappa^2) psi_inc(x)
    psi = psi_inc + psi_scat ,
    =>  psi = psi_inc + kappa^2 * G_kappa ( v * psi )
    =>  (I - kappa^2 * G_kappa * diag(v)) psi = psi_inc        (Lippmann-Schwinger)

with the periodic Green symbol (zeros of (Delta + kappa^2) -> -delta):

    Ghat(xi) = 1 / (|xi|^2 - kappa^2)

k = 2*pi*f is the free-space wavenumber, f = number of wavelengths across the
unit domain.  The incident field is a pure plane wave at the REAL wavenumber k,
exp(i k d.x); the complex wavenumber kappa = k(1 + i*gamma) enters only through
the propagator.  gamma > 0 is a small relative absorption (loss tangent) that
regularizes the otherwise resonant propagator.  With absorption proportional to
k the CONTINUUM propagator norm is uniform in k,
sup_xi |kappa^2 Ghat| = (1 + gamma^2) / (2 gamma)   [= 10.025 at gamma = 0.05],
attained at |xi| = k sqrt(1 - gamma^2), of which 1/(2 gamma) = 10.0 is only the
gamma -> 0 leading order (an earlier revision of this docstring quoted the
leading order as if it were the norm).  So the frequency f is the only parameter
that varies across the study and the contrast coupling v_max/(2 gamma) is
likewise k-independent.  That identity is a statement about the continuum symbol:
on an N x N grid the symbol is sampled at a discrete set of |xi|, the attained
norm does depend on k, and code/diag_propnorm.py measures the spread.

The discrete forward operator is solved to tight tolerance with a batched
Bi-CGSTAB.  All routines work in complex128 on the chosen device.
"""

import math
import numpy as np
import torch

PI = math.pi


# ----------------------------------------------------------------------------
# grid / symbol helpers
# ----------------------------------------------------------------------------
def mode_indices(N, device, dtype):
    """Integer Fourier mode indices for a periodic grid of N points on [0,1)."""
    return torch.fft.fftfreq(N, d=1.0 / N, device=device).to(dtype)


def green_symbol(N, k_arr, gamma, device, dtype=torch.float64):
    """Return Ghat of shape (B, N, N) for a batch of wavenumbers k_arr (B,)."""
    idx = mode_indices(N, device, dtype)
    kx = 2.0 * PI * idx
    XI2 = kx[:, None] ** 2 + kx[None, :] ** 2          # (N, N) real
    kappa = k_arr.to(torch.complex128) * (1.0 + 1j * gamma)
    kappa2 = (kappa ** 2)[:, None, None]               # (B,1,1) complex
    return 1.0 / (XI2.to(torch.complex128)[None, :, :] - kappa2)


# ----------------------------------------------------------------------------
# batched Bi-CGSTAB
# ----------------------------------------------------------------------------
def bicgstab(matvec, b, tol=1e-10, maxiter=400, x0=None, verbose=False):
    """Batched Bi-CGSTAB for A x = b.  b: (B, N, N) complex.  Returns x, info.

    IMPORTANT: all samples share one iteration loop, so a sample that has
    already converged must be FROZEN.  Otherwise its (near-zero) residual keeps
    feeding the recurrence -- alpha/omega become 0/0 and the converged sample
    silently turns into NaN while the loop continues for its slower siblings.
    The returned residual is recomputed by backward substitution r = b - A x so
    it is a true residual, independent of the recurrence's own bookkeeping.
    """
    B = b.shape[0]
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    r0 = r.clone()
    bnorm = torch.linalg.norm(b.reshape(B, -1), dim=1).clamp_min(1e-30)
    res = torch.linalg.norm(r.reshape(B, -1), dim=1) / bnorm
    done = res < tol

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
        rho = torch.where(act, rho, torch.ones_like(rho))            # frozen rho=1
        rho = torch.where(rho.abs() < 1e-300, torch.ones_like(rho), rho)
        beta = (rho / rho_old) * (alpha / omega)
        beta = torch.where(act, beta, torch.zeros_like(beta))
        p = r + beta[:, None, None] * (p - omega[:, None, None] * v)
        v = matvec(p)
        denom = ip(r0, v)
        alpha = rho / torch.where(denom.abs() < 1e-300, denom + 1e-30, denom)
        alpha = torch.where(act, alpha, torch.zeros_like(alpha))
        s = r - alpha[:, None, None] * v                # frozen -> s = r
        t = matvec(s)
        tt = ip(t, t)
        omega = ip(t, s) / torch.where(tt.abs() < 1e-300, tt + 1e-30, tt)
        # keep omega strictly non-zero for frozen samples so the NEXT iteration's
        # beta = (rho/rho_old)*(alpha/omega) cannot evaluate 0/0; the step that is
        # actually applied uses omega_step, which is zero for frozen samples.
        omega_step = torch.where(act, omega, torch.zeros_like(omega))
        omega = torch.where(act, omega, torch.ones_like(omega))
        x = x + alpha[:, None, None] * p + omega_step[:, None, None] * s
        r = s - omega_step[:, None, None] * t           # frozen -> r unchanged
        rho_old = rho
        it_done = it + 1
        res = torch.linalg.norm(r.reshape(B, -1), dim=1) / bnorm
        done = done | (~torch.isfinite(res)) | (res < tol)

    # honest backward-substitution residual (independent of the recurrence)
    res = torch.linalg.norm((b - matvec(x)).reshape(B, -1), dim=1) / bnorm
    if verbose:
        print("  bicgstab iters=%d  max_rel_res=%.3e" % (it_done, float(res.max())))
    return x, res


def bicgstab_naive(matvec, b, tol=1e-10, maxiter=400, return_loop_res=False):
    """The NAIVE batched Bi-CGSTAB, kept only to demonstrate the failure mode.

    Same recurrence as :func:`bicgstab` but WITHOUT masking the converged
    samples.  This is what a straightforward vectorised implementation looks
    like, and it is unsafe: a converged sample keeps contributing to the shared
    scalars, the breakdown of ``alpha`` or ``omega`` contaminates the whole
    vectorised update, and the sample is destroyed while the loop keeps
    iterating for its slower siblings.  ``diag_solver_fig.py`` quantifies the
    corruption.

    With ``return_loop_res=True`` the residual maintained *inside* the loop is
    returned as well, so that one can check whether that quantity is a valid
    health indicator (it is not -- see the caller).
    """
    B = b.shape[0]
    x = torch.zeros_like(b)
    r = b - matvec(x)
    r0 = r.clone()
    bnorm = torch.linalg.norm(b.reshape(B, -1), dim=1).clamp_min(1e-30)
    res = torch.linalg.norm(r.reshape(B, -1), dim=1) / bnorm
    loop_res = res.clone()
    done = res < tol
    rho_old = torch.ones(B, dtype=b.dtype, device=b.device)
    alpha = torch.ones(B, dtype=b.dtype, device=b.device)
    omega = torch.ones(B, dtype=b.dtype, device=b.device)
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)

    def ip(a, c):
        return torch.sum(torch.conj(a) * c, dim=(-2, -1))

    for it in range(maxiter):
        if bool(done.all()):
            break
        rho = ip(r0, r)
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta[:, None, None] * (p - omega[:, None, None] * v)
        v = matvec(p)
        alpha = rho / ip(r0, v)
        s = r - alpha[:, None, None] * v
        t = matvec(s)
        omega = ip(t, s) / ip(t, t)
        x = x + alpha[:, None, None] * p + omega[:, None, None] * s
        r = s - omega[:, None, None] * t
        rho_old = rho
        res = torch.linalg.norm(r.reshape(B, -1), dim=1) / bnorm
        loop_res = torch.where(done, loop_res, res)
        done = done | (res < tol)

    res = torch.linalg.norm((b - matvec(x)).reshape(B, -1), dim=1) / bnorm
    if return_loop_res:
        return x, res, loop_res
    return x, res


def bicgstab_zeromask(matvec, b, tol=1e-10, maxiter=400):
    """The TEMPTING-BUT-WRONG freeze: mask the converged samples to zero.

    This is the natural first attempt at 'freezing': set alpha, beta and omega to
    zero wherever the sample has converged.  It removes the immediate problem but
    creates a fresh 0/0, because the next iteration evaluates
    beta = (rho/rho_old)*(alpha/omega) with alpha = omega = 0.  Every frozen
    sample turns into NaN.  Kept to quantify the corruption in
    ``diag_solver_fig.py``; do not use for data generation.
    """
    B = b.shape[0]
    x = torch.zeros_like(b)
    r = b - matvec(x)
    r0 = r.clone()
    bnorm = torch.linalg.norm(b.reshape(B, -1), dim=1).clamp_min(1e-30)
    res = torch.linalg.norm(r.reshape(B, -1), dim=1) / bnorm
    done = res < tol
    rho_old = torch.ones(B, dtype=b.dtype, device=b.device)
    alpha = torch.ones(B, dtype=b.dtype, device=b.device)
    omega = torch.ones(B, dtype=b.dtype, device=b.device)
    p = torch.zeros_like(b)
    v = torch.zeros_like(b)

    def ip(a, c):
        return torch.sum(torch.conj(a) * c, dim=(-2, -1))

    for it in range(maxiter):
        if bool(done.all()):
            break
        act = ~done
        rho = ip(r0, r)
        beta = (rho / rho_old) * (alpha / omega)
        beta = torch.where(act, beta, torch.zeros_like(beta))
        p = r + beta[:, None, None] * (p - omega[:, None, None] * v)
        v = matvec(p)
        alpha = rho / ip(r0, v)
        alpha = torch.where(act, alpha, torch.zeros_like(alpha))
        s = r - alpha[:, None, None] * v
        t = matvec(s)
        omega = ip(t, s) / ip(t, t)
        omega = torch.where(act, omega, torch.zeros_like(omega))
        x = x + alpha[:, None, None] * p + omega[:, None, None] * s
        r = s - omega[:, None, None] * t
        rho_old = rho
        res = torch.linalg.norm(r.reshape(B, -1), dim=1) / bnorm
        done = done | (~torch.isfinite(res)) | (res < tol)

    res = torch.linalg.norm((b - matvec(x)).reshape(B, -1), dim=1) / bnorm
    return x, res
# ----------------------------------------------------------------------------
# forward model
# ----------------------------------------------------------------------------
def incident_plane_wave(N, k_arr, theta_arr, device, dtype=torch.float64):
    """Incident field exp(i*k*(cos t, sin t).x) with REAL k, shape (B,N,N).

    Using a real-k plane wave keeps the incident field a pure spatial tone at
    frequency f = k/(2 pi); the loss enters only through the propagator, so the
    frequency f remains the single varying parameter of the study.
    """
    x = torch.arange(N, device=device, dtype=dtype) / N
    X, Y = torch.meshgrid(x, x, indexing="ij")
    kap = k_arr.to(torch.complex128)
    dx = torch.cos(theta_arr.to(torch.complex128))
    dy = torch.sin(theta_arr.to(torch.complex128))
    phase = kap[:, None, None] * (dx[:, None, None] * X.to(torch.complex128)
                                  + dy[:, None, None] * Y.to(torch.complex128))
    return torch.exp(1j * phase)


def solve_ls(v, k_arr, theta_arr, gamma, N, tol=1e-10, maxiter=400, device="cpu",
             return_info=False):
    """Solve the discrete Lippmann-Schwinger system.

    v : (B, N, N) real contrast in [0, v_max]
    k_arr : (B,) wavenumbers; theta_arr : (B,) incident angles (rad)
    Returns psi (B,N,N) complex128.
    """
    v = v.to(torch.complex128)
    Ghat = green_symbol(N, k_arr, gamma, device)
    kappa2 = (k_arr.to(torch.complex128) * (1.0 + 1j * gamma)) ** 2
    kappa2 = kappa2[:, None, None]

    def matvec(psi):
        return psi - kappa2 * torch.fft.ifft2(Ghat * torch.fft.fft2(v * psi))

    psi_inc = incident_plane_wave(N, k_arr, theta_arr, device)
    psi, res = bicgstab(matvec, psi_inc, tol=tol, maxiter=maxiter)
    if return_info:
        return psi, res
    return psi


# ----------------------------------------------------------------------------
# contrast field generators
# ----------------------------------------------------------------------------
def _random_smooth_field(N, B, corr, gen, device, dtype=torch.float64):
    """Gaussian-correlated zero-mean unit-std random field, shape (B,N,N)."""
    idx = mode_indices(N, device, dtype)
    kx = 2.0 * PI * idx
    K2 = kx[:, None] ** 2 + kx[None, :] ** 2
    # Gaussian spectral filter: correlation length corr (in domain units)
    filt = torch.exp(-0.5 * (corr ** 2) * K2.to(dtype))       # (N,N)
    white = torch.randn((B, N, N), generator=gen, device=device, dtype=dtype)
    W = torch.fft.fft2(white) * filt
    g = torch.fft.ifft2(W).real
    g = g / g.reshape(B, -1).std(dim=1).clamp_min(1e-12)[:, None, None]
    return g


def make_contrast(N, B, v_max, corr, gen, device, kind="smooth",
                  dtype=torch.float64):
    """Generate contrast fields.

    kind = 'smooth'   : smooth random media, v in [0, v_max]
    kind = 'blocks'   : piecewise-constant blobs (hard phantom), v in {0, v_max}
    """
    if kind == "smooth":
        g = _random_smooth_field(N, B, corr, gen, device, dtype)
        v = 0.5 * (1.0 + torch.tanh(g))
        v = v * v_max
        v = v / v.reshape(B, -1).amax(dim=1).clamp_min(1e-12)[:, None, None] * v_max
        return v
    elif kind == "blocks":
        v = torch.zeros((B, N, N), device=device, dtype=dtype)
        x = torch.arange(N, device=device, dtype=dtype) / N
        X, Y = torch.meshgrid(x, x, indexing="ij")
        for b in range(B):
            nb = int(torch.randint(3, 7, (1,), generator=gen).item())
            for _ in range(nb):
                cx = float(torch.rand((1,), generator=gen)) * 0.9 + 0.05
                cy = float(torch.rand((1,), generator=gen)) * 0.9 + 0.05
                r = float(torch.rand((1,), generator=gen)) * 0.10 + 0.04
                m = (X - cx) ** 2 + (Y - cy) ** 2 < r ** 2
                v[b][m] = 1.0
        return v * v_max
    raise ValueError(kind)


# ----------------------------------------------------------------------------
# Born / Neumann physics baselines
# ----------------------------------------------------------------------------
def born_approx(v, k_arr, theta_arr, gamma, N, order=1, device="cpu"):
    """Truncated Born (Neumann) series psi ~ sum_{n=0}^{order} (kappa^2 G V)^n psi_inc."""
    v = v.to(torch.complex128)
    Ghat = green_symbol(N, k_arr, gamma, device)
    kappa2 = ((k_arr.to(torch.complex128) * (1.0 + 1j * gamma)) ** 2)[:, None, None]

    def op(psi):
        return kappa2 * torch.fft.ifft2(Ghat * torch.fft.fft2(v * psi))

    psi_inc = incident_plane_wave(N, k_arr, theta_arr, device)
    psi = psi_inc.clone()
    term = psi_inc.clone()
    for _ in range(order):
        term = op(term)
        psi = psi + term
    return psi
