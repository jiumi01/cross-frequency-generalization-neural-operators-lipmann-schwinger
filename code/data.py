# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Dataset construction for the cross-frequency Lippmann-Schwinger benchmark.

Everything is generated from the SAME contrast distribution (correlation length
fixed, v_max ~ U[0.05, 1.0], incident angle ~ U[0, 2pi)).  Only the frequency f
is changed between the training band and the test frequencies, so the study is
strictly single-variable in f.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P


# ----------------------------------------------------------------------------
def sample_params(n, f_lo, f_hi, vmax_lo, vmax_hi, seed, device, f_fixed=None):
    """Draw (v_max, theta, f).

    ORDER MATTERS: v_max and theta are drawn BEFORE f so that, for a fixed
    seed, changing the frequency band changes ONLY f.  Training bands, contrast
    arms and the test grid therefore share exactly the same scattering media,
    which keeps every comparison strictly single-variable (paired design).
    """
    g = torch.Generator(device=device).manual_seed(int(seed))
    vmax = torch.rand(n, generator=g, device=device) * (vmax_hi - vmax_lo) + vmax_lo
    theta = torch.rand(n, generator=g, device=device) * (2 * np.pi)
    if f_fixed is None:
        f = torch.rand(n, generator=g, device=device) * (f_hi - f_lo) + f_lo
    else:
        f = torch.full((n,), float(f_fixed), device=device)
    return f, vmax, theta


def sample_batch(n, seed, device, f_lo=None, f_hi=None, vmax_lo=0.05, vmax_hi=1.0,
                 f_fixed=None, n_grid=64, gamma=None, kind="smooth", return_info=False):
    """Draw n samples, generate contrast, solve the LS system exactly.

    gamma may be a scalar (the usual single-value run), None (the manuscript
    default), or a CALLABLE of the per-sample frequency tensor.  The callable
    form is what the gamma-compensation arm uses: gamma(f) is resolved per
    sample after f is drawn, so a single batch can carry a frequency-dependent
    regularisation.  physics.solve_ls broadcasts a (B,) gamma against (B,)
    wavenumbers, so nothing downstream needs to know.
    """
    f, vmax, theta = sample_params(n, f_lo, f_hi, vmax_lo, vmax_hi, seed, device,
                                   f_fixed=f_fixed)
    if callable(gamma):
        gamma = gamma(f)
    else:
        gamma = C.GAMMA if gamma is None else gamma
    g = torch.Generator(device=device).manual_seed(int(seed) + 7777)
    v = P.make_contrast(n_grid, n, 1.0, C.MM_REF, g, device, kind=kind)  # unit scale
    v = v * vmax[:, None, None]
    k = 2 * np.pi * f
    psi, res = P.solve_ls(v, k, theta, gamma, n_grid, tol=1e-9, maxiter=900,
                          device=device, return_info=True)
    out = dict(v=v, psi=psi, k=k, f=f, theta=theta, vmax=vmax)
    if return_info:
        out["res"] = res
    return out


# ----------------------------------------------------------------------------
def build_train_cache(n_total=4096, chunk=512, seed0=1000, out_dir=None,
                      device="cuda", f_lo=1.0, f_hi=4.0, vmax_lo=0.05, vmax_hi=1.0,
                      budget=None, gamma=None):
    """Generate the training set in resumable chunks (idempotent)."""
    import time
    out_dir = out_dir or os.path.join(C.DATA, "train")
    os.makedirs(out_dir, exist_ok=True)
    n_chunks = int(np.ceil(n_total / chunk))
    t0 = time.time()
    for ci in range(n_chunks):
        path = os.path.join(out_dir, "chunk_%04d.npz" % ci)
        if os.path.exists(path):
            continue
        if budget is not None and time.time() - t0 > budget:
            print("budget reached, stopping at chunk %d" % ci)
            break
        n = min(chunk, n_total - ci * chunk)
        d = sample_batch(n, seed0 + ci, device, f_lo=f_lo, f_hi=f_hi,
                         vmax_lo=vmax_lo, vmax_hi=vmax_hi, return_info=True,
                         gamma=gamma)
        np.savez(path,
                 v=d["v"].detach().cpu().numpy().astype(np.float32),
                 psi=d["psi"].detach().cpu().numpy().astype(np.complex64),
                 k=d["k"].detach().cpu().numpy().astype(np.float32),
                 f=d["f"].detach().cpu().numpy().astype(np.float32),
                 theta=d["theta"].detach().cpu().numpy().astype(np.float32),
                 vmax=d["vmax"].detach().cpu().numpy().astype(np.float32),
                 res=d["res"].detach().cpu().numpy().astype(np.float32))
        print("chunk %04d written (%d samples, max_res=%.2e)"
              % (ci, n, float(d["res"].max())))
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".npz"))
    print("train cache: %d chunks in %s" % (len(files), out_dir))
    return out_dir


class CachedSet(torch.utils.data.Dataset):
    def __init__(self, folder):
        self.files = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                            if f.endswith(".npz"))
        self.lens = []
        self.data = []
        for f in self.files:
            z = np.load(f)
            self.data.append(z)
            self.lens.append(z["v"].shape[0])
        self.cum = np.cumsum([0] + self.lens)
        self.n = int(self.cum[-1])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        ci = int(np.searchsorted(self.cum, i, side="right") - 1)
        j = i - int(self.cum[ci])
        z = self.data[ci]
        return (torch.from_numpy(z["v"][j]), torch.from_numpy(z["psi"][j]),
                float(z["k"][j]), float(z["theta"][j]), float(z["f"][j]),
                float(z["vmax"][j]))

    def max_res(self):
        return max(float(np.max(z["res"])) for z in self.data)

    def freq_hist(self):
        f = np.concatenate([z["f"] for z in self.data])
        return dict(min=float(f.min()), max=float(f.max()),
                    n=int(f.size), mean=float(f.mean()))


# C2 ablation (external review round 4): the baseline input encodes f as a
# single normalised scalar channel, and the review asks whether the error step
# at the band edge is an artefact of that low-bandwidth encoding rather than a
# generalisation boundary.  The "ff" embedding keeps the scalar channel and
# APPENDS K sin/cos pairs at geometrically spaced bands 2^0..2^(K-1), so the
# model sees a strictly richer (superset) representation of f: if the cliff
# survives, it is not caused by representation starvation.
FF_K = 7

def in_channels(emb="none", anchor="none", order=2, channels=None):
    """Input channel count for a given frequency embedding / anchor.

    base 4 = [v, Re psi_inc, Im psi_inc, f/F_MAX];  "ff" appends 2*FF_K
    frequency features (C2);  "born" appends Re/Im of the first `order` Born
    corrections, i.e. the analytic part of the solution (anchored arm).  The
    default order=2 reproduces the original four-channel anchor, so every
    existing checkpoint keeps loading with its original input width.
    """
    n = 4 if emb == "none" else 4 + 2 * FF_K
    if anchor != "born":
        return n
    return n + 2 * (order if channels is None else channels)


def born_series(v, k, theta, order=2, gamma=None, N=None):
    """[B1, ..., B_order]: successive Born corrections of the incident field.

    One pass over the Neumann recursion, no linear solve, no label:
    B1 = kappa^2 G (v psi_inc) and B_{j+1} = kappa^2 G (v B_j).  These carry
    the frequency dependence of the propagator ANALYTICALLY -- the spectral
    locus of B1 moves with k by construction -- which is exactly the part the
    trained operator cannot extrapolate (C3b showed the model loses phase out
    of band, not energy).  Computable at any frequency, including ones never
    trained on.

    `order` is a hyperparameter, not a free lunch: code/anchor_higher_order.py
    measures that the anchor's out-of-band plateau falls geometrically with it
    while the Born contraction factor rho stays below one, and then TURNS
    (v=1.00: 0.516 at M=5, 0.517 at M=6) once rho > 1 and the fit becomes an
    asymptotic summation on a divergent series.
    """
    gamma = C.GAMMA if gamma is None else gamma
    N = C.N if N is None else N
    dev = v.device
    vc = v.to(torch.complex128)
    Ghat = P.green_symbol(N, k, gamma, dev)
    kappa2 = ((k.to(torch.complex128) * (1.0 + 1j * gamma)) ** 2)[:, None, None]

    def op(psi):
        return kappa2 * torch.fft.ifft2(Ghat * torch.fft.fft2(vc * psi))

    psi_inc = P.incident_plane_wave(N, k, theta, dev).to(torch.complex128)
    out, cur = [], op(psi_inc)
    for _ in range(order):
        out.append(cur)
        cur = op(cur)
    return out


def anchor_terms(v, k, theta, gamma=None, N=None, order=2):
    """(B1, ..., B_order); order=2 reproduces the original (B1, B2) pair."""
    return tuple(born_series(v, k, theta, order=order, gamma=gamma, N=N))


def fit_anchor_coeffs(v, k, theta, u, order, cond_ceiling=1e12):
    """Least-squares complex coefficients on [B1..B_order], in-band.

    Returns a real vector of length 2*order (Re/Im interleaved), or None when
    the Gram matrix is singular past `cond_ceiling` -- in which case the fit
    would be a rounding artefact dressed as a measurement and must be refused
    rather than reported.
    """
    import numpy as _np
    terms = born_series(v, k, theta, order=order)
    n = u.shape[0]
    y = u.reshape(n, -1)
    m = torch.stack([t.reshape(n, -1) for t in terms], dim=1)
    gram = torch.einsum("njd,nkd->jk", m.conj(), m).cpu().numpy()
    rhs = torch.einsum("njd,nd->j", m.conj(), y).cpu().numpy()
    cond = float(_np.linalg.cond(gram))
    if not _np.isfinite(cond) or cond > cond_ceiling:
        return None, cond
    c = _np.linalg.solve(gram, rhs)
    out = torch.zeros(2 * order, dtype=torch.float64)
    for j, z in enumerate(c):
        out[2 * j] = float(z.real)
        out[2 * j + 1] = float(z.imag)
    return out, cond

def fourier_feats(f, K=FF_K):
    """NeRF-style embedding of the normalised frequency, returns (B, 2K)."""
    fn = (f / C.F_MAX).reshape(-1, 1).to(f.dtype)
    ang = 2 * np.pi * fn * (2.0 ** torch.arange(K, dtype=f.dtype, device=f.device))
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

def build_inputs(v, psi_inc, f, emb="none", anchor="none", born=None):
    """Network input tensor: [v, Re psi_inc, Im psi_inc, f/F_MAX].

    emb="ff" appends the Fourier-feature channels (in_channels("ff") total).
    anchor="born" appends Re/Im of (B1..B_order) from born_series(); the terms
    are passed in because the caller already has (k, theta) in hand, and
    recomputing them would repeat the FFT work in the training step.
    """
    fin = (f / C.F_MAX).view(-1, 1, 1).expand(-1, v.shape[-2], v.shape[-1])
    base = torch.stack([v, psi_inc.real, psi_inc.imag, fin], dim=1)
    if emb == "ff":
        fe = fourier_feats(f)[:, :, None, None].expand(
            -1, -1, v.shape[-2], v.shape[-1])
        base = torch.cat([base, fe], dim=1)
    elif emb != "none":
        raise ValueError(emb)
    if anchor == "none":
        return base
    if anchor == "born":
        if born is None:
            raise ValueError("anchor='born' needs the Born terms")
        # born_series returns (B, N, N) per order; stack them as channel planes
        ch = []
        for term in born:
            for part in (term.real, term.imag):
                ch.append(part.to(base.dtype)[:, None])
        return torch.cat([base] + ch, dim=1)
    raise ValueError(anchor)


def anchored_u(u_net, born, ab):
    """u = sum_j c_j B_j + u_net, the anchored residual operator.

    `ab` is a real vector of length 2*order holding [Re c_1, Im c_1, ...]; the
    coefficients are fitted (and then fine-tuned) on the band only and shipped
    in the checkpoint, so the analytic part is fixed before any out-of-band
    frequency is seen.  The additive form is what makes the arm safe: if the
    learned residual collapses out of band -- the failure mode C3b measured --
    what is left is the anchor, not the incident field.

    With order=2 this is exactly the original a*B1 + b*B2 + u_net.
    """
    out = u_net
    for j, t in enumerate(born):
        c = torch.complex(ab[2 * j], ab[2 * j + 1])
        out = out + c * t
    return out


def incident(v, k, theta, device):
    n, n_grid = v.shape[0], v.shape[-1]
    return P.incident_plane_wave(n_grid, k, theta, device)


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    build_train_cache(a.n, a.chunk, out_dir=a.out, device=dev, budget=a.budget)
