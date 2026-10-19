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

def in_channels(emb="none", anchor="none"):
    """Input channel count for a given frequency embedding / anchor.

    base 4 = [v, Re psi_inc, Im psi_inc, f/F_MAX];  "ff" appends 2*FF_K
    frequency features (C2);  "born" appends Re/Im of the first two Born
    corrections, i.e. the analytic part of the solution (anchored arm).
    """
    n = 4 if emb == "none" else 4 + 2 * FF_K
    return n + 4 if anchor == "born" else n


def anchor_terms(v, k, theta, gamma=None, N=None):
    """(B1, B2): the first two Born corrections of the incident field.

    One pass over the Neumann recursion, no linear solve, no label: B1 = kappa^2
    G (v psi_inc) and B2 = kappa^2 G (v B1).  These carry the frequency
    dependence of the propagator ANALYTICALLY -- the spectral locus of B1 moves
    with k by construction -- which is exactly the part the trained operator
    cannot extrapolate (C3b showed the model loses phase out of band, not
    energy).  Computable at any frequency, including ones never trained on.
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
    t1 = op(psi_inc)
    t2 = op(t1)
    return t1, t2

def fourier_feats(f, K=FF_K):
    """NeRF-style embedding of the normalised frequency, returns (B, 2K)."""
    fn = (f / C.F_MAX).reshape(-1, 1).to(f.dtype)
    ang = 2 * np.pi * fn * (2.0 ** torch.arange(K, dtype=f.dtype, device=f.device))
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

def build_inputs(v, psi_inc, f, emb="none", anchor="none", born=None):
    """Network input tensor: [v, Re psi_inc, Im psi_inc, f/F_MAX].

    emb="ff" appends the Fourier-feature channels (in_channels("ff") total).
    anchor="born" appends Re/Im of (B1, B2) from anchor_terms(); the pair is
    passed in because the caller already has (k, theta) in hand, and recomputing
    it would double the FFT work in the training step.
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
            raise ValueError("anchor='born' needs the Born pair")
        b1, b2 = born
        # anchor_terms returns (B, N, N); stack them as channel planes
        ch = [t.to(base.dtype)[:, None] for t in (b1.real, b1.imag,
                                                  b2.real, b2.imag)]
        return torch.cat([base] + ch, dim=1)
    raise ValueError(anchor)


def anchored_u(u_net, born, ab):
    """u = a B1 + b B2 + u_net, the anchored residual operator.

    `ab` is a length-4 real vector [Re a, Im a, Re b, Im b]; it is trained on
    the band only, alongside the network, and shipped in the checkpoint, so the
    analytic part is fixed before any out-of-band frequency is seen.  The
    additive form is what makes the arm safe: if the learned residual collapses
    out of band -- the failure mode C3b measured -- what is left is the anchor,
    not the incident field.
    """
    b1, b2 = born
    a = torch.complex(ab[0], ab[1])
    b = torch.complex(ab[2], ab[3])
    return a * b1 + b * b2 + u_net


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
