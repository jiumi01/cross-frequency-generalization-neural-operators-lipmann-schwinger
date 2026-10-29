# Author: Longgang Wang
"""Frequency x resolution cross design (review item 4, path 3).

Why this file exists
--------------------
The fixed-grid sweep couples frequency f with the number of grid points per
wavelength q = N/f: raising f at N=64 lowers q, so "the error jumps at the
band edge" and "the error jumps when q falls below a threshold" predict the
same curve.  The uniform 128^2 retraining bounds that confound but cannot
separate it, because it moves f and q together again.

This design separates them by measuring the SAME model on a grid of (f, N)
cells rather than on a single row.  The two hypotheses make different
predictions about where the jump sits in f:

    H_freq   the jump is indexed by the trained band edge   -> onset(f) = 4.25
             at every N, and only the jump height changes with N
    H_q      the jump is indexed by points per wavelength   -> onset(f) ~
             4.25 * N/64, i.e. 6.4 at N=96, 8.5 at N=128, 12.75 at N=192

The two are not nested, so one measurement discriminates them.

Paired media across grids
--------------------------
All grids must discretise ONE continuous medium, otherwise the comparison is
between medium draws rather than between meshes.  A master contrast is drawn
at 384^2 (384/64 = 6, /96 = 4, /128 = 3, /192 = 2, all integer) and restricted
to each grid by spectral truncation, which is exact sampling of the
band-limited interpolant.  The Gaussian correlation filter (corr = 0.12 domain
units) leaves essentially no energy above mode ~2, so the truncation is
lossless to well below the reported tolerances; the normalisation to v_max is
applied ONCE on the master field so that every grid sees the identical object.

Grids are limited to {64, 96, 128, 192}: adding 160 would require a master of
lcm(...) = 960^2 for an exact restriction, and the 4-level ladder already
spans a factor 3 in q.
"""
import os
import sys
import json
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P

MASTER = 384
# Training and evaluation share this ladder.  192 was in the first design and
# was dropped: at 192^2 the step cost on a 12 GB card in WDDM mode is 20-25x
# the 128^2 step rather than the 2.25x the size implies, and three levels
# already span a factor of two in points per wavelength at a fixed frequency
# (q = 15.1, 22.6, 30.1 at f = 4.25), which is what the discrimination needs.
GRIDS = (64, 96, 128)
# 4.25/8.5, 3.0/6.0, 4.5/9.0, 1.5/3.0 and their N-scaled partners give exact
# matched-q pairs across the band edge; 4.0 and 4.75 keep the locator rules of
# the main sweep unchanged.
FREQS = [1.5, 2.5, 3.0, 3.5, 4.0, 4.25, 4.5, 4.75, 5.0, 6.0, 8.0, 8.5, 9.0, 12.0]
N_MEDIA = 200
EDGE = 4.0


# --------------------------------------------------------------- media
def master_media(n, seed, vmax=0.25, device="cuda"):
    """Draw n contrast fields at MASTER resolution and the matching angles.

    The angle is drawn per medium and reused for every (f, N) cell, so cells
    differ only in f and N.
    """
    g = torch.Generator(device=device).manual_seed(int(seed))
    theta = torch.rand(n, generator=g, device=device) * 2 * np.pi
    g2 = torch.Generator(device=device).manual_seed(int(seed) + 7777)
    v = P.make_contrast(MASTER, n, vmax, C.MM_REF, g2, device)
    return v, theta


def restrict(v, N, master=MASTER):
    """Spectrally restrict a MASTER-grid field to an N-grid (exact sampling)."""
    if N == master:
        return v
    assert master % N == 0, "restriction needs an integer ratio"
    ratio = master // N
    V = torch.fft.fftshift(torch.fft.fft2(v), dim=(-2, -1))
    h = N // 2
    c = master // 2
    Vc = V[..., c - h:c + h, c - h:c + h] / (ratio * ratio)
    out = torch.fft.ifft2(torch.fft.ifftshift(Vc, dim=(-2, -1))).real
    return out.clamp_min(0.0)


def check_restriction(device="cpu", verbose=True):
    """Falsification probe: restriction must reproduce point samples of the
    band-limited interpolant, and must be lossless up to the filter's decay."""
    g = torch.Generator(device=device).manual_seed(7)
    v = P.make_contrast(MASTER, 2, 0.25, C.MM_REF, g, device)
    worst = 0.0
    for N in GRIDS:
        r = restrict(v, N)
        assert r.shape[-1] == N
        # point samples of the master field at the coarse nodes
        step = MASTER // N
        pts = v[..., ::step, ::step]
        worst = max(worst, float((r - pts).abs().max()))
    if verbose:
        print("[restriction] max |restrict(v) - v[::step,::step]| = %.3e" % worst)
    return worst
