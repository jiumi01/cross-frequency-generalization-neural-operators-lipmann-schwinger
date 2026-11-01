# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""P1-a pre-check: is the BA2 failure an out-of-band residual, and how wide a
band can the residual keep without buying it back?

BA2 (code/run_ba2.py) is u = sum_j c_j B_j + r_theta.  It holds the plateau at
v = 0.25 (0.026 against the trained operator's 0.963) and destroys it at
v >= 0.50, where the in-band error reaches 6.3 and 7.7e3.  The reading the
manuscript would like to make is that the residual, not the anchor, is
responsible: the residual is the only term in the construction that has never
seen an out-of-band frequency, so if it carries out-of-band energy it can only
be wrong there.

That reading is testable WITHOUT training.  Projecting the residual of an
already-trained checkpoint, u = A + Pi_R r_theta, changes nothing in band if R
covers the training band, and removes every mode the residual would have added
above it.  If the plateau returns to the anchor's level under projection, the
cause is located and a projected arm is worth training; if it does not, the
residual is not the carrier and the training run would have been wasted.

This is a pre-check, not a result: the projected model is not the model that
would be trained, because training would adapt to the constraint.  It bounds
the effect from below, which is what a go/no-go decision needs.

Mode radius: the domain is the unit cell, f is wavelengths across it, so the
wavenumber is 2*pi*f and the integer FFT mode index of the band edge is
m = f_train = 4.  R is therefore measured in mode index and compared with 4.

    python code/proj_precheck.py --device cuda
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import evaluate as E

# arm -> (run folder, contrast, head depth, channels handed to the network)
ARMS = [("a025", "ba_fno_a025_m8_s0", 0.25, 8, None),
        ("a050", "ba_fno_a050_m8_s0", 0.50, 8, 4),
        ("a100", "ba_fno_a100_m7_s0", 1.00, 7, 2)]
IN_BAND = (1.5, 2.5, 3.5, 4.0)
OUT_BAND = (6.0, 8.0, 10.0, 12.0)
# None = as trained, 0 = residual removed (anchor alone), int = mode radius
RADII = (None, 0, 4, 5, 6, 8)
N = 128


def radial_mask(n, r_cut, device, dtype=torch.complex64):
    """Keep Fourier modes with |m| <= r_cut on an n x n periodic grid."""
    idx = torch.fft.fftfreq(n, d=1.0 / n, device=device).to(torch.float64)
    r = torch.sqrt(idx[:, None] ** 2 + idx[None, :] ** 2)
    return (r <= float(r_cut) + 1e-9).to(dtype)


def project(u, r_cut, mask_cache):
    """Zero every mode of the residual above r_cut, on the grid it was made on."""
    if r_cut is None:
        return u
    if r_cut == 0:
        return torch.zeros_like(u)
    n = u.shape[-1]
    key = (n, float(r_cut))
    if key not in mask_cache:
        mask_cache[key] = radial_mask(n, r_cut, u.device, torch.complex64)
    return torch.fft.ifft2(torch.fft.fft2(u) * mask_cache[key])


@torch.no_grad()
def cell(model, ck, f, vmax, order, channels, r_cut, device, seed=50000):
    d = D.sample_batch(N, seed, device, vmax_lo=vmax, vmax_hi=vmax,
                       f_fixed=f, return_info=True, gamma=C.GAMMA)
    x, psi_inc, born = E.build(C.N, d["k"], d["theta"], d["v"], device,
                               anchor="born", order=order, channels=channels)
    o = model(x)
    u = torch.complex(o[:, 0], o[:, 1])
    u = project(u, r_cut, cell.mask_cache)
    ab = ck["anchor_ab"].to(device)
    u = D.anchored_u(u, born, ab)
    e = (torch.linalg.norm((psi_inc + u - d["psi"]).reshape(N, -1), dim=1) /
         torch.linalg.norm(d["psi"].reshape(N, -1), dim=1))
    return float(e.mean())


cell.mask_cache = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(C.RES, "proj_precheck.json"))
    a = ap.parse_args()
    dev = a.device
    rows = []
    for tag, run, vmax, order, ch in ARMS:
        ckpt = os.path.join(C.RUNS, run, "checkpoint.pt")
        if not os.path.exists(ckpt):
            print("[skip] %s: no checkpoint" % run)
            continue
        model, ck = E.load_model(ckpt, dev)
        for r in RADII:
            inb = [cell(model, ck, f, vmax, order, ch, r, dev) for f in IN_BAND]
            out = [cell(model, ck, f, vmax, order, ch, r, dev) for f in OUT_BAND]
            rec = dict(arm=tag, run=run, vmax=vmax, r_cut=r,
                       in_band=float(np.mean(inb)), plateau=float(np.mean(out)),
                       in_curve=inb, out_curve=out)
            rows.append(rec)
            print("%-5s v=%.2f R=%-4s in-band %.4g  plateau %.4g"
                  % (tag, vmax, "None" if r is None else r,
                     rec["in_band"], rec["plateau"]), flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    ref = {}
    for r in rows:
        ref.setdefault(r["arm"], {})["R%s" % r["r_cut"]] = dict(
            in_band=r["in_band"], plateau=r["plateau"])
    out = dict(n=N, in_band_freqs=list(IN_BAND),
               out_band_freqs=list(OUT_BAND), radii=[-1 if r is None else r
                                                     for r in RADII],
               rows=rows, by_arm=ref)
    C.save_json(out, a.out)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
