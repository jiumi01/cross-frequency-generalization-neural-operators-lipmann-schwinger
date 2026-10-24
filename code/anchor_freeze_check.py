# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Where does BA2's contrast boundary come from: the residual, or the head?

`code/proj_precheck.py` projected the residual of the trained BA2 checkpoints
onto a band of radius R and the plateau did not move (v=0.50: 18.6 unprojected,
21.4 at R=4, 20.6 at R=6).  So the residual's out-of-band energy is not what
carries the failure -- which contradicts the reading the arm was built on.

But removing the residual altogether is not the right control either, because
the head's coefficients were fine-tuned jointly with the residual: at v=0.50 the
checkpoint's coefficients combined with no residual give 21.3, while the same
truncated Born series fitted in band by least squares gives 0.187
(results/anchor_higher_order.json).  That gap is the tell.  The joint fit moves
the coefficients to compensate for what the residual does not get right in
band, and a truncated Born series with rho > 1 amplifies that displacement
geometrically out of band.

This script separates the two by holding the residual fixed and swapping only
the coefficients:

  A  trained coefficients + full residual      (the arm as evaluated)
  B  least-squares coefficients + full residual(the arm with the head frozen)
  C  least-squares coefficients + no residual  (the anchor alone)
  D  least-squares coefficients + residual low-passed at R=5

If B is close to C and both are far below A, the carrier is the fine-tuning of
the head, not the residual's bandwidth: the fix is to freeze the head, and a
projected-residual arm would be a wasted training run.

    python code/anchor_freeze_check.py --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import evaluate as E
import physics as P
from proj_precheck import radial_mask
from train import load_cache

# arm tag, run folder, contrast, head depth, channels, cache
ARMS = [("a025", "ba_fno_a025_m8_s0", 0.25, 8, None, "train_a025"),
        ("a050", "ba_fno_a050_m8_s0", 0.50, 8, 4, "train_a050"),
        ("a100", "ba_fno_a100_m7_s0", 1.00, 7, 2, "train_a100")]
IN_BAND = (1.5, 2.5, 3.5, 4.0)
OUT_BAND = (6.0, 8.0, 10.0, 12.0)
N = 128
R_PROJ = 5


def ls_coeffs(cache, order, dev):
    """The head's coefficients as train.py seeds them: an in-band LS fit.

    `order` must be the EFFECTIVE depth -- the number of Born terms the model
    actually sums.  `--anchor-channels` truncates the term list handed to both
    the network and anchored_u, so an arm launched with order=8 and
    channels=4 sums four terms, and fitting eight here would describe a model
    that was never run.
    """
    d = load_cache(cache, dev)
    nf = min(512, d["v"].shape[0])
    with torch.no_grad():
        vf = d["v"][:nf].float()
        kf, thf = d["k"][:nf], d["th"][:nf]
        pif = P.incident_plane_wave(vf.shape[-1], kf, thf,
                                    vf.device).to(torch.complex128)
        uf = d["psi"][:nf].to(torch.complex128) - pif
        c, cond = D.fit_anchor_coeffs(vf, kf, thf, uf, order)
    return c.to(dev).float(), cond


def project(u, mask):
    return torch.fft.ifft2(torch.fft.fft2(u) * mask)


@torch.no_grad()
def cell(model, ab, order, channels, f, vmax, mode, mask, dev, seed=50000):
    d = D.sample_batch(N, seed, dev, vmax_lo=vmax, vmax_hi=vmax,
                       f_fixed=f, return_info=True, gamma=C.GAMMA)
    x, psi_inc, born = E.build(C.N, d["k"], d["theta"], d["v"], dev,
                               anchor="born", order=order, channels=channels)
    o = model(x)
    u = torch.complex(o[:, 0], o[:, 1])
    if mode == "none":
        u = torch.zeros_like(u)
    elif mode == "proj":
        u = project(u, mask)
    u = D.anchored_u(u, born, ab)
    e = (torch.linalg.norm((psi_inc + u - d["psi"]).reshape(N, -1), dim=1) /
         torch.linalg.norm(d["psi"].reshape(N, -1), dim=1))
    return float(e.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(C.RES,
                                                  "anchor_freeze_check.json"))
    a = ap.parse_args()
    dev = a.device
    rows = []
    for tag, run, vmax, order, ch, cache in ARMS:
        ckpt = os.path.join(C.RUNS, run, "checkpoint.pt")
        if not os.path.exists(ckpt):
            print("[skip] %s" % run)
            continue
        model, ck = E.load_model(ckpt, dev)
        ab_tr = ck["anchor_ab"].to(dev)
        # the number of Born terms the arm actually sums
        eff = ch if ch is not None else order
        ab_ls, cond = ls_coeffs(os.path.join(C.DATA, cache), eff, dev)
        print("\n%s v=%.2f  head sums %d terms  fitted c1=%+.3f%+.3fj  "
              "LS c1=%+.3f%+.3fj (cond %.1e)"
              % (tag, vmax, eff, ab_tr[0].item(), ab_tr[1].item(),
                 ab_ls[0].item(), ab_ls[1].item(), cond), flush=True)
        mask = radial_mask(C.N, R_PROJ, dev, torch.complex64)
        for name, ab, mode in (("A trained ab + residual", ab_tr, "full"),
                               ("B LS ab + residual", ab_ls, "full"),
                               ("C LS ab, no residual", ab_ls, "none"),
                               ("D LS ab + resid |m|<=%d" % R_PROJ, ab_ls, "proj")):
            inb = [cell(model, ab, order, ch, f, vmax, mode, mask, dev)
                   for f in IN_BAND]
            out = [cell(model, ab, order, ch, f, vmax, mode, mask, dev)
                   for f in OUT_BAND]
            rec = dict(arm=tag, vmax=vmax, variant=name,
                       in_band=float(np.mean(inb)), plateau=float(np.mean(out)),
                       in_curve=inb, out_curve=out)
            rows.append(rec)
            print("   %-26s in-band %.4g  plateau %.4g"
                  % (name, rec["in_band"], rec["plateau"]), flush=True)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
    C.save_json(dict(n=N, in_band_freqs=list(IN_BAND),
                     out_band_freqs=list(OUT_BAND), r_proj=R_PROJ, rows=rows),
                a.out)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
