# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""The LS-fitted analytic head ALONE, under the formal evaluation protocol.

The BA2-FRZ chain (code/run_ba2_frz.py) pre-registers its FAIL gate against
"1.5 x the LS head alone".  The anchor_freeze_check number for that head
(0.512 at v=0.50, 1.228 at v=1.00) was measured on the pre-check protocol --
N=128, four in-band and four out-of-band frequencies, one sample seed.  A
gate is only meaningful when both sides sit on the same protocol, so this
script recomputes the LS head under the FORMAL one: n=200, the full f_test
grid, the same per-contrast seeds the run evaluations use, the same error
definition (psi_inc + u against psi).

Zero training: the coefficients are the in-band least-squares fit on the
512-sample training-cache prefix, exactly as train.py seeds them, and the
network output is identically zero, so u = sum_j c_j B_j.

    python code/ls_head_eval.py --device cuda
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
from train import load_cache

CACHE = {"0.50": ("train_a050", 8, 4),
         "1.00": ("train_a100", 7, 2)}
FIT_N = 512  # train.py fits on this prefix; deviate and the gate is a lie


def ls_coeffs(cache, eff, dev):
    """The head's coefficients exactly as train.py seeds them.

    `eff` is the EFFECTIVE depth -- the number of Born terms the model sums.
    `--anchor-channels` truncates the term list handed to the network AND to
    anchored_u, so an arm with order=8 and channels=4 sums FOUR terms; the
    fit must be at four or the coefficients describe a model that was never
    run (the a050 LS head would read 98 instead of 0.5 on the plateau).
    """
    d = load_cache(os.path.join(C.DATA, cache), dev)
    import data as _D
    # train.py uses the first FIT_N entries of the training split; the cache
    # loader returns the split already, so take the same prefix
    idx = torch.arange(FIT_N, device=dev)
    with torch.no_grad():
        vf = d["v"][idx].float()
        kf, thf = d["k"][idx], d["th"][idx]
        pif = P.incident_plane_wave(vf.shape[-1], kf, thf,
                                    vf.device).to(torch.complex128)
        uf = (d["psi"][idx].to(torch.complex128) - pif)
        c, cond = D.fit_anchor_coeffs(vf, kf, thf, uf, eff)
    if c is None:
        raise SystemExit("LS fit REFUSED (cond=%.2e) on %s" % (cond, cache))
    return c.to(dev).float(), cond


class Zero(torch.nn.Module):
    """The trained residual, removed: two zero channels on the input grid."""
    def forward(self, x):
        z = torch.zeros(x.shape[0], 2, x.shape[-2], x.shape[-1],
                        device=x.device, dtype=x.dtype)
        return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(C.RES, "ls_head_alone.json"))
    a = ap.parse_args()
    dev = a.device
    res = dict(n=a.n, f_test=list(C.F_TEST), arms={})
    zero = Zero()
    for tag, (cache, order, ch) in CACHE.items():
        # Three heads, one protocol.  "truncated" is what every a050/a100 arm
        # has actually run: fitted at the nominal order, summed over the first
        # `channels` terms.  "full" is what run_ba2.py's docstring CLAIMS the
        # arms run: fitted at the nominal order and summed over all of them.
        # "shallow" is the direct channels-term fit (anchor_freeze_check's
        # variant C).  The gap between them is the defect, not a modelling
        # choice.
        eff = ch if ch is not None else order
        ab_full, cond = ls_coeffs(cache, order, dev)
        ab_shallow = (ls_coeffs(cache, eff, dev)[0] if ch is not None
                      else ab_full)
        variants = {"truncated": (ab_full, ch), "full": (ab_full, None)}
        if ch is not None:
            variants["shallow"] = (ab_shallow, ch)
        sweep_all = {}
        for vname, (ab, sum_ch) in variants.items():
            sweep = {}
            for vi, vmax in enumerate(C.VMAX_LIST):
                rows = []
                for f in C.F_TEST:
                    e, _ = E.eval_cell(zero, f, vmax, n=a.n, device=dev,
                                       seed=60000 + vi, emb="none",
                                       anchor="born", ab=ab, order=order,
                                       channels=sum_ch)
                    rows.append(dict(f=float(f), mean=float(np.mean(e)),
                                     median=float(np.median(e)),
                                     q90=float(np.quantile(e, 0.9))))
                sweep["%.2f" % vmax] = rows
                print("%s %-9s v=%.2f  in %.4g  plat %.4g" %
                      (tag, vname, vmax,
                       np.mean([r["mean"] for r in rows if r["f"] <= 4.0]),
                       np.mean([r["mean"] for r in rows if r["f"] >= 5.0])),
                      flush=True)
            sweep_all[vname] = sweep
        res["arms"][tag] = dict(cache=cache, order=order, channels=ch,
                                eff_depth=eff, cond=cond,
                                ab=[float(x) for x in ab_full.cpu().flatten()],
                                sweep=sweep_all["truncated"],
                                variants=sweep_all)
    C.save_json(res, a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
