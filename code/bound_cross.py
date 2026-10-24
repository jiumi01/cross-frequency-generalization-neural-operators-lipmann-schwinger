# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Why the squared-sum bound fails at f = 4, and whether it is bookkeeping.

The manuscript quotes a bound

    E <= sqrt(eps^2 + delta^2)

for the error of a model scored against a refined reference, with eps its error
against the native-grid reference and delta the reference-solution
discretisation difference.  The bound holds at f >= 5 and fails at f = 4
(0.195 measured against 0.188 predicted).  Two explanations are possible and
they have opposite consequences:

  (a) bookkeeping.  eps is normalised by ||psi_64|| while delta and E are
      normalised by ||psi_256||.  A squared sum of quantities with different
      denominators is not an identity, and the discrepancy is of the same order
      as the failure (3.3%).

  (b) physics.  The two error vectors are correlated, so the cross term
      2*rho*eps*delta is positive and the orthogonal bound is simply wrong.

This file separates them by computing everything on one denominator and
reporting the cosine between the model error and the discretisation difference
per sample.  Same media, same seed, same protocol as grid_conv_three.py.

Usage
-----
    python code/bound_cross.py --device cpu
Output: results/bound_cross.json
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import physics as P
from evaluate import load_model
from grid_conv_three import downsample, make_media, solve_at

SEED = 61001
VMAX = 0.25
FREQS = (4, 5)
CKPT = os.path.join(C.RUNS, "main_fno_a025_s0", "checkpoint.pt")
OUT = os.path.join(C.RES, "bound_cross.json")


def flat(x):
    return x.reshape(x.shape[0], -1)


def run(n, dev):
    g = torch.Generator(device=dev).manual_seed(SEED)
    theta = torch.rand(n, generator=g, device=dev) * 2 * np.pi
    v256 = make_media(n, SEED, VMAX, 256, dev)
    v128 = downsample(v256)
    v64 = downsample(v128)

    model, ck = load_model(CKPT, dev)
    model.eval()

    rows = []
    t0 = time.time()
    for f in FREQS:
        kv = 2 * np.pi * torch.full((n,), float(f), dtype=torch.float64, device=dev)
        inc = P.incident_plane_wave(64, kv, theta, dev).to(torch.complex64)
        fv = torch.full((n,), float(f), device=dev)
        with torch.no_grad():
            o = model(D.build_inputs(v64.float(), inc, fv, emb="none"))
        pred = (inc + torch.complex(o[:, 0], o[:, 1])).to(torch.complex128)

        psi256 = solve_at(v256, f, theta, 256, dev)
        psi64 = solve_at(v64, f, theta, 64, dev)
        t64 = psi64
        t256 = downsample(downsample(psi256))

        a = flat(pred - t64)          # model error on the native grid
        b = flat(t64 - t256)          # discretisation difference
        s = flat(t256)                # common denominator

        na = torch.linalg.norm(a, dim=1)
        nb = torch.linalg.norm(b, dim=1)
        ns = torch.linalg.norm(s, dim=1)
        nab = torch.linalg.norm(a + b, dim=1)
        dot = (a.conj() * b).real.sum(dim=1)

        e64 = na / torch.linalg.norm(flat(t64), dim=1)
        e256 = nab / ns
        d = nb / ns
        # same-denominator version
        e64p = na / ns
        bound_same = torch.sqrt(e64p ** 2 + d ** 2)
        bound_mixed = torch.sqrt(e64 ** 2 + d ** 2)
        cos = dot / (na * nb).clamp_min(1e-30)
        cross = 2 * dot / ns ** 2          # the term the bound drops

        # the form an earlier draft used: substitute the MEAN error and the MEAN
        # difference into the same expression.  By concavity of sqrt this is
        # smaller than the mean of the per-sample bounds, so comparing it with
        # the mean error tests a different inequality.
        agg = torch.sqrt(e64.mean() ** 2 + d.mean() ** 2)

        rows.append(dict(
            f=f, n=n,
            e64=float(e64.mean()), e256=float(e256.mean()), delta=float(d.mean()),
            e64_same_denom=float(e64p.mean()),
            bound_same_denom=float(bound_same.mean()),
            bound_mixed_denom=float(bound_mixed.mean()),
            bound_aggregate=float(agg),
            ratio_aggregate=float(e256.mean() / agg),
            ratio_same=float((e256 / bound_same).mean()),
            ratio_mixed=float((e256 / bound_mixed).mean()),
            cos_mean=float(cos.mean()), cos_sd=float(cos.std()),
            cos_negative_frac=float((cos < 0).float().mean()),
            cross_term=float(cross.mean()),
            norm_ratio_t64_t256=float(
                (torch.linalg.norm(flat(t64), dim=1) / ns).mean()),
            holds_same=bool((e256 <= bound_same + 1e-12).all()),
            holds_mixed=bool((e256 <= bound_mixed + 1e-12).all()),
            # how often the pointwise inequality itself is violated, and by how
            # much.  A mean that satisfies the bound while a fraction of the
            # samples violate it would make "the bound holds" a statement about
            # the estimator, not about the inequality -- so it is recorded.
            viol_same=float((e256 > bound_same + 1e-12).float().mean()),
            exceed_max_same=float((e256 - bound_same).max()),
            ratio_max_same=float((e256 / bound_same).max()),
            viol_mixed=float((e256 > bound_mixed + 1e-12).float().mean())))
        print("[bound] f=%g  E=%.4f  eps=%.4f (same-denom %.4f)  delta=%.4f  "
              "bound same=%.4f mixed=%.4f  ratio %.3f/%.3f  cos=%+.3f"
              % (f, rows[-1]["e256"], rows[-1]["e64"], rows[-1]["e64_same_denom"],
                 rows[-1]["delta"], rows[-1]["bound_same_denom"],
                 rows[-1]["bound_mixed_denom"], rows[-1]["ratio_same"],
                 rows[-1]["ratio_mixed"], rows[-1]["cos_mean"]), flush=True)
        del psi256
        if dev == "cuda":
            torch.cuda.empty_cache()

    res = dict(n=n, seed=SEED, vmax=VMAX, ckpt=os.path.relpath(CKPT, C.ROOT)
               if hasattr(C, "ROOT") else CKPT, rows=rows,
               device=str(dev), seconds=round(time.time() - t0, 1))
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    print("\nwrote %s" % OUT)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=128)
    a = ap.parse_args()
    run(a.n, a.device)


if __name__ == "__main__":
    main()
