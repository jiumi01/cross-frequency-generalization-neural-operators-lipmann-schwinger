# Author: Longgang Wang
"""Evaluate the multi-grid FNO on the full (frequency, grid) cross.

One cell = (f, N).  The media are drawn ONCE at 384^2 and spectrally
restricted to each N, so the cells differ in f and N only and a cell-to-cell
difference is not a difference between medium draws.  Each prediction is
scored against the reference solution computed on its own grid, which is the
protocol every other table in the paper uses.

Usage
-----
    python code/cross_eval.py --seeds 0 1 2
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
import physics as P
import data as D
from evaluate import load_model
import cross_design as X

dev = "cuda" if torch.cuda.is_available() else "cpu"


def eval_seed(seed, n=200, freqs=None, grids=None, vmax=0.25, media_seed=60002,
              stack=32):
    freqs = freqs or X.FREQS
    grids = grids or X.GRIDS
    name = "cross_fno_a025_s%d" % seed
    model, ck = load_model(os.path.join(C.RUNS, name, "checkpoint.pt"), dev)
    print("[seed %d] epoch=%s val=%.4f params=%.2fM"
          % (seed, ck.get("epoch"), ck.get("val"), ck["cfg"]["params"] / 1e6),
          flush=True)

    v_master, theta = X.master_media(n, media_seed, vmax=vmax, device=dev)
    rows = []
    with torch.no_grad():
        for N in grids:
            v = X.restrict(v_master, N)
            for f in freqs:
                fv = torch.full((n,), float(f), device=dev)
                k = 2 * np.pi * fv
                psi = P.solve_ls(v, k, theta, C.GAMMA, N, tol=1e-9, maxiter=900,
                                 device=dev)
                inc = P.incident_plane_wave(N, k, theta, dev).to(torch.complex64)
                err = torch.zeros(n, device=dev)
                for s0 in range(0, n, stack):
                    sl = slice(s0, min(s0 + stack, n))
                    x = D.build_inputs(v[sl].float(), inc[sl], fv[sl])
                    o = model(x)
                    u = torch.complex(o[:, 0], o[:, 1])
                    d = (inc[sl] + u).to(torch.complex128) - psi[sl]
                    err[sl] = (torch.linalg.norm(d.reshape(d.shape[0], -1), dim=1) /
                               torch.linalg.norm(psi[sl].reshape(d.shape[0], -1), dim=1))
                e = err.cpu().numpy()
                rows.append(dict(f=float(f), N=int(N), n=int(n),
                                 q=float(N / f),
                                 mean=float(e.mean()), std=float(e.std()),
                                 median=float(np.median(e)),
                                 q25=float(np.percentile(e, 25)),
                                 q75=float(np.percentile(e, 75)),
                                 per_sample=[float(z) for z in e]))
                print("  N=%3d f=%5.2f  q=%5.2f  eps=%.4f" %
                      (N, f, N / f, rows[-1]["mean"]), flush=True)
    return dict(arm="cross_fno_a025", seed=int(seed), vmax=vmax, n=n,
                media_seed=int(media_seed), grids=[int(g) for g in grids],
                freqs=[float(z) for z in freqs], rows=rows,
                ckpt_epoch=ck.get("epoch"), ckpt_val=ck.get("val"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    for s in a.seeds:
        t0 = time.time()
        res = eval_seed(s, n=a.n)
        out = os.path.join(C.RES, "cross_eval_s%d.json" % s)
        C.save_json(res, out)
        print("saved %s (%.0fs)" % (out, time.time() - t0), flush=True)
