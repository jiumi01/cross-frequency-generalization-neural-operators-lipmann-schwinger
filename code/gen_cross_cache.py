# Author: Longgang Wang
"""Per-grid training caches for the frequency x resolution cross design.

Protocol mirrors data/train_a025 (the reference arm) byte for byte -- f ~
U[1,4], v_max fixed at 0.25, seeds seed0+ci, chunk 512, 13516 samples
(12288 train + 1228 validation) -- and differs ONLY in the grid, so the
multi-grid arm differs from the reference arm only by the set of grids it is
trained on.

Training media are drawn natively at each grid (not restricted from a master):
the point of the arm is to learn the operator on each mesh, and the paired
restriction is only needed for evaluation, where the question is whether the
same medium scores differently on different meshes.

Usage
-----
    python code/gen_cross_cache.py            # all grids, resumable
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D

N_TOTAL = 13516
CHUNK = 512            # 64/96/128: matches the reference cache
CHUNK_BIG = 256        # 192^2: complex128 Krylov vectors do not fit at 512
dev = "cuda" if torch.cuda.is_available() else "cpu"


def gen(N, n_total=N_TOTAL):
    chunk = CHUNK if N <= 128 else CHUNK_BIG
    out_dir = os.path.join(C.DATA, "cross_N%03d" % N)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    n_chunks = int(np.ceil(n_total / chunk))
    for ci in range(n_chunks):
        path = os.path.join(out_dir, "chunk_%04d.npz" % ci)
        if os.path.exists(path):
            continue
        n = min(chunk, n_total - ci * chunk)
        d = D.sample_batch(n, 1000 + (ci if chunk == CHUNK else ci * 2), dev,
                           f_lo=1.0, f_hi=4.0,
                           vmax_lo=0.25, vmax_hi=0.25, n_grid=N,
                           return_info=True)
        np.savez(path,
                 v=d["v"].detach().cpu().numpy().astype(np.float32),
                 psi=d["psi"].detach().cpu().numpy().astype(np.complex64),
                 k=d["k"].detach().cpu().numpy().astype(np.float32),
                 f=d["f"].detach().cpu().numpy().astype(np.float32),
                 theta=d["theta"].detach().cpu().numpy().astype(np.float32),
                 vmax=d["vmax"].detach().cpu().numpy().astype(np.float32),
                 res=d["res"].detach().cpu().numpy().astype(np.float32))
        if ci % 4 == 0 or ci == n_chunks - 1:
            print("[N=%3d] chunk %03d/%03d  (%d samples, max_res=%.2e, %.0fs)"
                  % (N, ci, n_chunks - 1, n, float(d["res"].max()),
                     time.time() - t0), flush=True)
    print("[N=%3d] done: %d chunks in %s (%.0fs)"
          % (N, n_chunks, out_dir, time.time() - t0), flush=True)


if __name__ == "__main__":
    grids = [int(x) for x in sys.argv[1:]] or [64, 96, 128, 192]
    for N in grids:
        gen(N)
