# Author: Longgang Wang
"""Generate the 128^2 training cache for the grid-convergence retraining arm.

Mirrors the train_a025 protocol byte-for-byte except the grid: v_max fixed at
0.25, f ~ U[1,4], seeds seed0+ci, chunk = 512, 32 chunks (16384 samples).
The retrained FNO then differs from main_fno_a025_s0 ONLY by discretization.
"""
import os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D

dev = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = os.path.join(C.DATA, "train_a025_128")
os.makedirs(out_dir, exist_ok=True)
n_total, chunk = 16384, 512
t0 = time.time()
for ci in range(int(np.ceil(n_total / chunk))):
    path = os.path.join(out_dir, "chunk_%04d.npz" % ci)
    if os.path.exists(path):
        print("exists", path)
        continue
    n = min(chunk, n_total - ci * chunk)
    d = D.sample_batch(n, 1000 + ci, dev, f_lo=1.0, f_hi=4.0,
                       vmax_lo=0.25, vmax_hi=0.25, n_grid=128,
                       return_info=True)
    np.savez(path,
             v=d["v"].detach().cpu().numpy().astype(np.float32),
             psi=d["psi"].detach().cpu().numpy().astype(np.complex64),
             k=d["k"].detach().cpu().numpy().astype(np.float32),
             f=d["f"].detach().cpu().numpy().astype(np.float32),
             theta=d["theta"].detach().cpu().numpy().astype(np.float32),
             vmax=d["vmax"].detach().cpu().numpy().astype(np.float32),
             res=d["res"].detach().cpu().numpy().astype(np.float32))
    print("chunk %04d written (%d, max_res=%.2e, %.0fs)"
          % (ci, n, float(d["res"].max()), time.time() - t0), flush=True)
print("done: %d chunks in %s (%.0fs)"
      % (len([f for f in os.listdir(out_dir) if f.endswith('.npz')]),
         out_dir, time.time() - t0))
