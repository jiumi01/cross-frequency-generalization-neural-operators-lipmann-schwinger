# Author: Longgang Wang
"""Probe: verify train_a025 data protocol and 128-grid solver convergence at f=12."""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", dev)

# 1) protocol of train_a025
z = np.load(os.path.join(C.DATA, "train_a025", "chunk_0000.npz"))
print("chunk shape:", z["v"].shape, "vmax range:", float(z["vmax"].min()), float(z["vmax"].max()),
      "f range:", float(z["f"].min()), float(z["f"].max()))

# 2) 128-grid solver: batch of 8, f=12, vmax=0.25, tol 1e-9
g = torch.Generator(device=dev).manual_seed(60000 + 2)
v = P.make_contrast(128, 8, 1.0, C.MM_REF, g, dev) * 0.25
k = torch.full((8,), 2 * np.pi * 12.0, device=dev)
th = torch.rand(8, generator=g, device=dev) * 2 * np.pi
t0 = time.time()
psi, res = P.solve_ls(v, k, th, C.GAMMA, 128, tol=1e-9, maxiter=900, device=dev, return_info=True)
dt = time.time() - t0
print("128 solve f=12 n=8: %.1fs  max_rel_res=%.2e" % (dt, float(res.max())))

# 3) 64-grid same-media (downsampled) solve for reference
v64 = v.reshape(8, 64, 2, 64, 2).mean(dim=(2, 4))
psi64, res64 = P.solve_ls(v64, k, th, C.GAMMA, 64, tol=1e-9, maxiter=900, device=dev, return_info=True)
print("64  solve f=12 n=8: %.1fs  max_rel_res=%.2e" % (time.time() - t0 - dt, float(res64.max())))
# relative difference 64 vs 128 (downsampled 128 to 64)
psi128d = psi.reshape(8, 64, 2, 64, 2).mean(dim=(2, 4))
delta = (torch.linalg.norm((psi64 - psi128d).reshape(8, -1), dim=1) /
         torch.linalg.norm(psi128d.reshape(8, -1), dim=1))
print("delta(64 vs down128) @ f=12: mean=%.4f  per-sample=%s"
      % (float(delta.mean()), np.round(delta.cpu().numpy(), 4)))
