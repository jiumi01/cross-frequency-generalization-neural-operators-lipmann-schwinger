# Author: Longgang Wang
"""Zero-cost probe: how expensive is one (f, N) cell of the cross design?

Measures wall time of physics.solve_ls for a small batch at the extreme
corners of the planned frequency x grid design, so the schedule is set from
measurements rather than guesses.
"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P

dev = "cuda"
n = 16
for N in (64, 96, 128, 160, 192):
    for f in (1.5, 4.25, 12.0):
        g = torch.Generator(device=dev).manual_seed(1234)
        v = P.make_contrast(N, n, 1.0, C.MM_REF, g, dev) * 0.25
        k = torch.full((n,), 2 * 3.14159265358979 * f, device=dev)
        th = torch.rand(n, generator=g, device=dev) * 6.283185307179586
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        psi, res = P.solve_ls(v, k, th, C.GAMMA, N, tol=1e-9, maxiter=900,
                              device=dev, return_info=True)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        print("N=%3d f=%5.2f  %.2fs  max_res=%.2e  bytes=%d"
              % (N, f, dt, float(res.max()), psi.numel() * psi.element_size()), flush=True)
if dev == "cuda":
    print("peak GPU MB = %.0f" % (torch.cuda.max_memory_allocated() / 2**20))
