# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Diagnose BiCGSTAB robustness across (f, v_max) for the LS system."""
import sys, os
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N = C.N

print("f | vmax | nan | res>1e-6 | maxres | max|psi| | med_iters(est)")
for f in [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0]:
    for vmax in [0.05, 0.25, 0.5, 1.0]:
        B = 128
        g = torch.Generator(device=DEV).manual_seed(31)
        v = P.make_contrast(N, B, vmax, C.MM_REF, g, DEV, kind="smooth")
        k = torch.full((B,), 2 * np.pi * f, device=DEV)
        th = torch.rand(B, generator=g, device=DEV) * 2 * np.pi
        psi, res = P.solve_ls(v, k, th, C.GAMMA, N, tol=1e-9, maxiter=600,
                              device=DEV, return_info=True)
        nan = int(torch.isnan(res).sum() + torch.isnan(psi).sum().item() * 0)
        bad = int((res > 1e-6).sum())
        mx = float(res.amax())
        amp = float(psi.abs().nan_to_num(nan=0).amax())
        print("%4.1f | %.2f | %3d | %3d | %8.2e | %9.2e"
              % (f, vmax, nan, bad, mx, amp))

print("\n--- single-sample breakdown trace (worst case found) ---")
f, vmax = 10.0, 1.0
g = torch.Generator(device=DEV).manual_seed(31)
v = P.make_contrast(N, 8, vmax, C.MM_REF, g, DEV, kind="smooth")
k = torch.full((8,), 2 * np.pi * f, device=DEV)
th = torch.rand(8, generator=g, device=DEV) * 2 * np.pi
for mx in [200, 600, 2000]:
    psi, res = P.solve_ls(v, k, th, C.GAMMA, N, tol=1e-9, maxiter=mx,
                          device=DEV, return_info=True)
    print("  maxiter=%4d  res=%s" % (mx, ["%.1e" % r for r in res.cpu().numpy()]))
