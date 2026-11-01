# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Find the hard samples of a cache chunk and compare solver strategies."""
import sys, os, time
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

FLO, FHI = 1.0, 8.0
SEEDS = [1002, 1003, 1004, 1005]
tot = dict(bad_bicg=0, bad_bicg_hi=0, bad_bicg_res=0, n=0)
worst = []
for s in SEEDS:
    d = D.sample_batch(512, s, DEV, f_lo=FLO, f_hi=FHI, return_info=True)
    res = d["res"]
    m = (~torch.isfinite(res)) | (res > 1e-6)
    print("seed %d : bad=%d  nan=%d  max_res=%.3e"
          % (s, int(m.sum()), int(torch.isnan(res).sum()), float(res.amax())))
    if m.any():
        bad = m.nonzero().flatten()
        print("   bad f:", ["%.2f" % x for x in d["f"][bad][:8].cpu().numpy()],
              " vmax:", ["%.2f" % x for x in d["vmax"][bad][:8].cpu().numpy()])
        np.savez(os.path.join(C.DATA, "hard_seed%d.npz" % s),
                 v=d["v"].cpu().numpy().astype(np.float32),
                 k=d["k"].cpu().numpy().astype(np.float32),
                 theta=d["theta"].cpu().numpy().astype(np.float32),
                 f=d["f"].cpu().numpy().astype(np.float32),
                 vmax=d["vmax"].cpu().numpy().astype(np.float32),
                 res=res.cpu().numpy().astype(np.float32))
        idx = bad[:6]
        v = d["v"][idx]; k = d["k"][idx]; th = d["theta"][idx]
        for tag, kw in [("bicg maxiter=600", dict(maxiter=600)),
                        ("bicg maxiter=4000", dict(maxiter=4000)),
                        ("bicg maxiter=4000 tol1e-11", dict(maxiter=4000, tol=1e-11))]:
            t0 = time.time()
            psi, r = P.solve_ls(v, k, th, C.GAMMA, C.N, device=DEV,
                                return_info=True, **kw)
            print("   %-26s res=%s (%.1fs)"
                  % (tag, ["%.1e" % x for x in r.cpu().numpy()[:6]], time.time() - t0))
