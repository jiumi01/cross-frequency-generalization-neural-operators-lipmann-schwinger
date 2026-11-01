# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Cost of the learned operator against the reference solver.

A benchmark paper about learned solvers owes the reader the one number that
decides whether the learned operator is worth using: how long a solve takes.
The physics baseline in the paper (truncated Born) is an *accuracy* baseline,
not a cost one, so this script supplies the missing axis by timing

  (i)  the batched Bi-CGSTAB reference solve, and
  (ii) a forward pass of each learned architecture,

on the identical 64x64 Helmholtz problem.  Timing is wall clock with an explicit
CUDA synchronise, averaged over repeats after warm-up, and reported per sample
so the batched solver is compared against the batched forward pass fairly.

Writes results/cost.json.
"""
import os
import sys
import time
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import models as M


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def timeit(fn, reps=5, warm=2, dev="cuda"):
    for _ in range(warm):
        fn()
    sync(dev)
    t = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default=C.RES)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    archs = [
        ("FNO w40 M20", dict(name="fno", in_ch=4, out_ch=2, width=40, modes=20,
                             n_layers=4, proj=128, norm="layer")),
        ("U-Net base32", dict(name="unet", in_ch=4, out_ch=2, base=32, depth=3)),
        ("U-Net base76", dict(name="unet", in_ch=4, out_ch=2, base=76, depth=3)),
        ("DeepONet w32", dict(name="deeponet", in_ch=4, out_ch=2, width=32,
                              p=128, n_grid=C.N)),
    ]
    res = {"batch": a.batch, "reps": a.reps, "device": dev,
           "solver": {}, "model": {}}

    print("device=%s batch=%d" % (dev, a.batch))

    # ---- reference solver -------------------------------------------------
    # The contrast is a smooth random field at the standard correlation length,
    # because the Bi-CGSTAB iteration count depends on the medium and timing it
    # on a constant field would understate the cost.
    import physics as P
    g = torch.Generator(device=dev).manual_seed(4242)
    v = P.make_contrast(C.N, a.batch, 0.25, C.MM_REF, g, dev)
    for f in [1.5, 4.0, 8.0, 12.0]:
        k = torch.full((a.batch,), 2 * np.pi * f, device=dev, dtype=torch.float64)
        th = torch.zeros(a.batch, device=dev, dtype=torch.float64)

        def once():
            return P.solve_ls(v, k, th, C.GAMMA, C.N, tol=1e-9, maxiter=900,
                              device=dev)

        t = timeit(once, reps=a.reps, dev=dev)
        res["solver"]["%.1f" % f] = dict(
            total_ms=t * 1e3, per_sample_ms=t * 1e3 / a.batch, k=float(k[0]))
        print("  solver  f=%5.1f  %8.2f ms/batch  %7.3f ms/sample"
              % (f, t * 1e3, t * 1e3 / a.batch))

    # ---- learned forwards -------------------------------------------------
    x = torch.randn(a.batch, 4, C.N, C.N, device=dev)
    for label, kw in archs:
        model = M.build_model(**kw).to(dev).eval()
        n_par = sum(p.numel() for p in model.parameters())
        t = timeit(lambda: model(x), reps=a.reps, dev=dev)
        res["model"][label] = dict(
            total_ms=t * 1e3, per_sample_ms=t * 1e3 / a.batch, params=int(n_par))
        print("  %-14s %8.2f ms/batch  %7.3f ms/sample  (%.2f M params)"
              % (label, t * 1e3, t * 1e3 / a.batch, n_par / 1e6))

    # headline ratio, taken at the highest frequency (the solver's worst case)
    fmax = max(res["solver"], key=float)
    s = res["solver"][fmax]["per_sample_ms"]
    for label in res["model"]:
        res["model"][label]["speedup_vs_solver_at_f%s" % fmax] = \
            s / res["model"][label]["per_sample_ms"]
    C.save_json(res, os.path.join(a.out, "cost.json"))
    print("wrote %s" % os.path.join(a.out, "cost.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
