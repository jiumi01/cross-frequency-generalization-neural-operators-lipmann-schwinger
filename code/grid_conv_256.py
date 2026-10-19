# Author: Longgang Wang
"""Systematic 256^2 convergence study (ChatGPT review item 1).

The reviewer asks for a systematic 64^2 / 128^2 / 256^2 convergence study
covering the in-band region, the band edge, and the high-frequency region,
instead of a single verification point.  We run the three-grid solve on the
SAME media batch as the existing verify_256 point (seed 60002, vmax = 0.25,
n = 8) at f in {2.0, 4.0, 4.25, 8.0, 12.0}: 2 and 4 lie inside the training
band [1,4], 4.25 is the band edge where the cliff starts, and 8 and 12 are
out of band.  For each f we report the two-grid relative differences
delta_128_vs_256 and delta_64_vs_128, so the reviewer's three regions are
each covered by a measured convergence step.
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P

dev = "cuda" if torch.cuda.is_available() else "cpu"
GAMMA = C.GAMMA

def downsample(x, f=2):
    B = x.shape[0]
    N = x.shape[-1] // f
    return x.reshape(B, N, f, N, f).mean(dim=(2, 4))

def rel_l2(pred, target):
    num = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-12)
    return num / den

def main():
    n = 8
    vmax = 0.25
    seed = 60002                      # same media batch as verify_256
    freqs = [2.0, 4.0, 4.25, 8.0, 12.0]

    g = torch.Generator(device=dev).manual_seed(seed)
    theta = torch.rand(n, generator=g, device=dev) * 2 * np.pi
    # media drawn at 256^2 and block-downsampled to 128^2 and 64^2, exactly as
    # in grid_convergence.py verify_256
    v256 = P.make_contrast(256, n, vmax, C.MM_REF, g, dev) * 1.0
    v128 = downsample(v256)
    v64 = downsample(v128)

    rows = []
    t0 = time.time()
    for f in freqs:
        fv = torch.full((n,), float(f), device=dev)
        k = 2 * np.pi * fv
        psi256 = P.solve_ls(v256, k, theta, GAMMA, 256, tol=1e-9, maxiter=900, device=dev)
        psi128 = P.solve_ls(v128, k, theta, GAMMA, 128, tol=1e-9, maxiter=900, device=dev)
        psi64 = P.solve_ls(v64, k, theta, GAMMA, 64, tol=1e-9, maxiter=900, device=dev)
        d128v256 = rel_l2(downsample(psi256), psi128)
        d64v128 = rel_l2(psi64, downsample(psi128))
        d64v256 = rel_l2(psi64, downsample(downsample(psi256)))
        r = dict(f=f,
                 delta_128_vs_256=float(d128v256.mean()),
                 delta_64_vs_128=float(d64v128.mean()),
                 delta_64_vs_256=float(d64v256.mean()))
        rows.append(r)
        print("[256] f=%-5g  d128vs256=%.4f  d64vs128=%.4f  d64vs256=%.4f  (%.1fs)"
              % (f, r["delta_128_vs_256"], r["delta_64_vs_128"], r["delta_64_vs_256"],
                 time.time() - t0), flush=True)

    out = dict(stage="A2", arm=vmax, n=n, seed=seed, rows=rows, device=dev)
    C.save_json(out, os.path.join(C.RES, "grid_conv_256.json"))
    print("saved", os.path.join(C.RES, "grid_conv_256.json"), flush=True)
    print("total %.1fs" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
