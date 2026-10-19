# Author: Longgang Wang
"""Grid-convergence study (reviewer gap): does the 64^2 discretization change
the cliff/plateau conclusions?

Stage A  reference-solution convergence: same continuous media (128^2 draw,
downsampled to 64^2), solve at both grids for f in {4,8,12} x all contrasts,
report delta(f) = mean ||psi64 - down(psi128)|| / ||down(psi128)||.
Plus one 256^2 verification point to show the convergence trend.

Stage B  cross-grid evaluation of the trained FNO (resolution-invariant):
feed the 128^2 input directly, compare eps(f) curves against the 64^2-labelled
curve, at v_max = 0.25 and 1.00.
"""
import os, sys, json, time, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
import models as M
from evaluate import load_model

dev = "cuda" if torch.cuda.is_available() else "cpu"
N64 = 64

def downsample(x, f=2):
    """Block-mean downsample (B, fN, fN) -> (B, N, N)."""
    B = x.shape[0]
    N = x.shape[-1] // f
    return x.reshape(B, N, f, N, f).mean(dim=(2, 4))

def rel_l2(pred, target):
    num = torch.linalg.norm((pred - target).reshape(pred.shape[0], -1), dim=1)
    den = torch.linalg.norm(target.reshape(target.shape[0], -1), dim=1).clamp_min(1e-12)
    return num / den

def media_batch(n, seed, vmax, n_grid, device):
    """Same-media protocol as evaluate.eval_cell: vmax fixed, f drawn later."""
    g = torch.Generator(device=device).manual_seed(int(seed) + 7777)
    v = P.make_contrast(n_grid, n, 1.0, C.MM_REF, g, device) * vmax
    return v

def solve(v, f, theta, n_grid, device):
    fv = torch.as_tensor(f, dtype=torch.float64, device=device)
    if fv.dim() == 0:
        fv = fv.reshape(1).expand(v.shape[0])
    k = 2 * np.pi * fv
    return P.solve_ls(v, k, theta, C.GAMMA, n_grid, tol=1e-9, maxiter=900,
                      device=device)

# ---------------------------------------------------------------- Stage A
def stage_a(n=200, out=None):
    rows = {}
    for vi, vmax in enumerate(C.VMAX_LIST):
        seed = 60000 + vi
        g = torch.Generator(device=dev).manual_seed(seed)
        theta = torch.rand(n, generator=g, device=dev) * 2 * np.pi
        v128 = media_batch(n, seed, vmax, 128, dev)
        v64 = downsample(v128)
        fr = []
        for f in [4.0, 8.0, 12.0]:
            psi128 = solve(v128, f, theta, 128, dev)
            psi64 = solve(v64, f, theta, 64, dev)
            p128d = downsample(psi128)
            d = rel_l2(psi64, p128d)
            fr.append(dict(f=f, mean=float(d.mean()), std=float(d.std()),
                           median=float(np.median(d.cpu().numpy())),
                           q90=float(np.quantile(d.cpu().numpy(), 0.9))))
        rows["%.2f" % vmax] = fr
        print("[A] vmax=%.2f  delta:" % vmax,
              " ".join("%.3f" % r["mean"] for r in fr), flush=True)
    # 256^2 verification point: f=12, vmax=0.25, n=8
    nv = 8
    seed = 60002
    g = torch.Generator(device=dev).manual_seed(seed)
    theta = torch.rand(nv, generator=g, device=dev) * 2 * np.pi
    v256 = media_batch(nv, seed, 0.25, 256, dev)
    v128 = downsample(v256, 2)
    v64 = downsample(v128, 2)
    psi256 = solve(v256, 12.0, theta, 256, dev)
    psi128 = solve(v128, 12.0, theta, 128, dev)
    psi64 = solve(v64, 12.0, theta, 64, dev)
    d128v256 = rel_l2(downsample(psi256, 2), psi128)
    d64v128 = rel_l2(psi64, downsample(psi128))
    rows["verify_256"] = dict(f=12.0, vmax=0.25, n=nv,
                              delta_128_vs_256=float(d128v256.mean()),
                              delta_64_vs_128=float(d64v128.mean()))
    print("[A] 256 verify @f=12 vmax=.25: d128vs256=%.3f  d64vs128=%.3f"
          % (rows["verify_256"]["delta_128_vs_256"],
             rows["verify_256"]["delta_64_vs_128"]), flush=True)
    C.save_json(dict(stage="A", n=n, rows=rows, device=dev), out)
    print("saved", out)

# ---------------------------------------------------------------- Stage B
@torch.no_grad()
def stage_b(ckpt_names, f_test, n=100, out=None):
    res = {}
    for name, arm in ckpt_names:
        ckpath = os.path.join(C.RUNS, name, "checkpoint.pt")
        model, ck = load_model(ckpath, dev)
        print("[B] %s epoch=%s val=%.4f" % (name, ck.get("epoch"), ck.get("val")), flush=True)
        vi = {"0.25": 2, "1.00": 4}[arm]
        seed = 60000 + vi
        g = torch.Generator(device=dev).manual_seed(seed)
        theta = torch.rand(n, generator=g, device=dev) * 2 * np.pi
        vmax = float(arm)
        v128 = media_batch(n, seed, vmax, 128, dev)
        v64 = downsample(v128)
        vmax_t = torch.full((n,), vmax, device=dev)
        rows = []
        for f in f_test:
            fv = torch.full((n,), float(f), device=dev)
            k = 2 * np.pi * fv
            # 128-grid reference label
            psi128 = solve(v128, f, theta, 128, dev)
            psi64 = solve(v64, f, theta, 64, dev)
            # model on 64-grid input (native training resolution)
            inc64 = P.incident_plane_wave(64, k, theta, dev).to(torch.complex64)
            x64 = D.build_inputs(v64.float(), inc64, fv.float())
            o64 = model(x64)
            u64 = torch.complex(o64[:, 0], o64[:, 1])
            p64 = (inc64 + u64).to(torch.complex128)
            # model on 128-grid input (resolution extrapolation; FNO invariant)
            inc128 = P.incident_plane_wave(128, k, theta, dev).to(torch.complex64)
            x128 = D.build_inputs(v128.float(), inc128, fv.float())
            o128 = model(x128)
            u128 = torch.complex(o128[:, 0], o128[:, 1])
            p128 = (inc128 + u128).to(torch.complex128)
            e64 = rel_l2(p64, psi64)
            e128 = rel_l2(p128, psi128)
            e64d = rel_l2(p64, downsample(psi128))
            rows.append(dict(f=f,
                             eps_64=float(e64.mean()),
                             eps_128in=float(e128.mean()),
                             eps_64_vs_down128=float(e64d.mean()),
                             delta_64vs128=float(rel_l2(psi64, downsample(psi128)).mean())))
        res[name] = dict(arm=arm, rows=rows)
        print("[B] %s: f | eps64 | eps128 | eps64d | delta"
              % name, flush=True)
        for r in rows:
            print("   %5.2f | %.3f | %.3f | %.3f | %.3f"
                  % (r["f"], r["eps_64"], r["eps_128in"], r["eps_64_vs_down128"],
                     r["delta_64vs128"]), flush=True)
    C.save_json(dict(stage="B", n=n, res=res), out)
    print("saved", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["A", "B", "AB"])
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    t0 = time.time()
    if a.stage in ("A", "AB"):
        stage_a(n=a.n, out=os.path.join(C.RES, "grid_conv_ref.json"))
    if a.stage in ("B", "AB"):
        stage_b([("main_fno_a025_s0", "0.25"), ("main_fno_a100_s0", "1.00")],
                C.F_TEST, n=a.n, out=os.path.join(C.RES, "grid_conv_model.json"))
    print("total %.1fs" % (time.time() - t0))
