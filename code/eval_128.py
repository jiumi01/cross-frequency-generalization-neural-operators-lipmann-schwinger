# Author: Longgang Wang
"""Evaluate the 128^2-retrained FNO (grid-convergence arm C).

Same media protocol as the main evaluation (v_max fixed, seed 60002 for the
0.25 arm), solved on the 128^2 grid, model fed at 128^2. Output the eps(f)
curve to results/eval_main_fno_a025_128_s0.json plus a compact summary for
the paper table.
"""
import os, sys, json, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
from evaluate import load_model, eval_cell

dev = "cuda" if torch.cuda.is_available() else "cpu"

def eval_128_grid(name, vmax=0.25, n=200, f_test=None, out=None):
    ckpath = os.path.join(C.RUNS, name, "checkpoint.pt")
    model, ck = load_model(ckpath, dev)
    print("[C] %s epoch=%s val=%.4f params=%.2fM"
          % (name, ck.get("epoch"), ck.get("val"), ck["cfg"]["params"]/1e6), flush=True)
    f_test = f_test or C.F_TEST
    vi = {"0.05":0, "0.10":1, "0.25":2, "0.50":3, "1.00":4}["%.2f" % vmax]
    seed = 60000 + vi
    g = torch.Generator(device=dev).manual_seed(seed)
    theta = torch.rand(n, generator=g, device=dev) * 2 * np.pi
    vmax_t = torch.full((n,), float(vmax), device=dev)
    rows = []
    with torch.no_grad():
        for f in f_test:
            fv = torch.full((n,), float(f), device=dev)
            k = 2 * np.pi * fv
            g2 = torch.Generator(device=dev).manual_seed(seed + 7777)
            v = P.make_contrast(128, n, 1.0, C.MM_REF, g2, dev) * vmax_t[:, None, None]
            psi = P.solve_ls(v, k, theta, C.GAMMA, 128, tol=1e-9, maxiter=900,
                             device=dev)
            inc = P.incident_plane_wave(128, k, theta, dev).to(torch.complex64)
            x = D.build_inputs(v.float(), inc, fv.float())
            o = model(x)
            u = torch.complex(o[:, 0], o[:, 1])
            e = (torch.linalg.norm(((inc + u).to(torch.complex128) - psi).reshape(n, -1), dim=1) /
                 torch.linalg.norm(psi.reshape(n, -1), dim=1))
            rows.append(dict(f=f, mean=float(e.mean()), std=float(e.std()),
                             median=float(np.median(e.cpu().numpy()))))
            print("   f=%5.2f  eps128=%.4f" % (f, rows[-1]["mean"]), flush=True)
    res = dict(name=name, arm="a025", n=n, vmax=vmax, rows=rows,
               ckpt_epoch=ck.get("epoch"), ckpt_val=ck.get("val"),
               params=int(ck["cfg"]["params"]))
    C.save_json(res, out)
    print("saved", out)
    return res

if __name__ == "__main__":
    out = os.path.join(C.RES, "eval_main_fno_a025_128_s0.json")
    eval_128_grid("main_fno_a025_128_s0", vmax=0.25, n=200, out=out)
