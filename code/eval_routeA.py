# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China
"""Route A control: strictly periodic incident waves vs the manuscript protocol.

ChatGPT reviewer Q1 asks whether the periodic Green function + FFT wrapping is
consistent with an incident plane wave of arbitrary direction and frequency.
This script implements the Route A control: the SAME media, the SAME integer
frequency set, and the same trained FNO, compared under two incidence
protocols:

  * axial   : theta = 0 (d = (1,0)).  Then f*cos(theta) = f and
              f*sin(theta) = 0 are both integers for every integer f, so the
              plane wave IS a periodic function on the unit torus and the FFT
              wrap introduces no jump at the grid boundary.
  * random  : theta ~ U[0, 2pi), the manuscript protocol, in which the plane
              wave is generically aperiodic and the wrap is a modelling
              choice (defended in Remark 1).

If the generalisation cliff is an artefact of the boundary mismatch, the two
protocols should separate; if the wrap does not covary with f and enters both
sides of the paired comparison equally, the axial curve should reproduce the
same in-band flatness, the same band-edge rise and the same plateau.  No
retraining is needed: the FNO was trained on f in [1,4] and is evaluated
zero-shot beyond.
"""
import os
import sys
import json
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
import evaluate as E

F_INT = np.arange(1, 13, dtype=np.float64)          # integer f: periodic axial
F_INT_LIST = [float(f) for f in F_INT]


def media(vmax, n, vi, device):
    """Same media seeds as the manuscript sweep (eval_cell uses 70000+vi)."""
    g = torch.Generator(device=device).manual_seed(70000 + vi)
    return P.make_contrast(C.N, n, vmax, C.MM_REF, g, device)


@torch.no_grad()
def run_protocol(model, v, f, theta, device, emb):
    """Solve LS under one incidence protocol and score the model zero-shot."""
    n = v.shape[0]
    kk = torch.full((n,), 2.0 * np.pi * f, device=device, dtype=torch.float64)
    if np.isscalar(theta):
        th = torch.full((n,), theta, device=device, dtype=torch.float64)
    else:
        th = torch.from_numpy(np.asarray(theta, dtype=np.float64)).to(device)
    psi_inc = P.incident_plane_wave(C.N, kk, th, device).to(torch.complex64)
    psi = P.solve_ls(v, kk, th, C.GAMMA, C.N, device=device, tol=1e-10)
    fv = torch.full((n,), f, device=device, dtype=torch.float32)
    x = D.build_inputs(v.float(), psi_inc, fv, emb=emb)
    o = model(x)
    u = torch.complex(o[:, 0], o[:, 1])
    psi_hat = psi_inc + u
    e = (torch.linalg.norm((psi_hat - psi.to(torch.complex64)).reshape(n, -1), dim=1)
         / torch.linalg.norm(psi.reshape(n, -1), dim=1).clamp_min(1e-12))
    return e.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(C.RUNS, "main_fno_a025_s0",
                                                   "checkpoint.pt"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--vmax", default="all",
                    help="contrast arm(s): all (5 arms) or a value like 0.25")
    ap.add_argument("--f-list", default="",
                    help="comma list to override F_INT (e.g. 1,2,3,4,5,6,12)")
    ap.add_argument("--out", default=os.path.join(C.RES, "routeA.json"))
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device=%s" % dev, flush=True)
    model, ck = E.load_model(a.ckpt, dev)
    emb = ck["cfg"].get("f_emb", "none")
    print("ckpt=%s  emb=%s  val=%.4f  epoch=%s"
          % (a.ckpt, emb, ck.get("val"), ck.get("epoch")), flush=True)

    vmaxes = ([float(v) for v in a.vmax.split(",")] if a.vmax != "all"
              else list(C.VMAX_LIST))
    fs = ([float(f) for f in a.f_list.split(",")] if a.f_list
          else F_INT_LIST)

    rng = np.random.RandomState(20261018)           # fixed random-theta stream
    results = dict(ckpt=a.ckpt, emb=emb, n=a.n, f_list=fs,
                   f_train=C.F_TRAIN, gamma=C.GAMMA,
                   protocol="axial theta=0 (periodic) vs random theta (manuscript)",
                   vmax_list=vmaxes, tag=a.tag)
    for vi, vmax in enumerate(vmaxes):
        v = media(vmax, a.n, vi, dev)
        rows = []
        for f in fs:
            e_ax = run_protocol(model, v, f, 0.0, dev, emb)
            th_r = rng.uniform(0.0, 2.0 * np.pi, size=a.n)
            e_rd = run_protocol(model, v, f, th_r, dev, emb)
            rows.append(dict(
                f=f,
                axial_mean=float(e_ax.mean()), axial_std=float(e_ax.std()),
                axial_med=float(np.median(e_ax)), axial_q90=float(np.quantile(e_ax, 0.9)),
                rand_mean=float(e_rd.mean()), rand_std=float(e_rd.std()),
                rand_med=float(np.median(e_rd)), rand_q90=float(np.quantile(e_rd, 0.9)),
                ratio=float(e_ax.mean() / max(e_rd.mean(), 1e-12))))
            print("vmax=%.2f f=%4.1f  axial=%.4f  random=%.4f  ratio=%.3f"
                  % (vmax, f, e_ax.mean(), e_rd.mean(),
                     e_ax.mean() / max(e_rd.mean(), 1e-12)), flush=True)
        results["vmax_%.2f" % vmax] = rows
    C.save_json(results, a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
