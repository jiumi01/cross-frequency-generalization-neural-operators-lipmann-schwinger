# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Band-down evaluation: behaviour of band-trained operators BELOW the band.

The training band is f in [1, 4] and the manuscript's test grid F_TEST samples
the right-hand side.  This script evaluates the main FNO arms on the LEFT side,
f in {0.25, 0.5, 0.75, 1.0}, with the same paired protocol as the right side:
the same (v_max, theta) media are solved fresh at every frequency by the batched
Bi-CGSTAB, so f remains the only variable.  The trivial incident baseline
||psi - psi_inc|| / ||psi|| is recorded per (v_max, f) from the same solved
media, so the model errors can be read against the same-arm baseline exactly as
on the right side.  No training is involved: the checkpoint is the band-trained
model whose right-side behaviour the manuscript already characterises.
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import physics as P
import data as D
import models as M
import evaluate as E

F_BELOW = [0.25, 0.5, 0.75, 1.0]


@torch.no_grad()
def cell(model, f, vmax, n=200, device="cuda", seed=50000, emb="none",
         anchor="none", ab=None, gamma=None):
    """Errors and the trivial baseline for one (f, v_max) cell.

    Returns (e, triv): per-sample relative-L2 errors of the model and the
    per-sample trivial incident-field error, both on the SAME solved media.
    """
    gamma = C.GAMMA if gamma is None else gamma
    d = D.sample_batch(n, seed, device, vmax_lo=vmax, vmax_hi=vmax,
                       f_fixed=f, return_info=True, gamma=gamma)
    x, psi_inc = E.build(C.N, d["k"], d["theta"], d["v"], device, emb=emb)
    o = model(x)
    u = torch.complex(o[:, 0], o[:, 1])
    if anchor == "born":
        born = D.anchor_terms(d["v"], d["k"], d["theta"])
        u = D.anchored_u(u, born, ab.to(device))
    psi_hat = psi_inc + u
    psi = d["psi"]
    num = torch.linalg.norm((psi_hat - psi).reshape(n, -1), dim=1)
    den = torch.linalg.norm(psi.reshape(n, -1), dim=1).clamp_min(1e-12)
    e = (num / den).cpu().numpy()
    triv = (torch.linalg.norm((psi_inc - psi).reshape(n, -1), dim=1) /
            den).cpu().numpy()
    return e, triv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default=os.path.join(C.RUNS, "main_fno_*",
                                                      "checkpoint.pt"))
    ap.add_argument("--f", default=None,
                    help="comma-separated frequencies; default %s"
                         % ",".join(str(x) for x in F_BELOW))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(C.RES, "banddown.json"))
    a = ap.parse_args()
    f_list = [float(x) for x in (a.f or "").split(",")] if a.f else F_BELOW
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("band-down evaluation: f=%s n=%d device=%s"
          % (f_list, a.n, dev), flush=True)

    runs = {}
    for ck in sorted(glob.glob(a.pattern)):
        name = os.path.basename(os.path.dirname(ck))
        if name.startswith("_") or name.startswith("probe"):
            continue
        # finished-run rule, identical to evaluate.py --all
        if not os.path.exists(os.path.join(os.path.dirname(ck), "result.json")):
            print("skip (still training)", name, flush=True)
            continue
        model, ckpt = E.load_model(ck, dev)
        cfg = ckpt["cfg"]
        emb = cfg.get("f_emb", "none")
        anchor = cfg.get("anchor", "none")
        ab = ckpt.get("anchor_ab", None)
        if anchor == "born" and ab is None:
            raise SystemExit("%s: anchor=born but no anchor_ab in checkpoint"
                             % name)
        arch, arm, seed, tag = E.parse_name(name)
        print("[%s] arch=%s arm=%s seed=%s emb=%s anchor=%s"
              % (name, arch, arm, seed, emb, anchor), flush=True)
        sweep = {}
        for vi, vmax in enumerate(C.VMAX_LIST):
            rows = []
            for fi, f in enumerate(f_list):
                e, triv = cell(model, f, vmax, n=a.n, device=dev,
                               seed=60000 + vi, emb=emb, anchor=anchor, ab=ab)
                rows.append(dict(f=f, mean=float(e.mean()), std=float(e.std()),
                                 sem=float(e.std() / np.sqrt(len(e))),
                                 median=float(np.median(e)),
                                 q90=float(np.quantile(e, 0.9)),
                                 triv_mean=float(np.mean(triv)),
                                 triv_std=float(np.std(triv))))
                print("  vmax=%.2f f=%.2f mean=%.4f triv=%.4f"
                      % (vmax, f, e.mean(), triv.mean()), flush=True)
            sweep["%.2f" % vmax] = rows
        runs[name] = dict(name=name, arch=arch, arm=arm, seed=seed, tag=tag,
                          f_emb=emb, anchor=anchor, params=int(cfg["params"]),
                          ckpt_epoch=ckpt.get("epoch"), ckpt_val=ckpt.get("val"),
                          f_test=f_list, n=a.n, gamma=C.GAMMA,
                          model=cfg["model"], modes=cfg.get("modes"),
                          sweep=sweep)
    out = dict(f_test=f_list, vmax_list=C.VMAX_LIST, runs=runs)
    C.save_json(out, a.out)
    print("wrote %s (%d runs)" % (a.out, len(runs)), flush=True)


if __name__ == "__main__":
    main()
