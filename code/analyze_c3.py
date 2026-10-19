# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C3 verdict: loss-induced pessimistic collapse vs. physical boundary.

Reads the six evaluation JSONs (baseline main_fno_a025, energy-regularised
en_fno_a025; same cache, same protocol, only the loss differs) and, in
addition, recomputes the correction-energy ratio |u_hat| / |u| and the phase
alignment cos-angle between u_hat and u directly from the checkpoints at a
subset of test frequencies (the eval JSONs store errors only, and the
energy-ratio readout is exactly the quantity the pessimism hypothesis is
about).  Writes results/c3_verdict.json.

The two probes are independent instruments: the error curve comes from the
standard evaluation protocol, the energy/alignment curves from a fresh
forward pass over freshly drawn paired media.  If the energy penalty recovers
the out-of-band energy WITHOUT recovering the error, the model has learned to
emit well-scaled but wrong corrections -- which is itself evidence against
"the model merely plays it safe".
"""
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import physics as P
from evaluate import load_model

F_IN = (1.5, 2.5, 3.5)
F_PLAT = tuple(f for f in C.F_TEST if f >= 6.0)
ARMS = {"baseline": "main_fno_a025",        # rel-L2 only
        "energy": "en_fno_a025",            # + in-band energy penalty (C3)
        "energy_ob": "eb_fno_a025"}         # + out-of-band Born-anchored (C3b)
ENERGY_F = (4.0, 5.0, 6.0, 8.0, 10.0, 12.0)   # probe frequencies
N_PROBE = 256


def curve(prefix):
    acc = {round(float(f), 2): [] for f in C.F_TEST}
    for s in (0, 1, 2):
        p = os.path.join(C.RES, "eval_%s_s%d.json" % (prefix, s))
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run the arm and its evaluation" % p)
        d = C.load_json(p)
        if float(d.get("gamma", 0)) != C.GAMMA:
            raise SystemExit("%s: gamma protocol mismatch" % p)
        for r in d["sweep"]["0.25"]:
            acc[round(float(r["f"]), 2)].append(float(r["mean"]))
    f = sorted(acc)
    return (f, [float(np.mean(acc[x])) for x in f],
            [float(np.std(acc[x])) for x in f])


def onset_interval(f, e, inb, k=3.0):
    thr = k * inb
    for i, x in enumerate(f):
        if e[i] > thr:
            lo = f[i - 1] if i > 0 else f[0]
            return [float(lo), float(x)]
    return None


@torch.no_grad()
def energy_probe(prefix, device):
    """Mean |u_hat|/|u| and alignment <u_hat,u>/(|u_hat||u|) at probe freqs.

    Fresh paired media (seed 90000 + i), v_max = 0.25, seed-mean over the
    three checkpoints of the arm.  Computed from the checkpoints, NOT from the
    eval JSONs, so the two instruments cannot share a failure mode.
    """
    models = []
    for s in (0, 1, 2):
        m, _ = load_model(os.path.join(C.RUNS, "%s_s%d" % (prefix, s),
                                       "checkpoint.pt"), device)
        models.append(m)
    out = {}
    for i, f in enumerate(ENERGY_F):
        d = D.sample_batch(N_PROBE, 90000 + i, device, vmax_lo=0.25,
                           vmax_hi=0.25, f_fixed=f)
        k = d["k"]
        psi_inc = P.incident_plane_wave(C.N, k, d["theta"], device).to(
            torch.complex64)
        x = D.build_inputs(d["v"].float(), psi_inc, torch.full(
            (N_PROBE,), float(f), device=device))
        u_true = d["psi"] - psi_inc
        rt, at = [], []
        for m in models:
            o = m(x)
            u = torch.complex(o[:, 0], o[:, 1])
            nu = torch.linalg.norm(u.reshape(N_PROBE, -1), dim=1)
            nt = torch.linalg.norm(u_true.reshape(N_PROBE, -1), dim=1)
            rt.append(float((nu / nt.clamp_min(1e-12)).mean()))
            # per-sample alignment of the two complex fields, flattened
            dot = torch.sum(u.reshape(N_PROBE, -1) *
                            u_true.reshape(N_PROBE, -1).conj(), dim=1)
            at.append(float((torch.abs(dot) /
                            (nu * nt).clamp_min(1e-12)).mean()))
        out["%.2f" % f] = dict(energy_ratio=float(np.mean(rt)),
                               alignment=float(np.mean(at)))
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = dict(arms={})
    for label, prefix in ARMS.items():
        f, m, sd = curve(prefix)
        e4 = m[f.index(4.0)]
        e5 = m[f.index(5.0)]
        e425 = m[f.index(4.25)]
        inb = float(np.mean([m[f.index(x)] for x in F_IN]))
        plat = float(np.mean([m[f.index(x)] for x in F_PLAT]))
        out["arms"][label] = dict(
            run=prefix, f=f, mean=m, sd=sd,
            in_band=inb, e4=e4, e425=e425, e5=e5,
            ratio=e5 / e4 if e4 > 0 else None,
            jump=e425 / e4 if e4 > 0 else None,
            plateau=plat,
            onset=onset_interval(f, m, inb), seeds=3,
            energy=energy_probe(prefix, device))
    a = out["arms"]["baseline"]
    k_lo = max(x["e4"] / x["in_band"] for x in out["arms"].values())
    k_hi = min(x["e425"] / x["in_band"] for x in out["arms"].values())
    out["verdict"] = dict(
        onset_k_lo=float(k_lo), onset_k_hi=float(k_hi),
        cliff_interval_baseline=a["onset"],
        cliff_interval_energy=out["arms"]["energy"]["onset"],
        cliff_interval_energy_ob=out["arms"]["energy_ob"]["onset"],
        cliff_interval_identical=all(
            out["arms"][k]["onset"] == a["onset"] for k in out["arms"]),
        plateau_baseline=a["plateau"],
        plateau_energy=out["arms"]["energy"]["plateau"],
        plateau_energy_ob=out["arms"]["energy_ob"]["plateau"],
        in_band_ratio_energy=float(a["in_band"] /
                                   out["arms"]["energy"]["in_band"]),
        in_band_ratio_energy_ob=float(a["in_band"] /
                                      out["arms"]["energy_ob"]["in_band"]))
    C.save_json(out, os.path.join(C.RES, "c3_verdict.json"))
    v = out["verdict"]
    for lab, arm in out["arms"].items():
        print("%-10s in-band %.4f  e(4) %.4f  e(5) %.4f  plateau %.4f  "
              "jump %.2fx  onset %s"
              % (lab, arm["in_band"], arm["e4"], arm["e5"], arm["plateau"],
                 arm["jump"], arm["onset"]))
    print("onset rule window k in (%.2f, %.2f]; identical=%s"
          % (v["onset_k_lo"], v["onset_k_hi"], v["cliff_interval_identical"]))
    for lab, arm in out["arms"].items():
        print(lab + " energy probe (|u_hat|/|u|, alignment):")
        for fk, r in arm["energy"].items():
            print("   f=%-5s ratio %.3f  align %.3f"
                  % (fk, r["energy_ratio"], r["alignment"]))
    print("wrote results/c3_verdict.json")


if __name__ == "__main__":
    main()
