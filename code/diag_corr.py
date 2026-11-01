# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Correction diagnostic: what does the model emit, and in which direction?

The plateau of the cross-frequency error sits at the level of the trivial
predictor u_theta == 0 (predict the incident field).  That fact alone does not
say which mechanism produced it, and the two candidate mechanisms are different
claims:

  * the correction has collapsed toward zero -- the model returns the incident
    field and nothing else;
  * the correction is still there but points the wrong way, so it adds more
    error than it removes.

They separate on three numbers computed here per test frequency:

    ratio(f)      = E[ ||u_hat||_2 / ||u||_2 ]          energy retained
    signed(f)     = E[ Re<u_hat,u> / (||u_hat|| ||u||) ]  direction, WITH sign
    err_ratio(f)  = E[ ||u_hat - u||_2 / ||u||_2 ]      model error / trivial error

Why the sign matters, and why an earlier version of this file was not enough:
the per-sample identity

    ||u_hat - u||^2 = ||u||^2 + ||u_hat||^2 - 2 Re<u_hat, u>

shows that the error relative to the trivial baseline is

    err_ratio^2 = 1 + ratio^2 - 2 * ratio * signed.

Taking the ABSOLUTE VALUE of the inner product -- which the first version did,
reporting |<u_hat,u>| -- deletes the sign and therefore deletes exactly the term
that decides whether the model is better or worse than predicting zero.  With
ratio = 0.58, a signed correlation of +0.58 gives err_ratio = 0.82 (better than
the baseline), while the same magnitude with a minus sign gives 1.42 (42% WORSE
than the baseline).  The two are indistinguishable in the magnitude, so the
magnitude cannot support any statement about the error level.  This is why the
signed quantity is computed and reported here, and why the derived err_ratio is
reported alongside it rather than left to be composed from the two means: the
identity holds per sample, not for the expectations, so the ratio is measured
directly.

ratio -> 0 means collapse.  ratio = O(1) with signed -> 0 means incoherence.
ratio = O(1) with signed < 0 means the correction is actively harmful, and
err_ratio > 1 is its signature.

Runs on GPU by default; pass --device cpu to avoid contending with a training
or evaluation job.  Read-only with respect to every existing archive: writes
corr_energy.json.
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import evaluate as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="main_fno_a025")
    ap.add_argument("--vmax", type=float, default=0.25)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--device", default="cuda",
                    help="cuda (default) or cpu; cpu avoids contending with a "
                         "running training/evaluation job and is deterministic")
    ap.add_argument("--out", default=os.path.join(C.RES, "corr_energy.json"))
    a = ap.parse_args()
    if a.device == "cuda" and not torch.cuda.is_available():
        print("[note] no CUDA device; falling back to cpu")
        a.device = "cpu"
    dev = a.device

    ckpts = sorted(glob.glob(os.path.join(C.RUNS, a.arm + "*", "checkpoint.pt")))
    if not ckpts:
        print("no checkpoints for %s" % a.arm); return
    print("seeds:", [os.path.basename(os.path.dirname(p)) for p in ckpts])

    F = list(C.F_TEST)
    # The sweep in evaluate.evaluate_run calls eval_cell with seed 60000 + vi,
    # where vi indexes the v_max list; matching that seed here is what makes the
    # words "the same paired media" true.  The first version of this file used
    # seed 50000 -- the eval_cell DEFAULT, not the one the sweep passes -- so it
    # measured statistically equivalent but different media while its docstring
    # claimed the curves' own samples.
    vi = C.VMAX_LIST.index(a.vmax)
    acc = {f: dict(ratio=[], signed=[], absalign=[], err_ratio=[], proj=[],
                   nu=[], nuh=[], npsi=[]) for f in F}
    for cp in ckpts:
        model, _ = E.load_model(cp, dev)
        for f in F:
            d = D.sample_batch(a.n, 60000 + vi, dev, vmax_lo=a.vmax,
                               vmax_hi=a.vmax, f_fixed=f, gamma=C.GAMMA)
            psi = d["psi"]
            x, psi_inc = E.build(C.N, d["k"], d["theta"], d["v"], dev)
            with torch.no_grad():
                o = model(x)
            psi_hat = psi_inc + torch.complex(o[:, 0], o[:, 1])
            u = (psi.to(torch.complex128)
                 - psi_inc.to(torch.complex128))
            uh = (psi_hat.to(torch.complex128)
                  - psi_inc.to(torch.complex128))
            uf = u.reshape(a.n, -1)
            hf = uh.reshape(a.n, -1)
            nu = torch.linalg.norm(uf, dim=1)
            nh = torch.linalg.norm(hf, dim=1)
            ne = torch.linalg.norm((hf - uf), dim=1)
            # <u_hat, u> with the complex inner product convention.  The REAL
            # PART is kept: its sign is the term that decides whether the
            # correction helps or hurts, and the per-sample identity
            #   ||u_hat-u||^2 = ||u||^2 + ||u_hat||^2 - 2 Re<u_hat,u>
            # is what ties these numbers to the error the tables report.
            re_ip = (hf.conj() * uf).sum(dim=1).real
            den = (nh * nu).clamp_min(1e-30)
            acc[f]["ratio"] += [float(t) for t in (nh / nu.clamp_min(1e-30)).cpu()]
            acc[f]["signed"] += [float(t) for t in (re_ip / den).cpu()]
            # the magnitude of the same quantity, kept so the paper can state
            # what the magnitude would have said and why it is not enough
            acc[f]["absalign"] += [float(t) for t in (re_ip.abs() / den).cpu()]
            # measured directly, not composed from the two means above: the
            # identity is per-sample, so E[err_ratio^2] != 1 + E[ratio]^2 - ...
            acc[f]["err_ratio"] += [float(t) for t in (ne / nu.clamp_min(1e-30)).cpu()]
            acc[f]["proj"] += [float(t) for t in
                               (re_ip / (nu * nu).clamp_min(1e-30)).cpu()]
            acc[f]["nu"] += [float(t) for t in nu.cpu()]
            acc[f]["nuh"] += [float(t) for t in nh.cpu()]
            acc[f]["npsi"] += [float(t) for t in
                               torch.linalg.norm(psi.reshape(a.n, -1), dim=1).cpu()]

    res = dict(arm=a.arm, vmax=a.vmax, n_per_seed=a.n,
               n_seeds=len(ckpts), device=dev, f=F,
               ratio={}, signed={}, absalign={}, err_ratio={}, proj={},
               resolution={})
    for f in F:
        res["ratio"]["%g" % f] = float(np.mean(acc[f]["ratio"]))
        res["signed"]["%g" % f] = float(np.mean(acc[f]["signed"]))
        res["absalign"]["%g" % f] = float(np.mean(acc[f]["absalign"]))
        res["err_ratio"]["%g" % f] = float(np.mean(acc[f]["err_ratio"]))
        res["proj"]["%g" % f] = float(np.mean(acc[f]["proj"]))
        res["resolution"]["%g" % f] = float(
            np.mean(acc[f]["nu"]) / np.mean(acc[f]["npsi"]))
    # Self-check of the per-sample identity against the aggregated numbers.
    # It is NOT expected to hold exactly for the means (Jensen), so it is
    # reported as a residual rather than asserted.
    resid = []
    for f in F:
        r = res["ratio"]["%g" % f]
        s = res["signed"]["%g" % f]
        e = res["err_ratio"]["%g" % f]
        resid.append(abs(e * e - (1 + r * r - 2 * r * s)))
    res["identity_residual_max"] = float(max(resid))
    C.save_json(res, a.out)

    print("   f    E|uh|/|u|  E[signed]  E[|align|]  E|uh-u|/|u|  E[Re<uh,u>/|u|^2]  E|u|/E|psi|")
    for f in F:
        print("%6.2f   %8.4f   %8.4f   %8.4f   %10.4f   %12.4f   %10.4f"
              % (f, res["ratio"]["%g" % f], res["signed"]["%g" % f],
                 res["absalign"]["%g" % f], res["err_ratio"]["%g" % f],
                 res["proj"]["%g" % f], res["resolution"]["%g" % f]))
    inb = [f for f in F if f <= C.F_TRAIN[1] + 1e-9]
    out = [f for f in F if f >= 6.0]
    print("in band : ratio %.4f  signed %.4f  |align| %.4f  err_ratio %.4f"
          % (np.mean([res["ratio"]["%g" % f] for f in inb]),
             np.mean([res["signed"]["%g" % f] for f in inb]),
             np.mean([res["absalign"]["%g" % f] for f in inb]),
             np.mean([res["err_ratio"]["%g" % f] for f in inb])))
    print("plateau : ratio %.4f  signed %.4f  |align| %.4f  err_ratio %.4f"
          % (np.mean([res["ratio"]["%g" % f] for f in out]),
             np.mean([res["signed"]["%g" % f] for f in out]),
             np.mean([res["absalign"]["%g" % f] for f in out]),
             np.mean([res["err_ratio"]["%g" % f] for f in out])))
    print("identity residual (max |e^2 - (1+r^2-2rs)| over f): %.4f"
          % res["identity_residual_max"])
    print("NOTE err_ratio > 1 means the model is WORSE than predicting the "
          "incident field alone.")


if __name__ == "__main__":
    main()
