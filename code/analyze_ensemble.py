# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China
"""Ensemble-divergence analysis (supplementary experiment 2).

Question: can the disagreement of the three training seeds serve as a
detector of out-of-band failure?  If the per-sample relative-L2 errors of
seeds s0/s1/s2 diverge exactly where the mean error jumps, the ensemble
spread is a calibrated signal for 'do not trust the surrogate here'; if the
spread stays flat while the error rises, the models fail confidently and
the plateau reflects missing information rather than estimator variance.

Uses only the already-shipped percell_*.npz files (per-sample errors, kept
by evaluate.py by design), so the analysis is a pure re-read of the
published numbers.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RES = C.RES


def load_percell(arm, seed):
    p = os.path.join(RES, "percell_main_fno_%s_s%d.npz" % (arm, seed))
    d = np.load(p)
    return d["per"], d["f_test"], d["vmax"], d["n"]


def main():
    arms = ["a005", "a010", "a025", "a050", "a100"]
    out = {}
    for arm in arms:
        pers = [load_percell(arm, s)[0] for s in (0, 1, 2)]
        per = np.stack(pers, axis=0)          # (3 seeds, 1 vmax, 15 f, 200)
        per = per[:, 0, :, :]                 # (3, 15, 200)
        err_mean = per.mean(axis=0)           # (15, 200) mean over seeds
        err_sd = per.std(axis=0)              # (15, 200) divergence
        fs = load_percell(arm, 0)[1]
        rows = []
        for fi, f in enumerate(fs):
            rows.append(dict(f=float(f),
                             err=float(err_mean[fi].mean()),
                             div=float(err_sd[fi].mean()),
                             err_q90=float(np.quantile(err_mean[fi], 0.9)),
                             div_q90=float(np.quantile(err_sd[fi], 0.9))))
        # paired per-sample correlation in band (f <= 4) and out of band (f >= 5)
        inb = np.concatenate([err_mean[i].ravel() for i in range(15)
                              if fs[i] <= 4.0])
        inb_d = np.concatenate([err_sd[i].ravel() for i in range(15)
                                if fs[i] <= 4.0])
        oob = np.concatenate([err_mean[i].ravel() for i in range(15)
                              if fs[i] >= 5.0])
        oob_d = np.concatenate([err_sd[i].ravel() for i in range(15)
                                if fs[i] >= 5.0])
        corr_in = float(np.corrcoef(inb, inb_d)[0, 1]) if inb.size > 2 else float("nan")
        corr_oob = float(np.corrcoef(oob, oob_d)[0, 1]) if oob.size > 2 else float("nan")
        # divergence amplification across the cliff: div(6-12) / div(in-band)
        div_in = float(np.mean([err_sd[i].mean() for i in range(15) if fs[i] <= 4.0]))
        div_out = float(np.mean([err_sd[i].mean() for i in range(15) if fs[i] >= 6.0]))
        err_in = float(np.mean([err_mean[i].mean() for i in range(15) if fs[i] <= 4.0]))
        err_out = float(np.mean([err_mean[i].mean() for i in range(15) if fs[i] >= 6.0]))
        out[arm] = dict(rows=rows, corr_in=corr_in, corr_oob=corr_oob,
                        div_in=div_in, div_out=div_out,
                        err_in=err_in, err_out=err_out,
                        div_amp=div_out / max(div_in, 1e-12),
                        err_amp=err_out / max(err_in, 1e-12))
        print("arm=%s  err_in=%.4f err_out=%.4f (amp %.1fx) | "
              "div_in=%.4f div_out=%.4f (amp %.1fx) | corr in=%.2f oob=%.2f"
              % (arm, err_in, err_out, err_out / max(err_in, 1e-12),
                 div_in, div_out, div_out / max(div_in, 1e-12),
                 corr_in, corr_oob), flush=True)
        print("  per-f: " + "  ".join("%.1f:%.3f/%.3f" % (r["f"], r["err"], r["div"])
                                      for r in rows), flush=True)
    C.save_json(out, os.path.join(RES, "ensemble_divergence.json"))
    print("saved", os.path.join(RES, "ensemble_divergence.json"), flush=True)


if __name__ == "__main__":
    main()
