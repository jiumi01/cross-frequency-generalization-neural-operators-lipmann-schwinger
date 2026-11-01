# Author: Longgang Wang
"""Per-medium trivial baseline on the *evaluation* media.

Why the existing baseline is not paired
---------------------------------------
`spectrum.json` records the trivial baseline ||u||/||psi|| as a mean over 600
media (`diag_spectrum.py`, n=600), while the evaluation archives
(`percell_*.npz`) hold 200 media drawn from the same seed base 60000+vi with
n=200.  `data.sample_params` draws (v_max, theta, f) in that order from one
generator, and `physics.make_contrast` fills an (B,N,N) tensor sequentially, so
the two draws do share media and angles -- this was checked, not assumed, and
the check is reproduced below (`prefix_check`).  What they do not share is the
*sample*: the 600-media mean contains the 200 evaluation media plus 400 others,
so

  * the ratio "plateau / baseline" quoted in the paper mixes a 200-media
    numerator with a partially overlapping 600-media denominator, and
  * a paired bootstrap of that ratio was impossible, which is why the ratio is
    the one headline scalar in Table `tab:ci` without an interval.

This script recomputes the baseline with the exact call the evaluator uses --
`sample_batch(200, 60000 + vi, f_fixed=f)` -- so numerator and denominator
share media, incidence angle and grid element by element, and the ratio can be
bootstrapped over media the same way `ci_plateau.py` bootstraps everything else.

The ratio is defined as (mean model error over f>=6) / (mean baseline over
f>=6), i.e. a ratio of means, which is what `make_results_tex.plateau_macros`
already quotes.  Keeping the definition and changing only the denominator is
deliberate: the point of this run is to remove a pairing defect, not to change
the statistic the paper reports.

Cost: 5 contrasts x 15 frequencies x 200 solves at 64^2, ~1.5 s per cell on an
RTX 3080 Ti.  Cells are cached, so the run is resumable and idempotent.
"""
import os, sys, json, hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import data as D
import physics as P

EVAL_SEED0 = 60000      # same base as evaluate.eval_cell
EVAL_N = 200            # same count as evaluate.eval_cell
OUT = os.path.join(C.RES, "triv_paired.json")
CELLS = os.path.join(C.RES, "triv_paired")


def _md5(t):
    return hashlib.md5(np.ascontiguousarray(
        t.detach().cpu().numpy().astype(np.float64)).tobytes()).hexdigest()[:16]


def prefix_check(dev):
    """n=200 and n=600 from one seed base: same media and angles, or not?

    Recorded rather than eyeballed because the whole pairing argument rests on
    it.  If a future change to the draw order broke the prefix property, the
    600-media baseline would silently be a different population and the
    comparison in the paper would stop being a comparison at all.
    """
    _, v2, t2 = D.sample_params(EVAL_N, None, None, 0.25, 0.25,
                                EVAL_SEED0, dev, f_fixed=4.0)
    _, v6, t6 = D.sample_params(600, None, None, 0.25, 0.25,
                                EVAL_SEED0, dev, f_fixed=4.0)
    ok = bool(torch.equal(v2, v6[:EVAL_N]) and torch.equal(t2, t6[:EVAL_N]))
    print("[prefix] n=200 vs n=600 share v_max and theta: %s" % ok, flush=True)
    return ok


def cell(vi, vmax, f, dev):
    """Per-medium baseline for one (contrast, frequency) cell, cached."""
    os.makedirs(CELLS, exist_ok=True)
    p = os.path.join(CELLS, "v%.2f_f%05.2f.npz" % (vmax, f))
    if os.path.exists(p):
        d = np.load(p)
        # npz returns 0-d arrays for scalars, and those are unhashable; the
        # digest is used only as a set member, so make it a plain str here.
        return d["triv"], str(d["v_md5"]), str(d["theta_md5"])
    d = D.sample_batch(EVAL_N, EVAL_SEED0 + vi, dev,
                       vmax_lo=vmax, vmax_hi=vmax, f_fixed=f)
    psi = d["psi"]
    inc = P.incident_plane_wave(C.N, d["k"], d["theta"], dev)
    u = psi.to(torch.complex128) - inc
    triv = (torch.linalg.norm(u.reshape(EVAL_N, -1), dim=1)
            / torch.linalg.norm(psi.reshape(EVAL_N, -1), dim=1)).cpu().numpy()
    vm, tm = _md5(d["v"]), _md5(d["theta"])
    np.savez(p, triv=triv, v_md5=vm, theta_md5=tm)
    return triv, vm, tm


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true",
                    help="required: this writes results/triv_paired.json, which "
                         "the manuscript's plateau-to-baseline ratio reads")
    a = ap.parse_args()
    if not a.go:
        raise SystemExit("refusing to run without --go (see module docstring)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("[triv_paired] n=%d per cell, seed base %d, device=%s"
          % (EVAL_N, EVAL_SEED0, dev), flush=True)
    paired_prefix = prefix_check(dev)

    out = dict(n=EVAL_N, seed0=EVAL_SEED0, f=list(C.F_TEST),
               vmax_list=list(C.VMAX_LIST), prefix_ok=paired_prefix, data={})
    for vi, vmax in enumerate(C.VMAX_LIST):
        arm = {}
        md5v, md5t = [], []
        for f in C.F_TEST:
            triv, vm, tm = cell(vi, vmax, f, dev)
            arm["f=%g" % f] = [float(x) for x in triv]
            md5v.append(vm)
            md5t.append(tm)
        # Pairing across frequency, measured: the same medium and the same
        # incidence angle must reappear at all 15 test frequencies.
        arm["_v_md5_unique"] = len(set(md5v))
        arm["_theta_md5_unique"] = len(set(md5t))
        out["data"]["%.2f" % vmax] = arm
        mean_by_f = [float(np.mean(arm["f=%g" % f])) for f in C.F_TEST]
        print("v=%.2f  baseline %.3f..%.3f  distinct v=%d theta=%d"
              % (vmax, min(mean_by_f), max(mean_by_f),
                 len(set(md5v)), len(set(md5t))), flush=True)

    # Cross-check against the 600-media baseline the paper currently quotes.
    try:
        spec = json.load(open(os.path.join(C.RES, "spectrum.json"),
                              encoding="utf-8"))
        diffs = []
        for vmax in C.VMAX_LIST:
            for f in C.F_TEST:
                # spectrum.json keys the contrast arm as "%.2f" but the value
                # "0.10" is stored as "0.1"; resolve by float, not by string.
                skey = None
                for k in spec["data"]:
                    if abs(float(k) - float(vmax)) < 1e-9:
                        skey = k
                a600 = spec["data"][skey]["f=%g" % f]["triv"]
                a200 = np.mean(out["data"]["%.2f" % vmax]["f=%g" % f])
                diffs.append(abs(a200 - a600) / a600)
        out["rel_diff_vs_600"] = dict(mean=float(np.mean(diffs)),
                                      max=float(np.max(diffs)))
        print("[cross-check] |n=200 - n=600|/n=600 : mean %.4f  max %.4f"
              % (np.mean(diffs), np.max(diffs)), flush=True)
    except Exception as e:
        print("[cross-check] skipped (%s)" % e, flush=True)

    C.save_json(out, OUT)
    print("saved %s" % OUT)


if __name__ == "__main__":
    main()
