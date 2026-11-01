# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C1 verdict: conditioning confound vs. frequency boundary.

Reads the six evaluation JSONs of the two paired arms (baseline main_fno_a025,
compensation gc_fno; same media, same protocol, only the regularisation
differs) and writes results/c1_verdict.json.  The paper's table and figure are
generated from that file, so the manuscript numbers and this script cannot
drift apart.

The comparison the verdict rests on:
  * baseline: gamma = 0.05 constant; the attained operator norm spans 2.2x
    over the test grid (4.53 at f = 2.5 to 10.02 from f = 4 on; non-monotone,
    see code/diag_propnorm.py and Table "propnorm" of the manuscript);
  * C1: gamma(f) from gamma_comp.json; attained norm constant at 2.9075,
    which sits 3.4x BELOW the baseline's band-edge value, so the compensated
    arm is the better-conditioned of the two at every frequency.
If the cliff were a conditioning artefact it would move or vanish under the
compensation.  It does not: the rise starts in the same [4.0, 4.25] grid
interval in both arms, and the band-edge ratio e(5)/e(4) grows rather than
shrinks (7.5 -> 18.5).

The two arms sit at different couplings v_max/(2 gamma) -- the compensation
raises gamma from 0.05 to 0.15-0.18 on the test grid -- so their ABSOLUTE error
levels are not comparable; only the onset location and the step shape are, and
the level difference is what the gamma-robustness ladder (Table "gamma") already
predicts.  The attained norms are therefore computed here from the same
instrument that built the compensation (gamma_comp.attained_norm) rather than
transcribed: an earlier revision of this script hard-coded "3.44x (2.91 ..
10.03)" as the baseline's spread, but 2.91 is the compensated TARGET (0.97 x
min_f attained(f, 0.02)), not a baseline value; the baseline's own spread on
this grid is 2.2x, and 3.44 is the ratio of the baseline's band-edge norm to
the target -- a third, different quantity.  All three are read off the data.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import gamma_comp as GC

F_IN = (1.5, 2.5, 3.5)                 # in-band reference frequencies
F_PLAT = tuple(f for f in C.F_TEST if f >= 6.0)
ARMS = {"baseline": "main_fno_a025", "compensated": "gc_fno"}


def curve(prefix):
    """seed-mean error curve at v_max = 0.25, plus its across-seed SD."""
    acc = {round(float(f), 2): [] for f in C.F_TEST}
    for s in (0, 1, 2):
        p = os.path.join(C.RES, "eval_%s_s%d.json" % (prefix, s))
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run the arm and its evaluation" % p)
        d = C.load_json(p)
        for r in d["sweep"]["0.25"]:
            acc[round(float(r["f"]), 2)].append(float(r["mean"]))
    f = sorted(acc)
    return (f, [float(np.mean(acc[x])) for x in f],
            [float(np.std(acc[x])) for x in f])


def attained_curves():
    """Attained operator norm at every test frequency, for both arms.

    A property of the forward operator, not of a model, so it carries no seed
    spread.  Computed on the CPU (a 64x64 symbol needs no GPU) through the same
    instrument that built the compensation table.
    """
    base = [GC.attained_norm(f, C.GAMMA, C.N) for f in C.F_TEST]
    comp = [GC.attained_norm(f, GC.comp_gamma(f), C.N) for f in C.F_TEST]
    return base, comp


def onset_interval(f, e, inb, k=3.0):
    """First grid interval in which the error leaves k x the in-band level.

    Returned as an interval, not a point: the test grid resolves the band edge
    to 0.25, so the honest statement is which interval the rise starts in.
    The factor is 3, not the manuscript's 6: that rule is defined against the
    absolute band-edge error e(4) and would compare unlike quantities here,
    because the compensation arm's in-band level is 5.5x lower.  Judging
    "leaves its in-band level" against the in-band level itself is the fair
    form, and at k=3 both arms sit far from the borderline (11.8x and 6.0x
    respectively), so the verdict does not hinge on the constant.
    """
    thr = k * inb
    for i, x in enumerate(f):
        if e[i] > thr:
            lo = f[i - 1] if i > 0 else f[0]
            return [float(lo), float(x)]
    return None


def main():
    out = dict(arms={})
    curves = {}
    for label, prefix in ARMS.items():
        f, m, sd = curve(prefix)
        curves[label] = (f, m)
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
            onset=onset_interval(f, m, inb),
            seeds=3)
    a, g = out["arms"]["baseline"], out["arms"]["compensated"]
    base_at, comp_at = attained_curves()
    gtab = C.load_json(os.path.join(C.RES, "gamma_comp.json"))
    target = float(gtab["target"])
    a["attained"], g["attained"] = base_at, comp_at
    gammas = [GC.comp_gamma(f) for f in C.F_TEST]
    # Window of onset-rule constants for which BOTH arms read the same interval.
    # The rule is arm-matched (k x that arm's own in-band mean), and the two
    # in-band means differ by 5.5x, so a constant tuned on one arm would move
    # the other's interval by a cell.  Reporting the window rather than a single
    # k makes the shared interval a measured fact instead of a choice.
    k_lo = max(a["e4"] / a["in_band"], g["e4"] / g["in_band"])
    k_hi = min(a["e425"] / a["in_band"], g["e425"] / g["in_band"])
    out["verdict"] = dict(
        attained_baseline_lo=float(min(base_at)),
        attained_baseline_hi=float(max(base_at)),
        attained_baseline_spread=float(max(base_at) / min(base_at)),
        attained_compensated=target,
        attained_compensated_max_dev=float(max(abs(v - target) for v in comp_at)),
        attained_level_ratio=float(max(base_at) / target),
        gamma_test_lo=float(min(gammas)),
        gamma_test_hi=float(max(gammas)),
        onset_k_lo=float(k_lo), onset_k_hi=float(k_hi),
        cliff_interval_baseline=a["onset"],
        cliff_interval_compensated=g["onset"],
        cliff_interval_identical=(a["onset"] == g["onset"]),
        jump_4_to_425_baseline=a["jump"],
        jump_4_to_425_compensated=g["jump"],
        ratio_baseline=a["ratio"], ratio_compensated=g["ratio"],
        plateau_ratio=float(a["plateau"] / g["plateau"]),
        in_band_ratio=float(a["in_band"] / g["in_band"]))
    C.save_json(out, os.path.join(C.RES, "c1_verdict.json"))
    v = out["verdict"]
    print("attained norm: baseline %.2f..%.2f (spread %.2fx)  "
          "compensated %.4f const (max dev %.1e)  level ratio %.2fx"
          % (v["attained_baseline_lo"], v["attained_baseline_hi"],
             v["attained_baseline_spread"], v["attained_compensated"],
             v["attained_compensated_max_dev"], v["attained_level_ratio"]))
    print("gamma(f) on the test grid: %.4f .. %.4f"
          % (v["gamma_test_lo"], v["gamma_test_hi"]))
    print("onset rule window k in (%.2f, %.2f]: both arms read %s"
          % (v["onset_k_lo"], v["onset_k_hi"], v["cliff_interval_baseline"]))
    print("cliff onset interval: baseline %s  compensated %s  identical=%s"
          % (v["cliff_interval_baseline"], v["cliff_interval_compensated"],
             v["cliff_interval_identical"]))
    print("jump e(4.25)/e(4):   baseline %.2fx  compensated %.2fx"
          % (v["jump_4_to_425_baseline"], v["jump_4_to_425_compensated"]))
    print("ratio e(5)/e(4):     baseline %.2f  compensated %.2f"
          % (v["ratio_baseline"], v["ratio_compensated"]))
    print("plateau ratio:        %.2f   in-band ratio: %.2f"
          % (v["plateau_ratio"], v["in_band_ratio"]))
    print("wrote results/c1_verdict.json")


if __name__ == "__main__":
    main()
