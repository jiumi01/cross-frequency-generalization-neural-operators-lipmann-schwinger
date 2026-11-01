# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C2 verdict: input-encoding artefact vs. frequency boundary.

Reads the six evaluation JSONs of the two paired arms (baseline main_fno_a025,
Fourier-feature ff_fno; same cache, same protocol, only the input encoding of f
differs -- scalar channel vs. scalar + 7 sin/cos pairs) and writes
results/c2_verdict.json.  The paper's C2 table, if the verdict warrants one,
is generated from that file.

The comparison the verdict rests on: the ff model sees a strict superset of
the baseline's frequency information (the scalar channel is retained), so if
the band-edge step were an artefact of the low-bandwidth scalar encoding, the
richer encoding should move or erase it.  The same arm-matched onset rule as
C1 is used: the error "leaves its in-band level" when it first exceeds k x
that arm's own in-band mean, reported as a grid interval together with the
window of rule constants k for which both arms read the same interval.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

F_IN = (1.5, 2.5, 3.5)                 # in-band reference frequencies
F_PLAT = tuple(f for f in C.F_TEST if f >= 6.0)
ARMS = {"baseline": "main_fno_a025", "fourier": "ff_fno"}


def curve(prefix):
    """seed-mean error curve at v_max = 0.25, plus its across-seed SD."""
    acc = {round(float(f), 2): [] for f in C.F_TEST}
    for s in (0, 1, 2):
        p = os.path.join(C.RES, "eval_%s_s%d.json" % (prefix, s))
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run the arm and its evaluation" % p)
        d = C.load_json(p)
        if d.get("f_emb", "none") != ("ff" if prefix.startswith("ff") else "none"):
            raise SystemExit("%s: unexpected f_emb=%s" % (p, d.get("f_emb")))
        for r in d["sweep"]["0.25"]:
            acc[round(float(r["f"]), 2)].append(float(r["mean"]))
    f = sorted(acc)
    return (f, [float(np.mean(acc[x])) for x in f],
            [float(np.std(acc[x])) for x in f])


def onset_interval(f, e, inb, k=3.0):
    """First grid interval in which the error leaves k x the in-band level."""
    thr = k * inb
    for i, x in enumerate(f):
        if e[i] > thr:
            lo = f[i - 1] if i > 0 else f[0]
            return [float(lo), float(x)]
    return None


def main():
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
            onset=onset_interval(f, m, inb),
            seeds=3)
    a, g = out["arms"]["baseline"], out["arms"]["fourier"]
    k_lo = max(a["e4"] / a["in_band"], g["e4"] / g["in_band"])
    k_hi = min(a["e425"] / a["in_band"], g["e425"] / g["in_band"])
    out["verdict"] = dict(
        onset_k_lo=float(k_lo), onset_k_hi=float(k_hi),
        cliff_interval_baseline=a["onset"],
        cliff_interval_fourier=g["onset"],
        cliff_interval_identical=(a["onset"] == g["onset"]),
        jump_4_to_425_baseline=a["jump"],
        jump_4_to_425_fourier=g["jump"],
        ratio_baseline=a["ratio"], ratio_fourier=g["ratio"],
        plateau_ratio=float(a["plateau"] / g["plateau"]),
        in_band_ratio=float(a["in_band"] / g["in_band"]),
        e5_ratio=float(a["e5"] / g["e5"]))
    C.save_json(out, os.path.join(C.RES, "c2_verdict.json"))
    v = out["verdict"]
    print("in-band mean:      baseline %.4f  fourier %.4f  (%.2fx)"
          % (a["in_band"], g["in_band"], v["in_band_ratio"]))
    print("eps(4) / eps(5):   baseline %.4f / %.4f   fourier %.4f / %.4f"
          % (a["e4"], a["e5"], g["e4"], g["e5"]))
    print("onset rule window k in (%.2f, %.2f]: both arms read %s"
          % (v["onset_k_lo"], v["onset_k_hi"], v["cliff_interval_baseline"]))
    print("cliff onset interval: baseline %s  fourier %s  identical=%s"
          % (v["cliff_interval_baseline"], v["cliff_interval_fourier"],
             v["cliff_interval_identical"]))
    print("jump e(4.25)/e(4):   baseline %.2fx  fourier %.2fx"
          % (v["jump_4_to_425_baseline"], v["jump_4_to_425_fourier"]))
    print("ratio e(5)/e(4):     baseline %.2f  fourier %.2f"
          % (v["ratio_baseline"], v["ratio_fourier"]))
    print("plateau ratio:        %.2f   eps(5) ratio: %.2f"
          % (v["plateau_ratio"], v["e5_ratio"]))
    print("wrote results/c2_verdict.json")


if __name__ == "__main__":
    main()
