# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""C4 verdict: does the TL-FNO protocol transfer across frequency here?

Reads the evaluation JSONs of the transfer arm (tl_fno_a025: pre-trained at
f = 2.5 on media paired with the baseline, fine-tuned on the baseline's
wide-band cache and prefix) and of the baseline arm, and writes
results/c4_verdict.json.  The judgment the review asked for:

  * TL beats the baseline out of band (f >= 5) -> transfer buys frequency
    generalisation on this benchmark and the manuscript's "only a wider
    band helps" must be qualified (and a compute-matched control follows);
  * TL matches the baseline in band and does not move the cliff -> the two
    protocols answer different questions -- [21] is supervised adaptation
    with target-frequency data, this manuscript is zero-shot
    extrapolation -- and the apparent conflict dissolves.

The seed-mean curves use the same arm-matched onset rule as C1-C3.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

F_IN = (1.5, 2.5, 3.5)
F_PLAT = tuple(f for f in C.F_TEST if f >= 6.0)
ARMS = {"baseline": "main_fno_a025",     # from scratch, wide band
        "tl": "tl_fno_a025",             # TL fine-tune, lr 1e-4
        "tl_warm": "tlw_fno_a025"}       # TL warm start, lr 1e-3 (favourable)


def curve(prefix):
    acc = {round(float(f), 2): [] for f in C.F_TEST}
    for s in (0, 1, 2):
        p = os.path.join(C.RES, "eval_%s_s%d.json" % (prefix, s))
        if not os.path.exists(p):
            raise SystemExit("missing %s -- run the arm and its evaluation" % p)
        d = C.load_json(p)
        if float(d.get("gamma", 0)) != C.GAMMA:
            raise SystemExit("%s: gamma protocol mismatch" % p)
        if d.get("train_data") and "train_f25" in str(d["train_data"]):
            raise SystemExit("%s: evaluated the PRE-TRAIN checkpoint by mistake" % p)
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
            onset=onset_interval(f, m, inb), seeds=3)
    a = out["arms"]["baseline"]
    k_lo = max(x["e4"] / x["in_band"] for x in out["arms"].values())
    k_hi = min(x["e425"] / x["in_band"] for x in out["arms"].values())
    out["verdict"] = dict(
        onset_k_lo=float(k_lo), onset_k_hi=float(k_hi),
        cliff_interval_baseline=a["onset"], cliff_interval_tl=out["arms"]["tl"]["onset"],
        cliff_interval_tl_warm=out["arms"]["tl_warm"]["onset"],
        cliff_interval_identical=all(
            out["arms"][k]["onset"] == a["onset"] for k in out["arms"]),
        in_band_ratio_tl=float(out["arms"]["tl"]["in_band"] / a["in_band"]),
        in_band_ratio_tl_warm=float(out["arms"]["tl_warm"]["in_band"] / a["in_band"]),
        e4_ratio_tl=float(out["arms"]["tl"]["e4"] / a["e4"]),
        e5_ratio_tl=float(out["arms"]["tl"]["e5"] / a["e5"]),
        e5_ratio_tl_warm=float(out["arms"]["tl_warm"]["e5"] / a["e5"]),
        plateau_ratio_tl=float(out["arms"]["tl"]["plateau"] / a["plateau"]),
        plateau_ratio_tl_warm=float(out["arms"]["tl_warm"]["plateau"] / a["plateau"]),
        jump_baseline=a["jump"],
        jump_tl=out["arms"]["tl"]["jump"],
        jump_tl_warm=out["arms"]["tl_warm"]["jump"],
        ratio_baseline=a["ratio"], ratio_tl=out["arms"]["tl"]["ratio"],
        ratio_tl_warm=out["arms"]["tl_warm"]["ratio"])
    C.save_json(out, os.path.join(C.RES, "c4_verdict.json"))
    v = out["verdict"]
    for lab, arm in out["arms"].items():
        print("%-9s in-band %.4f  e(4) %.4f  e(5) %.4f  plateau %.4f  "
              "jump %.2fx  onset %s"
              % (lab, arm["in_band"], arm["e4"], arm["e5"], arm["plateau"],
                 arm["jump"], arm["onset"]))
    print("TL / baseline ratios: in-band %.2fx (warm %.2fx)  e(4) %.2fx  "
          "e(5) %.2fx (warm %.2fx)  plateau %.2fx (warm %.2fx)"
          % (v["in_band_ratio_tl"], v["in_band_ratio_tl_warm"], v["e4_ratio_tl"],
             v["e5_ratio_tl"], v["e5_ratio_tl_warm"], v["plateau_ratio_tl"],
             v["plateau_ratio_tl_warm"]))
    print("onset rule window k in (%.2f, %.2f]; identical=%s"
          % (v["onset_k_lo"], v["onset_k_hi"], v["cliff_interval_identical"]))
    print("wrote results/c4_verdict.json")


if __name__ == "__main__":
    main()
