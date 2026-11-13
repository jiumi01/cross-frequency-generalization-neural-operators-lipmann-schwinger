# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Correlation-length robustness verdict (pre-registered, see
reviews/correlation_length_arm_plan_2026-11-12.md).

For each family arm (main_fno_a025_cl06 / _cl24, three seeds) at v_max = 0.25,
against the paired trivial baseline of the same family:

  r1      = epsilon(5) / epsilon(4)            (band-edge cliff, seed mean of
                                                per-seed node means)
  in      = mean error over the in-band nodes  (f <= 4)
  plateau = mean error over f >= 6
  ratio   = plateau / (paired trivial baseline over f >= 6)

Pre-registered reading:
  6 <= r1 <= 10 and ratio >= 0.9  ->  "shape robust across correlation length"
  otherwise                       ->  "conditional" (the scope sentence in the
                                       abstract/conclusion must narrow)

Aggregation follows the round-22 convention (R1): every statistic is a
per-seed node mean first, then a mean over seeds; the baseline is a single
deterministic draw (seed base 60000), so it aggregates directly.
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import common as C  # noqa: E402  (reads LS_MM_REF; this analyzer only uses F_TEST)

FAMILIES = {"0.12": None, "cl06": "0.06", "cl24": "0.24"}
ARM_FMT = "main_fno_a025_%s_s%d"
SEEDS = (0, 1, 2)
VMAX_KEY = "0.25"
F_IN = 4.0
F_EDGE, F_ONE = 4.0, 5.0
F_PLAT = 6.0


def curve(seed_json):
    d = json.load(open(seed_json, encoding="utf-8"))
    assert d["train_data"].rstrip("/").endswith(
        os.path.basename(d["train_data"])), "train_data unreadable"
    rows = d["sweep"][VMAX_KEY]
    by_f = {round(r["f"], 4): r["mean"] for r in rows}
    return by_f, d


def node_mean(by_f, lo=None, hi=None):
    vals = [m for f, m in by_f.items()
            if (lo is None or f >= lo - 1e-9) and (hi is None or f <= hi + 1e-9)]
    return sum(vals) / len(vals)


def family_stats(fam):
    arms = []
    if fam == "0.12":
        pat = os.path.join(C.RES, "eval_main_fno_a025_s*.json")
        files = sorted(glob.glob(pat))
        expect = "train_a025"
    else:
        files = [os.path.join(C.RES, "eval_%s.json" % ARM_FMT % (fam, s))
                 for s in SEEDS]
        expect = "train_a025_%s" % fam
    r1s, ins, plats = [], [], []
    for f in files:
        by_f, d = curve(f)
        assert expect in d["train_data"], (f, d["train_data"], expect)
        e4, e5 = by_f[F_EDGE], by_f[F_ONE]
        r1s.append(e5 / e4)
        ins.append(node_mean(by_f, hi=F_IN))
        plats.append(node_mean(by_f, lo=F_PLAT))
    mean = lambda v: sum(v) / len(v)  # noqa: E731
    return dict(n=len(files), r1=mean(r1s), inband=mean(ins),
                plateau=mean(plats), r1s=r1s)


def baseline_plateau(fam):
    name = "triv_paired.json" if fam == "0.12" else "triv_paired_%s.json" % fam
    d = json.load(open(os.path.join(C.RES, name), encoding="utf-8"))
    arm = d["data"][VMAX_KEY]
    vals = []
    for k, v in arm.items():
        if not k.startswith("f="):
            continue
        f = float(k[2:])
        if f >= F_PLAT - 1e-9:
            vals.extend(v)
    return sum(vals) / len(vals)


def main():
    out = {}
    rows = []
    for fam, mm in FAMILIES.items():
        st = family_stats(fam)
        base = baseline_plateau(fam)
        ratio = st["plateau"] / base
        verdict = ("robust" if (6.0 <= st["r1"] <= 10.0 and ratio >= 0.9)
                   else "conditional")
        out[fam] = dict(mm=mm, **st, baseline_plateau=base, ratio=ratio,
                        verdict=verdict)
        rows.append((fam, mm, st["n"], st["inband"], st["r1"], st["plateau"],
                     base, ratio, verdict))
    print("%-6s %-5s %2s %7s %6s %7s %7s %6s  %s" %
          ("fam", "mm", "n", "in", "r1", "plat", "base", "ratio", "verdict"))
    for r in rows:
        print("%-6s %-5s %2d %.4f %6.2f %.4f %.4f %.3f  %s" % r)
    ok = all(o["verdict"] == "robust" for k, o in out.items() if k != "0.12")
    out["_overall"] = ("robust" if ok else "conditional")
    C.save_json(out, os.path.join(C.RES, "cl_verdict.json"))
    print("overall:", out["_overall"], "-> results/cl_verdict.json")


if __name__ == "__main__":
    main()
