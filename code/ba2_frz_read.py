# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Read the BA2-FRZ chain against its pre-registered gates.

Inputs (all produced under the SAME formal protocol -- n=200, the f_test
grid, per-contrast seeds):

  results/eval_baf_fno_<tag>_m<m>_s<seed>.json   frozen-coefficient arms
  results/eval_ba_fno_<tag>_m<m>_s<seed>.json    fine-tuned-coefficient arms
  results/eval_main_fno_<tag>_s<seed>.json       trained operator (baseline)
  results/ls_head_alone.json                     LS head alone, same protocol

Statistics: per seed, the in-band mean (f <= 4) and the plateau mean (f >= 5)
at the arm's OWN training contrast; across the three seeds, a paired t test
(the seeds are shared between arms, so the paired difference is the licensed
reading -- the same protocol point the manuscript states in Section 5.4).

Pre-registered verdicts, per contrast:

  PASS    in-band <= trained FNO  AND  plateau <= 0.5 x trained FNO
  PARTIAL plateau <= trained FNO  but in-band worse
  FAIL    plateau > 1.5 x LS head alone

    python code/ba2_frz_read.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from p0_checks import paired

ARMS = {"a050": dict(tag="a050", m=8, vmax=0.50, cache="train_a050"),
        "a100": dict(tag="a100", m=7, vmax=1.00, cache="train_a100")}
SEEDS = (0, 1, 2)


def load(run):
    p = os.path.join(C.RES, "eval_%s.json" % run)
    if not os.path.exists(p):
        return None
    return C.load_json(p)


def band_means(d, vmax):
    """(in_band, plateau) means at one contrast from a run's archive."""
    rows = d["sweep"]["%.2f" % vmax]
    ib = np.mean([r["mean"] for r in rows if r["f"] <= 4.0])
    pl = np.mean([r["mean"] for r in rows if r["f"] >= 5.0])
    return float(ib), float(pl)


def collect(prefix, arm):
    got = []
    for s in SEEDS:
        if prefix == "main":
            run = "main_fno_%s_s%d" % (arm["tag"], s)
        else:
            run = "%s_fno_%s_m%d_s%d" % (prefix, arm["tag"], arm["m"], s)
        d = load(run)
        if d is None:
            return None
        got.append(band_means(d, arm["vmax"]))
    ib = [g[0] for g in got]
    pl = [g[1] for g in got]
    return dict(runs=["%s_s%d" % (prefix, s) for s in SEEDS],
                in_band=ib, plateau=pl)


def main():
    ls = C.load_json(os.path.join(C.RES, "ls_head_alone.json"))
    out = dict(arms={}, verdicts={})
    for tag, arm in ARMS.items():
        cols = {p: collect(p, arm) for p in ("main", "ba", "baf")}
        missing = [p for p, c in cols.items() if c is None]
        if missing:
            print("[skip] %s: no archives for %s" % (tag, missing))
            continue
        ls_rows = ls["arms"][tag]["sweep"]["%.2f" % arm["vmax"]]
        ls_plat = float(np.mean([r["mean"] for r in ls_rows if r["f"] >= 5.0]))
        ls_ib = float(np.mean([r["mean"] for r in ls_rows if r["f"] <= 4.0]))

        rec = dict(seeds=len(SEEDS), ls_head=dict(in_band=ls_ib,
                                                  plateau=ls_plat))
        for p, c in cols.items():
            rec[p] = dict(in_band=float(np.mean(c["in_band"])),
                          plateau=float(np.mean(c["plateau"])),
                          per_seed=c)
        # paired tests, frozen against each other arm
        for p in ("main", "ba"):
            rec["paired_baf_vs_%s" % p] = dict(
                in_band=paired(cols[p]["in_band"], cols["baf"]["in_band"],
                               "baf-%s in_band @%s" % (p, tag)),
                plateau=paired(cols[p]["plateau"], cols["baf"]["plateau"],
                               "baf-%s plateau @%s" % (p, tag)))

        baf_pl = np.mean(cols["baf"]["plateau"])
        baf_ib = np.mean(cols["baf"]["in_band"])
        main_pl = np.mean(cols["main"]["plateau"])
        main_ib = np.mean(cols["main"]["in_band"])
        if baf_ib <= main_ib and baf_pl <= 0.5 * main_pl:
            verdict = "PASS"
        elif baf_pl <= main_pl:
            verdict = "PARTIAL"
        elif baf_pl > 1.5 * ls_plat:
            verdict = "FAIL"
        else:
            verdict = "MIXED"   # between the gates; report, do not round
        rec["verdict"] = verdict
        rec["plateau_over_ls"] = float(baf_pl / ls_plat)
        out["arms"][tag] = rec
        out["verdicts"][tag] = verdict

        print("\n== v=%.2f ==" % arm["vmax"])
        print("  %-8s %-12s %-12s" % ("arm", "in_band", "plateau"))
        for p, name in (("ls_head", "LS head alone"), ("main", "trained FNO"),
                        ("ba", "BA2 (fine-tuned)"), ("baf", "BA2-FRZ")):
            r = rec[p]
            print("  %-8s %-12.4g %-12.4g" % (name, r["in_band"], r["plateau"]))
        for p in ("main", "ba"):
            t = rec["paired_baf_vs_%s" % p]
            print("  paired baf-%s  in_band: p=%.3f CI[%.3g,%.3g] | "
                  "plateau: p=%.3f CI[%.3g,%.3g]"
                  % (p, t["in_band"]["p"], t["in_band"]["ci"][0],
                     t["in_band"]["ci"][1], t["plateau"]["p"],
                     t["plateau"]["ci"][0], t["plateau"]["ci"][1]))
        print("  VERDICT: %s  (plateau / LS head alone = %.2f)"
              % (verdict, rec["plateau_over_ls"]))

    C.save_json(out, os.path.join(C.RES, "ba2_frz_read.json"))
    print("\nwrote", os.path.join(C.RES, "ba2_frz_read.json"))


if __name__ == "__main__":
    main()
