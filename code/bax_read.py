# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Read the BA2-X chain against its pre-registered gates.

Archive roles after the fit-depth/sum-depth fix:

  eval_bax_fno_*   corrected arm (fine-tuned coefficients, full head) -- THE
                   result this script reads out;
  eval_ba_fno_*    the archived fine-tuned arm, RE-SCORED under full head
                   semantics (its residual co-adapted to the truncated head,
                   so the pairing is broken; read as a diagnostic);
  eval_baf_fno_*   the frozen-on-truncated-head arm, re-scored (diagnostic);
  eval_main_fno_*  trained operator, unchanged (anchor=none, unaffected);
  ls_head_alone.json  heads alone, "full" variant supplies the FAIL gate;
  _archive_truncated_semantics/  backups of the pre-fix anchored archives.

Invariant check first: the a025 arms had channels=None, so nothing was
truncated and the re-scored archives must reproduce the backups to the last
bit.  A mismatch means the "fix" changed something else and every number
downstream is suspect.

Gates for bax (paired across the three shared seeds):

  PASS    in-band <= trained FNO AND plateau <= 0.5 x trained FNO
  PARTIAL plateau <= trained FNO but in-band worse
  FAIL    plateau > 1.5 x full LS head alone (0.578 @ v=0.50, 1.354 @ v=1.00)

    python code/bax_read.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from p0_checks import paired

ARMS = {"a050": dict(m=8, vmax=0.50, seeds=(0, 1, 2)),
        "a100": dict(m=7, vmax=1.00, seeds=(0, 1, 2, 3, 4, 5))}
# The ba/baf diagnostics are the pre-fix (buggy) arms re-scored under the fixed
# head; they were only ever run at three seeds, so they are collected on their
# own fixed seed list and the bax-vs-ba pairing uses the seeds common to both.
DIAG_SEEDS = (0, 1, 2)
BACKUP = os.path.join(C.RES, "_archive_truncated_semantics")


def load(run):
    p = os.path.join(C.RES, "eval_%s.json" % run)
    return C.load_json(p) if os.path.exists(p) else None


def band_means(d, vmax):
    """(in_band, plateau) under the conventions the manuscript's macros use:
    in-band from the archive's dedicated in_band entry (its own 1.0-4.0
    draw, seed 70000+vi), plateau as the mean over f >= 6 (the paper's
    plateau never quotes f = 5, the worst point, on its own)."""
    ib = None
    for r in d["in_band"]:
        if abs(r["vmax"] - vmax) < 1e-6:
            ib = float(r["mean"])
            break
    rows = d["sweep"]["%.2f" % vmax]
    pl = float(np.mean([r["mean"] for r in rows if r["f"] >= 6.0]))
    return ib, pl


def collect(prefix, tag, m, vmax, seeds):
    ib, pl, runs = [], [], []
    for s in seeds:
        run = ("main_fno_%s_s%d" % (tag, s) if prefix == "main"
               else "%s_fno_%s_m%d_s%d" % (prefix, tag, m, s))
        d = load(run)
        if d is None:
            return None
        a, b = band_means(d, vmax)
        ib.append(a)
        pl.append(b)
        runs.append(run)
    return dict(runs=runs, in_band=ib, plateau=pl)


def invariant_a025():
    """channels=None arms must reproduce the truncated-semantics archives."""
    bad = []
    for p in sorted(glob.glob(os.path.join(BACKUP, "eval_ba_fno_a025*.json"))):
        old = json.load(open(p, encoding="utf-8"))
        new = load(os.path.splitext(os.path.basename(p))[0][5:])
        if new is None:
            bad.append((os.path.basename(p), "missing re-scored archive"))
            continue
        for key in ("f_test", "n", "gamma"):
            if old.get(key) != new.get(key):
                bad.append((os.path.basename(p), "protocol field %s" % key))
        for v, rows in old["sweep"].items():
            for r_old, r_new in zip(rows, new["sweep"][v]):
                if abs(r_old["mean"] - r_new["mean"]) > 1e-9:
                    bad.append((os.path.basename(p),
                                "v=%s f=%s: %.6g -> %.6g"
                                % (v, r_old["f"], r_old["mean"],
                                   r_new["mean"])))
    return bad


def main():
    inv = invariant_a025()
    if inv:
        print("INVARIANT FAIL (a025 must be unchanged by the fix):")
        for name, why in inv:
            print("  %s: %s" % (name, why))
    else:
        print("invariant OK: every a025 archive reproduced bit-for-bit "
              "under the fixed semantics (nothing was truncated there)")

    ls = C.load_json(os.path.join(C.RES, "ls_head_alone.json"))
    out = dict(invariant_a025_ok=not inv, arms={}, verdicts={})
    for tag, arm in ARMS.items():
        # main and bax are the arms the verdict rests on; they share the arm's
        # seed list (a100 was extended to six seeds for W1).  ba/baf are
        # diagnostics that only ever had three.  Collection order main, ba,
        # baf, bax is preserved so the emitted JSON is byte-comparable.
        cols = {}
        for p in ("main", "ba", "baf", "bax"):
            seeds = arm["seeds"] if p in ("main", "bax") else DIAG_SEEDS
            cols[p] = collect(p, tag, arm["m"], arm["vmax"], seeds)
        missing = [p for p, c in cols.items() if c is None]
        if missing:
            print("[skip] %s: no archives for %s" % (tag, missing))
            continue
        ls_key = "0.50" if tag == "a050" else "1.00"
        # same conventions as band_means: f >= 6 plateau, f <= 4 in band
        full_plat = float(np.mean(
            [r["mean"] for r in ls["arms"][ls_key]["variants"]["full"]
             ["%.2f" % arm["vmax"]] if r["f"] >= 6.0]))
        full_ib = float(np.mean(
            [r["mean"] for r in ls["arms"][ls_key]["variants"]["full"]
             ["%.2f" % arm["vmax"]] if r["f"] <= 4.0]))
        rec = dict(ls_head_full=dict(in_band=full_ib, plateau=full_plat))
        for p, c in cols.items():
            rec[p] = dict(in_band=float(np.mean(c["in_band"])),
                          plateau=float(np.mean(c["plateau"])),
                          per_seed=c)
        # paired bax vs trained FNO on the arm's full (shared) seed list
        rec["paired_bax_vs_main"] = dict(
            in_band=paired(cols["main"]["in_band"], cols["bax"]["in_band"],
                           "bax-main in @%s" % tag),
            plateau=paired(cols["main"]["plateau"], cols["bax"]["plateau"],
                           "bax-main plat @%s" % tag))
        # paired bax vs the buggy re-scored arm, on the seeds common to both
        ncom = min(len(cols["ba"]["in_band"]), len(cols["bax"]["in_band"]))
        rec["paired_bax_vs_ba"] = dict(
            in_band=paired(cols["ba"]["in_band"][:ncom],
                           cols["bax"]["in_band"][:ncom],
                           "bax-ba in @%s" % tag),
            plateau=paired(cols["ba"]["plateau"][:ncom],
                           cols["bax"]["plateau"][:ncom],
                           "bax-ba plat @%s" % tag))
        bax_ib = rec["bax"]["in_band"]
        bax_pl = rec["bax"]["plateau"]
        main_ib = rec["main"]["in_band"]
        main_pl = rec["main"]["plateau"]
        if bax_ib <= main_ib and bax_pl <= 0.5 * main_pl:
            verdict = "PASS"
        elif bax_pl <= main_pl:
            verdict = "PARTIAL"
        elif bax_pl > 1.5 * full_plat:
            verdict = "FAIL"
        else:
            verdict = "MIXED"
        rec["verdict"] = verdict
        rec["plateau_over_main"] = float(bax_pl / main_pl)
        rec["plateau_over_ls_full"] = float(bax_pl / full_plat)
        out["arms"][tag] = rec
        out["verdicts"][tag] = verdict

        print("\n== v=%.2f ==" % arm["vmax"])
        print("  %-22s %-10s %-10s" % ("arm", "in_band", "plateau"))
        for p, name in (("ls_head_full", "LS head alone (full)"),
                        ("main", "trained FNO"),
                        ("ba", "BA2 re-scored"),
                        ("baf", "BA2-FRZ re-scored"),
                        ("bax", "BA2-X (fixed)")):
            r = rec[p]
            print("  %-22s %-10.4g %-10.4g" % (name, r["in_band"],
                                               r["plateau"]))
        for p in ("main", "ba"):
            t = rec["paired_bax_vs_%s" % p]
            print("  paired bax-%s  in: p=%.3f CI[%.3g,%.3g] | "
                  "plat: p=%.3f CI[%.3g,%.3g]"
                  % (p, t["in_band"]["p"], t["in_band"]["ci"][0],
                     t["in_band"]["ci"][1], t["plateau"]["p"],
                     t["plateau"]["ci"][0], t["plateau"]["ci"][1]))
        print("  VERDICT: %s  (plateau x%.2f of trained FNO, x%.2f of the "
              "full LS head)" % (verdict, rec["plateau_over_main"],
                                 rec["plateau_over_ls_full"]))

    C.save_json(out, os.path.join(C.RES, "bax_read.json"))
    print("\nwrote", os.path.join(C.RES, "bax_read.json"))


if __name__ == "__main__":
    main()
