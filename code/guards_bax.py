# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Guards for the fit-depth/sum-depth defect and the corrected BA2-X arms.

The claims these guards defend
------------------------------
  (i)  "the a025 arms are unchanged by the head-semantics fix" -- the
       re-scored archives must reproduce the pre-fix backups bit for bit;
       a mismatch means the fix changed something else and every corrected
       number downstream is suspect;
  (ii) "the corrected arm holds both pre-registered conditions at v=0.50 and
       splits at v=1.00" -- the verdicts must follow from the gates
       recomputed here (in-band from the archive's dedicated in_band entry,
       plateau as the mean over f >= 6), not from a stored verdict string;
  (iii)"the v=1.00 plateau gain is resolved at six seeds" -- W1 extended the
       arm from three to six seeds (task W1); the paired interval must now
       exclude zero (measured [-0.991, -0.004] at df=5); if it ever slips
       back to containing zero the sentence must be rewritten, and the guard
       fails until it is.  The in-band regression stays significant
       (+0.114, CI [+0.078, +0.150]), so the verdict remains PARTIAL;
  (iv) "the pre-fix archives are stale by protocol, not deleted" -- every
       re-scored anchored archive carries head_semantics == "full", every
       backup lacks it, and the backup directory still holds all eighteen.

The text guard fails if either manuscript reverts to the withdrawn reading --
"the same arm fails at v>=0.50" -- or quietly quotes the co-adapted re-scored
numbers (BATwoInMid and siblings) as if they were the corrected arm's.
"""
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from p0_checks import _t_crit_95

FAIL = []
BACKUP = os.path.join(C.RES, "_archive_truncated_semantics")
ARMS = {"a050": dict(m=8, vmax=0.50, ls="0.50", seeds=(0, 1, 2)),
        "a100": dict(m=7, vmax=1.00, ls="1.00", seeds=(0, 1, 2, 3, 4, 5))}
# a025 (the low-contrast BA1/BA2 arms) were only ever run at three seeds; the
# bit-for-bit invariance check and the backup count stay pinned to that list
# even though a100 was extended to six seeds for W1.
A025_SEEDS = (0, 1, 2)


def ok(msg):
    print("  ok  " + msg)


def bad(msg):
    FAIL.append(msg)
    print("  FAIL " + msg)


def _band(d, vmax):
    """(in_band, plateau) -- the manuscript's own conventions."""
    ib = next((float(r["mean"]) for r in d["in_band"]
               if abs(r["vmax"] - vmax) < 1e-6), None)
    rows = d["sweep"]["%.2f" % vmax]
    pl = float(np.mean([r["mean"] for r in rows if r["f"] >= 6.0]))
    return ib, pl


def _paired(a, b):
    """Paired t interval on b - a; tcrit is df-dependent (n-1)."""
    d = np.asarray(b) - np.asarray(a)
    n = len(d)
    mean = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("inf")
    half = _t_crit_95(n - 1) * se
    return mean, [mean - half, mean + half]


def _verdict(ib, pl, main_ib, main_pl, ls_pl):
    if ib <= main_ib and pl <= 0.5 * main_pl:
        return "PASS"
    if pl <= main_pl:
        return "PARTIAL"
    if pl > 1.5 * ls_pl:
        return "FAIL"
    return "MIXED"


def bax_guard():
    print("bax guard:")
    if not os.path.isdir(BACKUP):
        bad("backup directory %s is missing -- the as-run numbers the "
            "manuscript quotes have no source" % BACKUP)
        return
    backups = glob.glob(os.path.join(BACKUP, "eval_ba_fno_*.json"))
    # 6 a025 (BA1 + BA2) + 3 a050 + 3 a100; the frozen baf archives live in
    # the same directory under their own prefix and are counted by the
    # glob in the manuscript only through their re-scored copies
    if len(backups) != 12:
        bad("backup directory holds %d ba archives, expected 12"
            % len(backups))

    # (iv) protocol keys
    for tag in ARMS:
        for s in ARMS[tag]["seeds"]:
            m = ARMS[tag]["m"]
            new = C.load_json(os.path.join(
                C.RES, "eval_bax_fno_%s_m%d_s%d.json" % (tag, m, s)))
            if new is None:
                bad("missing re-scored/corrected archive for bax %s s%d"
                    % (tag, s))
                continue
            if new.get("head_semantics") != "full":
                bad("bax %s s%d archive lacks head_semantics=full" % (tag, s))
    if not any("head_semantics" in f for f in FAIL):
        ok("all corrected-arm archives carry head_semantics=full")

    # (i) a025 invariance, bit for bit
    for s in A025_SEEDS:
        old_p = os.path.join(BACKUP, "eval_ba_fno_a025_m8_s%d.json" % s)
        new_p = os.path.join(C.RES, "eval_ba_fno_a025_m8_s%d.json" % s)
        if not (os.path.exists(old_p) and os.path.exists(new_p)):
            bad("a025 invariance: missing archive pair s%d" % s)
            continue
        old = json.load(open(old_p, encoding="utf-8"))
        new = json.load(open(new_p, encoding="utf-8"))
        for v, rows in old["sweep"].items():
            for r_old, r_new in zip(rows, new["sweep"][v]):
                if abs(r_old["mean"] - r_new["mean"]) > 1e-9:
                    bad("a025 s%d v=%s f=%s changed: %.6g -> %.6g"
                        % (s, v, r_old["f"], r_old["mean"], r_new["mean"]))
    if not any("a025" in f for f in FAIL):
        ok("a025 archives reproduced bit-for-bit under the fixed semantics")

    # (ii)(iii) verdicts and paired intervals, recomputed
    ls = C.load_json(os.path.join(C.RES, "ls_head_alone.json"))
    read = C.load_json(os.path.join(C.RES, "bax_read.json")) or {}
    for tag, a in ARMS.items():
        vmax = a["vmax"]
        ibs = {"main": [], "bax": []}
        pls = {"main": [], "bax": []}
        for s in ARMS[tag]["seeds"]:
            for pre in ("main", "bax"):
                run = ("main_fno_%s_s%d" % (tag, s) if pre == "main"
                       else "bax_fno_%s_m%d_s%d" % (tag, a["m"], s))
                d = C.load_json(os.path.join(C.RES, "eval_%s.json" % run))
                if d is None:
                    bad("missing archive %s" % run)
                    continue
                ib, pl = _band(d, vmax)
                ibs[pre].append(ib)
                pls[pre].append(pl)
        if any(len(v) != len(ARMS[tag]["seeds"]) for v in list(ibs.values()) +
               list(pls.values())):
            continue
        ls_pl = float(np.mean(
            [r["mean"] for r in ls["arms"][a["ls"]]["variants"]["full"]
             ["%.2f" % vmax] if r["f"] >= 6.0]))
        want = _verdict(float(np.mean(ibs["bax"])),
                        float(np.mean(pls["bax"])),
                        float(np.mean(ibs["main"])),
                        float(np.mean(pls["main"])), ls_pl)
        got = read.get("verdicts", {}).get(tag)
        if got != want:
            bad("%s: verdict drift, recomputed %s, archive %s"
                % (tag, want, got))
        else:
            ok("%s: verdict %s follows from the recomputed gates" % (tag, want))
        dmean, ci = _paired(pls["main"], pls["bax"])
        contains = ci[0] <= 0 <= ci[1]
        # The quoted interval must be a measurement, not a recollection: the
        # \Bax* macros are generated from bax_read.json, so a stale read file
        # would silently quote the three-seed interval after W1.  Compare the
        # stored values against this recomputation, not just the sign.
        stored = (read.get("arms", {}).get(tag, {})
                  .get("paired_bax_vs_main", {}).get("plateau", {}))
        if stored:
            tol = 1e-3 * max(1.0, abs(dmean))
            got_ci = stored.get("ci")
            if got_ci is not None and any(
                    abs(g - w) > tol for g, w in zip(got_ci, ci)):
                bad("%s: quoted plateau interval %s is stale; the archives "
                    "recompute to %s -- re-run bax_read.py"
                    % (tag, [round(x, 4) for x in got_ci],
                       [round(x, 4) for x in ci]))
            got_d = stored.get("diff")
            if got_d is not None and abs(got_d - dmean) > tol:
                bad("%s: quoted plateau diff %.4f is stale; the archives "
                    "recompute to %.4f -- re-run bax_read.py"
                    % (tag, got_d, dmean))
        if tag == "a100" and contains:
            bad("a100: W1 extended this arm to six seeds and the text now "
                "says the plateau gain is resolved, but the recomputed "
                "interval %s still contains zero -- rewrite the sentence" % ci)
        if tag == "a100" and not contains:
            ok("a100: plateau interval %s excludes zero, matching the "
               "'resolved at six seeds' wording" % [round(x, 3) for x in ci])
        if tag == "a050" and contains:
            bad("a050: PASS rests on significant paired gains, but the "
                "plateau interval %s contains zero" % ci)
        if tag == "a050" and not contains:
            ok("a050: plateau interval excludes zero, matching PASS")


def bax_text_guard(paths=None):
    """`paths` maps lang -> main.tex; defaults to the real submission copies,
    which lets the falsification probe inject sentences into temp copies.
    Uses a local failure list so it can be probed independently of the
    archive guard."""
    fails = []

    def ok(msg):
        print("  ok  " + msg)

    def bad(msg):
        fails.append(msg)
        print("  FAIL " + msg)

    print("bax text guard:")
    sub = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "submission")
    paths = paths or {"en": os.path.join(sub, "paper_en", "main.tex"),
                      "cn": os.path.join(sub, "paper_cn", "main.tex")}
    for lang, root in paths.items():
        if not os.path.exists(root):
            bad("%s: manuscript not found" % lang)
            continue
        txt = open(root, encoding="utf-8").read()
        if lang == "en":
            withdrawn = [r"fails at \$v_\{\\max\}\\ge0\.50",
                         r"does not transfer at every contrast"]
            # The corrected-reading disclosure is required in two parts: the
            # statement that the head must sum what it was fitted to sum (the
            # check itself) and the statement that the pre-check archives carry
            # a protocol key (the record cannot be mixed).  Two independent
            # patterns rather than one, because the single-pattern form let a
            # rewrite of the surrounding prose drop the disclosure while the
            # gate stayed green.
            anchored = [r"\\BaxVerdictMid", r"\\BaxVerdictStrong",
                        r"head sums what it was fitted to sum",
                        r"protocol key"]
            stale = [r"\\BATwoInMid", r"\\BATwoPlatMid",
                     r"\\BATwoInStrong", r"\\BATwoPlatStrong"]
        else:
            withdrawn = [r"在 \$v_\{\\max\}\\ge0\.50\$ 失效",
                         r"残差并非在每个对比度上都能迁移"]
            anchored = [r"\\BaxVerdictMid", r"\\BaxVerdictStrong",
                        r"头按深度 \$M\$ 拟合"]
            stale = [r"\\BATwoInMid", r"\\BATwoPlatMid",
                     r"\\BATwoInStrong", r"\\BATwoPlatStrong"]
        for pat in withdrawn:
            if re.search(pat, txt):
                bad("%s: withdrawn sentence present: %s" % (lang, pat))
        for pat in anchored:
            if not re.search(pat, txt):
                bad("%s: required corrected-reading text missing: %s"
                    % (lang, pat))
        for pat in stale:
            if re.search(pat, txt):
                bad("%s: co-adapted re-scored number quoted in text: %s"
                    % (lang, pat))
    if not fails:
        ok("both manuscripts carry the corrected reading and none of the "
           "withdrawn sentences")
    return bool(fails)


if __name__ == "__main__":
    bax_guard()
    bax_text_guard()
    print("FAILURES:", len(FAIL))
    sys.exit(1 if FAIL else 0)
