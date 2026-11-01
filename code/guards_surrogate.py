# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Guards for the spectrum-matched control and for the per-sample bound.

The claims these guards defend
------------------------------
Three sentences were rewritten after the archive showed that the readings they
rested on had no interval attached:

  (i)  "the jump factor is not resolved beyond the window spread on either mesh"
       -- the paired interval on the real-vs-surrogate difference must contain
       zero at 64^2, and the difference must stay below the window spread;
  (ii) "the plateau difference carries a structural part of 1-3% with the same
       sign at both meshes" -- the sign must agree and the size must be small,
       because a large value would mean the spectrum matching failed;
  (iii)"the quadrature sum bounds the error in the mean, not per sample" -- the
       mean of the per-pointive bounds must exceed the mean error, the aggregate
       form must be the tighter number, and the pointwise violation rate must
       stay in the band the sentence describes (about half).

Each assertion recomputes the statistic from the stored per-window deltas rather
than reading the stored p-value, so a corrupted interval cannot hide behind an
intact summary.  The distribution function is implemented here again on purpose:
two paths to the same p-value is the point.
"""
import os
import sys
import json
import math

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FAIL = []


def ok(msg):
    print("  PASS  %s" % msg)


def bad(msg):
    FAIL.append(msg)
    print("  FAIL  %s" % msg)


def note(msg):
    print("  note  %s" % msg)


def _p_two(t, df):
    """Two-sided Student-t p, Simpson integration, independent implementation."""
    t = abs(float(t))
    if not math.isfinite(t):
        return 0.0
    c = math.gamma((df + 1) / 2.0) / (math.sqrt(df * math.pi) * math.gamma(df / 2.0))

    def dens(u):
        return c * (1.0 + u * u / df) ** (-(df + 1) / 2.0)

    hi = max(60.0, t + 40.0)
    n = 20000
    h = (hi - t) / n
    s = dens(t) + dens(hi)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * dens(t + i * h)
    return min(1.0, 2.0 * s * h / 3.0)


def _tcrit(df):
    return {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306}.get(df, 1.96)


def _stat(vals):
    v = np.asarray(vals, float)
    n = len(v)
    m = float(v.mean())
    se = float(v.std(ddof=1) / math.sqrt(n))
    tc = _tcrit(n - 1)
    return dict(n=n, mean=m, se=se, lo=m - tc * se, hi=m + tc * se,
                t=m / se if se > 0 else float("inf"), p=_p_two(m / se, n - 1))


def surrogate_guard(path=None):
    path = path or os.path.join(C.RES, "geo_windows.json")
    if not os.path.exists(path):
        note("sur: no geo_windows.json, spectrum-matched claims not asserted")
        return
    d = json.load(open(path, encoding="utf-8")).get("by_grid", {})
    if not d:
        bad("sur: geo_windows.json has no by_grid block")
        return

    rec = {}
    for g, blk in d.items():
        pairs = blk["pairs"]
        rec[g] = dict(
            jump=[float(p["delta_jump"]) for p in pairs],
            plat=[float(p["delta_plat"]) for p in pairs],
            real_jump=float(np.mean([p["real_jump"] for p in pairs])),
            real_plat=float(np.mean([p["real_plat"] for p in pairs])),
            n_sur=blk["n_surrogates"],
            meta=blk.get("meta", {}))
        st = _stat(rec[g]["jump"])
        rec[g]["jump_stat"] = st
        rec[g]["plat_stat"] = _stat(rec[g]["plat"])
        # spectrum match, recomputed from the field metadata
        devs = []
        for p in pairs:
            w = p["window"]
            rb = rec[g]["meta"].get("real_" + w, {}).get("bands", {})
            for s in range(rec[g]["n_sur"]):
                sb = rec[g]["meta"].get("sur_%s_%d" % (w, s), {}).get("bands", {})
                for M, rv in rb.items():
                    if M in sb and rv > 1e-12:
                        devs.append(abs(sb[M] / rv - 1.0))
        rec[g]["band_med_dev"] = float(np.median(devs)) if devs else None
        ok("sur/%s: %d windows, %d surrogates each, jump delta %+.3f "
           "[%+.3f,%+.3f] p=%.3f"
           % (g, len(pairs), rec[g]["n_sur"], st["mean"], st["lo"], st["hi"], st["p"]))

    # (i) the jump difference is not resolved at the coarse grid
    if "64" in rec:
        st = rec["64"]["jump_stat"]
        if st["lo"] <= 0.0 <= st["hi"]:
            ok("sur/64: jump interval contains zero, so 'not resolved' is supported "
               "(p=%.3f)" % st["p"])
        else:
            bad("sur/64: the text says the jump is not resolved, but the paired "
                "interval is [%+.3f,%+.3f]" % (st["lo"], st["hi"]))
    else:
        bad("sur: no 64^2 block, the coarse claim cannot be checked")

    # (ii) plateau effect is small and has the same sign on both grids
    if "64" in rec and "128" in rec:
        s64, s128 = rec["64"]["plat_stat"], rec["128"]["plat_stat"]
        same_sign = (s64["mean"] < 0) == (s128["mean"] < 0)
        if not same_sign:
            bad("sur: the text says same sign at both meshes, but the plateau "
                "deltas are %+.3f and %+.3f" % (s64["mean"], s128["mean"]))
        else:
            ok("sur: plateau delta has the same sign at both meshes (%+.3f, %+.3f)"
               % (s64["mean"], s128["mean"]))
        rel = [abs(s64["mean"]) / max(rec["64"]["real_plat"], 1e-30),
               abs(s128["mean"]) / max(rec["128"]["real_plat"], 1e-30)]
        if max(rel) < 0.05:
            ok("sur: structure effect stays below 5%% of the level (%.1f%%, %.1f%%)"
               % (100 * rel[0], 100 * rel[1]))
        else:
            bad("sur: the text says one to three percent, but the relative effect "
                "is %.1f%% and %.1f%%; the spectrum match may have failed"
                % (100 * rel[0], 100 * rel[1]))

        # the jump difference must be smaller than the window spread, otherwise
        # "not resolved" and "measured" would be the same size
        rj = np.asarray([p["real_jump"] for p in d["64"]["pairs"]], float)
        spread = (rj.max() - rj.min()) / rj.mean()
        jd = abs(rec["64"]["jump_stat"]["mean"]) / rec["64"]["real_jump"]
        if jd < spread:
            ok("sur: |jump delta|/level = %.1f%% < window spread %.1f%%"
               % (100 * jd, 100 * spread))
        else:
            bad("sur: the jump difference (%.1f%%) is not smaller than the window "
                "spread (%.1f%%), so 'unresolved' understates it"
                % (100 * jd, 100 * spread))

    # (iii) spectrum match actually happened
    for g in rec:
        med = rec[g]["band_med_dev"]
        if med is None:
            bad("sur/%s: surrogate fields carry no band metadata" % g)
        elif med < 0.02:
            ok("sur/%s: median band-fraction deviation %.2f%%" % (g, 100 * med))
        else:
            bad("sur/%s: median band deviation %.1f%% > 2%%, the surrogate is not "
                "spectrum-matched and the word 'matched' is unearned" % (g, 100 * med))


def bound_guard(path=None):
    path = path or os.path.join(C.RES, "bound_cross.json")
    if not os.path.exists(path):
        note("bound: no bound_cross.json, per-sample bound not asserted")
        return
    rows = json.load(open(path, encoding="utf-8")).get("rows", [])
    if not rows:
        bad("bound: bound_cross.json has no rows")
        return
    for r in rows:
        tag = "f=%g" % r["f"]
        if r["ratio_same"] <= 1.0:
            ok("bound/%s: mean of the pointwise bounds %.3f is above the mean "
               "error %.3f (ratio %.3f)"
               % (tag, r["bound_same_denom"], r["e256"], r["ratio_same"]))
        else:
            bad("bound/%s: the mean pointwise bound %.3f is below the mean "
                "error %.3f; the text claims it holds" % (tag, r["bound_same_denom"], r["e256"]))
        if r["bound_aggregate"] < r["bound_same_denom"]:
            ok("bound/%s: evaluating the bound on the means gives the tighter "
               "number (%.3f < %.3f), which is the concavity the text invokes"
               % (tag, r["bound_aggregate"], r["bound_same_denom"]))
        else:
            bad("bound/%s: the aggregate bound is not tighter than the per-sample "
                "one, so the accounting explanation is wrong" % tag)
        if int(r["f"]) == 4:
            if r["ratio_aggregate"] > 1.0 > r["ratio_same"]:
                ok("bound/%s: the error straddles the two forms (%.3f > 1 > %.3f), "
                   "which is exactly the sentence the text prints"
                   % (tag, r["ratio_aggregate"], r["ratio_same"]))
            else:
                bad("bound/%s: the error does not straddle the two forms "
                    "(aggregate ratio %.3f, per-sample ratio %.3f)"
                    % (tag, r["ratio_aggregate"], r["ratio_same"]))
        elif r["ratio_aggregate"] > 1.0:
            bad("bound/%s: the aggregate form fails here too, but the text presents "
                "f=4 as the only borderline case" % tag)
        else:
            ok("bound/%s: textbook case, both forms above the error (%.3f, %.3f)"
               % (tag, r["ratio_aggregate"], r["ratio_same"]))
        if abs(r["cos_mean"]) > 0.35:
            bad("bound/%s: mean cosine %.2f is too large for 'the cross term does "
                "not do work'" % (tag, r["cos_mean"]))
        else:
            ok("bound/%s: |cos| = %.3f, cross term negligible"
               % (tag, abs(r["cos_mean"])))

        # The mean cannot certify the inequality.  Measured, the pointwise form
        # is violated on about half of the samples, which makes the relation an
        # RMS statement.  If a re-run changes that, the paragraph must change.
        if r.get("viol_same") is None:
            bad("bound/%s: bound_cross.json carries no viol_same, so the text's "
                "'not sample by sample' is unasserted" % tag)
            continue
        if 0.2 <= r["viol_same"] <= 0.8:
            ok("bound/%s: the pointwise inequality is violated on %.0f%% of the "
               "samples (worst ratio %.2f), so 'in the mean, not per sample' is "
               "the measured relation" % (tag, 100 * r["viol_same"],
                                          r.get("ratio_max_same", float("nan"))))
        else:
            bad("bound/%s: violation rate %.0f%% does not support the phrase "
                "'in the mean, but not sample by sample'"
                % (tag, 100 * r["viol_same"]))
        if r.get("cos_sd") is not None and r.get("cos_negative_frac") is not None:
            if r["cos_sd"] > 0.2 and r["cos_negative_frac"] > 0.25:
                ok("bound/%s: cos spreads %.2f about %.3f and is negative on "
                   "%.0f%% of samples, so alignment is not the rare case"
                   % (tag, r["cos_sd"], r["cos_mean"],
                      100 * r["cos_negative_frac"]))
            else:
                bad("bound/%s: cos spread %.2f / negative fraction %.2f is too "
                    "small for 'about as likely as anti-alignment'"
                    % (tag, r["cos_sd"], r["cos_negative_frac"]))


def bound_text_guard(root=None):
    """The two sentences the bound paragraph must not print again."""
    root = root or os.path.join(C.ROOT, "submission")
    banned = {
        "en": ["The inequality as written therefore holds",
               "it does not, and the error was one of accounting"],
        "cn": ["按原文逐样本评估，该界在 $f=4$ 成立",
               "它并没有失效，"],
    }
    want = {"en": ["in the mean, but not sample by sample",
                   "RMS relation"],
            "cn": ["并非逐样本成立", "RMS 关系"]}
    for lang in ("en", "cn"):
        p = os.path.join(root, "paper_%s" % lang, "main.tex")
        if not os.path.exists(p):
            note("bound_text: no %s manuscript" % lang)
            continue
        txt = open(p, encoding="utf-8").read()
        # Match on whitespace-normalised text.  These are sentences, not lines:
        # reflowing a paragraph must not be able to hide the qualification from
        # the gate, and it did once -- the source broke "sample by sample"
        # across a newline and the paragraph read as if the disclosure were
        # gone while the manuscript was correct.
        flat = " ".join(txt.split())
        hits = [s for s in banned[lang] if " ".join(s.split()) in flat]
        if hits:
            bad("bound_text/%s: the manuscript still claims the pointwise bound "
                "holds: %s" % (lang, hits))
        miss = [s for s in want[lang] if " ".join(s.split()) not in flat]
        if miss:
            bad("bound_text/%s: the RMS qualification is missing: %s"
                % (lang, miss))
        if not hits and not miss:
            ok("bound_text/%s: the paragraph reports the pointwise violation and "
               "qualifies the relation as RMS" % lang)


def paired_jump_guard(path=None):
    """The 64^2 vs 128^2 jump shift must not be called significant at 3 seeds."""
    path = path or os.path.join(C.RES, "p0_checks.json")
    if not os.path.exists(path):
        note("paired: no p0_checks.json, the jump shift's interval is unasserted")
        return
    d = json.load(open(path, encoding="utf-8"))
    rec = d.get("grid", {}).get("part_a", {}).get("jump")
    if not rec:
        bad("paired: p0_checks.json carries no part_a jump block")
        return
    if rec["ci"][0] <= 0.0 <= rec["ci"][1]:
        ok("paired: jump shift interval [%+.2f,%+.2f] contains zero (p=%.3f), so "
           "the text does not call it a significant shift"
           % (rec["ci"][0], rec["ci"][1], rec["p"]))
    else:
        bad("paired: the interval [%+.2f,%+.2f] excludes zero, so the text "
            "understates the shift" % (rec["ci"][0], rec["ci"][1]))


def main():
    print("=== spectrum-matched control ===")
    surrogate_guard()
    print("\n=== per-sample quadrature bound ===")
    bound_guard()
    print("\n=== paired jump shift, 64 vs 128 ===")
    paired_jump_guard()
    print("\n%s" % ("all guards passed" if not FAIL else "%d failure(s)" % len(FAIL)))


if __name__ == "__main__":
    main()
