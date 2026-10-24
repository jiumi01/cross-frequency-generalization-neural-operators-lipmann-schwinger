# Author: Longgang Wang
"""Gates for the three-grid study and for the audited-vs-delivered parity.

Appended to check_claims by import so the two concerns stay readable: one is
about numbers (does the refinement support what the prose says), the other is
about the instrument (is the file being audited the file being submitted).
"""
import os, sys, json, hashlib, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

RES = []
FAIL = []


def ok(m):
    RES.append(("ok", m))
    print("  [ok]   " + m)


def bad(m):
    RES.append(("bad", m))
    FAIL.append(m)
    print("  [FAIL] " + m)


def note(m):
    RES.append(("note", m))
    print("  [note] " + m)


def _md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def delivery_guard(root=None):
    """The audited body must be the delivered body.

    Every numeric gate in this file reads paper/<lang>/main.tex.  On 2026-10-22
    that file was four days behind submission/paper_<lang>/main.tex and two
    subsections short, so four gates reported green about a manuscript that was
    never going to be sent -- and the style gate then failed on the delivered
    text the moment the two were synchronised.  The failure is silent and the
    direction of the mistake is always the same, so it is asserted rather than
    left to a habit.
    """
    root = root or C.ROOT
    for lang in ("en", "cn"):
        a = os.path.join(root, "submission", "paper_%s" % lang, "main.tex")
        b = os.path.join(root, "paper", lang, "main.tex")
        if not (os.path.exists(a) and os.path.exists(b)):
            note("delivery %s: one tree has no main.tex, parity not asserted" % lang)
            continue
        if _md5(a) != _md5(b):
            bad("delivery %s: the audited body is NOT the delivered body "
                "(paper/%s/main.tex %d bytes, submission/paper_%s/main.tex %d "
                "bytes) -- run code/sync_delivery.py"
                % (lang, lang, os.path.getsize(b), lang, os.path.getsize(a)))
        else:
            ok("delivery %s: the audited body is the delivered body (%s)"
               % (lang, _md5(a)[:8]))

    _pdf_parity(root)


def _pdf_text(p):
    """Text layer of a PDF, or None.  Two PDFs with equal text are the same
    manuscript even if their byte streams differ (timestamps, xref order)."""
    try:
        import subprocess
        r = subprocess.run(["pdftotext", p, "-"], capture_output=True, timeout=180)
        if r.returncode == 0:
            return r.stdout
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _pdf_parity(root):
    """The delivered PDF must be the PDF that was just compiled.

    The body parity above closed the same class for main.tex and left the PDF
    open: make_paper writes paper/<lang>/main.pdf, submission/ is a copy, and
    audit_paper reads submission/.  A stale copy there is audited as if it were
    the new build, which is exactly how the 2026-10-22 escape residue survived
    one full "verified" pass.  Page count and text hash, not bytes.
    """
    for lang in ("en", "cn"):
        a = os.path.join(root, "submission", "paper_%s" % lang, "main.pdf")
        b = os.path.join(root, "paper", lang, "main.pdf")
        if not (os.path.exists(a) and os.path.exists(b)):
            continue
        ta, tb = _pdf_text(a), _pdf_text(b)
        if ta is None or tb is None:
            note("delivery %s: pdftotext unavailable, PDF parity not asserted" % lang)
            continue
        if hashlib.md5(ta).hexdigest() != hashlib.md5(tb).hexdigest():
            bad("delivery %s: the delivered PDF is not the compiled PDF "
                "(%d vs %d characters of text) -- rebuild and copy, do not "
                "audit the older copy"
                % (lang, len(ta), len(tb)))
        else:
            ok("delivery %s: the delivered PDF carries the compiled text (%d chars)"
               % (lang, len(ta)))


def grid_guard(res=None):
    """Three-grid convergence and the plateau's dependence on the target.

    What is asserted is what the prose asserts, including the places where the
    claim FAILS: the quadrature bound holds on the plateau and does not hold in
    band, and the plateau is quoted as moving by a few percent rather than as
    invariant.  A guard that only checked the flattering half would pass on a
    manuscript that overclaimed.
    """
    p = os.path.join(res or C.RES, "grid_conv_three.json")
    if not os.path.exists(p):
        note("grid: results/grid_conv_three.json absent; three-grid claims unchecked")
        return
    d = json.load(open(p, encoding="utf-8"))

    ps = [r["p"] for k in ("ref_0.25", "ref_1.00") for r in d[k]["rows"]]
    # First order, and visibly not second: the ratio of successive differences
    # is 2.0-2.2, so p must lie in [1.0, 1.2].  A p outside that band would
    # mean the reference is not converging at the rate the text reports.
    if min(ps) >= 1.0 and max(ps) <= 1.2:
        ok("grid: observed order is %.2f-%.2f across both contrasts and every "
           "frequency -- first-order, which is what the prose reports"
           % (min(ps), max(ps)))
    else:
        bad("grid: observed order spans %.2f-%.2f, outside the reported "
            "first-order band" % (min(ps), max(ps)))

    # the 64^2 target really is far from the 256^2 one at the top frequency,
    # otherwise the whole exercise would be pointless and saying so would be
    # misleading rather than reassuring
    top = [r for r in d["ref_0.25"]["rows"] if r["f"] == 12.0][0]
    if top["d64v256"] > 0.3:
        ok("grid: the 64^2 reference differs from the 256^2 one by %.3f at "
           "f=12 -- the refinement is a real correction, not a formality"
           % top["d64v256"])
    else:
        bad("grid: d64v256 = %.3f at f=12 is small enough that the refinement "
            "study is not testing anything" % top["d64v256"])

    for tag, key in (("0.25", "model_0.25"), ("1.00", "model_1.00")):
        rows = d[key]["rows"]
        plat = lambda f: float(np.mean([r[f] for r in rows if r["f"] >= 6.0]))
        shift = 100.0 * (plat("e256") / plat("e64") - 1.0)
        rshift = 100.0 * ((plat("e256") / plat("b256")) /
                          (plat("e64") / plat("b64")) - 1.0)
        if 0.0 < shift < 10.0:
            ok("grid v=%s: refining the target two levels moves the plateau by "
               "%+.1f%%, and the plateau-to-baseline ratio by %+.1f%%"
               % (tag, shift, rshift))
        else:
            bad("grid v=%s: the plateau moves by %+.1f%% under refinement, "
                "which the prose does not claim" % (tag, shift))
        if abs(rshift) < abs(shift):
            ok("grid v=%s: the ratio moves less than the plateau itself, which "
               "is why the interpretation survives the refinement" % tag)
        else:
            bad("grid v=%s: the ratio moves more than the plateau (%+.1f%% "
                "against %+.1f%%), so the stated reason is wrong"
                % (tag, rshift, shift))

    # the quadrature bound: holds on the plateau, fails in band.  Both halves
    # are asserted because the prose says both.
    rows_b = {r["f"]: r for r in d["model_0.25"]["rows"]}
    rows_a = {r["f"]: r for r in d["ref_0.25"]["rows"]}
    holds, fails = [], []
    for f in sorted(rows_b):
        q = float(np.hypot(rows_b[f]["e64"], rows_a[f]["d64v256"]))
        (holds if rows_b[f]["e256"] <= q + 1e-9 else fails).append(f)
    if fails == [4.0] and len(holds) == 5:
        ok("grid: the quadrature bound holds at f=%s and fails only at f=4, "
           "exactly as the text states" % ", ".join("%g" % f for f in holds))
    else:
        bad("grid: the quadrature bound holds at %s and fails at %s, which is "
            "not the split the text reports"
            % ([float(x) for x in holds], [float(x) for x in fails]))

    # the input-path control is quoted, so it has to be the size the text says
    nat = rows_b[12.0]["e_native"] / rows_b[12.0]["e64"] - 1.0
    if abs(nat) < 0.05:
        ok("grid: the input path (downsampled vs native 64^2 medium) changes "
           "the plateau by %+.1f%% at f=12, of the same order as the target "
           "refinement" % (100 * nat))
    else:
        bad("grid: the input path changes the plateau by %+.1f%%, more than the "
            "text's 'of the same size'" % (100 * nat))


def ci_guard(ev=None, res=None):
    """The resampling intervals must reproduce the table they qualify.

    ci_plateau.py recomputes the in-band error and the plateau from the
    per-sample archive -- a different code path from the one that produced
    Table 1 -- so agreement here is an independent recomputation, not a
    transcription check.  Agreement to the printed precision is required; a
    mismatch means one of the two paths has drifted.
    """
    p = os.path.join(res or C.RES, "ci_plateau.json")
    if not os.path.exists(p):
        note("ci: results/ci_plateau.json absent; intervals unchecked")
        return
    d = json.load(open(p, encoding="utf-8"))
    by = {r["tag"]: r for r in d["rows"]}

    r = by.get("main_fno_a025")
    if r is None:
        bad("ci: the reference arm main_fno_a025 is missing from the intervals")
        return
    if abs(r["e_plat"] - 0.963) < 5e-4 and abs(r["e_in"] - 0.0854) < 5e-4:
        ok("ci: the interval pipeline independently reproduces the main-table "
           "values (in band %.4f, plateau %.4f)" % (r["e_in"], r["e_plat"]))
    else:
        bad("ci: the interval pipeline gives in band %.4f and plateau %.4f, "
            "which do not match Table 1's 0.0854 and 0.963"
            % (r["e_in"], r["e_plat"]))

    if r["e_in_lo"] < r["e_in"] < r["e_in_hi"] and \
            r["e_plat_lo"] < r["e_plat"] < r["e_plat_hi"]:
        ok("ci: every point estimate lies inside its own interval")
    else:
        bad("ci: a point estimate lies outside its own interval")

    # the family separation the text claims: FNO jump ratio against the U-Net's
    u = by.get("base_unet_a025")
    if u is not None:
        if r["r_lo"] > u["r_hi"]:
            ok("ci: the FNO and U-Net jump-ratio intervals are disjoint "
               "(%.2f-%.2f against %.2f-%.2f), so the difference is not a "
               "media artefact" % (r["r_lo"], r["r_hi"], u["r_lo"], u["r_hi"]))
        else:
            bad("ci: the FNO and U-Net jump-ratio intervals overlap "
                "(%.2f-%.2f against %.2f-%.2f); the published separation is not "
                "supported" % (r["r_lo"], r["r_hi"], u["r_lo"], u["r_hi"]))
        if abs(u["onset"] - 4.25) > 1e-9:
            ok("ci: the U-Net onset (f=%.2f) is not the FNO's (f=%.2f), which "
               "is what distinguishes a decay from a step" % (u["onset"], r["onset"]))
        else:
            bad("ci: both families report the same onset, so the step/decay "
                "distinction is not in the data")


def baseline_pairing_guard(res=None, root=None):
    """The trivial baseline must sit on the evaluation media, and say so.

    Three separate claims are asserted here, because they fail for different
    reasons:

    1. *Pairing across frequency.*  The same medium and the same incidence
       angle must reappear at all 15 test frequencies.  `baseline_paired.py`
       records an md5 of v and of theta per cell; one distinct digest per
       contrast is the only acceptable count.  Before this guard existed the
       pairing was asserted in prose ("every comparison is paired") and nothing
       measured it on the test grid -- the only bitwise check in
       `check_claims` compared two training caches at one contrast.
    2. *The switch was worth making, and is not a rescue.*  The 600-media
       baseline and the 200-media one must differ by more than 1% (otherwise
       the old denominator was fine and this is noise) and by less than 15%
       (otherwise the two are different populations and the earlier quoted
       range was simply wrong rather than imprecise).
    3. *The paired interval is tight, and agrees with the other path.*  A
       paired ratio interval that is as wide as its two marginal intervals
       would mean the pairing carried no information; and the ratio implied by
       the emitted macros must match the one in the interval table, because
       they are two instruments reading the same quantity through different
       aggregations.
    """
    res = res or C.RES
    root = root or C.ROOT
    p = os.path.join(res, "triv_paired.json")
    if not os.path.exists(p):
        note("baseline: results/triv_paired.json absent; pairing unchecked")
        return
    d = json.load(open(p, encoding="utf-8"))

    if not d.get("prefix_ok"):
        bad("baseline: n=200 and n=600 draws no longer share media and angles; "
            "the 600-media baseline is a different population")
    else:
        ok("baseline: the n=200 and n=600 draws share v_max and theta, so the "
           "old denominator was the same population at the wrong sample size")

    offenders = []
    for vk, arm in d["data"].items():
        if arm.get("_v_md5_unique") != 1 or arm.get("_theta_md5_unique") != 1:
            offenders.append((vk, arm.get("_v_md5_unique"),
                              arm.get("_theta_md5_unique")))
    if offenders:
        bad("baseline: the test media are not paired across frequency at %s "
            "(expected one distinct v and one distinct theta per contrast)"
            % offenders[:3])
    else:
        ok("baseline: one distinct v and one distinct theta digest at all "
           "%d frequencies in every contrast -- the test grid is paired "
           "element-wise, not only in expectation" % len(d["f"]))

    rd = d.get("rel_diff_vs_600") or {}
    mx = rd.get("max")
    if mx is None:
        note("baseline: no cross-check against the 600-media denominator")
    elif mx <= 0.01:
        bad("baseline: the 200- and 600-media denominators agree to %.2f%%, so "
            "this pairing fix changed nothing and its cost is unjustified"
            % (100 * mx))
    elif mx >= 0.15:
        bad("baseline: the 200- and 600-media denominators differ by up to "
            "%.1f%%, which is too large to be a sample-size effect -- the two "
            "draws are not the same population" % (100 * mx))
    else:
        ok("baseline: restricting the denominator to the evaluation media "
           "moves it by %.1f%% on average and %.1f%% at most, which is why the "
           "quoted ratio range moved" % (100 * rd.get("mean", 0.0), 100 * mx))

    c = os.path.join(res, "ci_plateau.json")
    if not os.path.exists(c):
        note("baseline: ci_plateau.json absent; ratio intervals unchecked")
        return
    ci = json.load(open(c, encoding="utf-8"))
    worst, weakest = 0.0, 1e9
    unpaired = []
    for r in ci["rows"]:
        if r.get("ratio") is None:
            bad("baseline: %s has no plateau-to-baseline ratio" % r["tag"])
            continue
        if not (r["ratio_lo"] < r["ratio"] < r["ratio_hi"]):
            bad("baseline: %s ratio %.3f lies outside its own interval "
                "[%.3f, %.3f]" % (r["tag"], r["ratio"], r["ratio_lo"],
                                  r["ratio_hi"]))
        if r.get("ratio_un_lo") is None:
            bad("baseline: %s has no unpaired counterfactual, so 'paired' is "
                "not shown to be doing anything" % r["tag"])
            continue
        wp = r["ratio_hi"] - r["ratio_lo"]
        wu = r["ratio_un_hi"] - r["ratio_un_lo"]
        shrink = wu / max(wp, 1e-12)
        # Tightness may only be claimed where the numerator is of the same
        # order as the denominator.  The head-plus-residual arm has a plateau
        # of 0.026 against a baseline of 0.93: its ratio is carried by the
        # denominator, which barely varies, so pairing has nothing to cancel
        # and its interval stays wide.  Asserting one number for all nine arms
        # would turn a 1.1x into a 40x.
        if r["ratio"] > 0.5:
            if shrink < 2.0:
                bad("baseline: %s ratio interval is only %.1fx narrower when "
                    "paired than when the two sides are resampled "
                    "independently, so the pairing carried no information"
                    % (r["tag"], shrink))
            worst = max(worst, 0.5 * wp / r["ratio"])
            weakest = min(weakest, shrink)
        else:
            unpaired.append((r["tag"], round(shrink, 1)))
    if worst >= 0.03:
        bad("baseline: the widest paired ratio interval among the operator "
            "arms is +-%.1f%%, which is no better than its components"
            % (100 * worst))
    else:
        ok("baseline: across the operator arms the paired ratio interval is at "
           "most +-%.1f%% and at least %.0fx narrower than its unpaired "
           "counterfactual" % (100 * worst, weakest))
    if unpaired:
        note("baseline: no tightness claim for %s -- numerator far below the "
             "denominator, so the ratio is carried by the baseline"
             % (", ".join("%s (shrink %.1fx)" % u for u in unpaired)))

    # Cross-instrument: the macros the paper prints against the interval table.
    gm = os.path.join(root, "submission", "paper_en", "gen_macros.tex")
    if not os.path.exists(gm):
        gm = os.path.join(root, "paper", "en", "gen_macros.tex")
    tag_of = {"0.05": "Weak", "0.10": "Soft", "0.25": "Mid",
              "0.50": "Strong", "1.00": "Max"}
    if os.path.exists(gm):
        raw = open(gm, encoding="utf-8").read()
        by = {}
        for r in ci["rows"]:
            if r["tag"].startswith("main_fno") and r.get("ratio"):
                by["%.2f" % r["vmax"]] = r["ratio"]
        drift = []
        for vk, tag in tag_of.items():
            m = re.search(r"\\newcommand\{\\PlatTriv%s\}\{([0-9.]+)\}" % tag, raw)
            if not m:
                drift.append((vk, "missing", None))
                continue
            if vk in by and abs(float(m.group(1)) - by[vk]) / by[vk] > 0.01:
                drift.append((vk, float(m.group(1)), by[vk]))
        if drift:
            bad("baseline: the ratio in gen_macros disagrees with the interval "
                "table at %s (macro, table)" % drift[:3])
        else:
            ok("baseline: the five printed plateau-to-baseline ratios agree "
               "with the interval table to 1%, through two aggregations")

        # The word "every" in the cliff section is a quantitative claim: it
        # holds only while the smallest ratio is above one.
        m = re.search(r"\\newcommand\{\\TrivRatioLo\}\{([0-9.]+)\}", raw)
        if not m:
            bad("baseline: TrivRatioLo missing, so 'every arm above its own "
                "baseline' is unmeasured")
        elif float(m.group(1)) < 1.0:
            bad("baseline: the smallest plateau-to-baseline ratio is %s, so the "
                "sentence claiming every arm sits above its own baseline is "
                "false" % m.group(1))
        else:
            ok("baseline: the smallest plateau-to-baseline ratio is %s > 1, so "
               "'every arm above its own baseline' holds" % m.group(1))

        # Two independent aggregations of one quantity.  The per-sample ratio
        # (CorrErrOut) comes from the checkpoints; the table-implied one
        # (CorrTableOut) is mean error over mean baseline.  With the unpaired
        # denominator they differed by 4%; with the paired one by 0.3%.  That
        # gap closing is the falsifiable signature that the denominator, not
        # the statistic, was what was wrong.
        e1 = re.search(r"\\newcommand\{\\CorrErrOut\}\{([0-9.]+)\}", raw)
        e2 = re.search(r"\\newcommand\{\\CorrTableOut\}\{([0-9.]+)\}", raw)
        if e1 and e2:
            a, b = float(e1.group(1)), float(e2.group(1))
            if abs(a - b) / a > 0.02:
                bad("baseline: the per-sample ratio %.3f and the table-implied "
                    "%.3f disagree by %.1f%% -- the denominator is not on the "
                    "same media as the numerator"
                    % (a, b, 100 * abs(a - b) / a))
            else:
                ok("baseline: the per-sample ratio (%.3f) and the table-implied "
                   "one (%.3f) agree to %.1f%%, two aggregations, one "
                   "denominator" % (a, b, 100 * abs(a - b) / a))


def _doi_offenders(pdf_text, bib_text):
    """DOIs printed in the rendered text that the bibliography does not supply.

    Split out so that the probe in test_gate_falsification can exercise it on
    synthetic text: a gate that has never been shown to fire is not evidence.
    """
    import re
    pat = re.compile(r"10\.3390/[A-Za-z0-9./\-]+")
    printed = set(m.rstrip(".") for m in pat.findall(pdf_text))
    known = set(pat.findall(bib_text))
    return printed, known, sorted(printed - known)


def doi_guard(root=None):
    """No DOI may be printed that the bibliography does not supply.

    The MDPI class derives a DOI from \articlenumber and prints it in the
    running footer.  With \articlenumber{0} that is the placeholder
    10.3390/app1010000, which appeared on all 42 pages of the delivered PDF and
    is the kind of artefact a reviewer reads as carelessness.  The check is
    stated generally rather than against that one string: every 10.3390/... in
    the rendered text must also occur in refs.bib, so a real reference DOI
    passes and a synthesised one cannot.
    """
    import re, subprocess
    root = root or C.ROOT
    pdf = os.path.join(root, "submission", "paper_en", "main.pdf")
    bib = os.path.join(root, "submission", "paper_en", "refs.bib")
    if not (os.path.exists(pdf) and os.path.exists(bib)):
        note("doi: delivered PDF or refs.bib absent, placeholder check skipped")
        return
    try:
        out = subprocess.run(["pdftotext", pdf, "-"], capture_output=True,
                             text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        note("doi: pdftotext unavailable, placeholder check skipped")
        return
    if out.returncode != 0:
        note("doi: pdftotext failed, placeholder check skipped")
        return
    printed, known, extra = _doi_offenders(
        out.stdout, open(bib, encoding="utf-8").read())
    if extra:
        bad("doi: the delivered PDF prints %d DOI(s) that no bibliography entry "
            "supplies: %s -- a placeholder must not reach a submission"
            % (len(extra), ", ".join(extra[:4])))
    else:
        ok("doi: all %d 10.3390 DOI(s) printed in the delivered PDF come from "
           "the bibliography; no placeholder is typeset" % len(printed))
