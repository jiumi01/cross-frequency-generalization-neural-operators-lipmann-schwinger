# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Verify every relational claim that the manuscripts state in prose.

The macro audit (`audit_paper.py`) checks that each generated NUMBER matches an
independent recomputation.  It cannot check the sentences built around those
numbers -- "the in-band error rises from X at v=0.05 to Y at v=1.00" is only true
if the error really is monotone in contrast, and "the largest ratio occurs in the
weakest arms" is only true if the argmax lands on v=0.05.  Those statements are
hard-coded in the .tex sources, so nothing else in the toolchain would catch a
contradiction.

Each check below mirrors one sentence of the manuscripts and reports PASS/FAIL
with the numbers it used.  Run after `make_paper.py --steps eval`.
"""
import os
import sys
import glob
import json
import argparse
import re
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FAIL = []
RES = C.RES          # overridable with --res, so the checker can be exercised
#                      against a scratch eval set without touching results/


def ok(msg):
    print("  [ok]   " + msg)


def bad(msg):
    FAIL.append(msg)
    print("  [FAIL] " + msg)


def note(msg):
    print("  [note] " + msg)


def load():
    ev = {}
    for p in sorted(glob.glob(os.path.join(RES, "eval_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        ev[d["name"]] = d
    return ev


def keys(ev, prefix):
    """Records matching `prefix` that were scored on the CURRENT test grid.

    Same reason as make_results_tex.runs_for: a stale eval file has a different
    number of frequency columns, and seed-averaging across grids would either
    raise on the shape mismatch or, if the shapes happened to agree, compare
    errors at different frequencies.
    """
    return sorted(n for n in ev
                  if n.startswith(prefix)
                  and list(ev[n].get("f_test") or []) == list(C.F_TEST))


def curve(ev, names, vmax):
    """Seed-averaged (f, mean, sem) for one contrast cell."""
    F = np.array([r["f"] for r in ev[names[0]]["sweep"]["%.2f" % vmax]])
    A = np.array([[r["mean"] for r in ev[n]["sweep"]["%.2f" % vmax]]
                  for n in names])
    sem = np.array([r["sem"] for r in ev[names[0]]["sweep"]["%.2f" % vmax]])
    return F, A.mean(axis=0), sem, A


def inband(ev, names, vmax):
    v = [r["mean"] for n in names for r in ev[n]["in_band"]
         if abs(r["vmax"] - vmax) < 1e-6]
    return float(np.mean(v)), float(np.std(v))


def _ratio(d, vmax):
    """Band-edge ratio eps(5)/eps(4) of a single eval record at one contrast."""
    rows = d["sweep"]["%.2f" % vmax]
    e4 = next(r["mean"] for r in rows if abs(r["f"] - 4.0) < 1e-9)
    e5 = next(r["mean"] for r in rows if abs(r["f"] - 5.0) < 1e-9)
    return e5 / e4 if e4 else float("nan")


def arm_stats(ev):
    st = {}
    for vmax in C.VMAX_LIST:
        tag = "a%03d" % int(round(vmax * 100))
        names = keys(ev, "main_fno_%s_" % tag)
        if not names:
            continue
        F, M, _, A = curve(ev, names, vmax)
        ib, _ = inband(ev, names, vmax)
        st[vmax] = dict(
            n=len(names), F=F, M=M,
            inband=ib,
            inband_freq_range=float(M[F <= 4.0].max() - M[F <= 4.0].min()),
            seed_spread=float(np.mean(A.std(axis=0)[F <= 4.0])),
            plateau=float(M[F >= 6].mean()),
            e4=float(M[np.isclose(F, 4.0)][0]),
            e5=float(M[np.isclose(F, 5.0)][0]),
        )
        s = st[vmax]
        s["ratio"] = s["e5"] / s["e4"]
    return st


def residual_guard(res_dir):
    """The residual claims must stay true on whatever corpus is delivered.

    These sentences are unusually easy to invalidate silently: regenerating the
    corpus, lowering the acceptance tolerance, or switching off the convergence
    freeze all move the distribution, and none of them touches the prose.  The
    claims checked here are the ones a referee asked for -- the whole-corpus
    distribution, the independent recomputation, and the detectable threshold --
    so each is bound to the measured artifact rather than to the generator.
    """
    p = os.path.join(res_dir, "residual_dist.json")
    if not os.path.exists(p):
        bad("residual_dist.json missing: the corpus-wide residual distribution, "
            "the independent recomputation and the gate sensitivity are all "
            "unmeasured, but the prose cites them")
        return
    d = json.load(open(p, encoding="utf-8"))
    ov = d["corpus"]["overall"]
    if ov["n"] < 50000:
        bad("corpus residual scan covers only %d samples; the prose says the "
            "whole campaign" % ov["n"])
    else:
        ok("corpus residual distribution measured over %d samples from %d arms"
           % (ov["n"], len(d["corpus"]["per_arm"])))
    if ov["n_over_gate"] == 0 and ov["max"] <= 1e-9 * 1.001:
        ok("no sample exceeds the 1e-9 acceptance gate (max %.2e, p99 %.2e)"
           % (ov["max"], ov["p99"]))
    else:
        bad("the corpus contains %d samples above the 1e-9 gate (max %.2e): the "
            "prose sentence about the generated corpus is wrong"
            % (ov["n_over_gate"], ov["max"]))
    rc = d.get("recheck")
    fl = (rc or {}).get("floor") if isinstance(rc, dict) else None
    rows = (rc or {}).get("per_arm") if isinstance(rc, dict) else None
    if not rows or not fl or fl.get("n", 0) < 100:
        bad("no independent recomputation of the recorded residual: the prose "
            "claims one")
    else:
        n = sum(r["n"] for r in rows)
        ref = fl["refined"]["max"]
        ship = fl["shipped"]["max"]
        rt = fl["roundtrip"]["max"]
        pert = fl["psi_storage_perturbation"]["p50"]
        # The recomputation from the shipped fields is precision-limited by the
        # single-precision storage.  If it were NOT above the gate, the prose
        # sentence explaining the floor would be describing a problem that does
        # not exist.
        if ship > fl["gate"]:
            ok("recomputation from the shipped fields is limited by the "
               "complex64 storage floor (%.1e against %.1e, psi perturbed by "
               "%.1e): it cannot resolve the %.0e gate, as the prose says"
               % (ship, rt, pert, fl["gate"]))
        else:
            bad("the shipped-field recomputation reaches %.1e, below the gate, "
                "so the prose's storage-floor explanation is not what is "
                "happening" % ship)
        if rt <= 3 * ship:
            ok("the floor is caused by the storage round-trip alone: round-trip "
               "%.1e vs shipped %.1e" % (rt, ship))
        else:
            bad("the shipped-field residual %.1e is not explained by the "
                "round-trip %.1e: something else is wrong" % (ship, rt))
        if ref <= fl["gate"]:
            ok("independently re-solved from the shipped inputs at working "
               "precision, the fields satisfy the equation to %.1e over %d "
               "samples, inside the %.0e gate" % (ref, n, fl["gate"]))
        else:
            bad("the working-precision re-solve only reaches %.1e over %d "
                "samples, so the recorded residuals are NOT reproduced and the "
                "gate claim fails" % (ref, n))
    nv = d.get("naive") or {}
    if not nv.get("n_total") or nv.get("clean", {}).get("n", 0) == 0:
        bad("the naive/frozen split was not measured: the separation and the "
            "detection threshold cited in the prose are unsupported")
    else:
        fr = nv.get("silent_fraction")
        if fr and fr > 0:
            ok("the naive recurrence silently ruins %.1f%% of the labels in a "
               "heterogeneous batch, while its own bookkeeping reports "
               "convergence" % (100 * fr))
        else:
            bad("no silent corruption was reproduced on a fresh heterogeneous "
                "batch: the prose's central hygiene finding is unsupported")
        ft = nv.get("frozen_true") or {}
        if ft.get("max") is not None and ft["max"] <= 1e-9:
            ok("the frozen recurrence's honest residual stays inside the gate "
               "on the same batch (max %.1e)" % ft["max"])
        else:
            bad("the frozen recurrence's honest residual reaches %s, outside "
                "the gate" % (ft.get("max"),))
        # The prose does NOT claim a finite separation any more.  The ruined
        # labels are non-finite, so there is no finite gap to place a gate in,
        # which is precisely why the in-loop check misses them.  The guard has
        # to verify that structure -- damage non-finite, survivors inside the
        # gate, no finite corrupted remainder -- AND it has to fail if a later
        # run produces finite corruption, because then the sentence would be
        # wrong in the other direction.  (This check itself caught the earlier
        # overclaim, which claimed a gap of `separation_orders` that was None.)
        nf = nv.get("nonfinite_fraction")
        cmax = (nv.get("clean") or {}).get("max")
        n_corr = (nv.get("corrupt") or {}).get("n", 0)
        if (nf and nf > 0 and n_corr == 0 and cmax is not None
                and cmax <= 1e-9 * 1.001):
            ok("every damaged label is non-finite (%.1f%% of the fresh batch) "
               "while the survivors stay inside the gate (max %.1e): the "
               "failure leaves no finite residual to threshold against, as the "
               "prose now says" % (100 * nf, cmax))
        else:
            bad("the prose says the damaged labels are non-finite and the "
                "survivors are inside the gate, but the measurement gives "
                "non-finite %.1f%%, finite-corrupt %d, clean max %s"
                % (100 * (nf or 0.0), n_corr, cmax))
        eta = None
        for row in nv["injection"]:
            if row["frac_flagged"] >= 0.999 and (eta is None or row["eta"] > eta):
                eta = row["eta"]
        if eta is None:
            bad("no injected perturbation size is flagged in every sample, so "
                "no detection threshold can be quoted")
        else:
            ok("gate sensitivity: a %.0e relative label error is flagged in "
               "every sample" % eta)


def seed_guard(ev):
    """The like-for-like claim must be a measurement, not a recollection.

    The manuscript asserted that the ablations ran "one or two seeds" while the
    main arms ran three.  That was accurate when written and silently became
    false once the replication pass gave every arm a third seed -- and no gate
    was reading the sentence, because it is a claim about the *protocol* rather
    than about a number in a table.  The prose now cites \\NumSeeds, so it is
    checked against the evaluation records here.

    W1 (2026-10-24) extended the v=1.00 anchored pair (main_fno_a100 and
    bax_fno_a100_m7) from three to six seeds; W2 (2026-10-25) extended the
    other seven arms the headline claims rest on, so nine arms now carry six.
    The property the manuscript actually claims is *within a comparison*: both
    sides of a paired test must share seeds, otherwise the error bars are not
    like for like.  Uniformity across the whole paper is neither claimed nor
    required -- the ablation arms stay at three -- so the check is: the
    extended arms all carry one identical seed set, and no arm at the same
    contrast tag carries seeds outside it.  A comparison between an extended
    arm and a base arm then runs on the shared subset; that degradation is
    reported, not failed, because the manuscript does not claim otherwise.
    """
    import re as _re
    fam = {}
    for n in ev:
        m = _re.match(r"^(.*)_s(\d+)$", n)
        if m:
            fam.setdefault(m.group(1), set()).add(int(m.group(2)))
    if not fam:
        bad("no per-seed evaluation records: the seed claim is unverifiable")
        return
    counts = {k: len(v) for k, v in fam.items()}
    base = Counter(counts.values()).most_common(1)[0][0]
    odd = sorted(k for k, c in counts.items() if c != base)
    if not odd:
        ok("all %d arms carry the same %d seeds, so the claim of a uniform "
           "seed protocol holds" % (len(fam), base))
        return
    # W2 (2026-10-25) extended every arm the headline claims rest on, so the
    # old rule -- "only a complete main+bax verdict pair may depart from the
    # base count" -- is now the wrong rule: nine arms carry six seeds and most
    # of them have no bax partner.  What still has to hold is the premise the
    # paired tests rest on: all extended arms share one identical seed set, and
    # no arm at their contrast tag sits outside it.  Pairing an extended arm
    # against a base arm silently runs on the shared subset, which is honest
    # only if it is stated, so that degradation is reported rather than failed.
    sets = {k: frozenset(fam[k]) for k in odd}
    if len({frozenset(v) for v in sets.values()}) != 1:
        bad("the extended arms disagree on their seed sets: %s"
            % {k: sorted(v) for k, v in sorted(sets.items())})
        return
    shared = next(iter(sets.values()))
    tags = {}
    for k in fam:
        m = _re.match(r"^(?:main|ba|bax|baf)_fno_(a\d+)", k)
        if m:
            tags.setdefault(m.group(1), set()).add(k)
    bad_any = False
    for tag, ks in sorted(tags.items()):
        inside = {k for k in ks if fam[k] <= shared}
        if inside != ks:
            for k in sorted(ks - inside):
                bad("%s at tag %s carries seeds %s outside the extended set "
                    "%s -- a paired test against it would compare different "
                    "pairs than the paper says"
                    % (k, tag, sorted(fam[k] - shared), sorted(shared)))
            bad_any = True
    if not bad_any:
        degraded = sorted({t for t, ks in tags.items()
                           if any(counts[k] == base for k in ks)})
        ok("%d arms carry the extended seed set %s; %d arms stay at %s%s"
           % (len(odd), sorted(shared), len(fam) - len(odd), base,
              ", so comparisons at %s run on the shared subset only"
              % "/".join(degraded) if degraded else ""))


def caption_guard(ev):
    """Generated captions are deliverable text too.

    Every prose check above reads ``main.tex``.  The table captions, however,
    are emitted by ``make_results_tex.py`` into ``paper/<lang>/gen_results.tex``
    and a stale one is read by a referee exactly as the prose is.  Observed
    2026-10-11: the band-table caption still asserted that widening "lowers the
    in-band error floor" a full revision after the prose had been corrected to
    say the opposite, because nothing in the toolchain was reading the caption.
    Bind each directional caption claim to the same numbers the prose is bound
    to, so the caption cannot drift back.
    """
    import re as _re
    # The tables were split into one file per label and the concatenated
    # gen_results.tex was deliberately deleted so a stale copy could not be
    # compiled.  This guard kept globbing for that deleted name, so from the
    # split onwards it reported "skipped" and checked nothing -- a guard that is
    # safe in direction (it never fires a false alarm) and useless in effect.
    # Read the split files instead.
    paths = sorted(glob.glob(os.path.join(C.ROOT, "paper", "*", "gen_tab_*.tex")))
    if not paths:
        note("no generated tables found; caption guard skipped")
        return

    ibs, plats = {}, {}
    for tag in ("b2", "b3", "b6", "b12"):
        n = keys(ev, "band_%s_" % tag)
        if not n:
            continue
        ib, _ = inband(ev, n, 0.25)
        F, M, _, _ = curve(ev, n, 0.25)
        ibs[tag] = ib
        plats[tag] = float(M[F >= 6].mean())
    if not ibs:
        note("band ablation not complete; caption guard skipped")
        return
    note("band in-band floors: " + ", ".join("%s=%.3f" % kv
                                             for kv in sorted(ibs.items())))

    pat_inband = _re.compile(r"lowers the in-band error floor|降低带内误差下限")
    pat_plateau = _re.compile(r"lowers the out-of-band plateau|降低带外平台")
    for p in paths:
        lang = os.path.basename(os.path.dirname(p))
        # Only the band table carries band-directional claims.  Keep scanning
        # every generated table -- if such a claim ever migrates, it is still
        # checked -- but suppress the "no such claim" note for the other ten, or
        # the guard buries its own signal under eleven lines per language.
        quiet = os.path.basename(p) != "gen_tab_band.tex"
        txt = open(p, encoding="utf-8").read()

        if pat_inband.search(txt):
            if "b12" in ibs and ibs["b12"] <= min(ibs.values()) + 1e-9:
                ok("[%s] caption claims the in-band floor is lowered with width "
                   "and the data agree (widest band %.3f is the lowest)"
                   % (lang, ibs["b12"]))
            else:
                bad("[%s] a generated caption still claims widening 'lowers the "
                    "in-band error floor', but the widest band's floor is %.3f "
                    "against %s -- rewrite the caption"
                    % (lang, ibs.get("b12", float("nan")),
                       ", ".join("%s=%.3f" % kv for kv in sorted(ibs.items()))))
        else:
            if not quiet:
                note("[%s] no 'lowers the in-band floor' caption claim" % lang)

        if pat_plateau.search(txt):
            if "b12" in plats and plats["b12"] <= min(plats.values()) + 1e-9:
                ok("[%s] caption claims the out-of-band plateau is lowered with "
                   "width and the data agree (widest band %.3f is the lowest)"
                   % (lang, plats["b12"]))
            else:
                bad("[%s] a generated caption claims widening 'lowers the "
                    "out-of-band plateau', but the widest band's plateau is "
                    "%.3f against %s"
                    % (lang, plats.get("b12", float("nan")),
                       ", ".join("%s=%.3f" % kv for kv in sorted(plats.items()))))

    # A third directional caption claim, added with the wide-band U-Net rows:
    # the band caption now asserts that the U-Net's in-band floor does not move
    # with the band width.  That is a claim about two specific arms, so it is
    # bound here rather than left to the numeric macros that happen to sit next
    # to it -- a macro cannot drift, but the sentence around it can.
    pat_unet_flat = _re.compile(r"unmoved by it|都不随带宽移动")
    nun, nub = keys(ev, "base_unet_a025_"), keys(ev, "uband_b6_")
    if nun and nub:
        ib_un4, _ = inband(ev, nun, 0.25)
        ib_un6, _ = inband(ev, nub, 0.25)
        rel = abs(ib_un6 - ib_un4) / max(ib_un4, 1e-30)
        for p in paths:
            lang = os.path.basename(os.path.dirname(p))
            if not pat_unet_flat.search(open(p, encoding="utf-8").read()):
                continue
            if rel <= 0.05:
                ok("[%s] caption says the U-Net in-band floor does not move with "
                   "the band width and the data agree (%.4f -> %.4f, %.1f%%)"
                   % (lang, ib_un4, ib_un6, 100 * rel))
            else:
                bad("[%s] a generated caption says the U-Net in-band floor is "
                    "unmoved by the band width, but it moves %.1f%% "
                    "(%.4f -> %.4f)"
                    % (lang, 100 * rel, ib_un4, ib_un6))
    else:
        note("wide-band U-Net arm missing; the U-Net in-band caption claim "
             "could not be bound")


ROOT = C.ROOT


def macro_name_guard(root=None):
    """A TeX control word stops at the first non-letter.

    ``\\newcommand{\\DAPlatM2}{0.300}`` does not define a macro called
    DAPlatM2.  It defines ``\\DAPlatM`` and leaves a literal ``2`` in the
    preamble, which is a fatal ``Missing \\begin{document}`` -- and it is
    fatal in the preamble because that is where the generated macro file is
    read, so every downstream gate reports clean while the PDF is never
    produced.  The exact macro name caused this on 2026-10-22, and the earlier
    \\ResP50 instance is the same class, so the class is closed here: no
    generated macro name may contain a digit, and no two may collide.
    """
    root = root or ROOT
    pat = re.compile(r"\\newcommand\{\\([A-Za-z@]+)\}")
    for lang, folder in (("en", os.path.join(root, "paper", "en")),
                         ("en-sub", os.path.join(root, "submission",
                                                 "paper_en")),
                         ("cn", os.path.join(root, "paper", "cn")),
                         ("cn-sub", os.path.join(root, "submission",
                                                 "paper_cn"))):
        p = os.path.join(folder, "gen_macros.tex")
        if not os.path.exists(p):
            continue
        names = pat.findall(open(p, encoding="utf-8").read())
        if not names:
            continue
        # every name the generator emits must also be checked for digits; the
        # regex above would not even see one, so scan the raw file too
        raw = open(p, encoding="utf-8").read()
        digit = re.findall(r"\\newcommand\{\\(\w*\d\w*)\}", raw)
        if digit:
            bad("macro names contain digits in %s: %s -- a TeX control word "
                "stops at the first non-letter, so these define the wrong "
                "macro and break the preamble"
                % (lang, ", ".join(sorted(set(digit))[:6])))
            continue
        dup = sorted(n for n in set(names) if names.count(n) > 1)
        if dup:
            bad("duplicate macro definitions in %s: %s" % (lang, ", ".join(dup)))
        else:
            ok("%s: %d generated macro names are legal control words and "
               "unique" % (lang, len(names)))


def macro_space_guard(root=None):
    """A control word eats the space that follows it.

    ``\\CapFactor changes`` does not print "2.6 changes", it prints
    "2.6changes": TeX terminates the control word at the space and swallows it.
    The generated-value macros are the worst place for this, because they appear
    in dense numeric sentences where a fused token still looks like a number and
    so survives every numeric gate -- measured on the delivered PDF, the
    sentence read "(0.084 against 0.085at 2.6-fold the parameters)".

    The class is closed here rather than instance by instance: every macro name
    that the generator emits is collected, every use that is followed by
    whitespace and then a letter is flagged, and the only permitted cases are
    declared in ``UNIT_SUFFIX_OK`` with the reading that makes them correct.
    Anything else fails, including a case that was introduced after this gate.
    """
    import re as _re
    root = root or C.ROOT
    names = set()
    for p in glob.glob(os.path.join(root, "paper", "*", "gen_macros.tex")):
        names |= set(_re.findall(r"\\newcommand\{\\([A-Za-z]+)\}",
                                 open(p, encoding="utf-8").read()))
    if not names:
        note("no generated macros found; macro spacing guard skipped")
        return
    # Declared exceptions.  In these the eaten space is the intended reading:
    # the macro expands to a count and the following "M" is the unit ("million"),
    # so "10.3M parameters" is what the sentence means.  Written out rather than
    # inferred, so a new occurrence cannot ride along silently.
    UNIT_SUFFIX_OK = {"FnoParams": "M", "UnetParams": "M",
                      "PmatchParams": "M", "DonParams": "M"}
    fails = []
    for lang in ("en", "cn"):
        mf = os.path.join(root, "paper", lang, "main.tex")
        if not os.path.exists(mf):
            continue
        for i, line in enumerate(open(mf, encoding="utf-8").read().split("\n"), 1):
            for m in _re.finditer(r"\\([A-Za-z]+)([ \t]+)(?=[A-Za-z(])", line):
                nm, gap = m.group(1), m.group(2)
                if nm not in names:
                    continue
                nxt = line[m.end():m.end() + 1]
                if nm in UNIT_SUFFIX_OK and nxt == UNIT_SUFFIX_OK[nm]:
                    continue
                fails.append("%s:%d \\%s" % (lang, i, nm))
            # The same fused token, reached the other way: a generated macro
            # written straight against a half-width bracket -- "\NDataTop(16\,384
            # total)" prints "12288(16,384 total)".  No space is swallowed here,
            # the value is simply welded to the bracket, and like the swallowed
            # space it survives every numeric gate because the digits are intact.
            # Only half-width brackets are flagged: in Chinese, a full-width
            # bracket follows the number with no space by convention.
            for m in _re.finditer(r"\\([A-Za-z]+)(?=[([])", line):
                if m.group(1) in names:
                    fails.append("%s:%d \\%s (welded to a half-width bracket)"
                                 % (lang, i, m.group(1)))
    if fails:
        bad("control word would eat its following space, or a generated value is "
            "welded to a bracket, either way fusing the token with what follows "
            "(write \\Name{} or \\Name\\ ): %s" % ", ".join(fails))
    else:
        ok("%d generated macro names scanned in both manuscripts; no control "
           "word is followed by a space that it would swallow, and no value is "
           "welded to a half-width bracket (the only space-then-letter uses are "
           "the %d declared count/unit pairs)"
           % (len(names), len(UNIT_SUFFIX_OK)))


def arch_label_guard(root=None):
    """Fig. 4's legend and Table III's rows must name the same four models.

    Class-level gate.  The defect it replaces was not a wrong number but a
    *label* drift: the figure said "U-Net (small)" while the table said
    "U-Net", and the figure caption then credited the lowest plateau to the
    base U-Net -- the matched U-Net is the one that attains it, the base one is
    merely the smaller model.  No numeric gate can see that, because both
    numbers are correct; the pairing is what is wrong.  The labels now come from
    ``common.ARCH_GROUPS``; this checks that the emitted table used that
    definition and that no stale label survives in the hand-written prose or in
    a figure caption.

    ``root`` is overridable so the gate can be pointed at a scratch manuscript
    by its falsifiability probe.
    """
    import re as _re
    root = root or C.ROOT
    want = [lab for _, lab in C.ARCH_GROUPS]
    # Same stale-glob defect as caption_guard: after the split the architecture
    # table lives in gen_tab_arch.tex, and looking only for the deleted
    # gen_results.tex made this gate skip itself silently.  A guard that can
    # never fire is worse than no guard, because it is counted as coverage.
    tex = sorted(glob.glob(os.path.join(root, "paper", "*", "gen_tab_arch.tex")))
    if not tex:
        note("no generated architecture table found; architecture-label guard skipped")
        return
    agreed = 0
    for p in tex:
        t = open(p, encoding="utf-8").read()
        # Scope the extraction to the tab:arch float.  An earlier version of this
        # gate scanned every "$x$ & y &" row in the file and picked up Table V's
        # mode-budget rows (labels "1"/"5"), i.e. it reported a label drift that
        # was not there.  Take the block from \label{tab:arch} to its
        # \end{tabular}, so only the comparison of interest is read.
        m_block = _re.search(r"\\label\{tab:arch\}(.*?)\\end\{tabular\}", t, _re.S)
        if not m_block:
            bad("%s: no tab:arch float found, so the legend/table label "
                "agreement cannot be verified" % os.path.basename(p))
            continue
        got = _re.findall(r"^\$[\d.]+\$\s*&\s*(.+?)\s*&", m_block.group(1), _re.M)
        if not got:
            bad("%s: tab:arch has no data rows, so the legend/table label "
                "agreement cannot be verified" % os.path.basename(p))
            continue
        # The set of labels must equal the declared set.  Do NOT require the
        # table to be rectangular: it is not.  Only the FNO was trained at every
        # contrast, so the 0.05 and 0.10 groups carry a single row while the
        # 0.25/0.50/1.00 groups carry all four.  An earlier version asserted
        # ``len(rows) % len(labels) == 0`` and failed on the correct table --
        # the invariant was about the shape of the artefact, not about the
        # property being checked.
        if sorted(set(got)) == sorted(want) and len(got) >= len(want):
            agreed += 1
        else:
            bad("%s: architecture rows are labelled %s while the shared "
                "definition says %s -- the Fig. 4 legend and Table III disagree"
                % (os.path.basename(p), sorted(set(got)), sorted(want)))
    stale = set()
    for p in sorted(glob.glob(os.path.join(root, "paper", "*", "main.tex"))):
        t = open(p, encoding="utf-8").read()
        for m in _re.finditer(r"U-Net\s*\([^)]*\)", t):
            if m.group(0) not in want:
                stale.add("%s: %s" % (os.path.basename(p), m.group(0)))
    if stale:
        bad("stale architecture label(s) in the manuscript: %s"
            % "; ".join(sorted(stale)))
    elif agreed:
        ok("architecture labels agree: Table III and the Fig. 4 legend both name "
           "%s, and no stale label remains in the prose" % ", ".join(want))
    else:
        bad("the architecture labels could not be agreed in any manuscript")


def _float_caption(text, label):
    """The caption of the float carrying ``label``, with braces balanced."""
    i = text.find("\\label{%s}" % label)
    if i < 0:
        return None
    j = text.rfind("\\caption{", 0, i)
    if j < 0:
        return None
    k = j + len("\\caption{")
    depth, out = 1, []
    while k < len(text) and depth:
        ch = text[k]
        if ch == "\\" and k + 1 < len(text):
            out.append(text[k:k + 2])
            k += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if not depth:
                break
        out.append(ch)
        k += 1
    return "".join(out)


def batch_label_guard(root=None):
    """A sample count named in a caption must come from the measurement, by macro.

    Class-level gate, over a class no numeric check can see.  The solver-hygiene
    campaign has exactly two legitimate sizes -- 1024 samples per heterogeneous
    batch and 8192 in aggregate (8 x 1024) -- and both the prose and the figure
    captions quote them.  The prose reaches them through ``\\SolverBatchN``; the
    English Fig. 11 caption spelled "per batch of 1024" out by hand, and the
    Chinese caption of the same figure gave no size at all.  So one figure, two
    languages, two different amounts of information, and neither bound to
    ``results/solver_fig.json``.

    The hand-written number happened to be correct, which is the point: no
    numeric gate could have fired, and the next batch-size change would leave
    the two captions disagreeing with each other and with the JSON.  What is
    fragile is the *form*, so the assertions are about form: both macros must
    exist and equal the measured values, and no caption may spell either value
    out as a literal.
    """
    import re as _re
    root = root or C.ROOT
    p = os.path.join(RES, "solver_fig.json")
    if not os.path.exists(p):
        note("solver_fig.json missing; batch-label guard skipped")
        return
    d = json.load(open(p, encoding="utf-8"))
    try:
        want = {"SolverBatchSize": int(d["freeze"]["batch"]),
                "SolverBatchN": int(d["freeze"]["aggregate"]["n_total"])}
    except (KeyError, TypeError, ValueError):
        bad("solver_fig.json carries no usable batch size, so the captions "
            "quoting one cannot be checked")
        return
    bound = 0
    for gp in sorted(glob.glob(os.path.join(root, "paper", "*", "gen_macros.tex"))):
        lang = os.path.basename(os.path.dirname(gp))
        t = open(gp, encoding="utf-8").read()
        for name, val in sorted(want.items()):
            m = _re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % name, t)
            if not m:
                bad("[%s] \\%s is not generated, so a caption cannot quote the "
                    "measured batch size" % (lang, name))
                continue
            lit = m.group(1).replace("\\,", "").replace(" ", "")
            if lit.strip("-") == "":
                bad("[%s] \\%s is still the placeholder %r" % (lang, name, lit))
                continue
            try:
                got = int(lit)
            except ValueError:
                bad("[%s] \\%s = %r is not an integer" % (lang, name, lit))
                continue
            if got != val:
                bad("[%s] \\%s = %d while the measurement says %d"
                    % (lang, name, got, val))
            else:
                bound += 1
    for mp in sorted(glob.glob(os.path.join(root, "paper", "*", "main.tex"))):
        lang = os.path.basename(os.path.dirname(mp))
        cap = _float_caption(open(mp, encoding="utf-8").read(), "fig:solver")
        if cap is None:
            bad("[%s] the fig:solver float has no caption, so the batch sizes it "
                "quotes cannot be checked" % lang)
            continue
        for name, val in sorted(want.items()):
            # the literal, allowing TeX thousands spacing (`1\,024`)
            pat = _re.compile(r"(?<![\\\d])" + r"\\?,?".join(str(val)) + r"(?!\d)")
            if pat.search(cap):
                bad("[%s] the fig:solver caption spells %d out by hand; quote "
                    "\\%s instead so it cannot drift" % (lang, val, name))
    if bound == 2 * len(want):
        ok("the Fig. 11 captions reach both measured batch sizes through macros "
           "(%s) and neither spells one out by hand"
           % ", ".join("%s=%d" % kv for kv in sorted(want.items())))
    elif bound == 0:
        bad("no manuscript binds a batch size to the measurement")


ABSTRACT_EN_LIMIT = 200   # MDPI submission cap, counted on the rendered PDF
ABSTRACT_CN_LIMIT = 400   # reading units: twice the EN cap (1 Latin word = 2 chars)


def pdf_text(pdf):
    """Rendered text of a built PDF, or None if pdftotext cannot read it.

    Module level so the falsifiability probe can drive the abstract gate with
    a synthetic render instead of building a PDF.
    """
    import subprocess
    try:
        r = subprocess.run(["pdftotext", pdf, "-"], capture_output=True,
                           text=True, timeout=120)
    except OSError:
        return None
    return r.stdout if r.returncode == 0 else None


def rendered_abstract_words(text, heading="Abstract", keywords="Keywords:"):
    """Word count of the abstract in rendered PDF text; returns (count, error).

    The heading line and bare page-number lines (lineno mode stamps one per
    page) are not words.  An absent heading or keyword line is an error, not
    a zero: a count of zero would pass every cap.

    The region also ends at the journal's front-matter sidebar.  That sidebar
    (``Received: / Revised: / Accepted: / Published: / Copyright: ... Submitted
    to Appl. Sci.``) sits between the abstract and the keyword line in the
    class's layout, and whether text extraction pulls it into the span depends
    on the reading order -- which changed the moment the line numbers became
    extractable, adding 32 words to the count without a word of the abstract
    being touched.  Ending the region at the first of several sentinels makes
    the count independent of that ordering.
    """
    i = text.find(heading)
    if i < 0:
        return 0, "abstract region not found in the rendered text"
    stops = [text.find(k, i + 1) for k in
             (keywords, "Received:", "Copyright:", "Submitted to")]
    stops = [j for j in stops if j > i]
    if not stops:
        return 0, "abstract region not found in the rendered text"
    j = min(stops)
    lines = []
    for ln in text[i:j].splitlines():
        s = ln.strip()
        if s and s != heading and not re.fullmatch(r"\d+", s):
            lines.append(s)
    return len(" ".join(lines).split()), None


def submission_element_guard(root=None):
    """Two submission-level elements that no numeric check can see.

    (a) The abstract word budget.  MDPI caps the abstract at about 200 words and
        the cap is a *hard* editorial gate; the abstract is also the one part of
        the manuscript that is rewritten most often, and it grew from 196 to 229
        words across one round of revision without anything flagging it.  The
        count is taken on the rendered PDF -- the source-based count that
        replaced the missing one reported 124 words for a 202-word abstract,
        because its non-greedy regex stopped at the first '}' (the one inside
        $v_{\\max}$) and its macro deletion hid every number-bearing macro.  A
        PDF older than main.tex is refused, so an edit without a rebuild cannot
        pass either.
    (b) Placeholder-shaped tokens in the delivered tables.  Two shapes have
        actually shipped: the `$--$` "locator not triggered" cell, which in math
        mode prints as two minus signs and reads as a range, and a bare `--`
        cell in the cost table, which outside math mode prints as an en-dash.
        Both are indistinguishable from a real interval at read time, and the
        numeric gate cannot see either because the surrounding digits are fine.

        `--` is NOT banned outright: it is legitimate inside a numeric range
        (`0.085--0.127`, `\\CliffLo--\\CliffHi`) and `---` is the em-dash used for
        asides.  Only the two observed cell shapes are forbidden.
    """
    import re as _re
    root = root or C.ROOT

    # (a) abstract budget ---------------------------------------------------
    #
    # EN is counted on the RENDERED PDF, because that is the text the
    # editorial office counts.  The source-based count this replaces was
    # wrong twice over and reported 124 words for the 202-word abstract that
    # shipped: its non-greedy \abstract{(.*?)} stopped at the first '}', the
    # one inside $v_{\max}$, so two thirds of the abstract was never seen,
    # and every number-bearing macro (\CliffLo, \BATwoIn, ...) was deleted
    # rather than counted.  The docstring below records the same class of
    # failure going unnoticed before (196 -> 229 words).
    #
    # The staleness direction is closed with it: an abstract edited without a
    # rebuild is refused, not silently counted from the stale render.
    en_tex = os.path.join(root, "paper", "en", "main.tex")
    en_pdf = os.path.join(root, "paper", "en", "main.pdf")
    if not os.path.exists(en_tex):
        bad("en: main.tex missing -- cannot check the abstract cap")
    elif not os.path.exists(en_pdf):
        bad("en: main.pdf missing -- the abstract cap is checked on the "
            "rendered text; run `--steps tex,build` first")
    elif os.path.getmtime(en_pdf) < os.path.getmtime(en_tex) - 1.0:
        bad("en: main.pdf is older than main.tex -- the abstract would be "
            "counted from a stale render; run `--steps tex,build` first")
    else:
        txt = pdf_text(en_pdf)
        if txt is None:
            bad("en: pdftotext could not read main.pdf")
        else:
            n, err = rendered_abstract_words(txt)
            if err:
                bad("en: %s" % err)
            elif n > ABSTRACT_EN_LIMIT:
                bad("en: abstract is %d words in the rendered PDF, over the "
                    "%d-word submission limit" % (n, ABSTRACT_EN_LIMIT))
            else:
                ok("en: abstract is %d words in the rendered PDF (limit %d)"
                   % (n, ABSTRACT_EN_LIMIT))
    # CN: whitespace tokenisation is meaningless for Chinese, so the
    # reading-unit convention the caption budget already uses (1 Latin word
    # = 2 characters) is kept, with the cap proportional to the EN one.  A
    # number-bearing macro is one Latin word once rendered, so it counts as
    # one here too.
    cn_tex = os.path.join(root, "paper", "cn", "main.tex")
    if not os.path.exists(cn_tex):
        bad("cn: main.tex missing -- cannot check the abstract cap")
    else:
        txt = open(cn_tex, encoding="utf-8").read()
        m = _re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", txt, _re.S)
        if not m:
            bad("cn: no abstract environment found")
        else:
            body = _re.sub(r"\\[A-Za-z]+\{([^}]*)\}", r"\1", m.group(1))
            body = _re.sub(r"\\[A-Za-z]+", " X ", body)
            cjk = len(_re.findall(r"[\u3400-\u9fff\u3000-\u303f\uff00-\uffef]",
                                  body))
            latin = len([w for w in _re.split(r"[^A-Za-z0-9.]+", body)
                         if w.strip()])
            units = cjk + 2 * latin
            if units > ABSTRACT_CN_LIMIT:
                bad("cn: abstract is %d reading units, over the %d-unit cap "
                    "(twice the EN word cap under the 1 Latin word = 2 "
                    "characters convention)" % (units, ABSTRACT_CN_LIMIT))
            else:
                ok("cn: abstract is %d reading units (cap %d)"
                   % (units, ABSTRACT_CN_LIMIT))

    # (b) placeholder-shaped cells ------------------------------------------
    #
    # Only CELL-SHAPED placeholders are forbidden, and the `&` boundary is part
    # of every pattern.  A bare `$--$` pattern was tried first and produced six
    # false positives on `$\CliffLo$--$\CliffHi$`: the closing `$` of one math
    # group, the range `--`, and the opening `$` of the next are character for
    # character the same token as a placeholder, so no local pattern can tell
    # them apart.  Tightening around the cell boundary removes the ambiguity
    # instead of loosening the rule.  Numeric ranges (`0.085--0.127`,
    # `\CliffLo--\CliffHi`) and the em-dash `---` therefore stay legal, and a
    # macro that resolves to `--` inside a cell is caught by
    # audit_paper.check_macros_resolved, which knows which macros are cited.
    shapes = [(r"&\s*--\s*&", "bare '--' as a table cell"),
              (r"&\s*--\s*\\\\", "bare '--' as a table cell"),
              (r"&\s*\$--\$\s*&", "'$--$' as a table cell"),
              (r"&\s*\$--\$\s*\\\\", "'$--$' as a table cell")]
    hits = []
    targets = [os.path.join(root, "paper", l, "main.tex") for l in ("en", "cn")]
    for lang in ("en", "cn"):
        targets += sorted(glob.glob(os.path.join(root, "paper", lang,
                                                 "gen_tab_*.tex")))
    for p in targets:
        if not os.path.exists(p):
            continue
        txt = open(p, encoding="utf-8").read()
        for pat, what in shapes:
            for m in _re.finditer(pat, txt):
                hits.append("%s:%d %s" % (os.path.relpath(p, root).replace("\\", "/"),
                                          txt[:m.start()].count("\n") + 1, what))
    if hits:
        bad("placeholder-shaped token(s) in delivered tables: %s" % hits[:6])
    else:
        ok("no placeholder-shaped cell in any delivered table")


def realgeo_guard(root=None):
    """Third level: real geology.  Design identity, window rule, wording.

    The third level exists to test whether the boundary and the jump survive a
    change of geometry, so its evidential value rests entirely on "nothing else
    moved".  That is a claim about the *design*, and it goes quietly false the
    moment the main protocol is edited and this rung is not re-run -- which no
    numeric audit of the *numbers* can see, because the third level's numbers
    would remain internally consistent while no longer being comparable to the
    rest of the paper.  The guard is therefore built around form:

      (a) protocol identity: the recorded loss tangent, test grid and contrast
          arms must still equal those of ``code/common.py``;
      (b) the window rule: three equal-width, equally spaced, non-overlapping
          windows, so the "no selection freedom" sentence is checkable rather
          than asserted;
      (c) wording strength, in both directions: the two families' plateau ranges
          mostly overlap, so neither manuscript may claim the plateau separates
          them -- and the sentence that refuses that claim must be present and
          bound to the measurement by macro.
    """
    import re as _re
    root = root or C.ROOT
    n_fail0 = len(FAIL)
    p = os.path.join(RES, "realgeo.json")
    if not os.path.exists(p):
        note("realgeo.json missing; third-level guard skipped")
        return
    d = json.load(open(p, encoding="utf-8"))
    s = d.get("summary") or {}
    src = d.get("source") or {}
    if not s:
        bad("realgeo.json has no summary, so the level-3 sentences cannot be "
            "checked against any measurement")
        return

    # -- (a) the design is the main protocol with the geometry swapped -------
    mism = []
    if abs(float(d.get("gamma", -1)) - C.GAMMA) > 1e-12:
        mism.append("gamma %s vs %s" % (d.get("gamma"), C.GAMMA))
    ft = [round(float(x), 3) for x in (d.get("f_test") or [])]
    if ft != [round(float(x), 3) for x in C.F_TEST]:
        mism.append("test grid has %d entries vs %d" % (len(ft), len(C.F_TEST)))
    vl = [round(float(x), 5) for x in (d.get("vmax_list") or [])]
    if vl != [round(float(x), 5) for x in C.VMAX_LIST]:
        mism.append("contrast arms %s vs %s" % (vl, C.VMAX_LIST))
    if mism:
        bad("level 3 is no longer run under the main protocol (%s), so 'only "
            "the geometry moves' is false; re-run code/make_realgeo.py"
            % "; ".join(mism))
    else:
        ok("level 3 is recorded under the current protocol: gamma=%.2f, %d test "
           "frequencies, %d contrast arms, all identical to code/common.py"
           % (C.GAMMA, len(C.F_TEST), len(C.VMAX_LIST)))

    # -- (b) the window rule -------------------------------------------------
    wins = src.get("windows") or {}
    names = ("west", "centre", "east")
    if len(wins) != 3 or any(n not in wins for n in names):
        bad("level 3 does not record exactly the three named windows (%s)"
            % sorted(wins))
    else:
        box = [wins[n] for n in names]
        wid = [b[1] - b[0] for b in box]
        gap = [box[i + 1][0] - box[i][1] for i in range(2)]
        if max(wid) - min(wid) > 1e-6:
            bad("the three windows are not equal width (%s), so the 'equally "
                "spaced rule' sentence is unsupported" % wid)
        elif max(gap) - min(gap) > 1e-6 or min(gap) < -1e-6:
            bad("the windows are not equally spaced without overlap: gaps %s"
                % gap)
        else:
            ok("three equal %.0f m windows, equally spaced with a %.0f m gap "
               "and zero overlap" % (wid[0], gap[0]))

    # -- (c) wording strength, both directions ------------------------------
    fam = s.get("family") or {}
    V = d.get("vmax_list") or []
    pairs, ov = [], 0
    for v in V:
        k = "%.2f" % float(v)
        r = (fam.get("real") or {}).get(k)
        y = (fam.get("synthetic") or {}).get(k)
        if not (r and y):
            continue
        hit = (r["plateau_lo"] <= y["plateau_hi"] and
               y["plateau_lo"] <= r["plateau_hi"])
        ov += int(hit)
        pairs.append((k, hit))
    n_arms = len(pairs)
    if n_arms == 0:
        bad("realgeo.json summary has no arm present in both families, so the "
            "plateau claim cannot be checked")
    elif ov * 2 < n_arms:
        bad("only %d of %d arms have overlapping plateau ranges, so the text "
            "cannot say the plateau fails to separate the two families"
            % (ov, n_arms))
    else:
        ok("the plateau ranges overlap or touch in %d of %d arms, so the "
           "text's refusal to claim a plateau separation is what the data "
           "supports" % (ov, n_arms))

    for gp in sorted(glob.glob(os.path.join(root, "paper", "*", "gen_macros.tex"))):
        lang = os.path.basename(os.path.dirname(gp))
        t = open(gp, encoding="utf-8").read()
        for name, val in (("RealPlateauOverlapArms", ov),):
            m = _re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % name, t)
            if not m:
                bad("[%s] \\%s is not generated, so the plateau sentence cannot "
                    "be bound to the measurement" % (lang, name))
            else:
                lit = m.group(1).strip()
                if lit.strip("-") == "" or int(float(lit)) != val:
                    bad("[%s] \\%s = %r while the data give %d"
                        % (lang, name, lit, val))

    # the mirror cross-check is a number the data-availability paragraph
    # quotes; require it to reach the text through a macro, not by hand
    cc = (src.get("cross_check") or {}).get("marmousi_mirror_300x298") or {}
    if "whole_sample_r" not in cc:
        bad("realgeo.json records no mirror cross-check, so the sentence about "
            "the independently hosted copy is unsupported")
    else:
        probe = "%.2f" % float(cc["whole_sample_r"])
        for gp in sorted(glob.glob(os.path.join(root, "paper", "*", "gen_macros.tex"))):
            lang = os.path.basename(os.path.dirname(gp))
            t = open(gp, encoding="utf-8").read()
            m = _re.search(r"\\newcommand\{\\RealMirrorR\}\{([^}]*)\}", t)
            if not m:
                bad("[%s] \\RealMirrorR is not generated" % lang)
            elif m.group(1).strip() != probe:
                bad("[%s] \\RealMirrorR = %r but the recorded correlation is %s"
                    % (lang, m.group(1).strip(), probe))

    # The three sentences that carry the level-3 verdict must actually be
    # there.  There are three rather than one because the verdict is not a
    # single scalar: the plateau fails to separate against the band-limited
    # control at 64^2, the structural part of the plateau difference survives
    # against a spectrum-matched control on both meshes, and the jump does not
    # order the two families at all.  A gate that required only the first
    # sentence would let a later edit drop the other two while everything
    # stayed green -- a check that tests the losing arm only.  The contract was
    # rewritten on 2026-10-22 when the spectrum-matched control replaced the
    # old "128^2 separation / order reversal" pair; keeping the old phrases
    # would have forced the text to say something the data no longer supports.
    want = {"en": ("does not separate", "\\RealPlateauOverlapArms"),
            "cn": ("并不分离", "\\RealPlateauOverlapArms")}
    want2 = {"en": ("higher on\nthe geological window", "\\GeoSurPlatDelta"),
             "cn": ("结构效应", "\\GeoSurPlatDevPct")}
    want3 = {"en": ("does not order the two families", "\\GeoJumpPCoarse"),
             "cn": ("从未把两族排出次序", "\\GeoJumpPCoarse")}
    for mp in sorted(glob.glob(os.path.join(root, "paper", "*", "main.tex"))):
        lang = os.path.basename(os.path.dirname(mp))
        if lang not in want:
            continue
        t = open(mp, encoding="utf-8").read()
        phrase, macro = want[lang]
        if phrase not in t:
            bad("[%s] the manuscript never states that the plateau does not "
                "separate the families; the third level is reported as if it "
                "supported a plateau claim" % lang)
        if macro not in t:
            bad("[%s] the non-separation sentence is not bound to "
                "\\%s" % (lang, macro.strip("\\")))
        for tag, tbl in (("the 128^2 separation", want2),
                         ("the order reversal", want3)):
            if lang not in tbl:
                continue
            ph, mac = tbl[lang]
            if ph not in t:
                bad("[%s] the manuscript never states %s, so the level-3 "
                    "verdict is reported in one half only" % (lang, tag))
            if mac not in t:
                bad("[%s] the sentence carrying %s is not bound to \\%s"
                    % (lang, tag, mac.strip("\\")))
        # a \label is not a citation: require the float to be *pointed at*, or
        # a manuscript could carry the label while never showing the reader the
        # figure that the level-3 sentences refer to. The level-3 table may
        # live in the supplementary material, in which case the main text
        # points at it by name, through the generated \\siTab map, instead of
        # by \ref, which would dangle in a separately compiled document.
        for need, why in (("fig:realgeo", "figure"),):
            if not _re.search(r"\\ref\s*\{[^}]*" + need + r"[^}]*\}", t):
                bad("[%s] the level-3 %s is never referenced from the text"
                    % (lang, why))
        if (not _re.search(r"\\ref\s*\{[^}]*tab:realgeo[^}]*\}", t)
                and "\\siTab{tab:si_realgeo}" not in t):
            bad("[%s] the level-3 table is never referenced from the text "
                "(either \\ref{tab:realgeo} or a \\siTab{tab:si_realgeo} "
                "pointer is required)" % lang)
    if len(FAIL) == n_fail0:
        ok("both manuscripts carry the non-separation sentence bound to the "
           "measurement, and both point at the level-3 float")


def plateau_normalised_guard(ev, root=None):
    """The own-plateau separation must hold seed by seed, not only on average.

    Class-level gate, over a failure mode no numeric check can see.  The claim
    the manuscripts draw from Figure 5c is an INEQUALITY between families on a
    quantity that is itself a ratio: at f = 5 the FNO is the only family sitting
    above its own out-of-band plateau.  Two things can go wrong and neither is
    visible in a mean curve.

    First, a mean curve can satisfy an inequality that individual seeds violate,
    so the separation would be an artefact of averaging.  We therefore test it
    on each seed separately and require unanimity.

    Second, and this is what produced the earlier "only the FNO rises past 1",
    dividing by a plateau is a ratio of two estimated quantities, so it has two
    defensible conventions: ratio of the seed-averaged curve to the seed-averaged
    plateau (what Figure 5c draws and what the \\Arch*NormFive macros report), or
    mean of the per-seed ratios (what a paired test needs).  They differ by a
    Jensen term.  The gate recomputes both and bounds the gap, so a claim read
    off panel (c) is only accepted when the two agree.

    The narrow scope matters: the earlier caption asserted "only the FNO rises
    past 1" without a frequency qualifier, which is FALSE.  Both U-Nets cross 1
    further out (at f = 12, where they are still climbing, so their f >= 6 mean
    underestimates their asymptote).  The inequality is only claimed at f = 5.
    """
    import re as _re
    root = root or C.ROOT
    fams = [("Fno", "main_fno_%s_"), ("Unet", "base_unet_%s_"),
            ("Don", "base_don_%s_"), ("Pmatch", "pmatch_unet_%s_")]
    viol = []

    def per_seed(pref, vm):
        """Per-seed eps(f=5) divided by that seed's own f>=6 plateau, keyed
        by seed index so a paired difference can align the two arms by seed.
        W1 extended the v=1.00 anchored arms to six seeds and left the
        architecture baselines at three, so a flat per-family array would
        mis-align (or fail to broadcast) exactly where the pairing matters;
        make_results_tex.py pairs on the common seed set for the same reason.
        """
        out = {}
        for n in keys(ev, pref):
            rows = ev[n]["sweep"]["%.2f" % vm]
            F = np.array([r["f"] for r in rows], float)
            M = np.array([r["mean"] for r in rows], float)
            out[int(re.search(r"_s(\d+)$", n).group(1))] = float(
                M[np.isclose(F, 5.0)][0] / M[F >= 6.0].mean())
        return out

    def _arr(mapping):
        return np.array([mapping[s] for s in sorted(mapping)], float)

    def per_seed_inband(pref, vm):
        """Per-seed in-band (f<=4) mean divided by that seed's own plateau."""
        names = keys(ev, pref)
        out = []
        for n in names:
            rows = ev[n]["sweep"]["%.2f" % vm]
            F = np.array([r["f"] for r in rows], float)
            M = np.array([r["mean"] for r in rows], float)
            out.append(float(M[F <= C.F_TRAIN[1] + 1e-9].mean()
                             / M[F >= 6.0].mean()))
        return np.array(out, float)

    for vmax, suf in [(0.25, ""), (1.00, "S")]:
        tag = "a%03d" % int(round(vmax * 100))
        per = {}
        for mac, pref in fams:
            if not keys(ev, pref % tag):
                continue
            per[mac] = per_seed(pref % tag, vmax)
        if "Fno" not in per or "Unet" not in per:
            continue
        fams_ok = True
        # The FNO must be above its own plateau at f=5 in EVERY seed ...
        if np.all(_arr(per["Fno"]) > 1.0):
            ok("v=%.2f: FNO is above its own plateau at f=5 in all %d seeds "
               "(min %.3f) -- the step reading survives per-seed checking"
               % (vmax, len(per["Fno"]), _arr(per["Fno"]).min()))
        else:
            bad("v=%.2f: FNO is at or below its own plateau at f=5 in some seed "
                "(%s) -- 'passes its own plateau' is false"
                % (vmax, ", ".join("%.3f" % v for v in _arr(per["Fno"]))))
            fams_ok = False
        # ... and both U-Nets must be below theirs in EVERY seed.
        for mac in ("Unet", "Pmatch"):
            if mac not in per:
                continue
            if np.all(_arr(per[mac]) < 1.0):
                ok("v=%.2f: %s is below its own plateau at f=5 in all seeds "
                   "(max %.3f) -- 'no step' survives per-seed checking"
                   % (vmax, mac, _arr(per[mac]).max()))
            else:
                bad("v=%.2f: %s reaches or exceeds its own plateau at f=5 in some "
                    "seed (%s) -- the shape contrast is unsupported"
                    % (vmax, mac, ", ".join("%.3f" % v
                                            for v in _arr(per[mac]))))
                fams_ok = False
        # Panel (c) is drawn at vmax = 0.25 only, and there the DeepONet is
        # below its plateau as well.  It is excluded from the claim because it
        # is the degenerate case: nearly all of its dynamic range is spent
        # inside the band, so its flatness carries no information about the edge.
        if vmax == 0.25 and "Don" in per:
            if np.all(_arr(per["Don"]) < 1.0):
                note("v=0.25: DeepONet also below its own plateau at f=5 "
                     "(max %.3f); it is excluded from the claim because its "
                     "in-band value already reaches %.3f of that plateau"
                     % (_arr(per["Don"]).max(),
                        float(np.mean(per_seed_inband(
                            "base_don_%s_" % tag, vmax)))))
            else:
                bad("v=0.25: DeepONet is at or above its own plateau at f=5 in "
                    "some seed (%s)" % ", ".join("%.3f" % v
                                                 for v in _arr(per["Don"])))
                fams_ok = False
        # The paired difference is taken over the seeds the two arms share,
        # which is the like-for-like subset and the only reading a paired
        # claim can rest on.
        common = sorted(set(per["Fno"]) & set(per["Unet"]))
        d = np.array([per["Fno"][s] - per["Unet"][s] for s in common], float)
        if len(d) and np.all(d > 0):
            ok("v=%.2f: paired FNO minus U-Net(base) at f=5 is positive in all "
               "%d shared seeds, mean %+.3f [%+.3f, %+.3f] (\\ArchFnoMinusUnet%s)"
               % (vmax, len(d), d.mean(), d.min(), d.max(), suf))
        else:
            bad("v=%.2f: paired FNO minus U-Net(base) at f=5 is not positive in "
                "every shared seed (%s)" % (vmax, ", ".join("%+.3f" % v
                                                            for v in d)))
            fams_ok = False
        if not fams_ok:
            viol.append(vmax)

    # --- macro <-> data consistency, under both averaging conventions --------
    checked = 0
    for gp in sorted(glob.glob(os.path.join(root, "paper", "*", "gen_macros.tex"))):
        lang = os.path.basename(os.path.dirname(gp))
        t = open(gp, encoding="utf-8").read()
        for vmax, suf in [(0.25, ""), (1.00, "S")]:
            tag = "a%03d" % int(round(vmax * 100))
            for mac, pref in fams:
                names = keys(ev, pref % tag)
                if not names:
                    continue
                rows0 = ev[names[0]]["sweep"]["%.2f" % vmax]
                F = np.array([r["f"] for r in rows0], float)
                A = np.array([[r["mean"] for r in
                               ev[n]["sweep"]["%.2f" % vmax]] for n in names],
                             float)
                M = A.mean(axis=0)
                i5 = int(np.argmin(np.abs(F - 5.0)))
                r1 = float(M[i5] / M[F >= 6.0].mean())          # figure convention
                r2 = float(np.mean(A[:, i5] / A[:, F >= 6.0].mean(axis=1)))
                rsd = float(np.std(A[:, i5] / A[:, F >= 6.0].mean(axis=1), ddof=1))
                for name, want in [("NormFive" + suf, r1),
                                   ("NormFiveSd" + suf, rsd)]:
                    m = _re.search(r"\\newcommand\{\\Arch%s%s\}\{([^}]*)\}"
                                   % (mac, name), t)
                    if not m:
                        bad("[%s] \\Arch%s%s is not generated" % (lang, mac, name))
                        continue
                    lit = m.group(1).strip()
                    if lit.strip("-") == "":
                        bad("[%s] \\Arch%s%s is still the placeholder %r"
                            % (lang, mac, name, lit))
                        continue
                    got = float(lit)
                    if abs(got - want) > 2e-3:
                        bad("[%s] \\Arch%s%s = %s but the measurement says %.3f"
                            % (lang, mac, name, lit, want))
                    else:
                        checked += 1
                if abs(r1 - r2) > 2e-3:
                    bad("[%s] \\Arch%sNormFive%s: the two averaging conventions "
                        "disagree (figure %.3f vs per-seed %.3f); a claim read "
                        "off Fig. 5c would be unsafe" % (lang, mac, suf, r1, r2))
    if checked:
        ok("%d normalised-plateau macros reproduce the per-seed recomputation "
           "under the figure convention" % checked)
    elif not viol:
        note("no architecture family available; plateau-normalised guard skipped")

    if viol:
        bad("the own-plateau separation fails seed-by-seed at v in {%s} -- the "
            "'FNO steps, the U-Nets do not' sentence must be re-checked"
            % ", ".join("%.2f" % v for v in viol))


def c1_guard(ev, root=None):
    """Section sec:c1: the conditioning confound, excluded by measurement.

    Five claims are made in prose and all five are re-derived here from the
    evaluation records, the caches and the propagator symbol -- deliberately
    NOT from results/c1_verdict.json, which is the file the table and the
    figure are generated from.  Reading the verdict file would make this gate
    agree with the manuscript by construction; a gate that cannot disagree
    with the manuscript is not a gate.

      1. the confound exists: at fixed gamma the attained norm spans more
         than a factor of two over the test grid;
      2. the compensation is real: gamma(f) holds the attained norm constant
         at every test frequency, inside the solver's gamma window, and the
         generated cache still passes the residual acceptance gate;
      3. the arms are paired: identical media (bitwise cache comparison) and
         identical protocol (run configs, modulo the regularisation itself);
      4. the cliff does not move: both arms' arm-matched onset interval is
         the same grid interval, for a whole window of rule constants;
      5. the step does not collapse: the band-edge ratio grows.
    """
    import gamma_comp as GC
    root = root or C.ROOT
    F = np.array(C.F_TEST, float)
    base, comp = keys(ev, "main_fno_a025_"), keys(ev, "gc_fno_")
    if not (base and comp):
        bad("C1 arms incomplete (baseline %d, compensated %d seeds) -- the "
            "discriminating experiment of Section sec:c1 is not fully "
            "evaluated" % (len(base), len(comp)))
        return
    # The baseline arm is one of the arms W2 extended to six seeds; the
    # compensated arm still has three.  Comparing the two six-seed means
    # against the three-seed mean would be an unpaired comparison wearing a
    # paired label, so the two arms are put on their shared seeds -- which is
    # the design the manuscript describes ("means over 3 seeds") and leaves
    # every C1 number unchanged.
    def _seed(n):
        m = re.search(r"_s(\d+)$", n)
        return int(m.group(1)) if m else -1
    bs = {_seed(n): n for n in base}
    cs = {_seed(n): n for n in comp}
    common = sorted(set(bs) & set(cs))
    if len(common) < 3:
        bad("C1 arms share only %d seeds (baseline %d, compensated %d) -- the "
            "paired design needs at least three"
            % (len(common), len(bs), len(cs)))
        return
    base = [bs[s] for s in common]
    comp = [cs[s] for s in common]
    ok("C1 arms compared on their %d shared seeds %s (baseline has %d, "
       "compensated %d)" % (len(common), common, len(bs), len(cs)))

    def seed_mean(names):
        A = np.array([[r["mean"] for r in ev[n]["sweep"]["0.25"]] for n in names])
        return A.mean(axis=0)

    mb, mc = seed_mean(base), seed_mean(comp)

    def onset(m, inb, k):
        thr = k * inb
        for i in range(len(F)):
            if m[i] > thr:
                return (float(F[i - 1]) if i else float(F[0]), float(F[i]))
        return None

    def inband(m):
        return float(np.mean([m[int(np.argmin(np.abs(F - x)))]
                              for x in (1.5, 2.5, 3.5)]))

    # --- (1) the confound exists -------------------------------------------
    att_b = np.array([GC.attained_norm(f, C.GAMMA, C.N) for f in F])
    spread = float(att_b.max() / att_b.min())
    if spread > 2.0:
        ok("conditioning confound present: attained norm spans %.2fx over "
           "the test grid at fixed gamma=%.2f (%.2f..%.2f)"
           % (spread, C.GAMMA, att_b.min(), att_b.max()))
    else:
        bad("attained-norm spread is only %.2f -- the confound Section "
            "sec:c1 claims to remove does not exist" % spread)

    # --- (2) the compensation is real ---------------------------------------
    gtab = C.load_json(os.path.join(C.RES, "gamma_comp.json"))
    target = float(gtab["target"])
    att_c = np.array([GC.attained_norm(f, GC.comp_gamma(f), C.N) for f in F])
    dev = float(np.max(np.abs(att_c - target)))
    if dev < 1e-9:
        ok("compensation holds the attained norm at %.4f at every test "
           "frequency (max dev %.1e)" % (target, dev))
    else:
        bad("compensated attained norm deviates from the target by %.1e -- "
            "'constant' is not supported" % dev)
    if target < att_b.max() - 1e-9:
        ok("compensated arm is the better-conditioned one everywhere "
           "(target %.2f vs baseline max %.2f, %.2fx lower)"
           % (target, att_b.max(), att_b.max() / target))
    else:
        bad("compensated target %.4f is not below the baseline's band-edge "
           "norm %.4f -- the 'better conditioned' sentence is false"
           % (target, att_b.max()))
    gl = min(GC.comp_gamma(float(f)) for f in F)
    gh = max(GC.comp_gamma(float(f)) for f in F)
    if 0.02 - 1e-12 <= gl and gh <= 0.20 + 1e-12:
        ok("gamma(f) on the test grid stays inside the solver's window "
           "[0.02, 0.20] (%.4f..%.4f)" % (gl, gh))
    else:
        bad("gamma(f) on the test grid leaves [0.02, 0.20] (%.4f..%.4f)"
            % (gl, gh))
    cfg = os.path.join(C.run_dir("gc_fno_s0", create=False), "config.json")
    if os.path.exists(cfg):
        mr = float(json.load(open(cfg, encoding="utf-8"))["cache_max_res"])
        if mr < 1e-9:
            ok("compensated cache max residual %.4e passes the 1e-9 "
               "acceptance gate" % mr)
        else:
            bad("compensated cache max residual %.2e fails the 1e-9 gate -- "
                "the solver-window sentence is unsupported" % mr)

    # --- (3) the arms are paired --------------------------------------------
    d_a = os.path.join(C.DATA, "train_a025")
    d_g = os.path.join(C.DATA, "train_gc")
    ch_a = sorted(os.listdir(d_a)) if os.path.isdir(d_a) else []
    ch_g = sorted(os.listdir(d_g)) if os.path.isdir(d_g) else []
    if ch_a != ch_g or not ch_a:
        bad("C1 caches are not the same chunk set (%d vs %d files)"
            % (len(ch_a), len(ch_g)))
    else:
        bad_key = None
        for fn in ch_a:
            x = np.load(os.path.join(d_a, fn))
            y = np.load(os.path.join(d_g, fn))
            for k in ("v", "f", "vmax", "theta"):
                if not np.array_equal(x[k], y[k]):
                    bad_key = (fn, k)
            x.close(); y.close()
            if bad_key:
                break
        if bad_key:
            bad("C1 media are NOT paired: %s differs in %s" % bad_key)
        else:
            ok("C1 media are paired: %d chunks identical in v, f, vmax, "
               "theta" % len(ch_a))
    cfg_a = os.path.join(C.run_dir("main_fno_a025_s0", create=False),
                         "config.json")
    if os.path.exists(cfg) and os.path.exists(cfg_a):
        ca = json.load(open(cfg_a, encoding="utf-8"))
        cg = json.load(open(cfg, encoding="utf-8"))
        skip = {"name", "data", "gamma_fn", "cache_max_res", "notes"}
        diff = {k for k in set(ca) | set(cg)
                if k not in skip and ca.get(k) != cg.get(k)}
        if diff:
            bad("C1 protocols differ beyond the regularisation: %s"
                % ", ".join(sorted(diff)))
        else:
            ok("C1 protocols match (same seed, budget, epochs; only the "
               "regularisation differs)")

    # --- (4) the cliff does not move ----------------------------------------
    on_b = onset(mb, inband(mb), 3.0)
    on_c = onset(mc, inband(mc), 3.0)
    if on_b is None or on_c is None:
        bad("onset not found for at least one C1 arm (%s / %s)"
            % (on_b, on_c))
    elif on_b == on_c:
        ok("cliff onset interval is identical in both arms: [%s, %s]"
           % (on_b[0], on_b[1]))
    else:
        bad("cliff onset moved under compensation: %s vs %s -- the "
            "conditioning reading would win" % (on_b, on_c))
    k_lo = max(mb[int(np.argmin(np.abs(F - 4.0)))] / inband(mb),
               mc[int(np.argmin(np.abs(F - 4.0)))] / inband(mc))
    k_hi = min(mb[int(np.argmin(np.abs(F - 4.25)))] / inband(mb),
               mc[int(np.argmin(np.abs(F - 4.25)))] / inband(mc))
    if k_lo < 3.0 <= k_hi:
        ok("both arms read the same onset for every rule constant k in "
           "(%.2f, %.2f]; k=3 is inside" % (k_lo, k_hi))
    else:
        bad("onset-rule window (%.2f, %.2f] does not contain k=3 -- the "
            "shared interval is an artefact of the constant" % (k_lo, k_hi))

    # --- (5) the step does not collapse --------------------------------------
    i4 = int(np.argmin(np.abs(F - 4.0)))
    i5 = int(np.argmin(np.abs(F - 5.0)))
    r_b, r_c = mb[i5] / mb[i4], mc[i5] / mc[i4]
    if r_c > r_b:
        ok("band-edge ratio grows under compensation (%.2f -> %.2f) -- the "
           "step does not collapse" % (r_b, r_c))
    else:
        bad("band-edge ratio does not grow (%.2f -> %.2f) -- 'grows rather "
            "than collapsing' is unsupported" % (r_b, r_c))


def disc_guard(ev, root=None):
    """Section sec:disc: the representation / loss / transfer discriminators.

    All claims are re-derived from the evaluation records (NOT from the
    c2/c3/c4 verdict JSONs, which generate Table tab:disc -- a gate that
    reads what the table reads cannot disagree with the table).  The energy
    probe claims are recomputed here from the checkpoints with a compact
    implementation written for this gate.

      1. every discriminator arm keeps its first onset exceedance at the
         band edge within one grid cell (f* in {4.0, 4.25} at k = 3);
      2. C2: the Fourier-feature arm is better in band, matched in params
         (< 1% delta), and its out-of-band plateau stays collapsed (> 0.9);
      3. C3: the in-band energy penalty arm reproduces the baseline (levels
         within 10% in band and 0.02 on the plateau);
      4. C3b: the out-of-band anchor recovers the energy (ratio > 1 at
         f = 12) while the alignment collapses (< 0.05) and the error gets
         worse (e(5) above the baseline's);
      5. C4: the warm-started TL arm matches the baseline in band (within
         5%) and out of band (e(5) within 10%), and the low-lr TL arm is
         worse in band.
    """
    import os
    import numpy as np

    def arm_curve(prefix):
        acc = {}
        for s in (0, 1, 2):
            p = os.path.join(C.RES, "eval_%s_s%d.json" % (prefix, s))
            if not os.path.exists(p):
                bad("disc: missing %s" % p)
                return None
            d = C.load_json(p)
            for r in d["sweep"]["0.25"]:
                acc.setdefault(round(float(r["f"]), 2), []).append(
                    float(r["mean"]))
        f = sorted(acc)
        return f, [float(np.mean(acc[x])) for x in f], acc

    def first_exceed(f, m, k=3.0, inb=None):
        if inb is None:
            inb = float(np.mean([m[f.index(x)] for x in (1.5, 2.5, 3.5)]))
        for x, e in zip(f, m):
            if e > k * inb:
                return x
        return None

    ARMS = {"baseline": "main_fno_a025", "ff": "ff_fno",
            "energy": "en_fno_a025", "energy_ob": "eb_fno_a025",
            "tl": "tl_fno_a025", "tl_warm": "tlw_fno_a025"}
    curves = {}
    for lab, pref in ARMS.items():
        c = arm_curve(pref)
        if c is None:
            return
        curves[lab] = c
    base_f, base_m, _ = curves["baseline"]
    base_inb = float(np.mean([base_m[base_f.index(x)] for x in (1.5, 2.5, 3.5)]))
    base_e5 = base_m[base_f.index(5.0)]
    base_plat = float(np.mean([base_m[base_f.index(x)]
                               for x in base_f if x >= 6.0]))
    # 1. onset at the band edge for every arm
    for lab in ARMS:
        f, m, _ = curves[lab]
        fx = first_exceed(f, m)
        if fx in (4.0, 4.25):
            ok("disc onset: %s first exceedance at f=%.2f (band edge within "
               "one grid cell)" % (lab, fx))
        else:
            bad("disc onset: %s first exceedance at f=%s" % (lab, fx))
    # 2. C2 representation
    f, m, _ = curves["ff"]
    inb = float(np.mean([m[f.index(x)] for x in (1.5, 2.5, 3.5)]))
    plat = float(np.mean([m[f.index(x)] for x in f if x >= 6.0]))
    if inb < base_inb:
        ok("disc C2: fourier in-band %.4f < baseline %.4f" % (inb, base_inb))
    else:
        bad("disc C2: fourier in-band %.4f not below baseline %.4f"
            % (inb, base_inb))
    if plat > 0.9:
        ok("disc C2: fourier plateau %.3f stays collapsed" % plat)
    else:
        bad("disc C2: fourier plateau %.3f not collapsed" % plat)
    pf = C.load_json(os.path.join(C.RES, "eval_ff_fno_s0.json"))["params"]
    pb = C.load_json(os.path.join(C.RES, "eval_main_fno_a025_s0.json"))["params"]
    if abs(pf - pb) / pb < 0.01:
        ok("disc C2: capacity matched (%d vs %d params, %.3f%%)"
           % (pf, pb, 100 * (pf - pb) / pb))
    else:
        bad("disc C2: capacity mismatch %d vs %d" % (pf, pb))
    # 3. C3 in-band penalty reproduces the baseline
    f, m, _ = curves["energy"]
    inb_e = float(np.mean([m[f.index(x)] for x in (1.5, 2.5, 3.5)]))
    plat_e = float(np.mean([m[f.index(x)] for x in f if x >= 6.0]))
    if abs(inb_e - base_inb) / base_inb < 0.1 and abs(plat_e - base_plat) < 0.02:
        ok("disc C3: in-band penalty arm reproduces baseline "
           "(in-band %.4f vs %.4f, plateau %.4f vs %.4f)"
           % (inb_e, base_inb, plat_e, base_plat))
    else:
        bad("disc C3: in-band penalty arm deviates from baseline "
            "(in-band %.4f vs %.4f, plateau %.4f vs %.4f)"
            % (inb_e, base_inb, plat_e, base_plat))
    # 4. C3b out-of-band anchor: probe recomputed from checkpoints
    try:
        import torch
        import data as D
        import physics as P
        from evaluate import load_model
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        probe = {}
        for lab, pref in (("baseline", "main_fno_a025"),
                          ("energy_ob", "eb_fno_a025")):
            ms = [load_model(os.path.join(C.RUNS, "%s_s%d" % (pref, s),
                                          "checkpoint.pt"), dev)[0]
                  for s in (0, 1, 2)]
            d = D.sample_batch(256, 90005, dev, vmax_lo=0.25, vmax_hi=0.25,
                               f_fixed=12.0)
            inc = P.incident_plane_wave(C.N, d["k"], d["theta"], dev).to(
                torch.complex64)
            x = D.build_inputs(d["v"].float(), inc,
                               torch.full((256,), 12.0, device=dev))
            rt, at = [], []
            with torch.no_grad():
                for mdl in ms:
                    o = mdl(x)
                    u = torch.complex(o[:, 0], o[:, 1])
                    ut = d["psi"] - inc
                    nu = torch.linalg.norm(u.reshape(256, -1), dim=1)
                    nt = torch.linalg.norm(ut.reshape(256, -1), dim=1)
                    dot = torch.sum(u.reshape(256, -1) *
                                    ut.reshape(256, -1).conj(), dim=1)
                    rt.append(float((nu / nt.clamp_min(1e-12)).mean()))
                    at.append(float((torch.abs(dot) /
                                     (nu * nt).clamp_min(1e-12)).mean()))
            probe[lab] = (float(np.mean(rt)), float(np.mean(at)))
        rb, ab = probe["baseline"]
        ro, ao = probe["energy_ob"]
        if ro > 1.0 and ao < 0.05 and rb < 0.3:
            ok("disc C3b probe at f=12: baseline ratio %.3f/align %.3f -> "
               "anchored ratio %.3f/align %.3f (energy recovered, phase lost)"
               % (rb, ab, ro, ao))
        else:
            bad("disc C3b probe at f=12: baseline %.3f/%.3f anchored %.3f/%.3f"
                % (rb, ab, ro, ao))
    except Exception as exc:                      # noqa: BLE001 - gate must report
        bad("disc C3b probe failed to recompute: %s" % exc)
    f, m, _ = curves["energy_ob"]
    e5_ob = m[f.index(5.0)]
    if e5_ob > base_e5:
        ok("disc C3b: e(5) %.3f worse than baseline %.3f (error not recovered)"
           % (e5_ob, base_e5))
    else:
        bad("disc C3b: e(5) %.3f not worse than baseline %.3f" % (e5_ob, base_e5))
    # 5. C4 transfer
    f, m, _ = curves["tl_warm"]
    inb_w = float(np.mean([m[f.index(x)] for x in (1.5, 2.5, 3.5)]))
    e5_w = m[f.index(5.0)]
    if abs(inb_w / base_inb - 1) < 0.05:
        ok("disc C4: warm-start TL in-band %.4f matches baseline %.4f"
           % (inb_w, base_inb))
    else:
        bad("disc C4: warm-start TL in-band %.4f vs baseline %.4f"
            % (inb_w, base_inb))
    if abs(e5_w / base_e5 - 1) < 0.10:
        ok("disc C4: warm-start TL e(5) %.3f matches baseline %.3f"
           % (e5_w, base_e5))
    else:
        bad("disc C4: warm-start TL e(5) %.3f vs baseline %.3f"
            % (e5_w, base_e5))
    f, m, _ = curves["tl"]
    inb_t = float(np.mean([m[f.index(x)] for x in (1.5, 2.5, 3.5)]))
    if inb_t > base_inb:
        ok("disc C4: low-lr TL in-band %.4f worse than baseline %.4f"
           % (inb_t, base_inb))
    else:
        bad("disc C4: low-lr TL in-band %.4f not worse than baseline %.4f"
            % (inb_t, base_inb))


def anchor_guard(ev, res=None):
    """Section sec:anchor: the analytic anchor and its flatness claim.

    The table and the macros are generated from results/anchor_baseline.json,
    so this gate deliberately does NOT trust that file for anything it can
    recompute.  Five claims are re-derived here from raw material:

      1. protocol parity: the anchor record declares the same sample count and
         seed scheme as the trained arms' evaluation records;
      2. fit stability: refitting (a, b) on a DIFFERENT prefix of the same
         training cache (samples N_FIT..2 N_FIT) still yields an out-of-band
         error well below the trivial baseline -- the result does not hinge on
         the particular fit prefix;
      3. paired superiority: on freshly drawn media at the protocol's own
         seeds, the anchor beats the trivial baseline sample by sample at
         f = 5 and f = 12, and its phase alignment with the true correction
         stays high at both;
      4. flatness versus step: the anchor's band-edge ratio stays near one at
         every contrast while the trained operator's exceeds five;
      5. plateau ordering: the anchor's plateau is below the trained
         operator's at every contrast tested.
    """
    import torch
    import data as D
    import physics as P
    res = res or RES
    p = os.path.join(res, "anchor_baseline.json")
    if not os.path.exists(p):
        bad("anchor: %s missing -- run code/anchor_baseline.py" % p)
        return
    d = json.load(open(p, encoding="utf-8"))

    # 1. protocol parity with the trained arms' evaluation records
    if int(d["n"]) != 200 or d.get("gamma") != C.GAMMA:
        bad("anchor record protocol mismatch: n=%s gamma=%s (expected "
            "n=200, gamma=%s)" % (d["n"], d.get("gamma"), C.GAMMA))
    else:
        ok("anchor record declares the trained arms' protocol (n=200, "
           "gamma=%s)" % C.GAMMA)
    if sorted(float(k) for k in d["arms"]) != [float(v) for v in C.VMAX_LIST]:
        bad("anchor record contrasts %s != the manuscript's %s"
            % (sorted(d["arms"]), C.VMAX_LIST))
        return

    # 2. fit stability on a different cache prefix
    N_FIT = int(d["n_fit"])
    cache = {0.05: "train_a005", 0.10: "train_a010", 0.25: "train_a025",
             0.50: "train_a050", 1.00: "train_a100"}

    def cache_batch(cache_name, lo, hi):
        zs = [np.load(q) for q in sorted(glob.glob(
            os.path.join(C.DATA, cache_name, "*.npz")))]
        need = int(np.ceil(hi / zs[0]["v"].shape[0]))
        v = torch.from_numpy(np.concatenate(
            [z["v"] for z in zs[:need]])).float()[lo:hi]
        psi = torch.from_numpy(np.concatenate(
            [z["psi"] for z in zs[:need]]))[lo:hi]
        k = torch.from_numpy(np.concatenate(
            [z["k"] for z in zs[:need]]))[lo:hi]
        th = torch.from_numpy(np.concatenate(
            [z["theta"] for z in zs[:need]]))[lo:hi]
        pi = P.incident_plane_wave(C.N, k, th, "cpu").to(torch.complex128)
        return v, k, th, (psi - pi).to(torch.complex128)

    def anchor_err(v, k, th, u, a, b):
        b1, b2 = D.anchor_terms(v, k, th)
        num = torch.linalg.norm((u - a * b1 - b * b2).reshape(u.shape[0], -1),
                                dim=1)
        den = torch.linalg.norm(u.reshape(u.shape[0], -1), dim=1).clamp_min(1e-12)
        return (num / den).numpy()

    v, k, th, u = cache_batch(cache[0.25], N_FIT, 2 * N_FIT)
    b1, b2 = D.anchor_terms(v, k, th)
    y = u.reshape(u.shape[0], -1)
    m = torch.stack([b1.reshape(u.shape[0], -1), b2.reshape(u.shape[0], -1)],
                    dim=1)
    a2, b2c = torch.linalg.solve(
        torch.einsum("njd,nkd->jk", m.conj(), m),
        torch.einsum("njd,nd->j", m.conj(), y))
    errs = []
    for f in (6.0, 10.0):
        dd = D.sample_batch(40, 60000, "cpu", vmax_lo=0.25, vmax_hi=0.25,
                            f_fixed=f)
        pi = P.incident_plane_wave(C.N, dd["k"], dd["theta"], "cpu").to(
            torch.complex128)
        uu = (dd["psi"] - pi).to(torch.complex128)
        e = anchor_err(dd["v"], dd["k"], dd["theta"], uu, a2, b2c)
        nu = torch.linalg.norm(uu.reshape(40, -1), dim=1)
        npsi = torch.linalg.norm(dd["psi"].reshape(40, -1), dim=1)
        errs.append((float(e.mean()), float((nu / npsi).mean())))
    worst = max(e / t for e, t in errs)
    if worst > 0.7:
        bad("anchor refit on cache samples %d..%d loses the out-of-band "
            "advantage (worst anchor/trivial ratio %.2f > 0.7) -- the result "
            "hinges on the fit prefix" % (N_FIT, 2 * N_FIT, worst))
    else:
        ok("anchor refit on a different cache prefix keeps the out-of-band "
           "advantage (worst anchor/trivial ratio %.2f)" % worst)

    # 3. paired superiority and phase alignment on fresh media
    for f in (5.0, 12.0):
        dd = D.sample_batch(40, 60000, "cpu", vmax_lo=0.25, vmax_hi=0.25,
                            f_fixed=f)
        pi = P.incident_plane_wave(C.N, dd["k"], dd["theta"], "cpu").to(
            torch.complex128)
        uu = (dd["psi"] - pi).to(torch.complex128)
        c1, c2 = D.anchor_terms(dd["v"], dd["k"], dd["theta"])
        pred = a2 * c1 + b2c * c2
        nu = torch.linalg.norm(uu.reshape(40, -1), dim=1)
        npsi = torch.linalg.norm(dd["psi"].reshape(40, -1), dim=1)
        e_anch = (torch.linalg.norm((uu - pred).reshape(40, -1), dim=1) / nu)
        e_triv = nu / torch.linalg.norm(
            dd["psi"].reshape(40, -1), dim=1).clamp_min(1e-12)
        dot = torch.sum(pred.reshape(40, -1) * uu.reshape(40, -1).conj(), dim=1)
        al = float((torch.abs(dot) / (
            torch.linalg.norm(pred.reshape(40, -1), dim=1) * nu).clamp_min(1e-12)
        ).mean())
        paired = float((e_anch - e_triv).mean())
        if paired >= 0:
            bad("anchor does not beat the trivial baseline sample-by-sample "
                "at f=%.0f (mean paired difference %+.3f)" % (f, paired))
        elif al < 0.7:
            bad("anchor phase alignment at f=%.0f is %.2f < 0.7 -- the "
                "phase-carrier story fails on fresh media" % (f, al))
        else:
            ok("f=%.0f: anchor beats trivial sample-by-sample (paired %+.3f) "
               "with alignment %.2f" % (f, paired, al))

    # 4. flatness versus step, every contrast
    flat, steps = [], []
    for vmax in C.VMAX_LIST:
        a = d["arms"]["%.2f" % vmax]
        flat.append(a["e5"] / a["e4"])
        names = [n for n in ev if n.startswith("main_fno_a%03d_"
                                             % int(round(vmax * 100)))]
        if not names:
            bad("no trained records for v=%.2f -- cannot state the contrast "
                "between the anchor and the trained operator" % vmax)
            continue
        A = np.array([[r["mean"] for r in ev[n]["sweep"]["%.2f" % vmax]]
                      for n in names]).mean(axis=0)
        F = np.array([r["f"] for r in ev[names[0]]["sweep"]["%.2f" % vmax]])
        e4 = float(A[np.isclose(F, 4.0)][0])
        e5 = float(A[np.isclose(F, 5.0)][0])
        steps.append(e5 / e4)
        # 5. plateau ordering
        plat_a = float(np.mean([r["mean"] for r in d["arms"]["%.2f" % vmax]
                                ["sweep"] if r["f"] >= 6.0]))
        plat_f = float(np.mean(A[np.array(F) >= 6]))
        if plat_a >= plat_f:
            bad("anchor plateau %.3f is NOT below the trained operator's %.3f "
                "at v=%.2f" % (plat_a, plat_f, vmax))
        # the complementarity claim has two directions and both are checked:
        # this one says the trained operator wins IN band, and it was the
        # direction without a gate when the win-factor macros first shipped --
        # one of them was computed upside down and printed 'the trained
        # operator wins, by 0.3x'
        ib_a = float(np.mean([r["mean"] for r in d["arms"]["%.2f" % vmax]
                              ["sweep"] if r["f"] <= 4.0]))
        ib_f = float(np.mean(A[np.array(F) <= 4.0]))
        if ib_f >= ib_a:
            bad("trained operator does NOT win in band at v=%.2f "
                "(FNO %.3f vs anchor %.3f) -- the complementarity prose is "
                "wrong" % (vmax, ib_f, ib_a))
    if flat and max(flat) > 1.2:
        bad("anchor band-edge ratio exceeds 1.2 (max %.2f) -- 'no step' is "
            "overstated" % max(flat))
    elif flat:
        ok("anchor band-edge ratio stays flat at every contrast (max %.2f) "
           "against the trained operator's %.2f--%.2f"
           % (max(flat), min(steps), max(steps)))
    if all(d["arms"]["%.2f" % v]["plateau"] <
           d["arms"]["%.2f" % v]["trivial_plateau"] for v in C.VMAX_LIST):
        ok("anchor plateau below the trivial baseline at every contrast")


def deep_guard(ev, res=None):
    """The deep analytic head: truncation, contraction, selection, transfer.

    The scan, the contraction measurement and the order-selection experiment
    are three separate zero-GPU instruments, and each could be wrong in a way
    that flatters the conclusion.  This gate re-derives from their records the
    eight claims the prose actually makes, rather than re-reading the macros:

      1. protocol parity: all three instruments ran at n=200 / gamma / the
         manuscript's contrasts and frequency grid;
      2. monotone decay where it should hold: at every contrast whose
         measured contraction factor is below one, the plateau falls with M;
      3. a turning point where it should appear: at the contrast whose
         contraction factor is largest, some M is worse than M-1, so the
         "deeper is better" reading is not available;
      4. the refusal is real: any order marked not-fitted has a Gram
         condition number past the declared ceiling;
      5. no catastrophic cancellation: every accepted fit's coefficients stay
         moderate, so a conditioning-warning fit is not silently reporting
         round-off as accuracy;
      6. in-band selection equals out-of-band optimum at every contrast;
      7. the frequency drift of the contraction factor is materially smaller
         than the cliff -- the number that licenses the corrected wording
         (rho is NOT frequency-independent, but its smooth drift cannot make
         a band-edge step);
      8. the head at the selected depth beats the trained operator's plateau
         at every contrast.

    Claim 7 is the one that exists because of a correction.  The manuscript
    originally said the residual was "near-frequency-independent"; measuring
    rho showed a 2.4x drift, so the prose was weakened to "smooth, and three
    times smaller than the cliff".  A gate that only checked the stronger
    claim would have gone on failing silently.
    """
    res = res or RES
    files = {"order": "anchor_higher_order.json",
             "rho": "anchor_contraction.json",
             "sel": "anchor_order_selection.json"}
    d = {}
    for k, fn in files.items():
        p = os.path.join(res, fn)
        if not os.path.exists(p):
            bad("deep head: %s missing -- run code/%s.py" % (fn, fn[:-5]))
            return
        d[k] = json.load(open(p, encoding="utf-8"))

    # 1. protocol parity
    if int(d["sel"]["n_eval"]) != 200 or d["sel"].get("gamma") != C.GAMMA:
        bad("deep head: selection record protocol mismatch n=%s gamma=%s"
            % (d["sel"]["n_eval"], d["sel"].get("gamma")))
    else:
        ok("deep head: all three instruments on the manuscript's protocol "
           "(n=200, gamma=%s)" % C.GAMMA)
    if sorted(float(k) for k in d["sel"]["arms"]) != [float(v)
                                                      for v in C.VMAX_LIST]:
        bad("deep head: contrasts %s != %s"
            % (sorted(d["sel"]["arms"]), C.VMAX_LIST))
        return

    def plateau(rec, M):
        o = rec["orders"].get(str(M), {})
        return o.get("plateau") if o.get("fitted") else None

    # 2. monotone decay where rho < 1
    for vk, rec in d["order"]["arms"].items():
        rho = d["rho"]["arms"][vk]["rho_level"]
        if rho >= 1.0:
            continue
        seq = [(M, plateau(rec, M)) for M in range(1, 9)]
        seq = [(M, p) for M, p in seq if p is not None]
        if len(seq) < 2:
            continue
        worst = max((seq[i + 1][1] / seq[i][1]) for i in range(len(seq) - 1))
        if worst < 1.0:
            ok("deep head: v=%s (rho=%.3f < 1) plateau falls monotonically "
               "with M, worst ratio %.2f" % (vk, rho, worst))
        else:
            bad("deep head: v=%s has rho=%.3f < 1 but the plateau rises "
                "-- the truncation reading is wrong" % (vk, rho))

    # 3. a turning point where rho is largest
    vk = max(d["rho"]["arms"], key=lambda k: d["rho"]["arms"][k]["rho_level"])
    rec = d["order"]["arms"][vk]
    seq = [(M, plateau(rec, M)) for M in range(1, 9)]
    seq = [(M, p) for M, p in seq if p is not None]
    turns = [i for i in range(len(seq) - 1) if seq[i + 1][1] > seq[i][1]]
    if turns:
        i = turns[0]
        ok("deep head: v=%s (rho=%.3f) turns at M=%d (%.3f -> %.3f) -- "
           "deeper is not monotonically better"
           % (vk, d["rho"]["arms"][vk]["rho_level"], seq[i][0], seq[i][1],
              seq[i + 1][1]))
    else:
        bad("deep head: v=%s has rho=%.3f > 1 but NO turning point -- either "
            "the scan is too short or the optimal-truncation story is "
            "unsupported" % (vk, d["rho"]["arms"][vk]["rho_level"]))

    # 4. the refusal is real
    ceil = float(d["order"].get("cond_ceiling", 1e12))
    refused = [o for rec in d["order"]["arms"].values()
               for M in range(1, 9)
               for o in [rec["orders"].get(str(M), {})]
               if o.get("fitted") is False]
    if refused:
        if all(o.get("cond", 0) > ceil for o in refused):
            ok("deep head: %d refused fit(s), all past the declared "
               "conditioning ceiling %.0e (largest cond %.1e)"
               % (len(refused), ceil, max(o["cond"] for o in refused)))
        else:
            bad("deep head: a fit was refused with cond below the ceiling -- "
                "the refusal criterion is not the one being enforced")
    else:
        note("deep head: no fit was refused at any (contrast, order)")

    # 5. no catastrophic cancellation
    worst_c = 0.0
    for rec in d["order"]["arms"].values():
        for M in range(1, 9):
            o = rec["orders"].get(str(M), {})
            if not o.get("fitted"):
                continue
            for re_, im in o.get("coeffs", []):
                worst_c = max(worst_c, abs(re_), abs(im))
    if worst_c < 100.0:
        ok("deep head: largest fitted |c_j| is %.2f -- no catastrophic "
           "cancellation in the ill-conditioned basis" % worst_c)
    else:
        bad("deep head: fitted |c_j| reaches %.1f -- the fit may be "
            "cancellation, not accuracy" % worst_c)

    # 6. in-band selection equals out-of-band optimum
    mism = [vk for vk, a in d["sel"]["arms"].items()
            if a.get("M_sel") != a.get("M_oracle")]
    if not mism:
        ok("deep head: in-band validation selects the out-of-band-optimal "
           "order at all %d contrasts" % len(d["sel"]["arms"]))
    else:
        bad("deep head: in-band selection misses the out-of-band optimum at "
            "v=%s -- the zero-shot order-selection claim fails"
            % ", ".join(mism))

    # 7. frequency drift is much smaller than the cliff
    st = arm_stats(ev)
    spread = max(d["rho"]["arms"][vk]["rho_spread"] for vk in d["rho"]["arms"])
    cliffs = [s["ratio"] for s in st.values() if np.isfinite(s["ratio"])]
    if cliffs:
        cliff = float(np.mean(cliffs))
        if spread < cliff / 2.0:
            ok("deep head: rho drifts %.2fx over f while the trained cliff is "
               "%.2fx -- the smooth drift cannot make the step"
               % (spread, cliff))
        else:
            bad("deep head: rho drift %.2fx is not small against the cliff "
                "%.2fx; the corrected wording is not licensed"
                % (spread, cliff))

    # 8. the head beats the trained plateau everywhere
    worse = []
    for vmax in C.VMAX_LIST:
        vk = "%.2f" % vmax
        if vmax not in st or vk not in d["sel"]["arms"]:
            continue
        if d["sel"]["arms"][vk]["plateau_sel"] >= st[vmax]["plateau"]:
            worse.append(vk)
    if worse:
        bad("deep head: plateau at the selected depth is not below the "
            "trained operator's at v=%s" % ", ".join(worse))
    else:
        ok("deep head: plateau at the selected depth is below the trained "
           "operator's at every contrast")

    # the learned-residual arms, read against their pre-registered criteria
    tag = "a%03d" % int(round(0.25 * 100))
    base = keys(ev, "main_fno_%s_" % tag)
    b1 = keys(ev, "ba_fno_%s_s" % tag)
    b2 = [n for n in keys(ev, "ba_fno_%s_" % tag) if "_m" in n]
    if base and b1:
        F, M1, _, _ = curve(ev, b1, 0.25)
        p1 = float(M1[F >= 6].mean())
        floor2 = d["sel"]["arms"]["0.25"]["plateau_m2"]
        if p1 > floor2:
            ok("deep head: BA1 (M=2 head + learned residual) plateau %.3f is "
               "WORSE than its own M=2 analytic floor %.3f -- reported as a "
               "negative result, the residual contaminates out of band"
               % (p1, floor2))
        else:
            bad("deep head: BA1 plateau %.3f beats its floor %.3f; the "
                "negative-result wording in the prose is stale" % (p1, floor2))
    if base and b2:
        F, M2, _, _ = curve(ev, b2, 0.25)
        ib2, _ = inband(ev, b2, 0.25)
        p2 = float(M2[F >= 6].mean())
        Fb, Mb, _, _ = curve(ev, base, 0.25)
        pbase = float(Mb[Fb >= 6].mean())
        ib_base, _ = inband(ev, base, 0.25)
        if ib2 <= ib_base and p2 <= 0.5 * pbase:
            note("deep head: BA2 PASS by pre-registration (in-band %.4f <= "
                 "%.4f and plateau %.3f <= half of %.3f)"
                 % (ib2, ib_base, p2, pbase))
        elif p2 <= pbase:
            note("deep head: BA2 PARTIAL by pre-registration (plateau %.3f "
                 "<= %.3f but in-band %.4f vs %.4f)"
                 % (p2, pbase, ib2, ib_base))
        else:
            note("deep head: BA2 FAIL by pre-registration (plateau %.3f > "
                 "%.3f)" % (p2, pbase))
    # The high-contrast arms are reported as FAILURES; if either ever passes,
    # the prose that calls them failures is stale and must be rewritten rather
    # than quietly left behind.  Both contrasts are checked because the
    # boundary sits between 0.25 and 0.50 and was located, not assumed.
    for v in (0.50, 1.00):
        tag = "a%03d" % int(round(v * 100))
        base = keys(ev, "main_fno_%s_" % tag)
        arms = [n for n in keys(ev, "ba_fno_%s_" % tag) if "_m" in n]
        if not (base and arms):
            continue
        F, M, _, _ = curve(ev, arms, v)
        p = float(M[F >= 6].mean())
        Fb, Mb, _, _ = curve(ev, base, v)
        pbase = float(Mb[Fb >= 6].mean())
        head = d["sel"]["arms"]["%.2f" % v]["plateau_sel"]
        if p > pbase:
            ok("deep head: the v=%.2f arm FAILS as reported (plateau %.3f > "
               "trained %.3f > head alone %.3f) -- the failure prose is "
               "current" % (v, p, pbase, head))
        else:
            bad("deep head: the v=%.2f arm now reaches plateau %.3f against "
                "the trained operator's %.3f -- the manuscript still calls it "
                "a failure, which is stale" % (v, p, pbase))


def main():
    global RES
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=None,
                    help="folder holding eval_*.json (default results/)")
    a = ap.parse_args()
    if a.res:
        RES = a.res
    ev = load()
    if not ev:
        print("no eval files -- run make_paper.py --steps eval first")
        return 1
    st = arm_stats(ev)
    have = sorted(st)
    if not have:
        # Survivable-but-fatal condition: every eval file on disk predates the
        # current protocol (f_test / n / gamma mismatch), so nothing could be
        # compared and the downstream max() would raise instead of reporting.
        bad("no contrast arm in %s matches the current test grid %s -- the "
            "evaluations are stale; run make_paper.py --steps eval" %
            (RES, C.F_TEST))
        return 1
    print("=== contrast arms present: %s ===" % ", ".join("%.2f" % v for v in have))
    for v in have:
        s = st[v]
        print("      v=%.2f  seeds=%d  in=%.3f (band range %.3f, seed sd %.3f)  "
              "e4=%.3f  e5=%.3f  plateau=%.3f  r=%.2f"
              % (v, s["n"], s["inband"], s["inband_freq_range"], s["seed_spread"],
                 s["e4"], s["e5"], s["plateau"], s["ratio"]))

    # --- (i) "for f<=4 the error is approximately independent of frequency" --
    worst = max(have, key=lambda v: st[v]["inband_freq_range"])
    s = st[worst]
    rel = s["inband_freq_range"] / max(s["inband"], 1e-12)
    drift = []
    for v in have:
        sel = st[v]["M"][st[v]["F"] <= 4.0]
        if sel.size and sel.min() > 0:
            drift.append(float(sel.max() / sel.min()))
    dmax = max(drift) if drift else float("nan")
    note("largest within-band spread: %.0f%% of the in-band mean (v=%.2f); "
         "largest max/min drift factor is %.2f" % (100 * rel, worst, dmax))
    inc0 = [v for v in have if st[v]["e4"] < 0.5 * st[v]["plateau"]]
    # "Flat inside the band" is asserted relative to the cliff: the within-band
    # variation must be smaller than the jump that happens at the band edge.
    if inc0 and np.isfinite(dmax):
        rmin = min(st[v]["ratio"] for v in inc0)
        if dmax < rmin:
            ok("in-band drift factor %.2f is smaller than the smallest cliff "
               "ratio %.2f -- the 'flat inside the band' claim holds "
               "(\\InBandDrift)" % (dmax, rmin))
        else:
            bad("in-band drift factor %.2f is NOT smaller than the smallest "
                "cliff ratio %.2f -- 'flat inside the band' is false" % (dmax, rmin))
    else:
        bad("cannot evaluate the in-band flatness claim (drift=%s, arms=%s)"
            % (dmax, inc0))

    # --- (ii) monotone rise of the in-band floor with contrast ---------------
    vs = sorted(have)
    mono = all(st[vs[i]]["inband"] <= st[vs[i + 1]]["inband"] + 1e-9
               for i in range(len(vs) - 1))
    if mono:
        ok("in-band error is monotone in contrast (%.3f -> %.3f)"
           % (st[vs[0]]["inband"], st[vs[-1]]["inband"]))
    else:
        bad("in-band error is NOT monotone in contrast: "
            + ", ".join("%.2f:%.3f" % (v, st[v]["inband"]) for v in vs))

    # --- "rises from InBandLo at 0.05 to InBandWorst at 1.00" ---------------
    if 1.00 in st:
        arg = max(have, key=lambda v: st[v]["inband"])
        if abs(arg - 1.00) < 1e-9:
            ok("the worst in-band arm really is v=1.00 (%.3f)"
               % st[1.00]["inband"])
        else:
            bad("\\InBandWorst is quoted at v=1.00 but the maximum is at "
                "v=%.2f (%.3f vs %.3f at 1.00)"
                % (arg, st[arg]["inband"], st[1.00]["inband"]))
        argp = max(have, key=lambda v: st[v]["plateau"])
        if abs(argp - 1.00) < 1e-9:
            ok("the largest plateau is also at v=1.00 (%.3f)" % st[1.00]["plateau"])
        else:
            bad("the largest plateau is at v=%.2f, not v=1.00" % argp)

    # --- (ii) cliff ratio does not vary systematically with contrast -------
    inc = [v for v in have if st[v]["e4"] < 0.5 * st[v]["plateau"]]
    if inc:
        rs = [st[v]["ratio"] for v in inc]
        spread = max(rs) / min(rs)
        note("identifiable ratios: " +
             ", ".join("v=%.2f r=%.2f" % (v, st[v]["ratio"]) for v in inc))
        if spread <= 1.6:
            ok("the identifiable cliff ratio does not vary systematically "
               "with contrast (spread %.2f over %d arms)" % (spread, len(inc)))
        else:
            bad("cliff ratio spread %.2f over %s is systematic -- the text says "
                "it does not vary with contrast"
                % (spread, ", ".join("%.2f" % v for v in inc)))
        note("inclusion rule e4 < plateau/2 keeps %s"
             % ", ".join("%.2f" % v for v in inc))

    # --- (ii) absolute jump persists at every contrast ----------------------
    badj = [v for v in vs if st[v]["e5"] <= st[v]["e4"]]
    if not badj:
        ok("eps(5) > eps(4) for every arm (rounded jump %.3f--%.3f)"
           % (min(st[v]["e5"] - st[v]["e4"] for v in vs),
              max(st[v]["e5"] - st[v]["e4"] for v in vs)))
    else:
        bad("eps(5) <= eps(4) at v=%s" % badj)

    # --- (iii) a plateau, not a continued rise -----------------------------
    badp = [v for v in vs if st[v]["plateau"] < 2.0 * st[v]["inband"]]
    if not badp:
        ok("every arm plateaus: min plateau/in-band ratio is %.1f"
           % min(st[v]["plateau"] / st[v]["inband"] for v in vs))
    else:
        bad("no cliff at v=%s (plateau < 2x in-band)" % badp)

    # --- (iii) "the same plateau appears in the median and the 90th pct" ----
    for v in vs:
        tag = "a%03d" % int(round(v * 100))
        nm = keys(ev, "main_fno_%s_" % tag)
        if not nm:
            continue
        rows = ev[nm[0]]["sweep"]["%.2f" % v]
        for lab in ("median", "q90"):
            y = np.array([r[lab] for r in rows if r["f"] >= 6.0], dtype=float)
            if y.size < 2 or y.mean() <= 0:
                continue
            cv = float(y.std() / y.mean())
            if cv <= 0.25:
                ok("v=%.2f: the %s also plateaus for f>=6 (CV %.2f)"
                   % (v, lab, cv))
            else:
                bad("v=%.2f: the %s does not plateau for f>=6 (CV %.2f) -- the "
                    "'same plateau in the median and the 90th percentile' "
                    "sentence is wrong" % (v, lab, cv))

    # --- "no power law is identifiable": the exponent moves with the window --
    al = []
    for v in vs:
        F, M = st[v]["F"], st[v]["M"]
        m = F > 4.0
        x, y = np.log(F[m] / 4.0), np.log(M[m])
        A = np.vstack([x, np.ones_like(x)]).T
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        yh = A @ coef
        r2 = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
        edge = np.log(st[v]["ratio"]) / np.log(5.0 / 4.0)
        al.append((v, float(coef[0]), float(r2), float(edge)))
    note("alpha by fit window: " +
         ", ".join("v=%.2f all-f=%.2f (R2=%.2f) edge=%.1f" % a for a in al))
    a_all_hi = max(a[1] for a in al)
    a_all_lo = min(a[1] for a in al)
    a_edge = max(a[3] for a in al)
    if a_all_hi < 1.0 and a_edge > 5.0:
        ok("the power-law exponent moves from %.2f (fit over all f>4, window "
           "dominated by the plateau) to %.1f (fit confined to the rise) -- "
           "no single exponent is identifiable" % (a_all_hi, a_edge))
    else:
        bad("the two fit windows give %.2f and %.1f -- the 'no exponent is "
            "identifiable' argument is not supported" % (a_all_hi, a_edge))

    # --- the step: edge error is a contrast-independent fraction of plateau -
    rise = [st[v]["e4"] / st[v]["plateau"] for v in vs]
    lo_, hi_ = min(rise), max(rise)
    if hi_ < 0.35 and hi_ / lo_ < 2.0:
        ok("the band-edge error is %.2f--%.2f of the plateau in every arm "
           "[+/-%.0f%%], i.e. nearly contrast-independent (\\RiseLo--\\RiseHi)"
           % (lo_, hi_, 100 * (hi_ - lo_) / (0.5 * (hi_ + lo_))))
    else:
        bad("the band-edge-to-plateau ratio spans %.2f--%.2f, too wide for the "
            "'nearly contrast-independent' claim" % (lo_, hi_))

    # --- the rise is over by f = 5 (one frequency unit past the edge) -------
    # The manuscript prints the window as \RiseWindow, which is the FIRST test
    # frequency past the edge at which EVERY arm has reached >=85% of its own
    # plateau, expressed in units of f/f_train.  Check the completion here and
    # report the window the generator will emit, so the sentence cannot drift
    # from the grid.
    late, comp = [], []
    for v in vs:
        F, M = st[v]["F"], st[v]["M"]
        plateau = st[v]["plateau"]
        e5 = float(M[np.isclose(F, 5.0)][0])
        late.append(e5 / plateau)
        out = F[F > C.F_TRAIN[1] + 1e-9]
        Mo = M[F > C.F_TRAIN[1] + 1e-9]
        hit = out[Mo >= 0.85 * plateau]
        comp.append(float(hit.min()) if hit.size else float("inf"))
    window = (max(comp) - C.F_TRAIN[1]) / C.F_TRAIN[1]
    if min(late) >= 0.85 and np.isfinite(window) and window <= 0.51:
        ok("the rise completes by f=%.1f in every arm (weakest arms at >=%.0f%% "
           "of plateau at f=5); in units of f/f_train that is a window of "
           "%.2f (\\RiseWindow)"
           % (max(comp), 100 * min(late), window))
    else:
        bad("at f=5 the weakest arm is only at %.0f%% of its plateau and the "
            "completion frequency is f=%.1f -- the transition is wider than the "
            "single frequency unit the text claims (\\RiseWindow would be %.2f)"
            % (100 * min(late), max(comp), window))

    # --- architecture: U-Net plateau is at least as high as FNO ------------
    for vmax in (0.25, 1.00):
        tag = "a%03d" % int(round(vmax * 100))
        fn = keys(ev, "main_fno_%s_" % tag)
        un = keys(ev, "base_unet_%s_" % tag)
        dn = keys(ev, "base_don_%s_" % tag)
        if not (fn and un):
            note("arch comparison at v=%.2f incomplete (%d fno / %d unet / %d don)"
                 % (vmax, len(fn), len(un), len(dn)))
            continue
        _, Mf, _, _ = curve(ev, fn, vmax)
        _, Mu, _, _ = curve(ev, un, vmax)
        F, Mf, _, _ = curve(ev, fn, vmax)
        pl = F >= 6
        # The manuscript now claims the OPPOSITE of what it claimed before the
        # real data landed: the U-Net reaches a LOWER plateau than the FNO while
        # being several times smaller, so the plateau is architecture-dependent
        # but not capacity-dependent.  Enforce that direction, and require the
        # DeepONet to sit near the FNO (which is what rules out "capacity").
        if Mu[pl].mean() < Mf[pl].mean() - 1e-9:
            ok("U-Net plateau is BELOW the FNO's at v=%.2f (%.3f vs %.3f) while "
               "%.1fx smaller -- plateau is not capacity"
               % (vmax, Mu[pl].mean(), Mf[pl].mean(),
                  np.mean([ev[n]["params"] for n in fn]) /
                  max(np.mean([ev[n]["params"] for n in un]), 1)))
        else:
            bad("U-Net plateau (%.3f) is NOT below the FNO (%.3f) at v=%.2f -- "
                "the 'architecture lowers the plateau without more capacity' "
                "sentence is not supported" % (Mu[pl].mean(), Mf[pl].mean(), vmax))
        if dn:
            _, Md, _, _ = curve(ev, dn, vmax)
            if Md[pl].mean() >= 0.8 * Mf[pl].mean():
                ok("DeepONet sits near the FNO at v=%.2f (%.3f vs %.3f) despite "
                   "being far smaller -- rules out a capacity reading"
                   % (vmax, Md[pl].mean(), Mf[pl].mean()))
            else:
                bad("DeepONet plateau (%.3f) is well below the FNO (%.3f) at "
                    "v=%.2f -- the 'gap is not capacity' reading is unsupported"
                    % (Md[pl].mean(), Mf[pl].mean(), vmax))
        # The manuscript says the parameter-matched U-Net does NOT beat the FNO.
        # That sentence is only meaningful if the pair really is matched, so the
        # match itself is checked before the comparison.
        pm = keys(ev, "pmatch_unet_%s_" % tag)
        if pm:
            _, Mp, _, _ = curve(ev, pm, vmax)
            pf = float(np.mean([ev[n]["params"] for n in pm]))
            ff = float(np.mean([ev[n]["params"] for n in fn]))
            if 0.8 <= pf / ff <= 1.25:
                ok("parameter-matched U-Net at v=%.2f: %.2f M vs FNO %.2f M "
                   "(ratio %.2f)" % (vmax, pf / 1e6, ff / 1e6, pf / ff))
            else:
                bad("the 'parameter-matched' U-Net at v=%.2f is %.2f M against "
                    "the FNO's %.2f M (ratio %.2f) -- not a matched pair, so the "
                    "cross-architecture sentence is unsupported"
                    % (vmax, pf / 1e6, ff / 1e6, pf / ff))
            if Mp[pl].mean() < Mf[pl].mean() - 1e-9:
                ok("parameter-matched U-Net plateau is also below the FNO's at "
                   "v=%.2f (%.3f vs %.3f) -- the effect survives matching"
                   % (vmax, Mp[pl].mean(), Mf[pl].mean()))
            else:
                bad("parameter-matched U-Net plateau (%.3f) is NOT below the "
                    "FNO's (%.3f) at v=%.2f -- the text says the matched model "
                    "reproduces the base model's advantage, but it does not"
                    % (Mp[pl].mean(), Mf[pl].mean(), vmax))
        else:
            note("parameter-matched U-Net at v=%.2f not evaluated yet" % vmax)

    # --- capacity ablation: more parameters, identical recipe --------------
    cw = keys(ev, "cap_w64_a025_")
    rf = keys(ev, "main_fno_a025_")
    if cw and rf:
        Fc, Mc, _, _ = curve(ev, cw, 0.25)
        Fr, Mr, _, _ = curve(ev, rf, 0.25)
        pl = Fc >= 6
        cf = float(np.mean([ev[n]["params"] for n in cw]))
        rr = float(np.mean([ev[n]["params"] for n in rf]))
        if cf / rr >= 1.5:
            ok("capacity ablation raises the parameter count %.2fx (%.1f M vs "
               "%.1f M)" % (cf / rr, cf / 1e6, rr / 1e6))
        else:
            bad("the capacity ablation is only %.2fx the parameter count "
                "(%.1f M vs %.1f M) -- too small to support 'capacity is not "
                "the bottleneck'" % (cf / rr, cf / 1e6, rr / 1e6))
        pc, pr = float(Mc[pl].mean()), float(Mr[pl].mean())
        if abs(pc - pr) <= 0.25 * max(pr, 1e-12):
            ok("out-of-band plateau unchanged by capacity (%.3f vs %.3f)"
               % (pc, pr))
        else:
            bad("the plateau moves from %.3f to %.3f under %.2fx capacity -- the "
                "'capacity is not the bottleneck' claim must be rewritten"
                % (pr, pc, cf / rr))
        # the cliff must not move either: eps(5)/eps(4) is the step indicator
        rc = _ratio(ev[cw[0]], 0.25)
        rr0 = _ratio(ev[rf[0]], 0.25)
        if np.isfinite(rc) and np.isfinite(rr0) and \
                0.5 <= rc / rr0 <= 2.0 and rc > 1.0:
            ok("cliff step preserved under capacity (r %.2f vs %.2f)" % (rc, rr0))
        else:
            bad("the cliff step changes from r=%.2f to r=%.2f under %.2fx "
                "capacity -- 'the cliff does not move' is unsupported"
                % (rr0, rc, cf / rr))
    else:
        note("capacity ablation not evaluated yet (%d cap / %d ref)"
             % (len(cw), len(rf)))

    # --- Born baseline: the trade-off at the band edge ---------------------
    weak = vs[0]
    names = keys(ev, "main_fno_a%03d_" % int(round(weak * 100)))
    if names:
        F, Mf, _, _ = curve(ev, names, weak)
        rows = ev[names[0]]["born_order1"]["%.2f" % weak]
        b_in = float(np.mean([r["mean"] for r in rows if r["f"] <= 4.0]))
        b_out = float(np.mean([r["mean"] for r in rows if r["f"] >= 6.0]))
        learned_in = st[weak]["inband"]
        learned_out = st[weak]["plateau"]
        if b_in > learned_in:
            ok("inside the training band the learned model is better than "
               "Born n=1 in the weakest arm (%.3f vs %.3f)" % (learned_in, b_in))
        else:
            bad("Born n=1 (%.3f) is not worse than the learned model (%.3f) "
                "inside the band -- the 'trade places' sentence is wrong"
                % (b_in, learned_in))
        gain = learned_out / max(b_out, 1e-12)
        if gain >= 2.0:
            ok("beyond the cliff Born n=1 wins by %.1fx in the weakest arm "
               "(%.3f vs %.3f)" % (gain, b_out, learned_out))
        else:
            bad("beyond the cliff Born n=1 wins by only %.2fx -- the "
                "'Born wins outside the band' sentence is too strong" % gain)
    # The manuscript makes the "orders of magnitude worse than predicting zero"
    # statement about the strongest arm only, so that is what is checked.
    top = vs[-1]
    names = keys(ev, "main_fno_a%03d_" % int(round(top * 100)))
    if names:
        b5 = np.array([r["mean"] for r in
                       ev[names[0]]["born_order5"]["%.2f" % top]])
        mx = float(np.nanmax(b5))
        order = int(np.floor(np.log10(mx))) if mx >= 10 else 0
        if order >= 3:
            ok("Born n=5 diverges at v=%.2f (max %.3g => %d orders of magnitude "
               "worse than predicting zero)" % (top, mx, order))
        else:
            bad("Born n=5 reaches only %.3g at v=%.2f -- fewer than 3 orders of "
                "magnitude; \\BornDivOrder would be %d" % (mx, top, order))
    for vmax in (0.25, 0.50):
        names = keys(ev, "main_fno_a%03d_" % int(round(vmax * 100)))
        if not names:
            continue
        b5 = np.array([r["mean"] for r in
                       ev[names[0]]["born_order5"]["%.2f" % vmax]])
        note("Born n=5 max at v=%.2f: %.3g" % (vmax, float(np.nanmax(b5))))

    # --- training-band width ablation -------------------------------------
    bands = [("b2", keys(ev, "band_b2_")), ("b3", keys(ev, "band_b3_")),
             ("b6", keys(ev, "band_b6_")), ("b12", keys(ev, "band_b12_"))]
    have_b = [(t, n) for t, n in bands if n]
    if have_b and st.get(0.25):
        ref_ib = st[0.25]["inband"]
        ref_plateau = st[0.25]["plateau"]
        rows = []
        for t, n in have_b:
            F, M, _, _ = curve(ev, n, 0.25)
            ib, _ = inband(ev, n, 0.25)
            rows.append((t, ib, float(M[F >= 6].mean())))
            print("      band %-3s  in=%.3f  plateau=%.3f" % (t, ib, rows[-1][2]))
        wide = [r for r in rows if r[0] == "b12"]
        if wide:
            t, ib, pl = wide[0]
            plats = [r[2] for r in rows]
            if pl <= min(plats) + 1e-9:
                ok("the widest band [1,12] gives the lowest plateau (%.3f vs "
                   "%s) -- widening lowers the out-of-band plateau"
                   % (pl, ", ".join("%.3f" % x for x in plats)))
            else:
                bad("the widest band [1,12] does NOT give the lowest plateau "
                    "(%.3f against %s) -- the 'widening lowers the plateau' "
                    "sentence is unsupported"
                    % (pl, ", ".join("%.3f" % x for x in plats)))
            # The manuscript does NOT claim the in-band floor improves with
            # width: at a fixed sample budget, spreading the same samples over a
            # wider band costs in-band accuracy.  Assert the direction the text
            # actually states, i.e. that the reference band is the best in band.
            ibs = [r[1] for r in rows]
            if ref_ib <= min(ibs) + 1e-9:
                ok("the reference band [1,4] has the lowest in-band floor of "
                   "all bands (%s) -- consistent with 'widening does not "
                   "monotonically improve the in-band floor'"
                   % ", ".join("%.3f" % x for x in ibs))
            else:
                bad("some other band has a lower in-band floor than the "
                    "reference [1,4] (%s) -- the 'not monotonically improving' "
                    "sentence must be re-checked" % ", ".join("%.3f" % x for x in ibs))
    else:
        note("band ablation not complete yet")

    # --- mode budget -------------------------------------------------------
    m2 = keys(ev, "mode_m02_")
    if m2 and st.get(0.25):
        F, M, _, _ = curve(ev, m2, 0.25)
        ib2, _ = inband(ev, m2, 0.25)
        # Anchor to the ends of the ACTUAL test grid -- the prose is macro-driven
        # off the same two points.  Looking f = 2.0 up by name is what broke here
        # when the grid was refined: the point stopped existing, the macro fell
        # back to a placeholder, and this check died with an IndexError instead
        # of reporting anything about the M=2 arm.
        i_lo, i_hi = int(np.argmin(F)), int(np.argmax(F))
        e_lo, e_hi = float(M[i_lo]), float(M[i_hi])
        note("M=2: in-band mean %.3f, eps(%.1f)=%.3f, eps(%.1f)=%.3f, "
             "reference in-band %.3f"
             % (ib2, F[i_lo], e_lo, F[i_hi], e_hi, st[0.25]["inband"]))
        if ib2 > st[0.25]["inband"]:
            ok("M=2 has a worse in-band error than M=20 (%.3f > %.3f): the band "
               "itself is not representable" % (ib2, st[0.25]["inband"]))
        else:
            bad("M=2 in-band error %.3f is NOT worse than M=20 %.3f -- the "
                "'M<f_train' branch is unsupported" % (ib2, st[0.25]["inband"]))
        if e_hi > e_lo:
            ok("M=2 error rises across the test range (%.3f at f=%.1f -> %.3f "
               "at f=%.1f)" % (e_lo, F[i_lo], e_hi, F[i_hi]))
        else:
            bad("M=2 error does not rise across the test range (%.3f at f=%.1f "
                "-> %.3f at f=%.1f)" % (e_lo, F[i_lo], e_hi, F[i_hi]))
        # "no longer flat" is a comparative claim: the in-band spread, relative
        # to its own level, must be much larger than the M=20 reference arm's.
        band = F <= C.F_TRAIN[1] + 1e-9
        r2 = float(M[band].max() / max(M[band].min(), 1e-12))
        ref_names = keys(ev, "main_fno_a025_")
        if ref_names:
            Fr, Mr, _, _ = curve(ev, ref_names, 0.25)
            rb = Fr <= C.F_TRAIN[1] + 1e-9
            rr = float(Mr[rb].max() / max(Mr[rb].min(), 1e-12))
            note("in-band spread ratio (max/min over f<=4): M=2 %.2f vs M=20 %.2f"
                 % (r2, rr))
            if r2 >= 2.0 * rr:
                ok("M=2 is no longer flat in-band: spread ratio %.2f is at least "
                   "twice the M=20 arm's %.2f" % (r2, rr))
            else:
                bad("M=2 spread ratio %.2f is not >= 2x the M=20 arm's %.2f -- "
                    "the 'no longer flat' sentence is unsupported" % (r2, rr))

    # --- the cliff ratio must be read arm-matched, not across contrasts -----
    top = vs[-1]
    wname = keys(ev, "main_fno_%s_" % ("a%03d" % int(round(vs[0] * 100))))
    if wname and abs(top - vs[0]) > 1e-9:
        d = ev[wname[0]]
        rows = d["sweep"]["%.2f" % top]
        Fx = np.array([r["f"] for r in rows])
        Mx = np.array([r["mean"] for r in rows])
        e4x = float(Mx[np.isclose(Fx, 4.0)][0])
        e5x = float(Mx[np.isclose(Fx, 5.0)][0])
        ibx = float(np.mean([r["mean"] for r in d["in_band"]
                             if abs(r["vmax"] - top) < 1e-6]))
        plx = float(Mx[Fx >= 6].mean())
        rx = e5x / e4x
        own = st[vs[0]]["ratio"]
        note("weak-contrast model swept over contrasts: " +
             ", ".join("v=%.2f r=%.2f" % (float(k), _ratio(d, float(k)))
                       for k in sorted(d["sweep"], key=float)))
        if ibx >= 0.5 * plx:
            ok("at v=%.2f the weak-contrast model is already at plateau level "
               "inside the band (%.3f against plateau %.3f): no cliff in that "
               "reading" % (top, ibx, plx))
        else:
            bad("at v=%.2f the weak-contrast model in-band error %.3f is still "
                "far below its plateau %.3f -- the 'already at plateau level' "
                "sentence is wrong" % (top, ibx, plx))
        if own > 3.0 and rx < own:
            ok("its band-edge ratio falls from %.2f on its own arm to %.2f at "
               "v=%.2f (\\CrossOwnRatio -> \\CrossRatio)" % (own, rx, top))
        else:
            bad("the band-edge ratio does not fall when the weak model is "
                "evaluated at v=%.2f (own %.2f vs %.2f)" % (top, own, rx))
    else:
        note("cross-contrast check needs at least two contrast arms")

    # --- mixed-contrast control --------------------------------------------
    # The prose claims the model trained on a spread of contrasts keeps the
    # step.  Two things have to hold: there IS a step, and the out-of-band
    # error is not wildly outside the range the single-contrast arms span.
    mx = keys(ev, "mixed_fno")
    if mx:
        rx = _ratio(ev[mx[0]], 0.25)
        if np.isfinite(rx) and rx > 1.0:
            ok("mixed-contrast control keeps the step at v=0.25 (r=%.2f)" % rx)
        else:
            bad("mixed-contrast control has no step at v=0.25 (r=%.2f) -- the "
                "'retains the step' sentence is wrong" % rx)
        plats = {}
        for vm in vs:
            r = ev[mx[0]]["sweep"].get("%.2f" % vm)
            if r:
                plats[vm] = float(np.mean([x["mean"] for x in r if x["f"] >= 6]))
        if plats:
            arms_pl = {vm: st[vm]["plateau"] for vm in plats if vm in st}
            lo, hi = min(plats.values()), max(plats.values())
            if arms_pl and hi <= 3.0 * max(arms_pl.values()):
                ok("mixed-contrast plateau %.3f--%.3f lies within the arms' "
                   "range (max arm %.3f)"
                   % (lo, hi, max(arms_pl.values())))
            else:
                bad("mixed-contrast plateau %.3f--%.3f is far outside the "
                    "single-contrast arms (max %.3f) -- the control must be "
                    "reported separately rather than as a robustness check"
                    % (lo, hi, max(arms_pl.values()) if arms_pl else float("nan")))
    else:
        note("mixed-contrast control not evaluated yet")

    # --- spectrum: delta_2 saturates inside / at the band edge -------------
    sp = os.path.join(RES, "spectrum.json")
    if not os.path.exists(sp):
        sp = os.path.join(C.RES, "spectrum.json")
    if os.path.exists(sp):
        d = json.load(open(sp, encoding="utf-8"))
        rows = d["data"]["0.25"]
        # diag_spectrum writes the row keys with "f=%g" and make_results_tex
        # reads them back the same way; this file used "f=%.1f", which looks
        # equivalent and is not -- it looked up "f=4.0" against a key "f=4"
        # and crashed with KeyError the first time the check was reached.
        vals = [(f, rows["f=%g" % f]["psi"]["2"]) for f in d["f"]]
        big = [f for f, v in vals if v >= 0.9]
        if big:
            ok("delta_2 reaches >=0.9 from f=%.1f on (saturates inside the "
               "extended range)" % min(big))
        else:
            bad("delta_2 never reaches 0.9")

    print("\n=== captions ===")
    caption_guard(ev)

    print("\n=== seed protocol ===")
    seed_guard(ev)

    print("\n=== residual claims ===")
    residual_guard(RES)

    print("\n=== architecture labels ===")
    arch_label_guard()

    print("\n=== plateau-normalised separation ===")
    plateau_normalised_guard(ev)

    print("\n=== C1: conditioning confound ===")
    c1_guard(ev)

    print("\n=== C2-C4: discriminators ===")
    disc_guard(ev)

    print("\n=== analytic anchor ===")
    anchor_guard(ev)
    deep_guard(ev)

    print("\n=== three-grid convergence and intervals ===")
    import guards_gridstats as G
    G.delivery_guard()
    G.grid_guard()
    G.ci_guard(ev)
    G.baseline_pairing_guard()
    G.doi_guard()
    # the module prints its own lines; only its failures are carried across, and
    # only once
    FAIL.extend(G.FAIL)

    print("\n=== macro spacing ===")
    macro_name_guard()
    macro_space_guard()

    print("\n=== text residue in the delivered files ===")
    import guards_text as T
    T.figure_guard()
    T.text_residue_guard()
    T.range_dash_guard()
    T.seed_claim_guard()
    FAIL.extend(T.FAIL)

    print("\n=== the three boundary locators, on every ladder ===")
    import guards_locators as LG
    LG.locator_guard()
    FAIL.extend(LG.FAIL)

    print("\n=== resolution: does the boundary belong to the problem or the mesh ===")
    import guards_res as RG
    RG.res_guard()
    print("\n=== the qualitative claims the resolution section prints ===")
    RG.claim_guard()
    FAIL.extend(RG.FAIL)

    print("\n=== spectrum-matched control and per-sample bound ===")
    import guards_surrogate as SG
    SG.surrogate_guard()
    SG.bound_guard()
    SG.bound_text_guard()
    SG.paired_jump_guard()
    FAIL.extend(SG.FAIL)

    print("\n=== fit-depth/sum-depth correction and the bax arms ===")
    import guards_bax as XB
    XB.bax_guard()
    XB.bax_text_guard()
    FAIL.extend(XB.FAIL)

    print("\n=== caption batch sizes ===")
    batch_label_guard()

    print("\n=== third level: real geology ===")
    realgeo_guard()

    print("\n=== submission elements ===")
    submission_element_guard()

    print("\n=== envelope documents ===")
    # The cover letter, README, CITATION.cff, the environment pins and the
    # wording of the Data Availability statement are delivered text that no
    # .tex-side gate can see.  Each of them drifted once while the manuscript
    # stayed correct, so they are checked as a class rather than one by one.
    import check_delivery_texts as DT
    dt = DT.run()
    for f in dt:
        bad(f)
    if not dt:
        ok("cover letter, README, both CITATION copies, the environment pins, "
           "the DOI wording and the archive agree with the manuscript")

    print("\n=== summary ===")
    if FAIL:
        print("  %d prose claims contradicted by the data:" % len(FAIL))
        for f in FAIL:
            print("   - " + f)
        return 1
    print("  every checked prose claim is supported by the data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
