# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Falsifiability probes for the two macro gates in audit_paper.

Before trusting a gate you must show it fires on a known-bad input.

probe 1 -- check_macros_resolved.  Reproduces, in a sandbox folder, the exact
defect that reached the PDF once: a macro whose placeholder is written with
doubled braces, i.e. \\newcommand{\\X}{{--}}, which both prints a stray en-dash
and used to slip past the `}{--}` pattern.
Expected: DoubleBrace and SingleBrace are reported (both cited), Uncited is
ignored, Good is ignored.

probe 2 -- check_macro_names.  A digit inside a macro name truncates it, because
TeX ends a control word at the first non-letter: \\newcommand{\\ResP50}{...}
defines \\ResP and leaves a literal "50" in the preamble.  That reached a real
build as a hard "Missing \\begin{document}" while every other gate stayed green.
Expected: the digit-bearing name is reported.

probe 3 -- check_claims.residual_guard.  This gate has to do three different
things on the three states the measurement can actually be in: accept a corpus
whose recomputation is limited by the single-precision storage floor AND whose
ruined labels are non-finite (that is the true state of the archive), reject a
corpus whose working-precision re-solve misses the gate, and reject a corpus
that contradicts the prose's non-finite claim by presenting finite corruption.
A gate that only ever accepts would be worse than none, so all three directions
are exercised here.

probe 13 -- audit_paper.check_labels.  A label defined twice is invisible to a
set, and LaTeX resolves every \\ref to the LAST definition, so the manuscript
shipped four cross-references that rendered the Discussion's number while
intending the discriminator subsection, while the gate reported "all refs
resolve".  Expected: the clean manuscript passes, the duplicated label fails
even though every ref still resolves, and a merely unreferenced label stays a
warning.
"""
import os
import re
import sys
import io
import glob
import json
import shutil
import tempfile
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import audit_paper as A
import check_claims as K

MACROS = (
    "\\newcommand{\\Good}{0.123}\n"
    "\\newcommand{\\DoubleBrace}{{--}}\n"
    "\\newcommand{\\SingleBrace}{--}\n"
    "\\newcommand{\\Uncited}{{--}}\n"
)
BODY = "text \\Good{} and \\DoubleBrace{} and \\SingleBrace{} end"

BAD_NAMES = (
    "\\newcommand{\\Fine}{1}\n"
    "\\newcommand{\\ResP50}{2}\n"
    "\\newcommand{\\Fine}{3}\n"
)


def run(folder, fn, *args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue().strip()


def probe_submission_elements():
    """Probe 10 -- submission_element_guard, both halves and both directions.

    The abstract budget must fire at 201 words and stay silent at 199.  The
    placeholder shapes must fire on `$--$` and on a bare `--` cell, and stay
    SILENT on a legitimate numeric range (`0.085--0.127`) and on an em-dash
    aside: a guard that flagged every en-dash would be turned off by the first
    person who saw it, and then the real defect would be invisible again.
    """
    d = tempfile.mkdtemp(prefix="_gate_probe10_")
    try:
        for lang in ("en", "cn"):
            os.makedirs(os.path.join(d, "paper", lang))

        def write(n_words, cell):
            body = " ".join(["word"] * n_words)
            with open(os.path.join(d, "paper", "en", "main.tex"), "w",
                      encoding="utf-8") as fh:
                fh.write("\\begin{abstract}\n%s\n\\end{abstract}\n" % body)
            with open(os.path.join(d, "paper", "en", "gen_tab_x.tex"), "w",
                      encoding="utf-8") as fh:
                fh.write("\\begin{tabular}{ccc}\n$f$ & $r$ & note \\\\\n"
                         "1.5 & 0.085 & %s \\\\\n\\end{tabular}\n" % cell)

        write(199, "a range 0.085--0.127 --- an aside")
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        clean = not K.FAIL

        write(201, "a range 0.085--0.127 --- an aside")
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        long_abs = bool(K.FAIL)

        write(199, "$--$")
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        dollar = bool(K.FAIL)

        write(199, "--")
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        bare = bool(K.FAIL)
        del K.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return clean, long_abs, dollar, bare


def probe_selfmath_wrap():
    """Probe 11 -- audit_paper.check_selfmath_macros.

    The defect shipped twice, once per language: a macro whose body is already
    `$2.5\\times10^{-8}$` was called as `$\\ResPsiStorePert$`.  pdfLaTeX turns
    `$$...$$` into a display-math toggle and produces a PDF with wrong spacing;
    XeLaTeX stops at the next CJK glyph and emits a 3-page file.  Both outcomes
    are silent to every numeric gate because the digits still print.
    Directions: `$\\X$` must be caught; `\\X`, `\\X{}` and `$\\Y$` (a macro whose
    body is NOT self-delimiting math) must all be accepted.
    """
    d = tempfile.mkdtemp(prefix="_gate_probe11_")
    try:
        os.makedirs(os.path.join(d, "paper", "en"))
        with open(os.path.join(d, "paper", "en", "gen_macros.tex"), "w",
                  encoding="utf-8") as fh:
            fh.write("\\newcommand{\\SelfM}{$2.5\\times10^{-8}$}\n"
                     "\\newcommand{\\PlainM}{0.123}\n")

        def write(body):
            with open(os.path.join(d, "paper", "en", "main.tex"), "w",
                      encoding="utf-8") as fh:
                fh.write(body + "\n")

        write("the perturbation \\SelfM{} is small")
        del A.FAIL[:]
        run(d, A.check_selfmath_macros, os.path.join(d, "paper", "en"))
        ok_braced = not A.FAIL
        write("the perturbation \\SelfM is small")
        del A.FAIL[:]
        run(d, A.check_selfmath_macros, os.path.join(d, "paper", "en"))
        ok_bare = not A.FAIL
        write("the value $\\PlainM$ is small")
        del A.FAIL[:]
        run(d, A.check_selfmath_macros, os.path.join(d, "paper", "en"))
        ok_plain = not A.FAIL
        write("the perturbation $\\SelfM$ is small")
        del A.FAIL[:]
        run(d, A.check_selfmath_macros, os.path.join(d, "paper", "en"))
        caught = bool(A.FAIL)
        del A.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return ok_braced, ok_bare, ok_plain, caught


def probe_plateau_normalised_gate():
    """The own-plateau separation gate has to fire in three directions.

    It asserts an INEQUALITY between families -- at f = 5 the FNO is above its
    own out-of-band plateau while both U-Nets are below theirs -- on a quantity
    that is itself a ratio of two estimated numbers.  Three things can go wrong
    and each needs a known-bad input of its own: the FNO stops overshooting, a
    U-Net reaches its plateau, or the macros the manuscript quotes drift away
    from the measurement.  A gate that only ever accepts is worse than none, so
    all three are exercised here against a throwaway copy of the real archive.
    """
    import check_claims as K
    import common as C

    d = tempfile.mkdtemp(prefix="_platenorm_probe_")
    res = os.path.join(d, "results")
    os.makedirs(res, exist_ok=True)

    def reset():
        for p in glob.glob(os.path.join(res, "eval_*.json")):
            os.remove(p)
        for p in sorted(glob.glob(os.path.join(C.RES, "eval_*.json"))):
            shutil.copy(p, res)

    def scale_at(freq, factor, prefixes):
        """Multiply eps at one frequency, for the given run prefixes."""
        for p in sorted(glob.glob(os.path.join(res, "eval_*.json"))):
            nm = os.path.basename(p)[5:-5]
            if not any(nm.startswith(x) for x in prefixes):
                continue
            dd = json.load(open(p, encoding="utf-8"))
            for vk in dd["sweep"]:
                for r in dd["sweep"][vk]:
                    if abs(r["f"] - freq) < 1e-9:
                        r["mean"] *= factor
            json.dump(dd, open(p, "w", encoding="utf-8"))

    def write_macros(drop=()):
        src = os.path.join(C.ROOT, "paper", "en", "gen_macros.tex")
        t = open(src, encoding="utf-8").read()
        for mac in drop:
            t = re.sub(r"\\newcommand\{\\%s\}\{[^}]*\}\n?" % mac, "", t)
        for lang in ("en", "cn"):
            pd = os.path.join(d, "paper", lang)
            os.makedirs(pd, exist_ok=True)
            with open(os.path.join(pd, "gen_macros.tex"), "w",
                      encoding="utf-8") as fh:
                fh.write(t)

    old_res = K.RES
    K.RES = res
    try:
        # (1) what is actually delivered: the separation holds in every seed
        reset(); write_macros()
        del K.FAIL[:]
        run(d, K.plateau_normalised_guard, K.load(), d)
        honest_ok = not K.FAIL

        # (2) the FNO no longer overshoots its plateau at f = 5
        reset(); write_macros(); scale_at(5.0, 0.4, ("main_fno_",))
        del K.FAIL[:]
        run(d, K.plateau_normalised_guard, K.load(), d)
        fno_rejected = bool(K.FAIL)

        # (3) a U-Net reaches its own plateau at f = 5
        reset(); write_macros(); scale_at(5.0, 6.0, ("base_unet_",))
        del K.FAIL[:]
        run(d, K.plateau_normalised_guard, K.load(), d)
        unet_rejected = bool(K.FAIL)

        # (4) the per-seed dispersion macro is no longer generated
        reset(); write_macros(drop=("ArchFnoNormFiveSd",))
        del K.FAIL[:]
        run(d, K.plateau_normalised_guard, K.load(), d)
        macro_rejected = bool(K.FAIL)
        del K.FAIL[:]
    finally:
        K.RES = old_res
        shutil.rmtree(d, ignore_errors=True)
    print("   honest archive accepted = %s ; FNO overshoot collapse rejected = %s"
          " ; U-Net plateau breach rejected = %s ; dropped dispersion macro "
          "rejected = %s" % (honest_ok, fno_rejected, unet_rejected,
                             macro_rejected))
    return honest_ok, fno_rejected, unet_rejected, macro_rejected


def probe_label_uniqueness():
    """Probe 13 -- check_labels must see a label defined twice.

    The real defect: \\label{sec:disc} was attached to the discriminator
    subsection and, one edit later, to the Discussion section as well.  LaTeX
    resolves every \\ref to the LAST definition, so four cross-references
    rendered the Discussion's number while intending the discriminators -- and
    the set-based check reported "all refs resolve" because both targets
    existed.  A set cannot see this class; the gate must keep a list.

    Three directions: a clean manuscript passes, the duplicated label fails
    *while every ref still resolves* (that is the state that fooled the old
    gate), and a merely unreferenced label stays a warning, not a failure --
    promoting it would turn the gate off on the first manuscript that labels a
    figure it never cites.
    """
    d = tempfile.mkdtemp(prefix="_gate_probe13_")
    try:
        os.makedirs(os.path.join(d, "paper", "en"))
        man = os.path.join(d, "paper", "en", "main.tex")

        def write(body):
            with open(man, "w", encoding="utf-8") as fh:
                fh.write(body)

        clean = ("\\section{Discriminators}\n\\label{sec:disc}\n"
                 "see Section~\\ref{sec:disc}\n"
                 "\\section{Discussion}\n\\label{sec:discussion}\n")
        dup = ("\\section{Discriminators}\n\\label{sec:disc}\n"
               "see Section~\\ref{sec:disc}\n"
               "\\section{Discussion}\n\\label{sec:disc}\n")
        unref = ("\\section{Discriminators}\n\\label{sec:disc}\n"
                 "see Section~\\ref{sec:disc}\n"
                 "\\section{Discussion}\n\\label{sec:discussion}\n"
                 "\\label{sec:ladder}\n")
        write(clean)
        del A.FAIL[:]
        out = run(d, A.check_labels, man)
        good = not A.FAIL
        write(dup)
        del A.FAIL[:]
        out_bad = run(d, A.check_labels, man)
        bad_dup = bool(A.FAIL)
        write(unref)
        del A.FAIL[:]
        out_unref = run(d, A.check_labels, man)
        unref_only_warns = not A.FAIL and "never referenced" in out_unref
        del A.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(out_bad)
    return good, bad_dup, unref_only_warns


def main():
    d = tempfile.mkdtemp(prefix="_gate_probe_")
    try:
        with open(os.path.join(d, "gen_macros.tex"), "w", encoding="utf-8") as fh:
            fh.write(MACROS)
        with open(os.path.join(d, "main.tex"), "w", encoding="utf-8") as fh:
            fh.write(BODY)
        out = run(d, A.check_macros_resolved, d)
        with open(os.path.join(d, "gen_macros.tex"), "w", encoding="utf-8") as fh:
            fh.write(BAD_NAMES)
        out2 = run(d, A.check_macro_names, d)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    print("-- probe 1 (placeholder form) --")
    print(out)
    caught = ("DoubleBrace" in out) and ("SingleBrace" in out)
    ignored = ("Uncited" not in out) and ("Good" not in out)
    print("PROBE 1: double-brace caught = %s ; uncited ignored = %s" % (caught, ignored))

    print("-- probe 2 (macro name form) --")
    print(out2)
    named = "ResP50" in out2
    duped = "Fine" in out2
    print("PROBE 2: digit name caught = %s ; duplicate caught = %s" % (named, duped))

    good3, bad3, fin3 = probe_residual_guard()
    print("PROBE 3: honest corpus accepted = %s ; tampered re-solve rejected = %s "
          "; finite-corruption overclaim rejected = %s" % (good3, bad3, fin3))

    good4, bad4t, bad4c = probe_arch_labels()
    print("PROBE 4: agreed labels accepted = %s ; drifted table rejected = %s "
          "; stale caption rejected = %s" % (good4, bad4t, bad4c))

    print("-- probe 5 (style gate, both languages) --")
    sty = probe_style_gate()
    print("PROBE 5: %s" % ("fires on bad and silent on clean in en and cn"
                           if all(sty.values()) else "MISBEHAVED: %s" % sty))

    print("-- probe 6 (generated-file freshness) --")
    fresh_stale, fresh_ok = probe_freshness_gate()
    print("PROBE 6: stale build blocked = %s ; fresh build allowed = %s"
          % (fresh_stale, fresh_ok))

    print("-- probe 7 (caption batch sizes) --")
    b_ok, b_lit, b_drift = probe_batch_label_gate()
    print("PROBE 7: %s" % ("macro-bound accepted and both drifts rejected"
                           if (b_ok and b_lit and b_drift)
                           else "MISBEHAVED: accepted=%s literal=%s drift=%s"
                                % (b_ok, b_lit, b_drift)))

    print("-- probe 8 (third level: real geology) --")
    r_ok, r_drift, r_word, r_float, r_over = probe_realgeo_gate()
    if all((r_ok, r_drift, r_word, r_float, r_over)):
        print("PROBE 8: honest record accepted and all four defects rejected")
    else:
        print("PROBE 8: MISBEHAVED accepted=%s drift=%s wording=%s float=%s "
              "overclaim=%s" % (r_ok, r_drift, r_word, r_float, r_over))

    print("-- probe 9 (macro spacing after a control word) --")
    msp = probe_macro_spacing()
    if all(msp.values()):
        print("PROBE 9: swallow caught, braced fix and unit exception accepted, "
              "glued-to-punctuation caught")
    else:
        print("PROBE 9: MISBEHAVED %s" % msp)

    print("-- probe 10 (submission elements: abstract budget + placeholder cells) --")
    s_clean, s_long, s_dollar, s_bare = probe_submission_elements()
    if s_clean and s_long and s_dollar and s_bare:
        print("PROBE 10: 199-word abstract and valid en-dash range accepted; "
              "201-word abstract, $--$ and bare -- cells all rejected")
    else:
        print("PROBE 10: MISBEHAVED clean=%s long=%s dollar=%s bare=%s"
              % (s_clean, s_long, s_dollar, s_bare))

    print("-- probe 11 (self-delimiting-math macro double-wrapped in '$') --")
    w_braced, w_bare, w_plain, w_caught = probe_selfmath_wrap()
    if w_braced and w_bare and w_plain and w_caught:
        print("PROBE 11: \\X and \\X{} and $\\Y$ accepted; $\\X$ on a "
              "self-delimiting-math macro rejected")
    else:
        print("PROBE 11: MISBEHAVED braced=%s bare=%s plain=%s caught=%s"
              % (w_braced, w_bare, w_plain, w_caught))

    print("-- probe 12 (own-plateau separation, seed by seed) --")
    p_honest, p_fno, p_unet, p_mac = probe_plateau_normalised_gate()
    if all((p_honest, p_fno, p_unet, p_mac)):
        print("PROBE 12: delivered archive accepted; FNO overshoot collapse, "
              "U-Net plateau breach and a dropped dispersion macro all rejected")
    else:
        print("PROBE 12: MISBEHAVED honest=%s fno=%s unet=%s macro=%s"
              % (p_honest, p_fno, p_unet, p_mac))

    print("-- probe 13 (label uniqueness) --")
    l_clean, l_dup, l_unref = probe_label_uniqueness()
    if l_clean and l_dup and l_unref:
        print("PROBE 13: unique labels accepted, duplicated label rejected, "
              "unreferenced label still only a warning")
    else:
        print("PROBE 13: MISBEHAVED clean=%s dup=%s unref=%s"
              % (l_clean, l_dup, l_unref))

    good = (caught and ignored and named and duped and good3 and bad3 and fin3
            and good4 and bad4t and bad4c and all(sty.values())
            and fresh_stale and fresh_ok
            and b_ok and b_lit and b_drift
            and r_ok and r_drift and r_word and r_float and r_over
            and all(msp.values())
            and s_clean and s_long and s_dollar and s_bare
            and w_braced and w_bare and w_plain and w_caught
            and p_honest and p_fno and p_unet and p_mac
            and l_clean and l_dup and l_unref)
    print("PROBE RESULT: %s" % ("all gates fire on known-bad input" if good else "GATE FAILED"))
    return 0 if good else 1


def _synthetic_residual(gate=1e-9, refined_max=9.8e-13, shipped_max=3.7e-8,
                        roundtrip_max=2.7e-8, nonfinite_fraction=0.216,
                        finite_corrupt=0):
    """A residual_dist.json shape matching the corrected schema.

    The corrected measurement reports what the damage actually looks like: the
    ruined labels are non-finite, so the finite corrupted remainder is empty and
    ``separation_orders`` is None.  ``finite_corrupt`` lets a probe reintroduce
    finite corruption, which the prose does not claim and the guard must reject.
    """
    def stats(v):
        return dict(n=128, p50=v, p90=v, p99=v, p99_9=v, p100=v, min=v, max=v)
    return {
        "corpus": {
            "per_arm": {a: dict(n=16384) for a in ("t1", "t2")},
            "overall": dict(n=196608, p50=3.3e-10, p99=9.8e-10, max=1.0e-9,
                            n_over_gate=0, n_over_bad=0),
            "gate": gate, "bad": 1e-6},
        "recheck": {
            "per_arm": [dict(arm="t1", n=128, agree_max_rel=318.0),
                        dict(arm="t2", n=128, agree_max_rel=42.0)],
            "floor": dict(n=256, gate=gate,
                          shipped=stats(shipped_max),
                          roundtrip=stats(roundtrip_max),
                          refined=stats(refined_max),
                          psi_storage_perturbation=stats(2.5e-8))},
        "naive": {
            "n_total": 1024, "fired_silently": 221, "silent_fraction": 0.216,
            "n_nonfinite": int(1024 * nonfinite_fraction),
            "nonfinite_fraction": nonfinite_fraction,
            "n_finite_corrupt": finite_corrupt,
            "clean": dict(n=803, min=2.4e-16, max=7.6e-10),
            "corrupt": (dict(n=finite_corrupt, min=1e-5, max=1e0)
                        if finite_corrupt else dict(n=0)),
            "frozen_true": stats(9.0e-11),
            "separation_orders": None,
            "injection": [dict(eta=e, batch=0, frac_flagged=f, median=1e-9)
                          for e, f in [(1e-3, 1.0), (1e-5, 1.0), (1e-7, 0.5)]]},
    }


def probe_residual_guard():
    """Three directions: honest accepted, broken re-solve rejected, and a
    finite-corruption corpus -- which contradicts the prose's non-finite claim
    -- rejected as well."""
    d = tempfile.mkdtemp(prefix="_res_probe_")
    try:
        # honest archive: recomputation limited by storage, re-solve inside gate,
        # and damaged labels non-finite
        with open(os.path.join(d, "residual_dist.json"), "w", encoding="utf-8") as fh:
            json.dump(_synthetic_residual(), fh)
        del K.FAIL[:]
        out_good = run(d, K.residual_guard, d)
        good = not K.FAIL
        # tampered: the independent re-solve no longer reaches the gate
        with open(os.path.join(d, "residual_dist.json"), "w", encoding="utf-8") as fh:
            json.dump(_synthetic_residual(refined_max=1e-4), fh)
        del K.FAIL[:]
        out_bad = run(d, K.residual_guard, d)
        bad = bool(K.FAIL)
        del K.FAIL[:]
        # overclaim in the other direction: finite corruption appears, so the
        # prose's "non-finite" sentence would be false
        with open(os.path.join(d, "residual_dist.json"), "w", encoding="utf-8") as fh:
            json.dump(_synthetic_residual(nonfinite_fraction=0.0,
                                          finite_corrupt=50), fh)
        del K.FAIL[:]
        out_fin = run(d, K.residual_guard, d)
        finite_rejected = bool(K.FAIL)
        del K.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    # Label the three scenarios.  Two of them are SUPPOSED to fail, so their
    # output carries `[FAIL]` lines; without a label those lines read as if the
    # gate itself had broken, which is exactly how this probe was misread once.
    for tag, expect, out in (("1", "must be ACCEPTED", out_good),
                             ("2", "must be REJECTED", out_bad),
                             ("3", "must be REJECTED", out_fin)):
        print("   [scenario %s] %s:" % (tag, expect))
        for ln in out.splitlines():
            print("   | " + ln)
    print("   accepted=%s ; rejected=%s ; rejected=%s" % (good, bad, finite_rejected))
    return good, bad, finite_rejected


def _tab_arch_block(labels, ragged=False):
    """Emit a tab:arch float in the exact shape make_results_tex produces.

    ``ragged`` reproduces the real table, which is NOT rectangular: only the FNO
    was trained at every contrast, so the first groups carry a single row while
    the last carry all four.  A gate that assumes a rectangular table fails on
    the correct artefact, which is exactly what happened once.
    """
    rows = []
    for i, v in enumerate(("0.05", "0.10", "0.25", "0.50", "1.00")):
        labs = labels[:1] if (ragged and i < 2) else labels
        for lab in labs:
            rows.append("$%s$ & %s & 10.25 & $0.09$ & $0.97$ & $1.02$ \\\\"
                        % (v, lab))
    return ("\\begin{table}[t]\n\\centering\n\\caption{c}\n\\label{tab:arch}\n"
            "\\begin{tabular}{clcccc}\n\\toprule\n"
            "$v_{\\max}$ & model & params (M) & $\\varepsilon_{in}$ & "
            "$\\varepsilon_{6}$ & $\\varepsilon_{12}$ \\\\\n\\midrule\n"
            + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# A second float whose data rows also look like "$1$ & 5 \\" -- the false
# positive control.  An earlier version of the gate scanned every such row in
# the file, picked up these mode-budget rows, and reported a label drift that
# did not exist.
_DECOY_FLOAT = ("\\begin{table}[t]\n\\centering\n\\label{tab:mode}\n"
                "\\begin{tabular}{cc}\n$1$ & 5 \\\\\n$5$ & 1 \\\\\n"
                "\\end{tabular}\n\\end{table}\n")


def probe_arch_labels():
    """Class-level gate for Fig. 4 / Table III label drift.

    Four directions: the agreed state must pass *even with an unrelated float
    carrying numeric row labels present* (false-positive control), a table whose
    rows use a stale label must fail, and a manuscript whose caption still
    carries the stale label must fail even when the table itself is correct --
    the original defect lived in the caption, not in the table.
    """
    d = tempfile.mkdtemp(prefix="_arch_probe_")
    want = [lab for _, lab in K.C.ARCH_GROUPS]
    try:
        os.makedirs(os.path.join(d, "paper", "en"))
        # One file per label since the split; the probe used to write the single
        # gen_results.tex, which the guard no longer reads -- so the probe would
        # have exercised nothing while still reporting a pass.
        gen = os.path.join(d, "paper", "en", "gen_tab_arch.tex")
        man = os.path.join(d, "paper", "en", "main.tex")
        with open(gen, "w", encoding="utf-8") as fh:
            fh.write(_tab_arch_block(want, ragged=True) + _DECOY_FLOAT)
        with open(man, "w", encoding="utf-8") as fh:
            fh.write("the U-Net (base) plateau is lower\n")
        del K.FAIL[:]
        out = run(d, K.arch_label_guard, d)
        good = not K.FAIL
        # table rows drifted
        with open(gen, "w", encoding="utf-8") as fh:
            fh.write(_tab_arch_block(["FNO", "U-Net (small)", "U-Net (matched)",
                                      "DeepONet"], ragged=True) + _DECOY_FLOAT)
        del K.FAIL[:]
        run(d, K.arch_label_guard, d)
        bad_table = bool(K.FAIL)
        # table correct, caption stale
        with open(gen, "w", encoding="utf-8") as fh:
            fh.write(_tab_arch_block(want, ragged=True) + _DECOY_FLOAT)
        with open(man, "w", encoding="utf-8") as fh:
            fh.write("the U-Net (small) plateau is the lowest of the three\n")
        del K.FAIL[:]
        run(d, K.arch_label_guard, d)
        bad_caption = bool(K.FAIL)
        del K.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print(out)
    return good, bad_table, bad_caption


def probe_style_gate():
    """The style gate must fire on a known-bad sample and stay silent on a clean
    one -- in BOTH languages.

    The Chinese half is the control that matters most.  The first version of this
    gate applied the English rule set to the Chinese manuscript and metered
    captions in whitespace-separated words; Chinese has no inter-word spaces, so
    `len(caption.split())` is 1 for a caption of any length and every Chinese
    caption passed.  A gate that can never fail is worse than no gate, so the
    long-Chinese-caption case is exercised explicitly.
    """
    import lint_style as S

    def scan_fails(text, lang):
        del S.FAIL[:]
        with contextlib.redirect_stdout(io.StringIO()):
            S.scan("probe.tex", text,
                   S.RULES_EN if lang == "en" else S.RULES_CN)
        return bool(S.FAIL)

    def caption_fails(text):
        del S.FAIL[:]
        with contextlib.redirect_stdout(io.StringIO()):
            S.captions("probe.tex", text)
        return bool(S.FAIL)

    res = dict(
        en_bad=scan_fails(
            "This claim---and the one right after it---is not new. "
            "Note that it holds.\n", "en"),
        en_ok=not scan_fails("The claim is not new and it holds.\n", "en"),
        # en-dash asides are a separate dash, so the `---` rule above cannot see
        # them; the range control is the false-positive guard, because `--` is
        # the correct spelling inside a numeric range.
        en_dash_bad=scan_fails(
            "The failure leaves no residual to threshold against -- no gap in "
            "which to place a gate -- which is why the check misses it.\n", "en"),
        en_dash_range_ok=not scan_fails(
            "The contrast runs over 0.05--1.00 and f in 1--12 throughout.\n", "en"),
        # `--` also marks "not applicable" in a table cell, and two such cells
        # in one row are two independent marks, not an aside.  A flat scan of
        # the line paired them anyway and this was the rule's *only* output in
        # either language -- its whole signal was one false positive.  Keeping
        # the control here means the pairing fix cannot be silently undone.
        en_dash_cells_ok=not scan_fails(
            "Bi-CGSTAB (reference) & -- & -- & 1.31 & 1.0 \\\\\n", "en"),
        cn_dash_cells_ok=not scan_fails(
            "Bi-CGSTAB（参考） & -- & -- & 1.31 & 1.0 \\\\\n", "cn"),
        # a comment never reaches the PDF, so it must not be scored as prose
        comment_ok=not scan_fails("% banner -- a -- b -- c, do not edit\n", "en"),
        # The blind spot that was live in the manuscript: the source is
        # hard-wrapped, so an aside whose two dashes land on different lines
        # must still be caught.  With `[^-\n]` it was invisible, and five such
        # asides sat in the English manuscript while the gate reported clean.
        en_dash_wrapped_bad=scan_fails(
            "the level above it --- measured\nfield data --- is named as "
            "absent rather than as a finding.\n", "en"),
        en_dash_wrapped_ok=not scan_fails(
            "the level above it, measured\nfield data, is named as absent "
            "rather than as a finding.\n", "en"),
        cn_bad=scan_fails(
            "该结论——在低频段尤其明显——并不新。值得注意的是它成立。\n", "cn"),
        cn_ok=not scan_fails("该结论并不新，且在该频段成立。\n", "cn"),
        cn_dash_wrapped_bad=scan_fails(
            "平台为 0.86——即在训练频带内部\n就已到达平台水平——其带缘比值随之下降。\n",
            "cn"),
        # The blind spot: a Chinese caption far over budget, then a short one.
        # Note the probe string has to be genuinely over 75 reading units; the
        # first version of this probe used 72 and therefore tested nothing.
        cn_cap_bad=caption_fails(
            "\\caption{%s}\n" % ("训练模型的实测误差与同预算下频带受限下界之比较，"
                                 * 4)),
        cn_cap_ok=not caption_fails("\\caption{实测误差与下界之比较。}\n"),
        # Control on the English side too: 60 words must fail, 5 must not.
        en_cap_bad=caption_fails("\\caption{%s}\n" % ("word " * 60)),
        en_cap_ok=not caption_fails("\\caption{a short caption here}\n"),
    )
    # The failure mode this gate actually shipped with: it skipped missing
    # files silently and then reported a pass, so pointing it at the wrong root
    # produced a green light over zero manuscripts.  An empty tree must be
    # caught, not blessed.
    empty = tempfile.mkdtemp(prefix="_style_empty_")
    try:
        del S.FAIL[:]
        with contextlib.redirect_stdout(io.StringIO()):
            S.run_over_root(empty)
        res["empty_tree_caught"] = bool(S.FAIL)
    finally:
        shutil.rmtree(empty, ignore_errors=True)
    del S.FAIL[:]
    for k in sorted(res):
        print("   %-18s %s" % (k, res[k]))
    return res


def probe_batch_label_gate():
    """Both directions for the caption batch-size gate.

    That gate is about *form*: a caption has to reach the measured sample counts
    through the generated macros instead of spelling them out.  The accepted
    case therefore has to pass with the macros present and correct, and two
    rejections are exercised -- a hand-written literal, and a macro that has
    drifted away from ``solver_fig.json`` -- because a gate that only ever
    accepts would be worse than none.
    """
    import check_claims as K
    d = tempfile.mkdtemp(prefix="_batch_probe_")
    res = os.path.join(d, "results")
    os.makedirs(res)
    with open(os.path.join(res, "solver_fig.json"), "w", encoding="utf-8") as fh:
        json.dump({"freeze": {"batch": 1024, "aggregate": {"n_total": 8192}}}, fh)
    for lang in ("en", "cn"):
        os.makedirs(os.path.join(d, "paper", lang))

    def write_macros(lang, size):
        with open(os.path.join(d, "paper", lang, "gen_macros.tex"), "w",
                  encoding="utf-8") as fh:
            fh.write("\\newcommand{\\SolverBatchSize}{%s}\n"
                     "\\newcommand{\\SolverBatchN}{8\\,192}\n" % size)

    def write_caption(lang, body):
        with open(os.path.join(d, "paper", lang, "main.tex"), "w",
                  encoding="utf-8") as fh:
            fh.write("\\caption{%s}\n\\label{fig:solver}\n" % body)

    old_res = K.RES
    K.RES = res
    try:
        for lang in ("en", "cn"):
            write_macros(lang, "1\\,024")
            write_caption(lang, "per batch of \\SolverBatchSize\\ samples, "
                                "\\SolverBatchN\\ in total")
        del K.FAIL[:]
        run(d, K.batch_label_guard, d)
        bound_ok = not K.FAIL

        write_caption("en", "per batch of 1024 heterogeneous samples")
        del K.FAIL[:]
        run(d, K.batch_label_guard, d)
        literal_rejected = bool(K.FAIL)
        write_caption("en", "per batch of \\SolverBatchSize\\ samples")

        write_macros("en", "2\\,048")
        del K.FAIL[:]
        run(d, K.batch_label_guard, d)
        drift_rejected = bool(K.FAIL)
        del K.FAIL[:]
    finally:
        K.RES = old_res
        shutil.rmtree(d, ignore_errors=True)
    print("   bound through macros = %s ; hand-written literal rejected = %s "
          "; drifted macro rejected = %s"
          % (bound_ok, literal_rejected, drift_rejected))
    return bound_ok, literal_rejected, drift_rejected


def _synthetic_realgeo(overlap_arms=3, gamma=0.05):
    """A minimal but structurally valid results/realgeo.json.

    ``overlap_arms`` sets how many of the five arms have plateau ranges that
    overlap between the two families, which is the quantity the wording gate
    turns on.
    """
    V = [0.05, 0.10, 0.25, 0.50, 1.00]
    fam = {"real": {}, "synthetic": {}}
    for i, v in enumerate(V):
        k = "%.2f" % v
        ov = i < overlap_arms
        fam["real"][k] = {"f_star": 4.25, "r5": 8.0, "in_band": 0.07,
                          "plateau": 0.70,
                          "plateau_lo": 0.60, "plateau_hi": 0.80}
        fam["synthetic"][k] = {"f_star": 4.50, "r5": 7.5, "in_band": 0.09,
                               "plateau": 0.90,
                               "plateau_lo": 0.70 if ov else 2.00,
                               "plateau_hi": 1.10 if ov else 3.00}
    return {
        "n": 200, "gamma": gamma,
        "f_test": list(K.C.F_TEST), "vmax_list": V,
        "summary": {"family": fam},
        "source": {
            "windows": {"west": [-200.0, 2800.0, 0.0, 3000.0],
                        "centre": [3000.0, 6000.0, 0.0, 3000.0],
                        "east": [6200.0, 9200.0, 0.0, 3000.0]},
            "cross_check": {"marmousi_mirror_300x298": {"whole_sample_r": 0.884}},
        },
    }


def probe_realgeo_gate():
    """Third direction of the level-3 gate: design, window rule, wording.

    The gate has to accept the honest record and reject two distinct defects --
    a protocol drift that makes 'only the geometry moves' false, and a
    manuscript that has dropped the refusal to claim a plateau separation.  The
    second is the one that matters editorially: the data support a boundary and
    a jump claim but not a plateau claim, so a text that quietly stops saying so
    is over-claiming without changing a single number.
    """
    import check_claims as K
    d = tempfile.mkdtemp(prefix="_realgeo_probe_")
    res = os.path.join(d, "results")
    os.makedirs(res)
    jp = os.path.join(res, "realgeo.json")

    def write_json(obj):
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(obj, fh)

    def write_paper(phrase_en, phrase_cn, drop_float=False):
        for lang, phrase in (("en", phrase_en), ("cn", phrase_cn)):
            os.makedirs(os.path.join(d, "paper", lang), exist_ok=True)
            with open(os.path.join(d, "paper", lang, "gen_macros.tex"), "w",
                      encoding="utf-8") as fh:
                fh.write("\\newcommand{\\RealPlateauOverlapArms}{3}\n"
                         "\\newcommand{\\RealMirrorR}{0.88}\n")
            body = phrase
            if not drop_float:
                body += " see \\ref{fig:realgeo} and \\ref{tab:realgeo}"
            with open(os.path.join(d, "paper", lang, "main.tex"), "w",
                      encoding="utf-8") as fh:
                fh.write("text %s\n\\label{fig:realgeo}\n\\label{tab:realgeo}\n"
                         % body)

    old_res = K.RES
    K.RES = res
    try:
        # (1) honest record, both sentences present -> accepted
        write_json(_synthetic_realgeo())
        write_paper("the plateau does not separate the two families "
                    "\\RealPlateauOverlapArms",
                    "平台高度并不分离两者 \\RealPlateauOverlapArms")
        del K.FAIL[:]
        run(d, K.realgeo_guard, d)
        honest_ok = not K.FAIL

        # (2) protocol drift: gamma no longer matches common.py
        write_json(_synthetic_realgeo(gamma=0.10))
        del K.FAIL[:]
        run(d, K.realgeo_guard, d)
        drift_rejected = bool(K.FAIL)

        # (3) wording: the refusal sentence is gone from the English text
        write_json(_synthetic_realgeo())
        write_paper("the plateau is preserved across the two families "
                    "\\RealPlateauOverlapArms",
                    "平台高度并不分离两者 \\RealPlateauOverlapArms")
        del K.FAIL[:]
        run(d, K.realgeo_guard, d)
        wording_rejected = bool(K.FAIL)

        # (4) the float is no longer referenced in either language
        write_paper("the plateau does not separate the two families "
                    "\\RealPlateauOverlapArms",
                    "平台高度并不分离两者 \\RealPlateauOverlapArms",
                    drop_float=True)
        del K.FAIL[:]
        run(d, K.realgeo_guard, d)
        float_rejected = bool(K.FAIL)

        # (5) the measurement no longer supports the non-separation sentence
        write_json(_synthetic_realgeo(overlap_arms=1))
        del K.FAIL[:]
        run(d, K.realgeo_guard, d)
        overclaim_rejected = bool(K.FAIL)
        del K.FAIL[:]
    finally:
        K.RES = old_res
        shutil.rmtree(d, ignore_errors=True)
    print("   honest record accepted = %s ; protocol drift rejected = %s ; "
          "missing refusal rejected = %s ; unreferenced float rejected = %s ; "
          "unsupported refusal rejected = %s"
          % (honest_ok, drift_rejected, wording_rejected, float_rejected,
             overclaim_rejected))
    return honest_ok, drift_rejected, wording_rejected, float_rejected, \
        overclaim_rejected


def probe_freshness_gate():
    """`--steps build` alone must not compile a manuscript whose generated
    bibliography predates it.  That happened: a paragraph citing four new
    references was added to the Chinese manuscript and the build stage was run
    on its own, so the Chinese PDF shipped with those four citations undefined
    and the reference list numbered against the previous order.  Both directions
    are checked, because a guard that always raises would also stop every
    correct build.
    """
    import make_paper as M
    d = tempfile.mkdtemp(prefix="_fresh_probe_")
    try:
        man = os.path.join(d, "main.tex")
        gen = os.path.join(d, "gen_bib.tex")
        with open(man, "w", encoding="utf-8") as fh:
            fh.write("body \\cite{x}\n")
        # the real generated set, taken from the build stage's own list
        for g in M.GEN_FILES:
            with open(os.path.join(d, g), "w", encoding="utf-8") as fh:
                fh.write("\\newcommand{\\x}{1}\n")
        base = os.path.getmtime(man)
        # stale: generated files 60 s older than the manuscript
        # the real generated set, taken from the build stage's own list
        for g in M.GEN_FILES:
            p = os.path.join(d, g)
            os.utime(p, (base - 60, base - 60))
        os.utime(man, (base, base))
        stale_caught = False
        try:
            M.check_generated_fresh(d, "probe")
        except RuntimeError as e:
            stale_caught = "gen_bib.tex" in str(e)
        # fresh: generated files 60 s newer
        # the real generated set, taken from the build stage's own list
        for g in M.GEN_FILES:
            p = os.path.join(d, g)
            os.utime(p, (base + 60, base + 60))
        clean_passes = True
        try:
            M.check_generated_fresh(d, "probe")
        except RuntimeError:
            clean_passes = False
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("   stale caught = %s ; fresh accepted = %s"
          % (stale_caught, clean_passes))
    return stale_caught, clean_passes


def probe_macro_spacing():
    """The control-word space swallow, and the declared exception to it.

    ``\\CapFactor changes`` prints "2.6changes" because TeX ends the control
    word at the space and eats it.  On the delivered PDF the sentence read
    "(0.084 against 0.085at 2.6-fold the parameters)", and no numeric gate could
    see it: the fused token still parses as a number.  Three directions are
    required here.  The defect must be caught; the braced fix must be accepted;
    and the declared unit exception (``\\FnoParams M`` meaning "10.3M") must stay
    silent, because a gate that always fires would force a rewrite of correct
    text.
    """
    import check_claims as K
    d = tempfile.mkdtemp(prefix="_macro_probe_")
    try:
        mac = os.path.join(d, "paper", "en", "gen_macros.tex")
        man = os.path.join(d, "paper", "en", "main.tex")
        os.makedirs(os.path.dirname(mac))
        with open(mac, "w", encoding="utf-8") as fh:
            fh.write("\\newcommand{\\CapFactor}{2.6}\n"
                     "\\newcommand{\\FnoParams}{10.25}\n"
                     "\\newcommand{\\NDataTop}{12\\,288}\n")

        def fires(body):
            with open(man, "w", encoding="utf-8") as fh:
                fh.write(body)
            del K.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                K.macro_space_guard(d)
            return bool(K.FAIL)

        res = dict(
            swallowed=fires("the error moves \\CapFactor changes with it.\n"),
            braced_ok=not fires("the error moves \\CapFactor{} changes with it.\n"),
            unit_ok=not fires("its \\FnoParams M budget is the reference.\n"),
            # A generated macro glued to punctuation is the same defect wearing
            # different clothes; it must not ride on the unit exception.
            glued_bad=fires("the ladder stops at \\NDataTop(16\\,384 total).\n"),
        )
        del K.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for k in sorted(res):
        print("   %-12s %s" % (k, res[k]))
    return res


if __name__ == "__main__":
    sys.exit(main())

