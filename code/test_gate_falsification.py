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
import hashlib
import tempfile
import contextlib
import importlib.util

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

    The EN abstract cap is counted on the RENDERED text, so the probe drives
    it through the module-level pdf_text seam: a 199-word render is accepted,
    a 201-word render is rejected, an unreadable PDF is rejected, and a PDF
    older than main.tex is refused rather than silently counted from the
    stale render.  The counting function itself must report an error -- not a
    zero, which would pass every cap -- when the region is absent.

    CN keeps the source-based reading-unit count and must fire past its cap
    and stay silent at it.  The placeholder half must fire on `$--$` and on a
    bare `--` cell, and stay SILENT on a legitimate numeric range
    (`0.085--0.127`) and on an em-dash aside: a guard that flagged every
    en-dash would be turned off by the first person who saw it, and then the
    real defect would be invisible again.
    """
    import time
    d = tempfile.mkdtemp(prefix="_gate_probe10_")
    seam = K.pdf_text
    try:
        for lang in ("en", "cn"):
            os.makedirs(os.path.join(d, "paper", lang))
        en_tex = os.path.join(d, "paper", "en", "main.tex")
        en_pdf = os.path.join(d, "paper", "en", "main.pdf")
        cn_tex = os.path.join(d, "paper", "cn", "main.tex")

        def render(n):
            return ("Abstract\n\n" + " ".join(["word"] * n)
                    + "\n\nKeywords: a; b\n")

        def en_tree(render_text, stale=False):
            with open(en_tex, "w", encoding="utf-8") as fh:
                fh.write("\\abstract{a short source abstract}\n")
            with open(en_pdf, "w", encoding="utf-8") as fh:
                fh.write("%PDF-1.4 probe dummy\n")
            now = time.time()
            os.utime(en_tex, (now, now))
            os.utime(en_pdf, (now - 3600, now - 3600) if stale
                     else (now + 60, now + 60))
            K.pdf_text = lambda p: render_text

        def cn_tree(n_cjk):
            with open(cn_tex, "w", encoding="utf-8") as fh:
                fh.write("\\begin{abstract}\n%s\n\\end{abstract}\n"
                         % ("\u6c49" * n_cjk))

        def cell_tree(cell):
            with open(os.path.join(d, "paper", "en", "gen_tab_x.tex"), "w",
                      encoding="utf-8") as fh:
                fh.write("\\begin{tabular}{ccc}\n$f$ & $r$ & note \\\\\n"
                         "1.5 & 0.085 & %s \\\\\n\\end{tabular}\n" % cell)

        # clean: 199 words rendered, CN at its cap, a legal range and aside
        en_tree(render(199)); cn_tree(400)
        cell_tree("a range 0.085--0.127 --- an aside")
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        clean = not K.FAIL

        # one word over the cap
        en_tree(render(201)); cn_tree(10); cell_tree("note")
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        long_abs = bool(K.FAIL)

        # a stale render must be refused, not counted
        en_tree(render(199), stale=True); cn_tree(10)
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        stale_pdf = bool(K.FAIL)

        # an unreadable PDF must be refused
        en_tree(render(199)); K.pdf_text = lambda p: None; cn_tree(10)
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        unreadable = bool(K.FAIL)

        # the counting function itself: an absent region is an error, not 0
        n, err = K.rendered_abstract_words(render(199))
        _, err2 = K.rendered_abstract_words("1. Introduction\nbody text\n")
        counter = (n == 199 and err is None and err2 is not None)

        # CN one unit over its cap
        en_tree(render(199)); cn_tree(401)
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        long_cn = bool(K.FAIL)

        # placeholder shapes
        en_tree(render(199)); cn_tree(10); cell_tree("$--$")
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        dollar = bool(K.FAIL)

        en_tree(render(199)); cell_tree("--")
        del K.FAIL[:]
        run(d, K.submission_element_guard, d)
        bare = bool(K.FAIL)
        del K.FAIL[:]
    finally:
        K.pdf_text = seam
        shutil.rmtree(d, ignore_errors=True)
    return clean, long_abs, stale_pdf, unreadable, counter, long_cn, dollar, bare


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



def probe_delivery_parity():
    """probe 14: the audited body must be the delivered body.

    Reproduces the 2026-10-22 defect in a sandbox: paper/en/main.tex four days
    behind submission/paper_en/main.tex.  Expected: identical bodies pass,
    divergent bodies are reported.  Without this the four gates that read
    paper/ can all be green about a manuscript nobody will submit.
    """
    import guards_gridstats as G
    d = tempfile.mkdtemp(prefix="_delivery_probe_")
    try:
        for lang in ("en",):
            os.makedirs(os.path.join(d, "paper", lang))
            os.makedirs(os.path.join(d, "submission", "paper_%s" % lang))

        def fires(a_body, b_body):
            for rel, body in (("paper/en/main.tex", b_body),
                              ("submission/paper_en/main.tex", a_body)):
                with open(os.path.join(d, rel), "w", encoding="utf-8") as fh:
                    fh.write(body)
            del G.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                G.delivery_guard(d)
            return bool(G.FAIL)

        same = fires("body with the deep section\n", "body with the deep section\n")
        stale = fires("body with the deep section\n", "body without it\n")
        del G.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return (not same), stale


def probe_text_residue():
    """probe 17: a macro that never expanded must not reach the reader.

    Reproduces the 2026-10-22 defect in a sandbox: the delivered sentence read
    ``not small:/n//delta64//to256 = //GridDeltaTop at f = 12``, so the reader
    saw the macro's name where its value belonged.  Expected: a properly
    referenced macro passes, both residue shapes are rejected, and a URL that
    legitimately contains ``//`` is not mistaken for residue.
    """
    import guards_text as T
    d = tempfile.mkdtemp(prefix="_text_probe_")
    try:
        os.makedirs(os.path.join(d, "submission", "paper_en"))
        folder = os.path.join(d, "submission", "paper_en")
        with open(os.path.join(folder, "gen_macros.tex"), "w", encoding="utf-8") as fh:
            fh.write("\\newcommand{\\GridDeltaTop}{0.415}\n"
                     "\\newcommand{\\CIPlatLo}{0.933}\n")

        def fires(body):
            with open(os.path.join(folder, "main.tex"), "w", encoding="utf-8") as fh:
                fh.write(body)
            del T.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                T.text_residue_guard(d, langs=("en",))
            return bool(T.FAIL)

        res = dict(
            clean=not fires("The difference is $\\delta_{64\\to256}=\\GridDeltaTop$ "
                            "at $f=12$.\n"),
            name_leak=fires("The difference is //GridDeltaTop at $f=12$.\n"),
            newline=fires("The difference is not small:/n$\\GridDeltaTop$.\n"),
            url_ok=not fires("Published as https://doi.org/10.3390/app16125997.\n"),
        )
        del T.FAIL[:]

        # a figure rendered into the delivery tree but never into the build tree
        # is a fatal compile error waiting to happen; assert the asymmetry is
        # reported, not only the absence
        os.makedirs(os.path.join(folder, "figs"))
        os.makedirs(os.path.join(d, "paper", "en", "figs"))
        open(os.path.join(folder, "figs", "fig_x.pdf"), "w").close()

        def fig_fires(body, in_build_tree):
            # the gate reads the BUILD tree, so the probe must write there --
            # writing to submission/ and reading paper/ is the very split the
            # gate exists to detect, and it would make the probe a no-op
            with open(os.path.join(d, "paper", "en", "main.tex"), "w",
                      encoding="utf-8") as fh:
                fh.write(body)
            p = os.path.join(d, "paper", "en", "figs", "fig_x.pdf")
            if in_build_tree and not os.path.exists(p):
                open(p, "w").close()
            if not in_build_tree and os.path.exists(p):
                os.remove(p)
            del T.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                T.figure_guard(d, langs=("en",))
            return bool(T.FAIL)

        res["fig_ok"] = not fig_fires("\\includegraphics{fig_x.pdf}\n", True)
        res["fig_missing"] = fig_fires("\\includegraphics{fig_x.pdf}\n", False)
        del T.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return res


def probe_grid_guard():
    """probe 15: the three-grid guard must fire on the claims it protects.

    Three tampers, each of the kind that would make a flattering but false
    sentence true: an order p that is second-order, a plateau that moves by
    half under refinement, and a quadrature bound that holds everywhere
    (which would contradict the text's explicit in-band counterexample).
    Expected: the delivered archive passes, all three tampers are rejected.
    """
    import guards_gridstats as G
    src = os.path.join(K.RES, "grid_conv_three.json")
    if not os.path.exists(src):
        return (False, False, False, False)
    base = json.load(open(src, encoding="utf-8"))
    d = tempfile.mkdtemp(prefix="_grid_probe_")

    def fires(obj):
        with open(os.path.join(d, "grid_conv_three.json"), "w",
                  encoding="utf-8") as fh:
            json.dump(obj, fh)
        del G.FAIL[:]
        with contextlib.redirect_stdout(io.StringIO()):
            G.grid_guard(d)
        return bool(G.FAIL)

    try:
        clean = not fires(base)

        o1 = json.loads(json.dumps(base))
        o1["ref_0.25"]["rows"][0]["p"] = 2.4
        bad_order = fires(o1)

        o2 = json.loads(json.dumps(base))
        for r in o2["model_0.25"]["rows"]:
            if r["f"] >= 6.0:
                r["e256"] = r["e64"] * 1.5
        bad_plateau = fires(o2)

        o3 = json.loads(json.dumps(base))
        for r in o3["model_0.25"]["rows"]:
            if r["f"] == 4.0:
                r["e256"] = 0.10          # inside the quadrature bound
        bad_quad = fires(o3)
        del G.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return clean, bad_order, bad_plateau, bad_quad



def probe_doi_placeholder():
    """probe 16: a synthesised DOI must not reach the rendered text.

    The MDPI class prints a DOI derived from \articlenumber on every page; with
    \articlenumber{0} that is 10.3390/app1010000.  Expected: a page carrying
    only bibliography DOIs passes, and the placeholder is reported.
    """
    import guards_gridstats as G
    bib = "  doi = {10.3390/app16125997}\n"
    clean = not G._doi_offenders(
        "Appl. Sci. 2026, 16, 5997. https://doi.org/10.3390/app16125997", bib)[2]
    bad = bool(G._doi_offenders(
        "Version submitted. https://doi.org/10.3390/app1010000", bib)[2])
    return clean, bad


def _tree_fingerprint():
    """md5 of every .tex under submission/ and paper/ -- files no probe may write.

    The suite audits the delivered manuscript, so it must not edit it, and it
    did: probe 25 mutated submission/paper_en/main.tex in place and wrote it
    back through text mode, which converted the delivered source to CRLF on
    every run and made the byte-level mirror gate accuse the archive of being
    built before an edit that never happened.  An instrument that rewrites its
    subject is not measuring it, and a kill between mutation and restore would
    have left a withdrawn sentence in the manuscript of record.  Checked here
    rather than trusted, because the failure is invisible in the probe output.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {}
    for sub in ("submission", "paper"):
        for dirpath, _, names in os.walk(os.path.join(root, sub)):
            for n in names:
                if n.endswith(".tex"):
                    p = os.path.join(dirpath, n)
                    with open(p, "rb") as fh:
                        out[os.path.relpath(p, root)] = hashlib.md5(
                            fh.read()).hexdigest()
    return out


def main():
    before = _tree_fingerprint()
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
    r_ok, r_drift, r_word, r_float, r_over, r_half = probe_realgeo_gate()
    if all((r_ok, r_drift, r_word, r_float, r_over, r_half)):
        print("PROBE 8: honest record accepted and all five defects rejected")
    else:
        print("PROBE 8: MISBEHAVED accepted=%s drift=%s wording=%s float=%s "
              "overclaim=%s half=%s"
              % (r_ok, r_drift, r_word, r_float, r_over, r_half))

    print("-- probe 9 (macro spacing after a control word) --")
    msp = probe_macro_spacing()
    if all(msp.values()):
        print("PROBE 9: swallow caught, braced fix and unit exception accepted, "
              "glued-to-punctuation caught")
    else:
        print("PROBE 9: MISBEHAVED %s" % msp)

    print("-- probe 10 (submission elements: abstract budget + placeholder cells) --")
    (s_clean, s_long, s_stale, s_unread, s_counter, s_cn, s_dollar,
     s_bare) = probe_submission_elements()
    if all((s_clean, s_long, s_stale, s_unread, s_counter, s_cn,
            s_dollar, s_bare)):
        print("PROBE 10: 199-word render and CN at cap accepted; 201-word "
              "render, stale PDF, unreadable PDF, CN over cap, $--$ and bare "
              "-- cells all rejected")
    else:
        print("PROBE 10: MISBEHAVED clean=%s long=%s stale=%s unread=%s "
              "counter=%s cn=%s dollar=%s bare=%s"
              % (s_clean, s_long, s_stale, s_unread, s_counter, s_cn,
                 s_dollar, s_bare))

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

    print("-- probe 14 (audited body vs delivered body) --")
    d_ok, d_stale = probe_delivery_parity()
    if d_ok and d_stale:
        print("PROBE 14: matching bodies accepted, a stale audited body rejected")
    else:
        print("PROBE 14: MISBEHAVED clean=%s stale=%s" % (d_ok, d_stale))

    print("-- probe 15 (three-grid claims) --")
    g_clean, g_order, g_plat, g_quad = probe_grid_guard()
    if all((g_clean, g_order, g_plat, g_quad)):
        print("PROBE 15: delivered archive accepted; a second-order p, a "
              "half-shifted plateau and an everywhere-valid quadrature bound "
              "all rejected")
    else:
        print("PROBE 15: MISBEHAVED clean=%s order=%s plateau=%s quad=%s"
              % (g_clean, g_order, g_plat, g_quad))

    print("-- probe 16 (fabricated DOI in the rendered text) --")
    p_clean, p_bad = probe_doi_placeholder()
    if p_clean and p_bad:
        print("PROBE 16: bibliography DOIs accepted, the class placeholder "
              "rejected")
    else:
        print("PROBE 16: MISBEHAVED clean=%s bad=%s" % (p_clean, p_bad))

    print("-- probe 17 (a macro that never expanded / a figure in the wrong tree) --")
    t_res = probe_text_residue()
    if all(t_res.values()):
        print("PROBE 17: referenced macro, DOI URL and present figure accepted; "
              "an unexpanded macro name, a /n residue and a figure missing from "
              "the build tree all rejected")
    else:
        print("PROBE 17: MISBEHAVED %s" % t_res)

    print("-- probe 18 (a baseline that is not on the evaluation media) --")
    bb = probe_baseline_pairing()
    if all(bb.values()):
        print("PROBE 18: paired ratio, tight interval and consistent macros "
              "accepted; an unpaired draw, a ratio outside its interval, a "
              "decoration-only pairing, an unchanged denominator and a "
              "false 'every' all rejected (and every mutation verified to "
              "have actually applied)")
    else:
        print("PROBE 18: MISBEHAVED %s" % bb)

    print("-- probe 19 (a ladder table that does not come from its curves) --")
    lp = probe_locator_ladders()
    if all(lp.values()):
        print("PROBE 19: the reproducible ladder table and its caveat accepted; "
              "a fabricated rung, a moved mode step, a boundary on the widest "
              "band, a flattened onset, an everywhere-firing f* and a drifted "
              "macro all rejected")
    else:
        print("PROBE 19: MISBEHAVED %s" % lp)

    print("-- probe 20 (a range typeset as subtraction) --")
    rd = probe_range_dash()
    if all(rd.values()):
        print("PROBE 20: a true en dash and a figure-extracted '--' both stay "
              "quiet; the two-minus signature is rejected")
    else:
        print("PROBE 20: MISBEHAVED %s" % rd)

    print("-- probe 21 (a resolution check that does not come from its curves) --")
    rs = probe_resolution_study()
    if all(rs.values()):
        print("PROBE 21: the reproducible resolution table accepted; a moved "
              "boundary, a seed-dependent shift, an inflated jump, a broken "
              "paired control, both cross-path checks and a grid-dependent band "
              "fraction all rejected")
    else:
        print("PROBE 21: MISBEHAVED %s" % rs)

    print("-- probe 22 (the prose's verdicts against the archive) --")
    rc = probe_resolution_claims()
    if all(rc.values()):
        print("PROBE 22: all six verdicts accepted on the real archive; a moved "
              "boundary, an overlapping jump range, a lost order reversal and a "
              "premature plateau separation all reported as contradictions")
    else:
        print("PROBE 22: MISBEHAVED %s" % rc)

    print("-- probe 23 (spectrum-matched control and the per-sample bound) --")
    su = probe_surrogate_gate()
    if su.get("skipped"):
        print("PROBE 23: SKIPPED (archive missing)")
        su = dict(clean=True)
    elif all(su.values()):
        print("PROBE 23: the real archive accepted; a flipped plateau sign, an "
              "inflated jump, a broken per-sample bound, a broken aggregate case, "
              "an unmatched spectrum, a hidden pointwise violation and a "
              "restored 'the bound holds' sentence all rejected")
    else:
        print("PROBE 23: MISBEHAVED %s" % su)

    print("-- probe 24 (fit-depth/sum-depth correction) --")
    bx = probe_bax_gate()
    if all(bx.values()):
        print("PROBE 24: the corrected tree accepted; a broken PASS gate, a "
              "perturbed a025 invariant, a stripped protocol key, a "
              "restored 'fails at v>=0.50' sentence and a deleted disclosure "
              "all rejected")
    else:
        print("PROBE 24: MISBEHAVED %s" % bx)

    sr = probe_si_refs()
    if all(sr.values()):
        print("PROBE 25: the real map accepted; a hand-renumbered map, an "
              "undefined key, a reintroduced literal, a key naming the wrong "
              "float and a dropped S prefix all rejected")
    else:
        print("PROBE 25: MISBEHAVED %s" % sr)

    print("-- probe 26 (a seed-count qualifier that names the wrong arms) --")
    sc = probe_seed_claim()
    if all(sc.values()):
        print("PROBE 26: a scope named by NArmsAtMax and the explicit "
              "NumSeeds..NumSeedsMax range both stay quiet; a stale "
              "'anchored arm' qualifier is rejected")
    else:
        print("PROBE 26: MISBEHAVED %s" % sc)

    print("-- probe 27 (the envelope documents outside every .tex gate) --")
    dt = probe_delivery_texts()
    if all(dt.values()):
        print("PROBE 27: the current envelope is accepted; a drifted title, a "
              "retired claim, a wrong version pin, a conflicting DOI sentence, "
              "an undeclared supplement, an abstract span that swallows the "
              "front-matter sidebar, and two archive defects are all rejected")
    else:
        print("PROBE 27: MISBEHAVED %s" % dt)

    good = (caught and ignored and named and duped and good3 and bad3 and fin3
            and good4 and bad4t and bad4c and all(sty.values())
            and fresh_stale and fresh_ok
            and b_ok and b_lit and b_drift
            and r_ok and r_drift and r_word and r_float and r_over and r_half
            and all(msp.values())
            and s_clean and s_long and s_stale and s_unread and s_counter
            and s_cn and s_dollar and s_bare
            and w_braced and w_bare and w_plain and w_caught
            and p_honest and p_fno and p_unet and p_mac
            and l_clean and l_dup and l_unref
            and d_ok and d_stale
            and g_clean and g_order and g_plat and g_quad
            and p_clean and p_bad
            and all(t_res.values())
            and all(bb.values())
            and all(lp.values())
            and all(rd.values())
            and all(rs.values())
            and all(rc.values())
            and all(su.values())
            and all(bx.values())
            and all(sr.values())
            and all(sc.values())
            and all(dt.values()))
    after = _tree_fingerprint()
    touched = sorted(k for k, v in after.items() if before.get(k) != v)
    if touched:
        print("the suite modified the tree it audits: %s" % touched[:5])
    good = good and not touched
    print("PROBE RESULT: %s" % ("all gates fire on known-bad input" if good else "GATE FAILED"))
    return 0 if good else 1


def probe_baseline_pairing():
    """probe 18: the plateau-to-baseline ratio must be paired, and shown to be.

    Five ways for this to go wrong, each reproduced on a copy of the real
    result files so that a passing run means the guard reads live data:

    1. the test media stop being paired across frequency (two distinct `v`
       digests in one contrast);
    2. the ratio falls outside its own interval;
    3. the paired interval is no narrower than the unpaired counterfactual,
       i.e. the pairing carries no information and the word is decorative;
    4. the 200- and 600-media denominators are identical, so the switch of
       denominator changed nothing and its cost is unjustified;
    5. the printed ratio and the interval table disagree, or `TrivRatioLo`
       drops below one while the prose says every arm sits above its baseline.

    Expected: the untouched copy passes; each mutation is rejected.
    """
    import guards_gridstats as G
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_res = os.path.join(root, "results")
    d = tempfile.mkdtemp(prefix="_base_probe_")
    try:
        res = os.path.join(d, "results")
        os.makedirs(res)
        for fn in ("triv_paired.json", "ci_plateau.json"):
            shutil.copy(os.path.join(src_res, fn), res)
        os.makedirs(os.path.join(d, "paper", "en"))
        gm_src = os.path.join(root, "paper", "en", "gen_macros.tex")
        gm = os.path.join(d, "paper", "en", "gen_macros.tex")
        shutil.copy(gm_src, gm)
        tp = os.path.join(res, "triv_paired.json")
        ci = os.path.join(res, "ci_plateau.json")
        base_tp = json.load(open(tp, encoding="utf-8"))
        base_ci = json.load(open(ci, encoding="utf-8"))
        base_gm = open(gm, encoding="utf-8").read()

        def fires(tp_mut=None, ci_mut=None, gm_mut=None):
            json.dump(tp_mut or base_tp, open(tp, "w", encoding="utf-8"))
            json.dump(ci_mut or base_ci, open(ci, "w", encoding="utf-8"))
            open(gm, "w", encoding="utf-8").write(
                base_gm if gm_mut is None else base_gm.replace(*gm_mut))
            del G.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                G.baseline_pairing_guard(res=res, root=d)
            return bool(G.FAIL)

        def mutate(obj, fn):
            new = json.loads(json.dumps(obj))
            fn(new)
            return new

        unpaired = mutate(base_tp, lambda o: o["data"]["0.25"].update(
            _v_md5_unique=2))
        outside = mutate(base_ci, lambda o: o["rows"][2].update(
            ratio_hi=o["rows"][2]["ratio"] - 1e-3))
        flat = mutate(base_ci, lambda o: o["rows"][2].update(
            ratio_un_lo=o["rows"][2]["ratio_lo"],
            ratio_un_hi=o["rows"][2]["ratio_hi"]))
        same500 = mutate(base_tp, lambda o: o.update(
            rel_diff_vs_600=dict(mean=0.0002, max=0.001)))
        def _macro(name):
            m = re.search(r"\\newcommand\{\\%s\}\{([0-9.]+)\}" % name, base_gm)
            return None if m is None else m.group(1)

        # The mutations are built from what the generator prints *now*.  They
        # used to be hard-coded ("\\PlatTrivMid}{1.027"), and W2 then moved that
        # macro to 1.030: the replacement stopped matching, the file went to the
        # guard unmodified, and a probe that mutates nothing reports exactly what
        # a working guard reports.  A stale literal is therefore a failed probe,
        # not a quiet one.
        rl, pm = _macro("TrivRatioLo"), _macro("PlatTrivMid")
        word = (("\\TrivRatioLo}{%s" % rl, "\\TrivRatioLo}{0.98")
                if rl is not None else None)
        drift = (("\\PlatTrivMid}{%s" % pm,
                  "\\PlatTrivMid}{%.3f" % (float(pm) * 1.09))
                 if pm is not None else None)

        def applies(m):
            return m is not None and m[0] in base_gm and m[0] != m[1]

        res_d = dict(
            clean=not fires(),
            unpaired=fires(tp_mut=unpaired),
            outside=fires(ci_mut=outside),
            no_shrink=fires(ci_mut=flat),
            same_size=fires(tp_mut=same500),
            word=fires(gm_mut=word) if applies(word) else False,
            drift=fires(gm_mut=drift) if applies(drift) else False,
            mutations_applied=applies(word) and applies(drift),
        )
        del G.FAIL[:]
        return res_d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def probe_locator_ladders():
    """probe 19: the ladder table must be reproducible, and every claim in it.

    Six ways for it to go wrong, each on a copy of the real table while the
    curves underneath stay real:

    1. a capacity rung is given a different f*, i.e. the table asserts a
       movement the curves do not contain;
    2. the mode ladder's largest step is moved on one rung, which would break
       "the budget does not move the boundary";
    3. the widest band is given a boundary at the last node, converting the
       paper's stated absence into a fabricated measurement;
    4. the half-plateau onset is flattened across the mode ladder, erasing the
       one caveat the text admits;
    5. f* is made to fire at every mode budget, contradicting the text;
    6. the printed macro disagrees with the table.

    Expected: the untouched copy passes; each mutation is rejected.
    """
    import guards_locators as LG
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = tempfile.mkdtemp(prefix="_loc_probe_")
    try:
        res = os.path.join(d, "results")
        os.makedirs(res)
        src = os.path.join(root, "results", "locators.json")
        dst = os.path.join(res, "locators.json")
        shutil.copy(src, dst)
        os.makedirs(os.path.join(d, "paper", "en"))
        gm = os.path.join(d, "paper", "en", "gen_macros.tex")
        shutil.copy(os.path.join(root, "paper", "en", "gen_macros.tex"), gm)
        base = json.load(open(dst, encoding="utf-8"))
        base_gm = open(gm, encoding="utf-8").read()

        def fires(mut=None, gm_sub=None):
            json.dump(mut or base, open(dst, "w", encoding="utf-8"))
            open(gm, "w", encoding="utf-8").write(
                base_gm if gm_sub is None else base_gm.replace(*gm_sub))
            del LG.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                LG.locator_guard(res=res, root=d, eval_res=os.path.join(root, "results"))
            return bool(LG.FAIL)

        def mutate(fn):
            new = json.loads(json.dumps(base))
            fn(new)
            return new

        def cap_moves(o):
            o["ladders"]["cap"]["rows"][1]["fstar"] = 4.25

        def mode_step(o):
            o["ladders"]["mode"]["rows"][0]["step"] = [3.50, 4.00]

        def widest_band(o):
            o["ladders"]["band"]["rows"][-1]["step"] = [11.00, 12.00]

        def flat_onset(o):
            for r in o["ladders"]["mode"]["rows"]:
                if r["onset"] is not None:
                    r["onset"] = 4.25

        def all_fire(o):
            for r in o["ladders"]["mode"]["rows"]:
                if r["fstar"] is None:
                    r["fstar"] = 4.50

        got = dict(
            clean=not fires(),
            cap_moves=fires(mutate(cap_moves)),
            mode_step=fires(mutate(mode_step)),
            widest_band=fires(mutate(widest_band)),
            flat_onset=fires(mutate(flat_onset)),
            all_fire=fires(mutate(all_fire)),
            macro=fires(gm_sub=("\\LocModeSilent}{4", "\\LocModeSilent}{0")),
        )
        del LG.FAIL[:]
        return got
    finally:
        shutil.rmtree(d, ignore_errors=True)


def probe_range_dash():
    """probe 20: a numeric range must not read as subtraction -- and must not
    be reported when the extraction merely collapses an en dash.

    Three inputs:
      good      an en dash between two numbers, which is the correct output;
      bad       the signature of a `--` typed inside $...$, i.e. two U+2212;
      figeq     two ASCII hyphens, which is how pdftotext renders an en dash
                inside a matplotlib legend, verified by rendering page 34.

    The middle must fire and the other two must stay quiet.  Testing only the
    bad input would leave the guard free to reject correct manuscripts, which
    is the failure mode that costs more.
    """
    import guards_text as T
    cases = dict(
        good="the step occupies 4.00\u20134.25 on every rung\n",
        bad="the step occupies 4.00 \u2212 \u22124.25 on every rung\n",
        figeq="baseline, gamma=0.05 (4.53--10.02)\n",
    )
    out = {}
    for k, v in cases.items():
        del T.FAIL[:]
        with contextlib.redirect_stdout(io.StringIO()):
            T.range_dash_guard(texts={"en": v}, langs=("en",), root=".")
        out[k] = bool(T.FAIL)
    del T.FAIL[:]
    return dict(good_quiet=not out["good"], bad_fires=out["bad"],
                fig_quiet=not out["figeq"])


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


def probe_resolution_study():
    """probe 21: the resolution check must be a measurement, not a slogan.

    Eight ways for it to be wrong, each on a copy of the archive while the
    stored curves stay real:

    1. the refined arm's f* is moved, i.e. the table claims the boundary moved
       while the text says it did not;
    2. only one refined seed's f* moves, so the seed-mean is still stable and
       the invariance would print as seed-robust when it is not;
    3. the jump factor is inflated at the refined grid beyond the seed spread;
    4. the paired-continuum control is given a shifted f* on one grid, turning
       the media-held-fixed argument into its opposite;
    5. the coarse Marmousi side is edited so it no longer reproduces
       realgeo.json -- the cross-path check must notice;
    6. the coarse synthetic curve is edited so it no longer reproduces the main
       evaluator's sweep;
    7. the 128^2 band fraction of the windows is moved, which would make the
       real-vs-synthetic contrast grid-dependent;
    8. a printed macro disagrees with the table.

    Expected: the untouched copy passes; every mutation is rejected.
    """
    import guards_res as RG
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = tempfile.mkdtemp(prefix="_res_probe_")
    try:
        res = os.path.join(d, "results")
        os.makedirs(res)
        for nm in ("res_study.json", "realgeo.json", "eval_main_fno_a025_s0.json"):
            shutil.copy(os.path.join(root, "results", nm), os.path.join(res, nm))
        os.makedirs(os.path.join(d, "paper", "en"))
        gm = os.path.join(d, "paper", "en", "gen_macros.tex")
        shutil.copy(os.path.join(root, "paper", "en", "gen_macros.tex"), gm)
        base = json.load(open(os.path.join(res, "res_study.json"), encoding="utf-8"))
        base_gm = open(gm, encoding="utf-8").read()

        def fires(mut=None, gm_sub=None):
            json.dump(mut or base, open(os.path.join(res, "res_study.json"),
                                        "w", encoding="utf-8"))
            open(gm, "w", encoding="utf-8").write(
                base_gm if gm_sub is None else base_gm.replace(*gm_sub))
            del RG.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                RG.res_guard(res=res, root=d)
            return bool(RG.FAIL)

        def mutate(fn):
            new = json.loads(json.dumps(base))
            fn(new)
            return new

        def shift_fstar(o):
            o["part_a"]["grids"]["128"]["fstar"] = 5.00

        def one_seed(o):
            o["part_a"]["grids"]["128"]["fstar_seeds"][0] = 4.00

        def inflate_jump(o):
            o["part_a"]["grids"]["128"]["jump"] *= 1.5
            o["part_a"]["grids"]["128"]["jump_seeds"] = [
                o["part_a"]["grids"]["128"]["jump"]] * 3

        def break_paired(o):
            o["part_b"]["grids"]["64"]["fstar"] = 5.00

        def break_realgeo(o):
            o["part_c"]["grids"]["64"]["per_geometry"]["geo_west"]["fstar"] = 5.00

        def break_eval(o):
            o["part_a"]["curves"]["64"][3]["s0"]["mean"] *= 1.5

        def move_band(o):
            for w in ("west", "centre", "east"):
                o["part_c"]["grids"]["128"]["per_geometry"]["geo_" + w]["bands"]["20"] += 0.05

        got = dict(
            clean=not fires(),
            shift_fstar=fires(mutate(shift_fstar)),
            one_seed=fires(mutate(one_seed)),
            inflate_jump=fires(mutate(inflate_jump)),
            break_paired=fires(mutate(break_paired)),
            break_realgeo=fires(mutate(break_realgeo)),
            break_eval=fires(mutate(break_eval)),
            move_band=fires(mutate(move_band)),
            macro=fires(gm_sub=("\\ResFstarFine}{", "\\ResFstarFine}{1")),
        )
        del RG.FAIL[:]
        return got
    finally:
        shutil.rmtree(d, ignore_errors=True)

def probe_resolution_claims():
    """probe 22: the verdicts in the prose must break when the data breaks.

    probe 21 asks whether the table reproduces its curves.  This one asks the
    complementary question, the one that failed in this paper's history: the
    sentences read "the boundary does not move", "the jump rises", "the jump
    reverses the two families' order", "the plateau separates them only at
    128^2" -- and each of those is a *claim about the data*, so a re-run that
    makes one of them false has to stop the gate rather than leave a paragraph
    quietly wrong.

    Four mutations, each applied to the per-seed curves so that nothing but the
    verdict changes:

    1. the refined arm's curve is flattened at 4.5, which moves f*;
    2. the refined arm's f=5 level is set to the coarse one, so the jump ranges
       overlap and the rise becomes seed noise;
    3. the synthetic control's f=5 level at 128^2 is lowered, so the jump no
       longer reverses the families' order;
    4. the lowest-abundance synthetic control's plateau at 64^2 is raised, so
       the plateau would separate the families at the coarse mesh too.

    Expected: the untouched archive passes all six verdicts; every mutation is
    reported as a contradiction.
    """
    import guards_res as RG
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = tempfile.mkdtemp(prefix="_res_claim_")
    try:
        res = os.path.join(d, "results")
        os.makedirs(res)
        shutil.copy(os.path.join(root, "results", "res_study.json"),
                    os.path.join(res, "res_study.json"))
        base = json.load(open(os.path.join(res, "res_study.json"), encoding="utf-8"))

        def fires(fn=None):
            obj = json.loads(json.dumps(base))
            if fn is not None:
                fn(obj)
            json.dump(obj, open(os.path.join(res, "res_study.json"), "w",
                                encoding="utf-8"))
            del RG.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                RG.claim_guard(res=res)
            return bool(RG.FAIL)

        def flat_edge(o):
            for r in o["part_a"]["curves"]["128"]:
                if abs(r["f"] - 4.5) < 1e-9:
                    for k in ("s0", "s1", "s2"):
                        r[k]["mean"] *= 0.4

        def level_jump(o):
            # Lowering the refined arm's f=5 level puts its per-seed jumps
            # inside the coarse arm's range, which is the case the paper's
            # "does not overlap" sentence would be wrong about.  Scaling rather
            # than copying matters: the two grids have different f=4 levels, so
            # copying the coarse value leaves the ratio above the coarse one
            # and the mutation would be invisible for an honest reason.
            for r in o["part_a"]["curves"]["128"]:
                if abs(r["f"] - 5.0) < 1e-9:
                    for k in ("s0", "s1", "s2"):
                        r[k]["mean"] *= 0.93

        def kill_flip(o):
            for name, rows in o["part_c"]["curves"]["128"].items():
                if not name.startswith("syn_"):
                    continue
                for r in rows:
                    if abs(r["f"] - 5.0) < 1e-9:
                        for k in ("s0", "s1", "s2"):
                            r[k]["mean"] *= 0.75

        def separate_coarse(o):
            rows = o["part_c"]["curves"]["64"].get("syn_2") or []
            for r in rows:
                if r["f"] >= 6.0 - 1e-9:
                    for k in ("s0", "s1", "s2"):
                        r[k]["mean"] *= 1.10

        got = dict(
            clean=not fires(),
            flat_edge=fires(flat_edge),
            level_jump=fires(level_jump),
            kill_flip=fires(kill_flip),
            separate_coarse=fires(separate_coarse),
        )
        del RG.FAIL[:]
        return got
    finally:
        shutil.rmtree(d, ignore_errors=True)


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
        # The inverted control for the masking above: a prose dash that merely
        # shares a line with a cell separator is not a placeholder and must
        # still be caught, or the mask would silence the rule everywhere a
        # table row appears.
        en_dash_after_amp_bad=scan_fails(
            "the reference solver & the head -- fitted inside the band -- "
            "agree to 2\\%.\n", "en"),
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
        # The second blind spot, measured 2026-10-23: the delimiters required
        # `[ \t]` on both sides of each dash, so an aside whose opening dash
        # starts a line (or whose closing dash ends one) was invisible -- a
        # line break is not a blank.  One such aside sat in the English
        # introduction while the gate reported clean.
        en_dash_linestart_bad=scan_fails(
            "the step without training\n-- an analytic head fitted inside "
            "the band plus a learned residual on top --\nearns its place.\n",
            "en"),
        en_dash_linestart_ok=not scan_fails(
            "the step without training\nand an analytic head fitted inside "
            "the band earns its place.\n", "en"),
        # The Chinese em-dash rule capped the aside body at 60 characters
        # while the English rule meters 140, so a long Chinese aside was
        # invisible; the cap now matches.
        cn_dash_long_bad=scan_fails(
            r"其他算子主干——消息传递\cite{brandstetter2022message} 与 "
            r"Transformer 式架构\cite{hao2024dpot}——同样可在本文数据上训练。" "\n",
            "cn"),
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
        # A *single* spaced en dash is the aside rule's blind spot: the pair
        # rule only ever saw dashes in twos, so sixteen one-dash asides shipped
        # while it reported clean (measured 2026-10-26).
        en_dash_single_bad=scan_fails(
            "the plateau is analytic Born truncation error -- removable by a "
            "construction fitted inside the band.\n", "en"),
        # The control on the other side: an en dash inside a range has no
        # spaces and must stay legal, or the rule would condemn every table.
        en_dash_single_range_ok=not scan_fails(
            "the ratio runs 7.16--8.07 over f in 1--12.\n", "en"),
        # A macro definition is code, not prose: its \ref argument is a
        # placeholder, and scoring it as a broken reference is how the
        # abbreviated-cross-reference rule fired on the macro that produces
        # exactly the compact table form it is meant to encourage.
        def_line_ok=not scan_fails(
            "\\newcommand{\\secref}[1]{Sec.~\\ref{#1}}\n", "en"),
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
        # (1) honest record, all three sentences present -> accepted.  Three,
        # not one: the level-3 verdict now rests on a spectrum-matched control,
        # so the fixture has to carry the structural part of the plateau
        # difference and the unresolved jump as well as the old refusal.
        honest_en = ("the plateau does not separate the two families "
                     "\\RealPlateauOverlapArms ; the plateau is higher on\n"
                     "the Marmousi window by $\\GeoSurPlatDelta$ ; the jump "
                     "factor does not order the two families, $\\GeoJumpPCoarse$")
        honest_cn = ("平台高度并不分离两者 \\RealPlateauOverlapArms；"
                     "结构效应为 $\\GeoSurPlatDevPct$ 个百分点；"
                     "跃变从未把两族排出次序 $\\GeoJumpPCoarse$")
        write_json(_synthetic_realgeo())
        write_paper(honest_en, honest_cn)
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
        write_paper(honest_en, honest_cn, drop_float=True)
        del K.FAIL[:]
        run(d, K.realgeo_guard, d)
        float_rejected = bool(K.FAIL)

        # (5) the measurement no longer supports the non-separation sentence
        write_json(_synthetic_realgeo(overlap_arms=1))
        write_paper(honest_en, honest_cn)
        del K.FAIL[:]
        run(d, K.realgeo_guard, d)
        overclaim_rejected = bool(K.FAIL)

        # (6) the old half survives and the two new halves are dropped: the
        # gate must not read this as "the refusal is present, all good".  This
        # is the case the earlier one-sentence contract could not see.
        write_json(_synthetic_realgeo())
        write_paper("the plateau does not separate the two families "
                    "\\RealPlateauOverlapArms",
                    "平台高度并不分离两者 \\RealPlateauOverlapArms")
        del K.FAIL[:]
        run(d, K.realgeo_guard, d)
        half_rejected = bool(K.FAIL)
        del K.FAIL[:]
    finally:
        K.RES = old_res
        shutil.rmtree(d, ignore_errors=True)
    print("   honest record accepted = %s ; protocol drift rejected = %s ; "
          "missing refusal rejected = %s ; unreferenced float rejected = %s ; "
          "unsupported refusal rejected = %s ; half-verdict rejected = %s"
          % (honest_ok, drift_rejected, wording_rejected, float_rejected,
             overclaim_rejected, half_rejected))
    return honest_ok, drift_rejected, wording_rejected, float_rejected, \
        overclaim_rejected, half_rejected


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



def probe_bax_gate():
    """probe 24: the fit-depth/sum-depth correction must be defended.

    The corrected arms (bax) reversed a boundary the previous draft printed.
    Four sentences rest on that reversal, so four mutations:

    1. inflate one corrected arm's in-band entry past the trained operator's,
       so the recomputed PASS verdict no longer follows from the gates;
    2. perturb one re-scored a025 number, breaking the bit-for-bit invariance
       that certifies the fix changed only the head's semantics;
    3. strip head_semantics from a corrected archive, the shape that would
       let a truncated-scored number sit beside a full-scored one;
    4. put a withdrawn sentence -- "fails at v>=0.50" -- back into a copy of
       each manuscript.

    Expected: the untouched tree passes all four; every mutation is reported.
    """
    import guards_bax as GB
    import common as C
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = tempfile.mkdtemp(prefix="_bax_probe_")
    try:
        res = os.path.join(d, "results")
        shutil.copytree(os.path.join(root, "results"), res,
                        ignore=shutil.ignore_patterns("_*log*", "*.log"))
        sub = os.path.join(d, "submission")
        for lang in ("en", "cn"):
            os.makedirs(os.path.join(sub, "paper_%s" % lang))
            shutil.copy(os.path.join(root, "submission", "paper_%s" % lang,
                                     "main.tex"),
                        os.path.join(sub, "paper_%s" % lang, "main.tex"))

        def run(with_txt=None):
            old_res, old_bak = C.RES, GB.BACKUP
            C.RES, GB.BACKUP = res, os.path.join(res,
                                                 "_archive_truncated_semantics")
            try:
                del GB.FAIL[:]
                with contextlib.redirect_stdout(io.StringIO()):
                    GB.bax_guard()
                    # the archive guard reports through GB.FAIL, the text
                    # guard through its return value (its own local list)
                    fired = bool(GB.FAIL)
                    if with_txt is not None:
                        fired = GB.bax_text_guard(paths=with_txt) or fired
                return fired
            finally:
                C.RES, GB.BACKUP = old_res, old_bak

        def mut(path, fn):
            o = json.load(open(path, encoding="utf-8"))
            fn(o)
            json.dump(o, open(path, "w", encoding="utf-8"))

        def break_pass(o):
            for r in o["in_band"]:
                if abs(r["vmax"] - 0.50) < 1e-6:
                    r["mean"] = 0.5  # far above the trained operator's 0.187

        def break_invariant(o):
            # sweep keys are contrasts, values are per-frequency records
            o["sweep"]["0.50"][0]["mean"] *= 1.000001

        def hide_protocol(o):
            o.pop("head_semantics", None)

        clean = not run()   # run() returns True when the guard fires
        p = os.path.join(res, "eval_bax_fno_a050_m8_s0.json")
        mut(p, break_pass); b1 = run(); mut(p, lambda o: None)
        shutil.copy(os.path.join(root, "results",
                                 "eval_bax_fno_a050_m8_s0.json"), p)
        p2 = os.path.join(res, "eval_ba_fno_a025_m8_s0.json")
        mut(p2, break_invariant); b2 = run()
        shutil.copy(os.path.join(root, "results",
                                 "eval_ba_fno_a025_m8_s0.json"), p2)
        p3 = os.path.join(res, "eval_bax_fno_a100_m7_s0.json")
        mut(p3, hide_protocol); b3 = run()
        shutil.copy(os.path.join(root, "results",
                                 "eval_bax_fno_a100_m7_s0.json"), p3)
        txt = {}
        for lang, sent in (("en", r"the same arm fails at $v_{\max}\ge0.50$, "
                                    "destroying the plateau."),
                           ("cn", r"同一臂在 $v_{\max}\ge0.50$ 失效。")):
            mp = os.path.join(sub, "paper_%s" % lang, "main.tex")
            t = open(mp, encoding="utf-8").read()
            open(mp, "w", encoding="utf-8").write(
                t.replace("\\begin{abstract}", sent + "\n\\begin{abstract}", 1)
                 if "\\begin{abstract}" in t else sent + "\n" + t)
            txt[lang] = mp
        b4 = run(with_txt=txt)
        # 5. the other direction: remove the disclosure itself.  A guard that
        #    only fires on the presence of a withdrawn sentence stays green
        #    when the required text is deleted, which is how the single-pattern
        #    form of this check passed a rewrite that dropped the whole
        #    paragraph.
        mp = os.path.join(sub, "paper_en", "main.tex")
        t = open(mp, encoding="utf-8").read()
        assert "head sums what it was fitted to sum" in t, \
            "MUTATION TARGET ABSENT: the disclosure sentence is gone"
        open(mp, "w", encoding="utf-8").write(
            t.replace("head sums what it was fitted to sum",
                      "head is used as fitted"))
        b5 = run(with_txt={"en": mp})
        open(mp, "w", encoding="utf-8").write(t)
        out = dict(clean=clean, pass_broken=b1, invariant_broken=b2,
                   protocol_hidden=b3, text_drift=b4,
                   disclosure_missing=b5)
        del GB.FAIL[:]
        return out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def probe_surrogate_gate():
    """probe 23: the spectrum-matched sentences must break when the archive does.

    Three sentences were rewritten after the intervals were computed: the jump
    is "not resolved beyond the window spread", the plateau "carries a
    structural part of one to three percent with the same sign at both meshes",
    and the quadrature sum "bounds the error in the mean, not sample by sample".
    Each is a claim about the archive, so each must stop the gate when the
    archive contradicts it.  Six mutations:

    1. flip the sign of the plateau delta at 128^2, so the two meshes disagree
       about the direction of the structure effect;
    2. multiply every jump delta by five, so the interval excludes zero and
       "not resolved" becomes false;
    3. raise the per-sample bound's ratio above one, so the bound no longer
       holds even in the mean at f=4;
    4. raise the aggregate ratio above one at f=5, which the text presents as
       the ordinary case;
    5. coarsen the surrogate's spectrum, so "matched" is unearned;
    6. hide the pointwise violation (viol_same := 0), the archive shape that
       would make the RMS qualification unearned.

    A seventh case tests the text side rather than the data side: a manuscript
    that puts the withdrawn sentence back must be rejected by the text guard,
    because the data guard cannot see prose.

    Expected: the untouched archive passes; every mutation is reported.
    """
    import guards_surrogate as SG
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = tempfile.mkdtemp(prefix="_sur_probe_")
    try:
        res = os.path.join(d, "results")
        os.makedirs(res)
        names = ("geo_windows.json", "bound_cross.json", "p0_checks.json")
        base = {}
        for n in names:
            src = os.path.join(root, "results", n)
            if not os.path.exists(src):
                return dict(skipped=True)
            base[n] = json.load(open(src, encoding="utf-8"))
            shutil.copy(src, os.path.join(res, n))

        def reset():
            for n in names:
                json.dump(base[n], open(os.path.join(res, n), "w",
                                        encoding="utf-8"))

        def fires(mut=None, file=None):
            reset()
            if mut is not None:
                obj = json.loads(json.dumps(base[file]))
                mut(obj)
                json.dump(obj, open(os.path.join(res, file), "w",
                                    encoding="utf-8"))
            del SG.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                SG.surrogate_guard(os.path.join(res, "geo_windows.json"))
                SG.bound_guard(os.path.join(res, "bound_cross.json"))
                SG.paired_jump_guard(os.path.join(res, "p0_checks.json"))
            return bool(SG.FAIL)

        def text_fires(sentence=None, mutator=None):
            """Re-run the text guard on a copy of the delivered manuscript."""
            src = os.path.join(root, "submission", "paper_en", "main.tex")
            if not os.path.exists(src):
                return None
            sub = os.path.join(d, "sub")
            os.makedirs(os.path.join(sub, "paper_en"), exist_ok=True)
            txt = open(src, encoding="utf-8").read()
            if mutator is not None:
                txt = mutator(txt)
            if sentence:
                # the front matter uses \abstract{...}, not the environment
                assert "\\abstract{" in txt, "no anchor for the drift probe"
                txt = txt.replace("\\abstract{",
                                  sentence + "\n\\abstract{", 1)
            open(os.path.join(sub, "paper_en", "main.tex"), "w",
                 encoding="utf-8").write(txt)
            del SG.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                SG.bound_text_guard(sub)
            return bool(SG.FAIL)

        def flip_plat(o):
            for p in o["by_grid"]["128"]["pairs"]:
                p["delta_plat"] = -p["delta_plat"]

        def big_jump(o):
            # Scaling every delta leaves the t statistic unchanged, because mean
            # and standard error scale together; a mutation that cannot move the
            # p-value is not a test of the sentence.  Making the deltas nearly
            # constant and positive is the mutation that does move it.
            for g in o["by_grid"]:
                for i, p in enumerate(o["by_grid"][g]["pairs"]):
                    p["delta_jump"] = 0.5 + 0.001 * i

        def break_bound(o):
            o["rows"][0]["ratio_same"] = 1.05

        def break_aggregate(o):
            for r in o["rows"]:
                if int(r["f"]) == 5:
                    r["ratio_aggregate"] = 1.02

        def coarse_spectrum(o):
            for name, meta in o["by_grid"]["64"]["meta"].items():
                if name.startswith("sur_"):
                    meta["bands"] = {k: v * 0.5 for k, v in meta["bands"].items()}

        def hide_violation(o):
            r = o["rows"][0]
            r["viol_same"] = 0.0
            r["ratio_max_same"] = 1.0

        out = dict(
            clean=not fires(),
            plat_sign=fires(flip_plat, "geo_windows.json"),
            jump_resolved=fires(big_jump, "geo_windows.json"),
            bound_broken=fires(break_bound, "bound_cross.json"),
            aggregate_broken=fires(break_aggregate, "bound_cross.json"),
            spectrum_unmatched=fires(coarse_spectrum, "geo_windows.json"),
            violation_hidden=fires(hide_violation, "bound_cross.json"),
        )
        tf = text_fires("The inequality as written therefore holds at $f=4$.")
        if tf is not None:
            clean_text = text_fires()
            out["text_drift"] = bool(tf) and not clean_text

            # The opposite direction, which is the one that bit: the required
            # sentence is present, but the source wraps it across a line break.
            # A literal-substring gate reads that as "qualification missing"
            # and reports a correct manuscript as wrong.  Flattening the file
            # and re-breaking the sentence guarantees the case is exercised
            # whatever the current wrapping happens to be.
            def reflow(t):
                flat = " ".join(t.split())
                key = "in the mean, but not sample by sample"
                assert key in flat, "anchor for the reflow probe is gone"
                return flat.replace(key, "in the mean, but not\nsample by sample", 1)

            out["text_reflow_silent"] = not text_fires(mutator=reflow)
        del SG.FAIL[:]
    finally:
        shutil.rmtree(d, ignore_errors=True)
    for k in sorted(out):
        print("   %-18s %s" % (k, out[k]))
    return out


def probe_si_refs():
    """probe 25: every cross-reference into the supplement must fail loudly.

    The main text cites SI floats through the generated map gen_si_refs.tex,
    keyed by the SI's own \\label.  This probe checks each assertion of
    check_si_refs.py separately, because they fail in different ways and a
    single mutation would only exercise one of them:

      clean            untouched manuscript passes;
      stale_map        a map entry renumbered by hand is caught even though
                       the main text is untouched (the map no longer matches
                       the SI, and it no longer matches SI_main.aux);
      undefined_key    a key with no definition typesets ?? in the PDF --
                       invisible to a text audit, so the gate must catch it;
      literal_back     a hand-written number reintroduced next to the macro;
      wrong_table      a key that exists but names the wrong float is caught
                       by the caption-vocabulary check;
      no_s_prefix      the SI dropping the S prefix while the main text still
                       cites S<n>.

    Runs against a sandbox copy: this probe used to mutate the delivered
    manuscript in place and write it back in text mode, which converted
    submission/paper_en/main.tex to CRLF on every suite run and made the
    byte-level mirror gate report a stale archive for a file that had not
    changed by one character.
    """
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gate = os.path.join(root, "code", "check_si_refs.py")
    d = tempfile.mkdtemp(prefix="_si_refs_probe_")
    try:
        os.makedirs(os.path.join(d, "submission", "paper_en"))
        os.makedirs(os.path.join(d, "paper", "en"))
        os.makedirs(os.path.join(d, "code"))
        for name in glob.glob(os.path.join(root, "submission", "paper_en",
                                          "*.tex")):
            shutil.copy(name, os.path.join(d, "submission", "paper_en",
                                           os.path.basename(name)))
        for name in glob.glob(os.path.join(root, "paper", "en", "*.tex")):
            shutil.copy(name, os.path.join(d, "paper", "en",
                                           os.path.basename(name)))
        # the gate re-derives the map by importing the generator
        shutil.copy(os.path.join(root, "code", "gen_si_labels.py"),
                    os.path.join(d, "code", "gen_si_labels.py"))

        main_tex = os.path.join(d, "submission", "paper_en", "main.tex")
        si_tex = os.path.join(d, "submission", "paper_en", "SI_main.tex")
        gen_tex = os.path.join(d, "paper", "en", "gen_si_refs.tex")
        src = open(main_tex, encoding="utf-8").read()
        si_src = open(si_tex, encoding="utf-8").read()
        gen_src = open(gen_tex, encoding="utf-8").read()

        def put(path, text):
            # byte-exact: utf-8, no newline translation, inside the sandbox
            open(path, "wb").write(text.encode("utf-8"))

        def run():
            p = subprocess.run([sys.executable, gate, "--root", d], cwd=root,
                               capture_output=True, text=True)
            return p.returncode, p.stdout

        out = {}
        rc, log = run()
        out["clean"] = (rc == 0)

        # The number this probe renumbers is read off the map, not written
        # into the probe.  The hard-coded "{3}" survived two restructures and
        # then stopped matching when the supplement was renumbered: str.replace
        # is silent on a miss, so the probe ran the gate against an unmodified
        # file, reported stale_map=False, and the suite failed for a reason
        # that pointed at the gate instead of at itself.  A probe whose
        # mutation can miss must assert that it landed.
        #
        # The key is taken from the manuscript for the same reason, one level
        # up: the probe used to name tab:si_born by hand, and when the SI and
        # the body were put on one set of labels that key stopped appearing in
        # the main text at all.  Reading the key from the text under test means
        # the probe follows a rename instead of silently testing nothing.
        keys = re.findall(r"\\siTab\{([^}]+)\}", src)
        assert keys, "the main text cites no SI table; this probe cannot run"
        # The substituted key is an SI table the main text never cites, and the
        # pair is chosen so that no word of its caption occurs in the sentence
        # being mutated.  Two earlier versions of this probe failed for reasons
        # that had nothing to do with the gate: the first substituted the
        # second key in citation order (tab:anchor for tab:arch -- captions and
        # sentences shared "arm" and "band", so the mutated sentence still
        # matched the caption it now named and the gate was right to pass); the
        # second minimised the overlap over all keys and still accepted a
        # non-zero clash.  A mutation that the gate cannot distinguish from the
        # truth tests nothing, so the clash is asserted to be exactly zero.
        caps = {m.group(2): {w[:5] for w in re.findall(r"[a-z]{5,}",
                                                       m.group(1).lower())}
                for m in re.finditer(
                    r"\\caption\{(.*?)\}\s*\\label\{(tab:[^}]+)\}", si_src, re.S)}
        ctx = {}
        for m in re.finditer(r"\\siTab\{([^}]+)\}", src):
            lo = max(0, m.start() - 220)
            ctx.setdefault(m.group(1), set()).update(
                w[:5] for w in re.findall(r"[a-z]{5,}",
                                          src[lo:m.end() + 220].lower()))
        uncited = [o for o in sorted(caps) if o not in set(keys)]
        assert uncited, ("every captioned SI table is cited by the main text; "
                         "this probe needs an uncited one to point at")
        best = None
        for t in sorted(set(keys)):
            if t not in ctx:
                continue
            for o in uncited:
                clash = len(ctx[t] & caps[o])
                if best is None or clash < best[0]:
                    best = (clash, t, o)
        assert best is not None, "this probe needs a cited and an uncited key"
        clash, target, other = best
        # The gate matches on five-character stems, so the comparison here is
        # on five-character prefixes too.  Comparing whole words instead let a
        # pair through whose captions shared a stem ("arch"/"archi...") and the
        # mutation then produced a sentence that still matched the float it
        # named -- an ambiguous mutation, not a gate defect.
        assert clash == 0, ("no uncited table is lexically disjoint from the "
                            "sentence citing %s (best clash %d); the mutation "
                            "would be ambiguous" % (target, clash))

        born = re.search(r"sitab@" + re.escape(target) + r"\\endcsname\{(\d+)\}",
                         gen_src)
        assert born, "the map does not number %s; update this probe" % target
        old_num = born.group(0)
        new_num = "sitab@%s\\endcsname{%d}" % (target, int(born.group(1)) + 1)
        mutated_map = gen_src.replace(old_num, new_num)
        assert mutated_map != gen_src, "the renumbering mutation did not land"
        put(gen_tex, mutated_map)
        rc, log = run()
        out["stale_map"] = (rc != 0 and "stale" in log)
        put(gen_tex, gen_src)

        mutated = src.replace("\\siTab{%s}" % target, "\\siTab{tab:si_nope}")
        assert mutated != src, "the undefined-key mutation did not land"
        put(main_tex, mutated)
        rc, log = run()
        out["undefined_key"] = (rc != 0 and "no such SI table label" in log)
        put(main_tex, src)

        mutated = src.replace("Supplementary Table~\\siTab{%s}" % target,
                              "Supplementary Table~3")
        assert mutated != src, "the literal mutation did not land"
        put(main_tex, mutated)
        rc, log = run()
        out["literal_back"] = (rc != 0 and "literal SI float number" in log)
        put(main_tex, src)

        mutated = src.replace("\\siTab{%s}" % target, "\\siTab{%s}" % other)
        assert mutated != src, "the wrong-table mutation did not land"
        put(main_tex, mutated)
        rc, log = run()
        # If the gate accepts the swap, decide whether it was *entitled* to.
        # Rule 6 asks only that some citing sentence share a distinctive stem
        # with the caption it names, and that is a one-sided test: a paragraph
        # about the architecture also mentions the submission checklist, so a
        # swap to that table is genuinely indistinguishable from the truth by
        # vocabulary alone.  The verdict is recomputed here with the gate's own
        # tokeniser and stem rule, so "ambiguous" is decided by the gate's
        # semantics and not by this probe's opinion of them.
        ambiguous = False
        if rc == 0:
            spec = importlib.util.spec_from_file_location("_csr_probe", gate)
            M = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(M)
            M.EN = os.path.join(d, "submission", "paper_en")
            M.PEN = os.path.join(d, "paper", "en")
            fl = M.floats(M.inline_inputs(M.strip_comments(si_src), M.EN))
            if other in fl:
                st = M.stems_of(fl[other]["caption"]) - M.CONNECTIVES
                others = set().union(*[M.stems_of(fl[k]["caption"])
                                       - M.CONNECTIVES
                                       for k in fl if k != other])
                dist = st - others
                msk = M.mask_math(M.mask_captions(mutated))
                for m in re.finditer(r"\\siTab\{([^}]+)\}", mutated):
                    if m.group(1) != other:
                        continue
                    ctxw = M.sentence_of(msk, m.start(), m.end())
                    if any(M.same_stem(w, dd) for w in ctxw - M.CONNECTIVES
                           for dd in dist):
                        ambiguous = True
        out["wrong_table"] = ((rc != 0
                               and "shares vocabulary with the caption" in log)
                              or ambiguous)
        put(main_tex, src)

        put(si_tex, si_src.replace(
            "\\renewcommand{\\thefigure}{S\\arabic{figure}}", ""))
        rc, log = run()
        out["no_s_prefix"] = (rc != 0 and "S prefix" in log)
        put(si_tex, si_src)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out


def probe_seed_claim():
    """probe 26: a seed-count qualifier that names the wrong arms must fire.

    The W2 replication pass gave nine arms six seeds.  The sentence "the
    $v_{\\max}=1.00$ anchored arm carries NumSeedsMax of them" was true when it
    was written and became false afterwards, while NumSeedsMax itself stayed
    correct -- no numeric gate could see the stale scope.  Three inputs are
    checked, because a guard that only rejects the bad case is free to reject
    correct manuscripts:

      good   the scope is named by NArmsAtMax, counted from the records;
      stale  the scope is a remembered arm name, which is the defect;
      rng    the explicit range NumSeeds..NumSeedsMax, which names no scope
             and must stay quiet.
    """
    import guards_text as T
    d = tempfile.mkdtemp(prefix="_seed_probe_")
    try:
        folder = os.path.join(d, "submission", "paper_en")
        os.makedirs(folder)
        cases = dict(
            good="The \\NArmsAtMax\\ arms the headline claims rest on carry "
                 "\\NumSeedsMax\\ of them.\n",
            stale="the $v_{\\max}=1.00$ anchored arm carries \\NumSeedsMax\\ "
                  "of them.\n",
            rng="\\NumSeeds{} to \\NumSeedsMax{} seeds are trained per arm.\n",
        )
        out = {}
        for k, body in cases.items():
            with open(os.path.join(folder, "main.tex"), "w",
                      encoding="utf-8") as fh:
                fh.write(body)
            del T.FAIL[:]
            with contextlib.redirect_stdout(io.StringIO()):
                T.seed_claim_guard(langs=("en",), root=d)
            out[k] = bool(T.FAIL)
        del T.FAIL[:]
        return dict(good_quiet=not out["good"], stale_fires=out["stale"],
                    range_quiet=not out["rng"])
    finally:
        shutil.rmtree(d, ignore_errors=True)


def probe_delivery_texts():
    """probe 27: the files an editor reads before the manuscript.

    Every other gate reads the .tex sources.  The cover letter, CITATION.cff,
    README, the environment pins and the Data Availability wording sit outside
    that, and each of them drifted while the manuscript stayed correct.  Five
    classes are exercised here, one mutation each, plus a two-case archive test
    on a synthetic tree (a seed the archive lacks, and a MANIFEST that describes
    a different tree).
    """
    import check_delivery_texts as D
    t = D.collect()
    out = {}

    # the untouched envelope must be accepted, or every rejection below is
    # meaningless
    out["clean"] = not D.run(texts=t)

    # The mutation must be derived from the title the manuscript actually
    # carries.  Hard-coding a word of it ("Discretized", until v42) turns the
    # probe into a no-op the moment the title is edited, and a no-op probe
    # passes without ever asking the gate anything -- worse than no probe,
    # because the suite still prints a green line for it.
    m = dict(t)
    canon = D.norm(D.tex_title(t["paper_en"], "Title") or "")
    token = None
    for cand in canon.split():
        if len(cand) >= 5 and t["cover"].count(cand) == 1:
            token = cand
            break
    if token is None:
        out["title"] = False
    else:
        m["cover"] = t["cover"].replace(token, "", 1)
        out["title"] = any("cover" in f for f in D.run(texts=m))

    m = dict(t)
    m["citation"] = (t["citation"]
                     + "\n# the plateau is the trivial incident-field baseline\n")
    out["retired"] = any("citation" in f for f in D.run(texts=m))

    m = dict(t)
    m["env"] = t["env"].replace("pytorch=2.6.0", "pytorch=2.13.0", 1)
    out["pins"] = any("pytorch" in f for f in D.run(texts=m))

    m = dict(t)
    m["cover"] = t["cover"].replace("on acceptance", "on publication", 1)
    out["doi"] = any("cover letter" in f for f in D.run(texts=m))

    m = dict(t)
    m["paper_cn"] = t["paper_cn"].replace("补充材料", "附录")
    out["supp"] = any("paper_cn" in f for f in D.run(texts=m))

    # The abstract cap is counted on rendered text, and the region used to run
    # from "Abstract" to "Keywords:", which swallows the journal's front-matter
    # sidebar whenever text extraction orders it there.  It did, once the line
    # numbers became extractable: the count rose by 32 words with no change to
    # the abstract.  A synthetic page with the sidebar present but outside the
    # region must count exactly the abstract's own words.
    import check_claims as K
    page = ("Abstract\n" + " ".join(["alpha"] * 12) +
            "\nReceived:\nRevised:\nAccepted:\nPublished:\n"
            "Copyright: (c) 2026 by the authors.\nSubmitted to Appl. Sci.\n"
            "Keywords:\nx\n")
    out["abstract_span"] = (K.rendered_abstract_words(page)[0] == 12)

    d = tempfile.mkdtemp(prefix="_deliv_probe_")
    try:
        live = os.path.join(d, "results")
        arc = os.path.join(d, "submission", "supplementary")
        os.makedirs(live)
        os.makedirs(os.path.join(arc, "results"))
        for s in range(6):
            open(os.path.join(live, "eval_arm_s%d.json" % s), "w").write("{}")
        for s in range(3):
            open(os.path.join(arc, "results", "eval_arm_s%d.json" % s),
                 "w").write("{}")
        open(os.path.join(arc, "MANIFEST.md"), "w").write("# m\n\nFiles: 99\n")
        fails = D.archive_failures(d)
        out["archive_seed"] = any("missing seeds" in f for f in fails)
        out["archive_manifest"] = any("MANIFEST claims" in f for f in fails)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return out


if __name__ == "__main__":
    sys.exit(main())
