# Author: Longgang Wang
"""Gates for defects that live in the typeset text rather than in the numbers.

Every other gate in this project asks whether a number is true.  None of them
asks whether the number reached the page.  On 2026-10-22 the delivered 42-page
PDF printed, in the middle of the one paragraph the external reviewer had asked
for:

    not small:/n//delta64//to256 = //GridDeltaTop at f = 12

The source line carried a literal ``/n`` and two ``//`` escapes, so the macro
never expanded: the reader was shown the *name* of the macro where its value
should be, and a reader cannot tell a missing value from a deliberately printed
symbol.  Four gates were green while this was on the page, because
``audit_paper`` looks for the project's known placeholder forms (``{--}``,
``??``) and a substituted-backslash form is not one of them.

So the class is closed here, with two instruments of different structure:

  A. the delivered body source -- exact, no line-wrap blind spots;
  B. the rendered PDF text -- end to end, catches residue introduced anywhere
     in the chain, including by a generator rather than by hand.

A gate that only reads the source would miss a value lost at compile time; a
gate that only reads the PDF would miss a macro split across a line break.
Both are cheap, so both are run.
"""
import os
import re
import sys
import subprocess

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


def _strip_comments(src):
    """Drop LaTeX comments so a URL in a comment is not read as body text."""
    out = []
    for line in src.splitlines():
        i, cut = 0, None
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                cut = i
                break
            i += 1
        out.append(line if cut is None else line[:cut])
    return "\n".join(out)


def defined_macros(folder):
    """Names defined in a generated macro file, as TeX will actually bind them."""
    p = os.path.join(folder, "gen_macros.tex")
    if not os.path.exists(p):
        return set()
    raw = open(p, encoding="utf-8").read()
    return set(re.findall(r"\\newcommand\{\\([A-Za-z@]+)\}", raw))


def _name_hits(text, names):
    """Macro names present as literal text, i.e. not preceded by a backslash."""
    hits = []
    for n in sorted(names):
        # a legal reference is \Name; anything else means the macro did not
        # expand, and the reader sees the name instead of the value
        for m in re.finditer(r"(?<![\\A-Za-z])" + re.escape(n) + r"(?![A-Za-z])",
                             text):
            a = max(0, m.start() - 24)
            hits.append((n, text[a:m.end() + 24].replace("\n", " ")))
            break
    return hits


def _escape_hits(text, label):
    """`//` and `/n` outside URLs: the shapes a substituted backslash leaves."""
    hits = []
    for line in text.splitlines():
        bare = re.sub(r"https?://\S*", " ", line)
        bare = re.sub(r"\bdoi:\S*", " ", bare)
        if "//" in bare:
            hits.append((label, "//", line.strip()[:110]))
        if re.search(r"(?<![\w/.])/n(?![\w/])", bare):
            hits.append((label, "/n", line.strip()[:110]))
    return hits


def pdf_text(path):
    for exe in ("pdftotext",):
        try:
            r = subprocess.run([exe, path, "-"], capture_output=True, timeout=180)
            if r.returncode == 0:
                return r.stdout.decode("utf-8", "replace")
        except (OSError, subprocess.SubprocessError):
            pass
    return ""


def range_dash_guard(root=None, langs=("en", "cn"), texts=None):
    """A numeric range must not reach the reader as two minus signs.

    ``\\newcommand{\\XLo}{4.00}`` and ``\\newcommand{\\XHi}{4.25}`` are safe in
    any context; a macro whose value is ``4.00--4.25`` is safe in text mode and
    wrong in math mode, where ``--`` typesets as two minus signs and a range
    reads as subtraction.  That is what happened when the locator endpoints
    were first exported as one macro and used as ``$\\LocRefStep$``.

    The check is on the rendered text, because the defect only exists after
    typesetting: the source is legal LaTeX.  Two adjacent minus signs with
    digits on both sides is the signature.

    The signature must be U+2212 twice, not two hyphens.  A first version
    accepted ASCII hyphens and immediately reported "(4.53--10.02)" from a
    figure legend; rendering that page showed a correct en dash, i.e. the
    extraction, not the typesetting, had collapsed it (matplotlib's font has no
    ToUnicode entry for it here).  Requiring the math-minus code point keeps
    the true defect -- which does produce U+2212 U+2212 -- and drops the figure
    text, and the distinction was confirmed on both artefacts rather than
    assumed.
    """
    root = root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    hits = []
    for lang in langs:
        if texts is not None:
            # probe path: the rendered text is supplied, so that the guard's
            # judgement can be tested without a PDF to hand
            text = texts.get(lang, "")
        else:
            pdf = os.path.join(root, "submission", "paper_%s" % lang, "main.pdf")
            if not os.path.exists(pdf):
                note("range dash %s: no delivered PDF, not asserted" % lang)
                continue
            text = pdf_text(pdf)
        if not text:
            note("range dash %s: rendered text unavailable, not asserted" % lang)
            continue
        for m in re.finditer(r"\d[\s\u00a0\u2009\u200a]*\u2212[\s\u00a0\u2009\u200a]*"
                             r"\u2212[\s\u00a0\u2009\u200a]*\d", text):
            ctx = text[max(0, m.start() - 60):m.end() + 40].replace("\n", " ")
            hits.append((lang, ctx.strip()))
    if hits:
        for lang, ctx in hits[:3]:
            bad("range dash %s: a numeric range is typeset as subtraction in %r"
                % (lang, ctx[:110]))
    else:
        ok("range dash: no numeric range in either delivered PDF is typeset "
           "as two minus signs")


def figure_guard(root=None, langs=("en", "cn")):
    """Every figure the body includes must exist in the tree that is compiled.

    The body writes ``\\includegraphics{fig_x.pdf}`` and relies on
    ``\\graphicspath{{figs/}}``, so a missing file is a fatal compile error.
    On 2026-10-22 ``fig_deep`` existed only under ``submission/paper_en/figs``
    because that one figure had been rendered into the delivery directory while
    everything else was rendered into ``paper/en/figs``.  The delivered PDF was
    therefore built by hand in ``submission/``, ``paper/en`` fell three days
    behind, and the difference was invisible until a rebuild was attempted.

    A figure present in the delivery tree but absent from the build tree is
    reported separately, because that asymmetry is the signature of the defect
    rather than its symptom.
    """
    root = root or C.ROOT
    pat = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")
    for lang in langs:
        pap = os.path.join(root, "paper", lang)
        body = os.path.join(pap, "main.tex")
        if not os.path.exists(body):
            note("figures %s: no build body, not asserted" % lang)
            continue
        names = sorted(set(os.path.basename(n.strip()) for n in
                           pat.findall(_strip_comments(
                               open(body, encoding="utf-8").read()))))
        if not names:
            note("figures %s: no \\includegraphics found" % lang)
            continue
        miss, stray = [], []
        for n in names:
            if not os.path.exists(os.path.join(pap, "figs", n)):
                miss.append(n)
                sub = os.path.join(root, "submission", "paper_%s" % lang, "figs", n)
                if os.path.exists(sub):
                    stray.append(n)
        if miss:
            bad("figures %s: %s referenced but absent from paper/%s/figs -- the "
                "build cannot produce a PDF%s"
                % (lang, ", ".join(miss), lang,
                   " (present only in submission/: the delivery tree was "
                   "compiled by hand)" if stray else ""))
        else:
            ok("figures %s: all %d included figures exist in the build tree"
               % (lang, len(names)))

def text_residue_guard(root=None, langs=("en", "cn")):
    """No defined macro may reach the reader as a name, and no escape residue."""
    root = root or C.ROOT
    for lang in langs:
        folder = os.path.join(root, "submission", "paper_%s" % lang)
        body = os.path.join(folder, "main.tex")
        if not os.path.exists(body):
            note("text %s: no delivered body, not asserted" % lang)
            continue
        names = defined_macros(folder)
        if not names:
            note("text %s: no generated macros found, only escape checks run" % lang)
        src = _strip_comments(open(body, encoding="utf-8").read())

        # instrument A: the source, where nothing can hide behind a line break
        hitA = _name_hits(src, names)
        escA = _escape_hits(src, "tex")
        # instrument B: the rendered text, where nothing can hide behind a %
        pdf = os.path.join(folder, "main.pdf")
        txt = pdf_text(pdf) if os.path.exists(pdf) else ""
        hitB = _name_hits(txt, names) if txt else []
        escB = _escape_hits(txt, "pdf") if txt else []

        if hitA or hitB:
            for n, ctx in (hitA + hitB)[:4]:
                bad("text %s: macro \\%s reaches the reader unexpanded: ...%s..."
                    % (lang, n, ctx))
        if escA or escB:
            for where, tok, ctx in (escA + escB)[:4]:
                bad("text %s: escape residue %r in %s: %s"
                    % (lang, tok, where, ctx))
        if not (hitA or hitB or escA or escB):
            ok("text %s: %d generated macros all expand to values, %s"
               % (lang, len(names),
                  "no escape residue in source or PDF" if txt
                  else "no escape residue in source (no PDF)"))
        elif not txt:
            note("text %s: PDF absent, only the source was checked" % lang)


def seed_claim_guard(root=None, langs=("en", "cn")):
    """A sentence quoting the maximum seed count must say which arms reached it.

    Before the W2 replication pass the sentence "the $v_{\\max}=1.00$ anchored
    arm carries NumSeedsMax of them" was true.  W2 gave nine arms six seeds, so
    the qualifier in the parentheses became false while NumSeedsMax itself
    stayed correct -- every numeric gate kept reporting green, because the
    number is right and only its scope went stale.  That is the class closed
    here: NumSeedsMax may be quoted only together with NArmsAtMax, which counts
    the arms at the maximum straight from the evaluation records, unless the
    sentence is the explicit range NumSeeds..NumSeedsMax.

    The check uses a character window rather than sentence splitting: "2.6.0"
    and "Fig. 3" contain periods, and a splitter that treats them as sentence
    ends would separate a correct qualifier from the number it qualifies.  A
    window can be fooled by a neighbouring sentence, which errs towards staying
    quiet; a splitter errs towards firing on correct text, and a guard that
    fires on correct text gets switched off.
    """
    root = root or C.ROOT
    RANGE = re.compile(r"NumSeeds\{\}\s*(?:to|至|到|--|\u2013)\s*\\?NumSeedsMax")
    WIN = 220
    for lang in langs:
        body = os.path.join(root, "submission", "paper_%s" % lang, "main.tex")
        if not os.path.exists(body):
            note("seeds %s: no delivered body, not asserted" % lang)
            continue
        src = _strip_comments(open(body, encoding="utf-8").read())
        pos = [m.start() for m in re.finditer(r"\\NumSeedsMax(?![A-Za-z])", src)]
        if not pos:
            note("seeds %s: NumSeedsMax is never quoted, nothing to assert"
                 % lang)
            continue
        bads = []
        for p in pos:
            win = src[max(0, p - WIN):p + WIN]
            if "NArmsAtMax" not in win and not RANGE.search(win):
                bads.append(" ".join(win.split())[:170])
        if bads:
            for w in bads[:3]:
                bad("seeds %s: NumSeedsMax quoted without NArmsAtMax: ...%s..."
                    % (lang, w))
        else:
            ok("seeds %s: all %d NumSeedsMax quotes name their scope "
               "(NArmsAtMax) or are the NumSeeds..NumSeedsMax range"
               % (lang, len(pos)))
