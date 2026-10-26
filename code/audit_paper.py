# Author: Longgang Wang <longgangwang@163.com>
# Affiliation: Luoyang Institute of Science and Technology, Luoyang 471000, Henan, China

"""Independent audit of the manuscript deliverables.

Checks performed
  1. no unresolved placeholders in either manuscript
  2. the on-disk gen_macros.tex equals a freshly regenerated one (catches any
     hand edit of a generated number)
  3. every \\label has a \\ref and vice versa
  4. every \\cite key exists in refs.bib, and every bib entry is cited
  5. the headline macros are re-derived INDEPENDENTLY from results/eval_*.json
     with plain numpy, i.e. not through the generator that wrote them
  6. the compiled PDFs: page count, embedded fonts, and text extraction

Exit code is non-zero if any hard check fails.
"""
import os
import re
import sys
import glob
import json
import argparse
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import make_results_tex as G

FAIL = []


def ok(msg):
    print("  [ok]   " + msg)


def bad(msg):
    FAIL.append(msg)
    print("  [FAIL] " + msg)


def warn(msg):
    print("  [warn] " + msg)


# ---------------------------------------------------------------------------
def check_placeholders(path):
    txt = open(path, encoding="utf-8").read()
    hits = re.findall(r"\\tbd\{[^}]*\}", txt)
    hits += re.findall(r"\bTODO\b|\bFIXME\b|\?\?+", txt)
    if hits:
        bad("%s: %d unresolved placeholders: %s"
            % (os.path.basename(path), len(hits), hits[:6]))
    else:
        ok("%s: no placeholders" % os.path.basename(path))


def check_macros_fresh(lang, folder):
    disk = os.path.join(folder, "gen_macros.tex")
    if not os.path.exists(disk):
        bad("%s: gen_macros.tex missing" % lang); return
    on_disk, _ = G.build(lang)
    if on_disk is None:
        bad("%s: generator produced nothing" % lang); return
    if on_disk.strip() == open(disk, encoding="utf-8").read().strip():
        ok("%s: gen_macros.tex is byte-identical to a fresh run" % lang)
    else:
        bad("%s: gen_macros.tex DIFFERS from a fresh run (hand-edited?)" % lang)


def check_macros_resolved(folder):
    """Every macro the prose actually uses must have resolved to a number.

    The generator emits `\\newcommand{\\X}{--}` as a fallback when the run it
    needs has not been evaluated yet.  That is fine while the grid is still
    running, but a `--` reaching the PDF is a visible defect, so any fallback
    that the manuscript *uses* is a hard failure.  Macros that nobody cites are
    ignored, which keeps genuinely optional quantities from blocking delivery.
    """
    disk = os.path.join(folder, "gen_macros.tex")
    main = os.path.join(folder, "main.tex")
    if not (os.path.exists(disk) and os.path.exists(main)):
        return
    # Match the fallback however it is braced: `{--}` and `{{--}}` are the same
    # defect, but a literal `}{--}` search only sees the first.  (One generator
    # path used to emit the double-braced form, which printed a stray en-dash
    # into the PDF while every gate reported "all macros resolved".)
    macro_val = re.findall(r"\\newcommand\{\\(\w+)\}\{(.+?)\}\s*$",
                           open(disk, encoding="utf-8").read(), re.M)
    fallback = {name for name, val in macro_val
                if re.sub(r"[{}]", "", val).strip() == "--"}
    if not fallback:
        ok("%s: every macro resolved (no '--' fallbacks)" % os.path.basename(folder))
        return
    used = set(re.findall(r"\\(\w+)", open(main, encoding="utf-8").read()))
    # The tables were split into one file per label and the concatenated
    # gen_results.tex was deliberately deleted.  This line kept reading that
    # deleted name, so from the split onwards the caption text of every
    # generated table was invisible to this gate: a fallback macro cited only
    # inside a table caption would have printed its `--` into the PDF and the
    # audit would still have said "unused fallback macro(s)".  Read the split
    # files.
    for g in sorted(glob.glob(os.path.join(folder, "gen_tab_*.tex"))):
        used |= set(re.findall(r"\\(\w+)", open(g, encoding="utf-8").read()))
    live = sorted(fallback & used)
    if live:
        bad("%s: %d macro(s) still at the '--' fallback but cited in the text: %s"
            % (os.path.basename(folder), len(live), live[:8]))
    else:
        ok("%s: %d unused fallback macro(s), none cited in the text"
           % (os.path.basename(folder), len(fallback)))


def check_macro_names(folder):
    """A macro name may not contain a digit, and no name may repeat.

    TeX ends a control word at the first non-letter, so
    ``\\newcommand{\\ResP50}{...}`` silently defines ``\\ResP`` and leaves a
    literal ``50`` behind.  Observed 2026-10-12: that produced a hard
    ``Missing \\begin{document}`` in the preamble while every other gate stayed
    green, because no gate had ever looked at the *name*.  Duplicate definitions
    are the mirror-image failure: the later one wins and the earlier value is
    silently dropped.
    """
    disk = os.path.join(folder, "gen_macros.tex")
    if not os.path.exists(disk):
        return
    names = re.findall(r"\\newcommand\{\\(\w+)\}", open(disk, encoding="utf-8").read())
    withdigits = sorted({n for n in names if any(c.isdigit() for c in n)})
    if withdigits:
        bad("%s: macro name(s) contain a digit, so TeX defines the truncated "
            "name instead: %s" % (os.path.basename(folder), withdigits[:6]))
    seen, dup = set(), set()
    for n in names:
        if n in seen:
            dup.add(n)
        seen.add(n)
    if dup:
        bad("%s: macro(s) defined more than once (the last silently wins): %s"
            % (os.path.basename(folder), sorted(dup)[:6]))
    if not withdigits and not dup:
        ok("%s: %d macro names are digit-free and unique"
           % (os.path.basename(folder), len(names)))


def expand_inputs(tex):
    """Read a .tex file with \\input{...} pulled in from the same folder.

    Labels and tables live in the auto-generated gen_tab_*.tex files, which the
    manuscript pulls in from the body; auditing main.tex alone would report
    every \\ref{tab:...} as dangling.
    """
    path = tex
    seen = set()
    out = []
    for line in open(tex, encoding="utf-8").read().splitlines():
        m = re.match(r"\s*\\input\{([^}]+)\}", line)
        if m:
            cand = m.group(1)
            if not cand.lower().endswith(".tex"):
                cand += ".tex"
            sub = os.path.join(os.path.dirname(tex), cand)
            if os.path.exists(sub) and sub not in seen:
                seen.add(sub)
                out.append(open(sub, encoding="utf-8").read())
                continue
        out.append(line)
    return "\n".join(out), seen


def strip_comments(txt):
    """Drop LaTeX comments so cite lists split across lines parse correctly."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", l) for l in txt.splitlines())


def check_selfmath_macros(folder):
    """A macro whose body is already `$...$` must not be wrapped in `$` again.

    Observed 2026-10-12 on `\\ResPsiStorePert`, whose body is
    ``$2.5\\times10^{-8}$``: the manuscript wrote ``$\\ResPsiStorePert$``, which
    expands to ``$$2.5\\times10^{-8}$$``.  Under pdfLaTeX the doubled dollars
    parse as a display-math toggle and the build survives with silently wrong
    spacing; under XeLaTeX the same construct stops at the following CJK
    character with ``Missing $ inserted`` and produces a 3-page PDF.  Neither
    the numeric audit (the number printed is still 2.5e-8) nor the macro-name
    gate can see it, so the check is on the *call site*: any macro emitted with
    a self-delimiting math body may not appear as ``$\\Name$``.
    """
    disk = os.path.join(folder, "gen_macros.tex")
    if not os.path.exists(disk):
        return
    mac = open(disk, encoding="utf-8").read()
    selfmath = re.findall(r"\\newcommand\{\\(\w+)\}\{\s*\$", mac)
    if not selfmath:
        return
    bads = []
    main = os.path.join(folder, "main.tex")
    if os.path.exists(main):
        txt = open(main, encoding="utf-8").read()
        for nm in selfmath:
            for m in re.finditer(r"\$\\" + nm + r"\$", txt):
                bads.append("main.tex:%d $\\%s$"
                            % (txt[:m.start()].count("\n") + 1, nm))
    if bads:
        bad("%s: self-delimiting-math macro double-wrapped in '$': %s"
            % (os.path.basename(folder), bads[:6]))
    else:
        ok("%s: %d self-delimiting-math macro(s), none double-wrapped"
           % (os.path.basename(folder), len(selfmath)))


def check_labels(tex):
    txt, extra = expand_inputs(tex)
    # Macro definitions are code: `\newcommand{\secref}[1]{Sec.~\ref{#1}}`
    # has a \ref whose argument is the placeholder #1, not a label, and a
    # gate that reads it as one reports a broken reference for a reference
    # that is generated correctly.  Same strip as lint_style uses; the
    # rendered prose is audited separately.
    txt = "\n".join("" if re.match(
        r"\s*\\(?:newcommand|renewcommand|providecommand|def|let|gdef|edef)\b",
        line) else line for line in txt.split("\n"))
    # A *list*, not a set: a label defined twice is invisible to a set, and
    # LaTeX silently resolves every \\ref to the LAST definition.  That is not
    # hypothetical -- \\label{sec:disc} was attached both to the discriminator
    # subsection and, one edit later, to the Discussion section, so four
    # cross-references rendered the Discussion's number while pointing at the
    # discriminators, and the set-based check reported "all refs resolve".
    lab_list = re.findall(r"\\label\{([^}]*)\}", txt)
    labels = set(lab_list)
    dup = sorted({l for l in lab_list if lab_list.count(l) > 1})
    refs = set()
    for m in re.findall(r"\\(?:ref|eqref|autoref)\{([^}]*)\}", txt):
        refs.update(m.split(","))
    miss = sorted(r for r in refs if r not in labels)
    unref = sorted(l for l in labels if l not in refs)
    if miss:
        bad("%s: \\ref with no \\label: %s" % (os.path.basename(tex), miss))
    elif dup:
        bad("%s: \\label defined more than once: %s" % (os.path.basename(tex), dup))
    else:
        ok("%s: all %d refs resolve (%d labels, %d from \\input files)"
           % (os.path.basename(tex), len(refs), len(labels), len(extra)))
    if unref:
        warn("%s: labels never referenced: %s" % (os.path.basename(tex), unref))
    return labels, refs


def check_bib(tex, bib):
    txt = strip_comments(open(tex, encoding="utf-8").read())
    keys_cited = set()
    for m in re.findall(r"\\cite[a-z]*\*?\{([^}]*)\}", txt):
        keys_cited.update(k.strip() for k in m.split(",") if k.strip())
    bibtxt = open(bib, encoding="utf-8").read()
    keys_bib = set(re.findall(r"@\w+\{([^,]+),", bibtxt))
    miss = sorted(keys_cited - keys_bib)
    unused = sorted(keys_bib - keys_cited)
    if miss:
        bad("%s: cited but not in bib: %s" % (os.path.basename(tex), miss))
    else:
        ok("%s: all %d citations present in refs.bib" % (os.path.basename(tex),
                                                         len(keys_cited)))
    if unused:
        warn("%s: bib entries never cited: %s" % (os.path.basename(tex), unused))
    # duplicate keys
    allk = re.findall(r"@\w+\{([^,]+),", bibtxt)
    dup = sorted({k for k in allk if allk.count(k) > 1})
    if dup:
        bad("%s: duplicate bib keys: %s" % (os.path.basename(bib), dup))
    return keys_bib


def check_bib_inline_tex(folder):
    """Class gate: no text-mode TeX special survives into the rendered list.

    The reference list is generated, so every character that is special in
    LaTeX text mode has to be escaped by the *renderer*, not by the .bib file:
    the .bib must stay valid BibTeX (a `_` is legal there), and the escaping is
    a property of the output.  A repository URL containing an underscore
    (`.../reverse_time_migration`) reached a real build as a hard
    "Missing $ inserted" that produced no PDF at all, which is why this
    assertion is about the class -- text-mode specials anywhere in a generated
    entry -- rather than about that one URL.  Math mode and escaped forms are
    removed first, because both make the characters legitimate.
    """
    p = os.path.join(folder, "gen_bib.tex")
    if not os.path.exists(p):
        bad("%s: gen_bib.tex missing" % folder)
        return
    hits = 0
    for i, line in enumerate(open(p, encoding="utf-8"), 1):
        if line.lstrip().startswith("%"):
            continue
        body = re.sub(r"\$[^$]*\$", "", line)
        for esc in ("\\_", "\\&", "\\%", "\\#", "\\textbackslash{}"):
            body = body.replace(esc, "")
        for ch, why in (("_", "subscript"), ("&", "alignment"),
                        ("#", "parameter"), ("%", "comment")):
            if ch in body:
                hits += 1
                bad("%s gen_bib.tex:%d has a bare %r, a %s character in text "
                    "mode; the renderer must escape it"
                    % (os.path.basename(folder), i, ch, why))
    if not hits:
        ok("%s: the rendered reference list carries no unescaped text-mode "
           "special" % os.path.basename(folder))
    bib_locators(folder)


# Entries whose venue issues no DOI and that have no preprint: a 1991 EAGE
# abstract volume, an APS meeting abstract, and two GitHub-hosted data sets.
# Each is listed with the reason, so the exception cannot be widened silently
# by adding a key without one.
NO_LOCATOR_OK = {
    "bourgeois1991marmousi": "EAGE abstract volume, 1991, no DOI",
    "millet2024acoustic": "APS DFD meeting abstract, no DOI or preprint",
    "mlamoureuxMarmousi": "GitHub data set, URL is the locator",
    "shenyaojinMarmousi": "GitHub data set, URL is the locator",
}


def bib_locators(folder):
    """Every reference must carry a DOI or an arXiv number.

    MDPI production asks for the DOI of every entry it can resolve, and the
    entries that lack one here are the proceedings papers, whose arXiv number
    is their only persistent locator.  The renderer used to drop that field
    (see make_bib_tex.render), so six references printed as "In Proceedings;
    2024." with nothing to find them by.  Checking the rendered file rather
    than the .bib is what catches a renderer that ignores a field it has.
    """
    p = os.path.join(folder, "gen_bib.tex")
    if not os.path.exists(p):
        return
    txt = open(p, encoding="utf-8").read()
    items = re.split(r"\\bibitem\{", txt)[1:]
    missing = []
    for it in items:
        key = it.split("}", 1)[0]
        body = it.split("}", 1)[1] if "}" in it else ""
        if "doi.org/" in body or re.search(r"(arXiv|arxiv):\s*\d", body):
            continue
        if key in NO_LOCATOR_OK:
            continue
        missing.append(key)
    if missing:
        for k in missing:
            bad("%s: reference %r has neither a DOI nor an arXiv number"
                % (os.path.basename(folder), k))
    else:
        ok("%s: every reference carries a DOI or an arXiv number "
           "(%d entries, %d documented exceptions)"
           % (os.path.basename(folder), len(items), len(NO_LOCATOR_OK)))


# ---------------------------------------------------------------------------
def rederive():
    """Recompute the headline numbers from the raw eval JSONs with plain numpy."""
    ev = {}
    for p in sorted(glob.glob(os.path.join(C.RES, "eval_*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        ev[d["name"]] = d

    def arm(vmax):
        tag = "a%03d" % int(round(vmax * 100))
        names = sorted(n for n in ev if n.startswith("main_fno_%s_" % tag))
        if not names:
            return None
        F = np.array([r["f"] for r in ev[names[0]]["sweep"]["%.2f" % vmax]])
        M = np.mean([[r["mean"] for r in ev[n]["sweep"]["%.2f" % vmax]]
                     for n in names], axis=0)
        ib = np.mean([r["mean"] for n in names for r in ev[n]["in_band"]
                      if abs(r["vmax"] - vmax) < 1e-6])
        plateau = float(np.mean(M[F >= 6]))
        e4 = float(M[np.isclose(F, 4.0)][0])
        e5 = float(M[np.isclose(F, 5.0)][0])
        cliff = e5 / e4
        m = F > 4.0
        x, y = np.log(F[m] / 4.0), np.log(M[m])
        A = np.vstack([x, np.ones_like(x)]).T
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        yh = A @ coef
        r2 = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
        return dict(n_seeds=len(names), inband=float(ib), plateau=plateau,
                    e4=e4, cliff=cliff, alpha=float(coef[0]), r2=float(r2),
                    # two-point slope at the band edge: the exponent the same
                    # data give when the fit is confined to the rise
                    edge_alpha=float(np.log(cliff) / np.log(5.0 / 4.0)),
                    edge_ratio=e4 / plateau if plateau else np.nan)

    arms = {v: arm(v) for v in C.VMAX_LIST}
    arms = {k: v for k, v in arms.items() if v}
    if not arms:
        return None
    # Same inclusion rule the generator applies, re-implemented here: the RATIO
    # eps(5)/eps(4) is only identifiable when the band-edge error is at most half
    # the plateau.  At strong contrast eps(4) is already at plateau level, so the
    # boundary exists but no ratio-based cliff does.
    cliffs = [v["cliff"] for k, v in arms.items()
              if np.isfinite(v["cliff"]) and v["e4"] < 0.5 * v["plateau"]]
    alphas = [v["alpha"] for v in arms.values() if np.isfinite(v["alpha"])]
    rises = [v["edge_ratio"] for v in arms.values() if np.isfinite(v["edge_ratio"])]
    edges = [v["edge_alpha"] for v in arms.values()
             if np.isfinite(v["edge_alpha"])]
    return dict(
        arms=arms,
        InBandLo="%.3f" % min(v["inband"] for v in arms.values()),
        InBandHi="%.3f" % max(v["inband"] for v in arms.values()),
        PlateauLo="%.3f" % min(v["plateau"] for v in arms.values()),
        PlateauHi="%.3f" % max(v["plateau"] for v in arms.values()),
        CliffLo="%.2f" % min(cliffs), CliffHi="%.2f" % max(cliffs),
        AlphaLo="%.2f" % min(alphas), AlphaHi="%.2f" % max(alphas),
        AlphaEdge="%.1f" % max(edges),
        RiseLo="%.2f" % min(rises), RiseHi="%.2f" % max(rises),
    )


def check_rederived(folder):
    got = rederive()
    if got is None:
        bad("no eval data to audit"); return
    txt = open(os.path.join(folder, "gen_macros.tex"), encoding="utf-8").read()
    have = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", txt))
    print("    arm table (re-derived):")
    for k in sorted(got["arms"]):
        a = got["arms"][k]
        print("      v=%.2f  seeds=%d  in=%.3f  plateau=%.3f  r=%.2f  "
              "alpha=%.2f (R2=%.2f)"
              % (k, a["n_seeds"], a["inband"], a["plateau"], a["cliff"],
                 a["alpha"], a["r2"]))
    for k in ("InBandLo", "InBandHi", "PlateauLo", "PlateauHi", "CliffLo",
              "CliffHi", "AlphaLo", "AlphaHi", "AlphaEdge", "RiseLo", "RiseHi"):
        if k not in have:
            bad("macro \\%s missing" % k); continue
        if have[k] == got[k]:
            ok("\\%s = %s (matches independent recomputation)" % (k, have[k]))
        else:
            bad("\\%s = %s on disk but independently recomputed %s"
                % (k, have[k], got[k]))


# ---------------------------------------------------------------------------
def check_pdf(pdf):
    if not os.path.exists(pdf):
        bad("%s missing" % os.path.basename(pdf)); return
    size = os.path.getsize(pdf)
    n = subprocess.run(["pdfinfo", pdf], capture_output=True, text=True,
                       encoding="utf-8", errors="backslashreplace")
    pages = "?"
    for ln in (n.stdout or "").splitlines():
        if ln.lower().startswith("pages:"):
            pages = ln.split(":")[1].strip()
    f = subprocess.run(["pdffonts", pdf], capture_output=True, text=True,
                       encoding="utf-8", errors="backslashreplace")
    fonts = [l for l in (f.stdout or "").splitlines()[2:] if l.strip()]
    # pdffonts columns: name  type  encoding  emb  sub  uni  object ID
    notemb = [l for l in fonts
              if len(l.split()) >= 4 and l.split()[3] == "no"]
    t = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True,
                       encoding="utf-8", errors="backslashreplace")
    chars = len((t.stdout or "").strip())
    ok("%s: %d pages, %d fonts, %d extracted chars, %.0f kB"
       % (os.path.basename(pdf), int(pages) if pages.isdigit() else -1,
          len(fonts), chars, size / 1024.0))
    if pages == "?" or int(pages) < 6:
        bad("%s: suspicious page count %s" % (os.path.basename(pdf), pages))
    if notemb:
        warn("%s: %d font rows not fully embedded" % (os.path.basename(pdf),
                                                      len(notemb)))
    if chars < 5000:
        bad("%s: only %d characters extracted (text/encoding problem?)"
            % (os.path.basename(pdf), chars))


def check_figures(folder):
    """Every figure must have a sane bounding box.

    A degenerate axis (a log scale fed an exact zero, an autoscaled outlier)
    produces a PDF centimetres wide and hundreds of metres tall; LaTeX then
    dies much later with "Dimension too large", far from the cause.
    """
    d = os.path.join(folder, "figs")
    if not os.path.isdir(d):
        bad("%s: figs/ missing" % os.path.basename(folder)); return
    import figures as G
    sizes = G.figure_sizes([d])
    if not sizes:
        bad("%s: no figure PDFs" % os.path.basename(folder)); return
    broke = {k: v for k, v in sizes.items() if v[0] > 1000 or v[1] > 1000}
    tiny = {k: v for k, v in sizes.items() if v[0] < 20 or v[1] < 20}
    if broke:
        bad("%s: degenerate figure bounding boxes: %s"
            % (os.path.basename(folder),
               {k: ("%.0fx%.0f" % v) for k, v in broke.items()}))
    if tiny:
        bad("%s: empty figure bounding boxes: %s"
            % (os.path.basename(folder),
               {k: ("%.0fx%.0f" % v) for k, v in tiny.items()}))
    if not broke and not tiny:
        w = max(v[0] for v in sizes.values())
        h = max(v[1] for v in sizes.values())
        ok("%s: %d figures, largest %.0f x %.0f pt"
           % (os.path.basename(folder), len(sizes), w, h))


def main():
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    for lang in ("en", "cn"):
        folder = os.path.join(C.ROOT, "paper", lang)
        tex = os.path.join(folder, "main.tex")
        bib = os.path.join(folder, "refs.bib")
        print("\n=== %s ===" % lang)
        check_placeholders(tex)
        check_macros_fresh(lang, folder)
        check_macro_names(folder)
        check_selfmath_macros(folder)
        check_macros_resolved(folder)
        check_labels(tex)
        check_bib(tex, bib)
        check_bib_inline_tex(folder)
        check_figures(folder)
        if lang == "en":
            check_rederived(folder)
        # The PDF that matters is the delivered one.  paper/<lang>/main.pdf is
        # a by-product of a local build and went stale once already (29 pages
        # audited against a 42-page delivery), so the submission copy is
        # preferred whenever it exists.
        _delivered = os.path.join(C.ROOT, "submission", "paper_%s" % lang,
                                  "main.pdf")
        check_pdf(_delivered if os.path.exists(_delivered)
                  else os.path.join(folder, "main.pdf"))
    print("\n=== summary ===")
    if FAIL:
        print("  %d FAILURES:" % len(FAIL))
        for f in FAIL:
            print("   - " + f)
        sys.exit(1)
    print("  all hard checks passed")


if __name__ == "__main__":
    main()
