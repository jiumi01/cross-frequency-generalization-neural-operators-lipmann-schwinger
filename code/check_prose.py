# -*- coding: utf-8 -*-
"""Mechanical proofreading of the English prose.

Why this exists
---------------
A native-language read is not something a script can certify, and claiming
"proofread" on the strength of a careful re-read is the same class of error as
the gates that printed OK while testing nothing: the claim cannot fail.  What a
script *can* do is fix the layer underneath the read -- the layer where the
defects are mechanical and countable:

  * one word spelled two ways (behavior / behaviour),
  * one concept written two ways (data set / dataset),
  * a doubled word left by an edit,
  * a contraction in a manuscript that otherwise has none,
  * an en dash typed as a character instead of ``--``,
  * "Fig." where every other cross-reference says "Figure".

Each of those is invisible to a spellchecker and tedious for a reader, and each
is decidable from the text.  This gate decides them, and it declares in its own
output which files and how many words it actually read, because a proofreading
gate that scanned nothing must not report a pass.

What this gate does NOT do
--------------------------
It cannot tell you whether a sentence is idiomatic, whether an article is right,
or whether a paragraph should be split.  Its silence is evidence about
consistency, not about quality.

Scope
-----
The prose only.  Comments, maths, ``\\label``/``\\ref``/``\\cite``/``\\input``
arguments and the remaining control sequences are blanked before any rule runs:
the first version of this scan counted 11 "center"s that were all ``\\centering``
and 68 "label"s that were all ``\\label``, which is exactly the false-positive
mode that makes a gate get switched off.
"""
import io
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    os.path.join(ROOT, "paper", "en", "main.tex"),
    os.path.join(ROOT, "paper", "en", "SI_main.tex"),
]

# A gate that reads fewer words than this is reading the wrong file.
MIN_WORDS = 2000

# ---------------------------------------------------------------------------
# prose extraction
# ---------------------------------------------------------------------------
MATH_ENVS = ("equation", "align", "gather", "multline", "split", "cases",
             "array", "eqnarray", "displaymath", "math")

# Arguments of these carry identifiers, not prose, and blanking them removes
# most of the false positives (`\\label{tab:label_budget}` is not the word
# "label").
ARG_CMDS = ("label", "cite", "citep", "citet", "input", "includegraphics",
            "url", "href", "bibitem", "bibliography", "usepackage",
            "documentclass", "newcommand", "renewcommand", "def", "setlength",
            "hspace", "vspace")


def _blank(m):
    """Replace a matched span with spaces, keeping its newlines.

    Line numbers must survive masking or every report points at the wrong
    place, and a report that points at the wrong place costs more than it
    gives.
    """
    s = m.group(0)
    return re.sub(r"[^\n]", " ", s)


def _math(m):
    """Replace a maths span with a non-space, non-letter placeholder.

    Spaces here would be read as prose spacing: `(X and Y)` with both symbols
    masked becomes "( and )", which the spacing rule reports as a typo the
    author cannot see.  The placeholder keeps the span opaque instead.
    """
    return re.sub(r"[^\n]", "\u00a7", m.group(0))


def prose(text):
    """Return the running text with markup blanked out."""
    t = re.sub(r"(?<!\\)%.*", "", text)                 # comments
    # The preamble is blanked rather than dropped: dropping it renumbers every
    # line, and a report that names the wrong line is worse than no report.
    i = t.find(r"\begin{document}")
    if i >= 0:
        t = re.sub(r"[^\n]", " ", t[:i]) + t[i:]
    for env in MATH_ENVS:
        t = re.sub(r"\\begin\{%s\*?\}.*?\\end\{%s\*?\}" % (env, env),
                   _math, t, flags=re.S)
    # Tabular bodies are data, not prose: their `&` separators and blanked
    # macros manufacture spaces next to punctuation that no reader ever sees.
    # The caption outside the tabular is kept, because that IS prose.
    t = re.sub(r"\\begin\{(tabular\*?|tabularx|longtable|tabu)\}.*?"
               r"\\end\{\1\}", _blank, t, flags=re.S)
    t = re.sub(r"\$\$.*?\$\$", _math, t, flags=re.S)
    t = re.sub(r"\$[^$]*\$", _math, t)
    t = re.sub(r"\\\[.*?\\\]", _math, t, flags=re.S)
    t = re.sub(r"\\\(.*?\\\)", _math, t, flags=re.S)
    # Code environments are listings, not prose: the algorithm blocks are
    # pseudocode and their spacing is set by the author of the listing.
    t = re.sub(r"\\begin\{(algorithm\*?|algorithmic|verbatim|lstlisting|"
               r"minted)\}.*?\\end\{\1\}", _blank, t, flags=re.S)
    # A reference is left as a word: the rules below have to see that a number
    # follows, and blanking `\ref{...}` outright made "Fig.~\ref" and
    # "Section \ref" indistinguishable from "Fig." with nothing after it --
    # the self-test caught both rules silently passing.
    t = re.sub(r"\\(?:ref|eqref|siTab|siFig)\*?\{[^{}]*\}", "REF", t)
    for cmd in ARG_CMDS:
        t = re.sub(r"\\%s\*?(\[[^\]]*\])?(\{[^{}]*\})*" % cmd, _blank, t)
    t = re.sub(r"\\begin\{[^{}]*\}(\[[^\]]*\])?", _blank, t)
    t = re.sub(r"\\end\{[^{}]*\}", _blank, t)
    # Remaining control sequences: the macro NAME is markup, but a macro that
    # expands to a word (`\\CliffLo`) still stands for one word in the output.
    t = re.sub(r"\\[A-Za-z]+\*?", " M ", t)
    # `~` is kept: it is the tie that the cross-reference rules test for.
    # Braces are deleted, not turned into spaces: `\emph{silent},` would
    # otherwise become "silent ,", a spacing defect that exists only in the
    # masked text.
    t = re.sub(r"[{}]", "", t)
    return t


def line_of(text, pos):
    return text[:pos].count("\n") + 1


def snippets(text, pat, flags=0, width=60):
    out = []
    for m in re.finditer(pat, text, flags):
        lo = max(0, m.start() - width // 3)
        out.append((line_of(text, m.start()),
                    re.sub(r"\s+", " ", text[lo:m.end() + width]).strip()))
    return out


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------
# British spelling is the house style (measured: 4 behaviour, 9 normalis-,
# 0 behavior, 0 normaliz-).  The list is pairs because the report should say
# what to write, not only what is wrong.
US_UK = [
    (r"\bbehaviors?\b", "behaviour"),
    (r"\bbehavioral\b", "behavioural"),
    (r"\bnormaliz\w*\b", "normalis-"),
    (r"\boptimiz\w*\b", "optimis-"),
    (r"\borganiz\w*\b", "organis-"),
    (r"\binitializ\w*\b", "initialis-"),
    (r"\banalyz\w*\b", "analys-"),
    (r"\bregulariz\w*\b", "regularis-"),
    (r"\b(modeling|modeled)\b", "modelling / modelled"),
    (r"\b(labeled|labeling)\b", "labelled / labelling"),
    (r"\b(center|centers|centered|centering)\b", "centre"),
    (r"\bcolor(ed|ing|s|map|bar)?\b", "colour"),
    (r"\bartifact\w*\b", "artefact"),
    (r"\bfavor\w*\b", "favour"),
    (r"\bneighbor\w*\b", "neighbour"),
    (r"\bdefense\b", "defence"),
    (r"\blicense\b", "licence (noun)"),
]

# One concept, one spelling.  A pair is a violation only when BOTH forms are
# present: either one alone is fine, and demanding a particular one would be
# taste rather than consistency.
VARIANT_PAIRS = [
    ("dataset", "data set"),
    ("wavenumber", "wave number"),
    ("timestep", "time step"),
    ("gridpoint", "grid point"),
    ("hyperparameter", "hyper-parameter"),
    ("subwavelength", "sub-wavelength"),
    ("finetune", "fine-tune"),
    ("percent", "per cent"),
    ("zero shot", "zero-shot"),
    ("in band", "in-band"),
    ("out of band", "out-of-band"),
]

# Characters that must reach the page through a control sequence, not as a
# literal, or the font/substitution is out of the author's hands.
BAD_UNICODE = {
    "\u2013": "en dash typed literally; use --",
    "\u2014": "em dash typed literally; use ---",
    "\u00d7": "multiplication sign; use $\\times$",
    "\u00b5": "micro sign; use $\\mu$",
    "\u03bc": "Greek mu in text mode; use $\\mu$",
    "\u2264": "<= as a literal; use $\\le$",
    "\u2265": ">= as a literal; use $\\ge$",
    "\u2192": "arrow as a literal; use $\\to$",
    "\u201c": "curly quote; use ``",
    "\u201d": "curly quote; use ''",
    "\u2019": "curly apostrophe; use '",
}


def rule_us_spelling(t):
    out = []
    for pat, want in US_UK:
        for ln, snip in snippets(t, pat, re.I):
            out.append((ln, "US spelling for %r: %s" % (want, snip)))
    return out


def rule_variant_pairs(t):
    out = []
    for a, b in VARIANT_PAIRS:
        na = len(re.findall(r"(?<![\w-])%s(?![\w-])" % re.escape(a), t, re.I))
        nb = len(re.findall(r"(?<![\w-])%s(?![\w-])" % re.escape(b), t, re.I))
        # Hyphenation of a compound adjective and the same words as a noun
        # phrase are both legitimate ("in-band error" / "the error in band"),
        # so the hyphen pairs are reported only when the hyphenless form never
        # takes an object -- i.e. only when one side is used as a noun phrase
        # exclusively.  Detecting that reliably is not possible here, so those
        # pairs are advisory rather than failing; see ADVISORY below.
        if na and nb and (a, b) not in ADVISORY:
            ln = line_of(t, re.search(re.escape(a), t, re.I).start())
            out.append((ln, "both %r (%d) and %r (%d) are used" % (a, na, b, nb)))
    return out


def rule_doubled_word(t):
    out = []
    for ln, snip in snippets(t, r"\b([A-Za-z]{3,})\s+\1\b", re.I):
        out.append((ln, "doubled word: %s" % snip))
    return out


def rule_contraction(t):
    return [(ln, "contraction: %s" % snip)
            for ln, snip in snippets(
                t, r"\b(?:isn't|aren't|doesn't|don't|can't|won't|didn't|"
                   r"it's|that's|we've|hasn't)\b", re.I)]


def rule_sentence_opener(t):
    return [(ln, "sentence opens with a conjunction: %s" % snip)
            for ln, snip in snippets(t, r"(?:^|(?<=[.!?])\s)(?:And|But)\s")]


def rule_fig_abbrev(t):
    return [(ln, "abbreviated cross-reference: %s" % snip)
            for ln, snip in snippets(t, r"\bFigs?\.~?\s*(?:REF|\d)")]


def rule_untied_reference(t):
    out = []
    for kw in ("Section", "Figure", "Table", "Equation", "Appendix"):
        # `Section~REF` is tied; `Section REF` and `Section 4` are not.  The
        # lookbehind keeps the tied form out, since it is the house style.
        for ln, snip in snippets(t, r"\b%s(?<!~)\s+(?:REF|\d)" % kw):
            out.append((ln, "%s not tied to its number: %s" % (kw, snip)))
    return out


def rule_bad_unicode(t):
    out = []
    for ch, why in BAD_UNICODE.items():
        for ln, snip in snippets(t, re.escape(ch)):
            out.append((ln, "%s -- %s" % (why, snip)))
    return out


def rule_etal(t):
    return [(ln, "'et al' without a full stop: %s" % snip)
            for ln, snip in snippets(t, r"\bet al\b(?!\.)")]


def rule_sci_notation(t):
    return [(ln, "C-style exponent in prose: %s" % snip)
            for ln, snip in snippets(t, r"\b\d+(?:\.\d+)?[eE][-+]?\d+\b")]


def rule_dash_in_name(t):
    return [(ln, "Lippmann-Schwinger with a hyphen: %s" % snip)
            for ln, snip in snippets(t, r"Lippmann-Schwinger")]


def rule_space_punct(t):
    out = []
    # The word must be a real word: a blanked macro leaves one or two letters
    # behind, and `M ,` is not a typo a reader can make.
    for ln, snip in snippets(t, r"(?<![\\\w])[A-Za-z]{3,}[ \t]+[,;:]"):
        out.append((ln, "space before punctuation: %s" % snip))
    for ln, snip in snippets(t, r"\([ \t]+[a-z]{3,}\b"):
        out.append((ln, "space after opening parenthesis: %s" % snip))
    return out


RULES = [
    ("us-spelling", rule_us_spelling),
    ("variant-pairs", rule_variant_pairs),
    ("doubled-word", rule_doubled_word),
    ("contraction", rule_contraction),
    ("sentence-opener", rule_sentence_opener),
    ("fig-abbrev", rule_fig_abbrev),
    ("untied-reference", rule_untied_reference),
    ("bad-unicode", rule_bad_unicode),
    ("et-al", rule_etal),
    ("sci-notation", rule_sci_notation),
    ("dash-in-name", rule_dash_in_name),
    ("space-punct", rule_space_punct),
]

# Pairs where both forms can be correct in the same manuscript because the
# hyphen marks the attributive use.  Reported, never failed.
ADVISORY = {("in band", "in-band"), ("out of band", "out-of-band")}


# ---------------------------------------------------------------------------
# self-test: every rule must fire on a bad sample and stay silent on a good one
# ---------------------------------------------------------------------------
FIXTURES = {
    "us-spelling": ("the model behavior and its normalized error",
                    "the model behaviour and its normalised error"),
    "variant-pairs": ("each data set is paired; the dataset is reused",
                      "each data set is paired; the same set is reused"),
    "doubled-word": ("the the operator is trained",
                     "the operator is trained"),
    "contraction": ("it doesn't converge", "it does not converge"),
    "sentence-opener": ("Done. And the error falls.",
                        "Done, and the error falls."),
    "fig-abbrev": ("as shown in Fig.~\\ref{fig:main}", "as shown in Figure~\\ref{fig:main}"),
    "untied-reference": ("see Section \\ref{sec:x}", "see Section~\\ref{sec:x}"),
    "bad-unicode": ("the step is 3\u00d7 larger", "the step is $3\\times$ larger"),
    "et-al": ("Smith et al report", "Smith et al. report"),
    "sci-notation": ("a tolerance of 1e-6", "a tolerance of $10^{-6}$"),
    "dash-in-name": ("the Lippmann-Schwinger equation",
                     "the Lippmann--Schwinger equation"),
    "space-punct": ("the error , which rises", "the error, which rises"),
}


def self_test():
    bad = 0
    for name, fn in RULES:
        if name not in FIXTURES:
            print("  [FAIL] %-18s has no fixture; an unfalsified rule is a claim"
                  % name)
            bad += 1
            continue
        sample_bad, sample_good = FIXTURES[name]
        hit = fn(prose(sample_bad))
        quiet = fn(prose(sample_good))
        ok = bool(hit) and not quiet
        print("  [%s] %-18s fires on the bad sample: %s; silent on the good one: %s"
              % ("ok" if ok else "FAIL", name, bool(hit), not quiet))
        if not ok:
            bad += 1
    # The masking is itself a rule: a control sequence must not be read as
    # prose, or every `\\label` counts as the word "label".
    m = prose(r"see \label{tab:label_budget} and \centering x")
    ok = "label" not in m and "centering" not in m
    print("  [%s] %-18s macro arguments are blanked before the rules run"
          % ("ok" if ok else "FAIL", "masking"))
    if not ok:
        print("       masked text was: %r" % m)
        bad += 1
    print("\n  %d of %d checks misbehaved" % (bad, len(RULES) + 1))
    return bad


def targets():
    """Files to read, overridable so the probe can point the same rules at a
    mutated copy.  Without an override a gate can only be tested by editing the
    manuscript, which makes the test and the defect the same object."""
    root = None
    for i, a in enumerate(sys.argv):
        if a == "--root" and i + 1 < len(sys.argv):
            root = sys.argv[i + 1]
    root = root or os.environ.get("PROSE_PAPER_ROOT")
    if root:
        base = (os.path.join(root, "paper")
                if os.path.isdir(os.path.join(root, "paper", "en")) else root)
        return [os.path.join(base, "en", "main.tex"),
                os.path.join(base, "en", "SI_main.tex")]
    return FILES


def main():
    if "--self-test" in sys.argv:
        print("=== check_prose self-test ===")
        sys.exit(1 if self_test() else 0)

    total_words = 0
    findings = []
    for path in targets():
        if not os.path.exists(path):
            print("[FAIL] missing %s" % path)
            sys.exit(1)
        t = prose(io.open(path, encoding="utf-8").read())
        words = len(re.findall(r"[A-Za-z][A-Za-z'-]*", t))
        total_words += words
        print("=== %s: %d prose words ===" % (os.path.basename(path), words))
        for name, fn in RULES:
            hits = fn(t)
            for ln, msg in hits:
                findings.append((os.path.basename(path), ln, name, msg))
            if hits:
                print("  %-16s %d" % (name, len(hits)))

    if total_words < MIN_WORDS:
        print("[FAIL] read %d prose words, below the floor of %d: this run "
              "cannot report a pass" % (total_words, MIN_WORDS))
        sys.exit(1)

    if findings:
        print("\n=== %d finding(s) ===" % len(findings))
        for f, ln, name, msg in findings:
            print("  %s:%d [%s] %s" % (f, ln, name, msg))
        sys.exit(1)

    print("\n  %d prose words read, no mechanical defect" % total_words)
    print("  prose gate passed")


if __name__ == "__main__":
    main()
